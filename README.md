# SLM for multi-turn security patching

A minimal, end-to-end pipeline that trains a sub-4B model (Qwen3-1.7B) to **fix security
vulnerabilities** in real repositories through multi-turn tool use, and closes the loop
**SFT → RL → eval** on the [PatchEval-Verified](https://github.com/patcheval-cve) benchmark.

For the design, reward shaping, results, and trade-offs, see **[WRITEUP.md](WRITEUP.md)**. This file is
just setup and how to run the three stages.

---

## What's here

```
pb/
  agent.py      multi-turn agent loop + system prompt + context trimming (shared by every policy)
  env.py        PatchEnv: the tools the agent calls (bash, view_file, replace_lines, submit) + guardrails
  sandbox.py    rootless udocker container; grade() = exploit blocked (fix-run.sh) AND tests pass (unit_test.sh)
  qwen.py       Qwen3-1.7B policy (native <tool_call> format; Turing SDPA fix)
  teacher.py    frontier teacher via Azure OpenAI Responses API (SFT-data generation only)
  data.py       PatchEval-Verified loader + fixed train/eval split (eval = 30, 10 per language)
  reward.py     shaped reward (increasing milestones) + anti-reward-hacking rules
  sft.py        LoRA SFT (Unsloth) — behavior cloning on successful teacher trajectories
  grpo.py       on-policy GRPO from the SFT checkpoint, with the shaped reward
collect.py      run a policy on a split, grade every attempt, write traces + summary (shardable)
run_workers.sh  run collect.py as N parallel workers (one udocker dir each)
runs/           the three reported stages (see "Results" below)
third_party/    PatchEval-Verified dataset + grading scripts (dependency)
```

---

## Setup

### 1. Two Python environments

Unsloth (used only by SFT) pins its own `transformers`, so SFT lives in a separate env from everything
else. Both are Python 3.11 with a CUDA 12.x GPU.

```bash
# main env: inference, teacher collection, GRPO, eval, grading
conda create -n patching python=3.11 -y && conda activate patching
pip install -r requirements.txt

# SFT-only env
conda create -n patching-unsloth python=3.11 -y && conda activate patching-unsloth
pip install -r requirements-sft.txt
```

> `torch` in both files is the `+cu128` build. If your driver needs a different CUDA, install the
> matching torch wheel first (`pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/<cuxxx>`).

### 2. Dataset

The benchmark lives under `third_party/PatchEval/`. If it isn't already populated:

```bash
git clone https://github.com/patcheval-cve/PatchEval third_party/PatchEval
```

`pb/data.py` reads `third_party/PatchEval/patcheval/datasets/patcheval_verified.json` and produces a
fixed split (eval = 30 CVEs, 10 per language; train = the rest). No network is needed at run time; the
CVE Docker images are pulled on demand by udocker.

### 3. Teacher credentials (only needed to regenerate SFT data)

Create `.env` in the repo root (already git-ignored):

```
AZURE_API_KEY=<your key>
AZURE_ENDPOINT=<https://.../openai/v1>
```

You do **not** need this to run the baseline, SFT, GRPO, or eval — only to re-collect teacher traces.

### 4. Environment variables used by every run

- `PYTHONPATH=.` — so `import pb...` resolves (run from the repo root).
- `CUDA_DEVICE_ORDER=PCI_BUS_ID` — makes `CUDA_VISIBLE_DEVICES` indices match `nvidia-smi`.
- `UDOCKER_DIR=<dir with ~6 GB free per worker>` — container storage. A RAM disk
  (`/dev/shm/$USER/udocker`) is ~10× faster than NFS for unpacking images.

The examples below export these inline. The GPU code auto-detects Turing cards (e.g. 2080 Ti) and applies
an SDPA memory fix; bf16 cards work unchanged.

---

## Running the three stages

All commands run from the repo root. Each writes to `runs/<name>/` (`traces.jsonl` = per-episode records,
`summary.json` = scores). Outputs go to a **fresh** `--name`; the reported runs already occupy
`baseline_eval`, `sft_adapter`, `sft_eval`, `grpo_adapter`, `grpo_eval`.

```bash
conda activate patching
export PYTHONPATH=. CUDA_DEVICE_ORDER=PCI_BUS_ID
export UDOCKER_DIR=/dev/shm/$USER/udocker && mkdir -p "$UDOCKER_DIR"
```

### 1. Baseline — the untrained Qwen3-1.7B

```bash
CUDA_VISIBLE_DEVICES=0 \
  python collect.py --policy qwen --split eval --name my_baseline_eval --attempts 1
```

### 2. SFT — behavior-clone the teacher's successful trajectories

**(a) Collect teacher trajectories** (optional — `runs/teacher_traces/` is already provided). Needs
`.env`. Runs 10 parallel workers and keeps grading until it has enough successes:

```bash
UDOCKER_ROOT=/dev/shm/$USER/udocker \
  bash run_workers.sh 10 --policy teacher --split train --limit 40 \
       --attempts 4 --name my_teacher_traces --reasoning low
```

**(b) Train the SFT adapter** — in the Unsloth env; `decisive-weight` upweights the loss on
`replace_lines`/`submit` turns so the student learns to commit, not just explore:

```bash
conda activate patching-unsloth
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. CUDA_DEVICE_ORDER=PCI_BUS_ID \
  python pb/sft.py --data runs/teacher_traces/traces.jsonl \
       --name my_sft_adapter --epochs 6 --decisive-weight 4
```

**(c) Evaluate it** — back in the main env:

```bash
conda activate patching
CUDA_VISIBLE_DEVICES=0 \
  python collect.py --policy qwen --split eval --name my_sft_eval \
       --adapter runs/my_sft_adapter/adapter --attempts 1
```

### 3. GRPO — on-policy RL from the SFT checkpoint

**(a) Train** with the shaped reward (grades each rollout with PatchEval; saves the policy adapter every
step to `runs/<name>/adapter/policy`):

```bash
CUDA_VISIBLE_DEVICES=0 \
  python pb/grpo.py --sft-adapter runs/sft_adapter/adapter --name my_grpo_adapter \
       --steps 20 --cves-per-step 2 --k 4 --reward shaped
```

**(b) Evaluate it** — note the adapter path ends in `/policy`:

```bash
CUDA_VISIBLE_DEVICES=0 \
  python collect.py --policy qwen --split eval --name my_grpo_eval \
       --adapter runs/my_grpo_adapter/adapter/policy --attempts 1
```

### Faster eval (optional): shard across GPUs

`collect.py` takes `--shard i/N`; run N processes that append to the same `runs/<name>/`:

```bash
CUDA_VISIBLE_DEVICES=0 UDOCKER_DIR=/dev/shm/$USER/ud0 python collect.py ... --shard 0/2 &
CUDA_VISIBLE_DEVICES=2 UDOCKER_DIR=/dev/shm/$USER/ud1 python collect.py ... --shard 1/2 &
wait
```

---

## Results

The three reported stages are in `runs/`:

| Folder | Stage |
|---|---|
| `runs/baseline_eval/` | untrained Qwen3-1.7B |
| `runs/teacher_traces/` | the 32 teacher trajectories SFT was trained on (14 CVEs: Go 11, Python 9, JavaScript 12) |
| `runs/sft_adapter/` + `runs/sft_eval/` | SFT model + its eval |
| `runs/grpo_adapter/` + `runs/grpo_eval/` | GRPO model (14 steps) + its eval |

Read any stage's score with:

```bash
cat runs/<name>/summary.json
```

The headline comparison and interpretation are in **[WRITEUP.md](WRITEUP.md)**.
