"""On-policy GRPO from the SFT checkpoint, with the shaped reward in pb/reward.py.

Per step, for each sampled train CVE: K rollouts with the current policy (guardrails on, context trimmed),
grade the (test-stripped) patch with PatchEval's fix-run.sh for the 1.0 objective, add dense milestone
credit (which pays for a successful str_replace and submit), group-normalise rewards -> advantage, and
update the policy LoRA by advantage-weighted NLL on assistant tokens + beta * KL(policy||sft) (k3).

Starts from the SFT adapter (trainable 'policy'); a frozen copy ('sft') is the KL reference. Saves the
policy adapter every step to runs/<name>/adapter/policy.
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from pb.agent import run_episode, trim_context
from pb.data import split
from pb.env import PatchEnv
from pb.qwen import CHAT_TOOLS, parse_tool_calls, pick_dtype
from pb.reward import codebleu_reward, gradeable_patch, milestones, reward
from pb.sandbox import grade, pull, remove_image

ASSISTANT, THINK, END = "<|im_start|>assistant\n", "<think>\n\n</think>\n\n", "<|im_end|>"


def tokenize(tok, messages, max_len=22000):
    """Assistant-only labels on the trimmed conversation (matches inference and SFT)."""
    messages = trim_context(messages)
    text = tok.apply_chat_template(messages, tools=CHAT_TOOLS, tokenize=False, enable_thinking=False)
    head, *turns = text.split(ASSISTANT)
    text = head + "".join(ASSISTANT + (t if t.startswith(THINK) else THINK + t) for t in turns)
    spans, pos = [], 0
    while (a := text.find(ASSISTANT + THINK, pos)) != -1:
        a += len(ASSISTANT + THINK)
        b = text.index(END, a) + len(END)
        spans.append((a, b)); pos = b
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    labels, si = [], 0
    for tid, (s, _) in zip(enc.input_ids, enc.offset_mapping):
        while si < len(spans) and s >= spans[si][1]:
            si += 1
        labels.append(tid if si < len(spans) and spans[si][0] <= s < spans[si][1] else -100)
    return enc.input_ids[:max_len], labels[:max_len]


class Policy:
    """base + trainable 'policy' adapter + frozen 'sft' adapter (KL reference). act() drives rollouts."""

    def __init__(self, model_name, sft_adapter=None, max_context=22000, max_new_tokens=1024):
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dtype = pick_dtype()  # installs the Turing SDPA fix
        self.tok = AutoTokenizer.from_pretrained(model_name)
        # With >1 visible GPU, split the model across them (naive model parallelism via accelerate hooks):
        # weights and, crucially, the lm_head vocab projection's activations are spread over both cards,
        # which is what blew the 11 GB budget on a single 2080 Ti. Cap each GPU below its capacity so there
        # is headroom for activations; cap GPU 0 lower so the heavy lm_head lands on a less-loaded device.
        n_gpu = torch.cuda.device_count()
        if n_gpu > 1:
            free0 = torch.cuda.get_device_properties(0).total_memory / 2**30
            mm = {0: f"{free0 * 0.45:.0f}GiB", **{i: f"{torch.cuda.get_device_properties(i).total_memory / 2**30 * 0.9:.0f}GiB" for i in range(1, n_gpu)}}
            device_map = "auto"
        else:
            mm, device_map = None, "cuda"
        base = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, device_map=device_map,
                                                    max_memory=mm, attn_implementation="sdpa")
        if sft_adapter:  # start from the SFT checkpoint; its frozen copy is the KL reference
            self.model = PeftModel.from_pretrained(base, sft_adapter, adapter_name="policy", is_trainable=True)
            self.model.load_adapter(sft_adapter, adapter_name="sft")
            self.ref = "sft"
        else:  # start from the base model: a fresh LoRA, with the base (adapter disabled) as the KL reference
            cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
            self.model = get_peft_model(base, cfg, adapter_name="policy")
            self.ref = None  # disable_adapter() -> base model
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.enable_input_require_grads()
        self.max_context, self.max_new_tokens = max_context, max_new_tokens

    @torch.no_grad()
    def act(self, messages):
        self.model.eval(); self.model.set_adapter("policy")
        enc = self.tok.apply_chat_template(messages, tools=CHAT_TOOLS, add_generation_prompt=True,
                                           enable_thinking=False, return_tensors="pt", return_dict=True)
        ids = enc["input_ids"].to(self.model.device)
        if ids.shape[1] + self.max_new_tokens > self.max_context:
            return "", []
        out = self.model.generate(ids, max_new_tokens=self.max_new_tokens, do_sample=True,
                                  temperature=1.0, top_p=1.0, top_k=0, pad_token_id=self.tok.eos_token_id)
        text = self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        del out, ids, enc; torch.cuda.empty_cache()
        return parse_tool_calls(text)


def token_logprobs(model, ids, labels, chunk=1024):
    """Gathered log-probs of the assistant tokens. The per-chunk lm_head projection ([chunk, vocab] in
    fp32) is checkpointed so its activations are recomputed in backward instead of retained -- without
    this the grad path accumulates one big logits graph per chunk over the whole rollout and OOMs. All
    tensors are placed explicitly so this is correct whether the model is on one device or split across
    GPUs (device_map), where the body and the lm_head can live on different cards."""
    core, head = model.base_model.model.model, model.base_model.model.lm_head
    emb_dev = core.embed_tokens.weight.device
    hidden = core(input_ids=ids.to(emb_dev)).last_hidden_state[:, :-1]
    tgt = labels[:, 1:].to(hidden.device); keep = tgt != -100
    h, y = hidden[keep], tgt[keep]
    hd = head.weight.device

    def proj(hc, yc):  # projection + gather for one chunk, on the lm_head's device
        lp = F.log_softmax(head(hc.to(hd)).float(), -1)
        return torch.gather(lp, 1, yc.to(hd)[:, None]).squeeze(1)

    out = [checkpoint(proj, h[i:i + chunk], y[i:i + chunk], use_reentrant=False)
           for i in range(0, len(y), chunk)]
    return torch.cat(out) if out else h.new_zeros(0, device=hd)


def grpo_step(policy, batch, beta, opt):
    policy.model.train()
    ntok = max(sum((l != -100).sum().item() for _, l, _ in batch), 1)
    stats = {"pg": 0.0, "kl": 0.0}
    for ids, labels, adv in batch:
        ids, labels = ids.cuda(), labels.cuda()
        policy.model.set_adapter("policy")
        logp = token_logprobs(policy.model, ids, labels)
        with torch.no_grad():
            if policy.ref == "sft":
                policy.model.set_adapter("sft")
                logp_ref = token_logprobs(policy.model, ids, labels)
                policy.model.set_adapter("policy")
            else:  # KL reference is the base model
                with policy.model.disable_adapter():
                    logp_ref = token_logprobs(policy.model, ids, labels)
        d = logp_ref - logp
        kl = torch.exp(d) - d - 1
        ((-adv * logp + beta * kl).sum() / ntok).backward()
        stats["pg"] += (-adv * logp).sum().item() / ntok
        stats["kl"] += kl.sum().item() / ntok
    torch.nn.utils.clip_grad_norm_([p for p in policy.model.parameters() if p.requires_grad], 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    return stats


def rollout(policy, task, max_turns, reward_mode="codebleu"):
    with PatchEnv(task) as env:
        ep = run_episode(policy, env, task, max_turns=max_turns)
        patch = env.diff()
        image, workdir, reference = task["image_url"], env.workdir, env.reference_patch
    ep["patch"] = patch
    m = milestones(ep, task)
    if reward_mode == "codebleu":  # dense, no Docker grading; reference is train-only
        r, resolved = codebleu_reward(patch, reference), False
    else:  # sparse shaped: PatchEval fix-run.sh objective + milestone credit
        gp = gradeable_patch(patch)
        resolved, _ = grade(image, gp, workdir) if gp else (False, "")
        r = reward(resolved, m)
    return {**ep, "reward": r, "resolved": resolved, "milestones": m}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--sft-adapter", default=None, help="start from this SFT adapter; omit to start from the base model")
    p.add_argument("--name", required=True)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--cves-per-step", type=int, default=2)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--max-turns", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--beta", type=float, default=0.02)
    p.add_argument("--reward", choices=["shaped", "codebleu"], default="shaped")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = Path("runs") / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    log = (out / "grpo_log.jsonl").open("a")

    tasks = split()["train"]
    rng = random.Random(args.seed)
    policy = Policy(args.model, args.sft_adapter)
    opt = torch.optim.AdamW([q for q in policy.model.parameters() if q.requires_grad], lr=args.lr)

    for step in range(1, args.steps + 1):
        t0 = time.time()
        batch, mean_r, n_res, acts = [], [], 0, {"valid_edit": 0, "submitted": 0}
        for task in rng.sample(tasks, args.cves_per_step):
            rolls = []
            try:
                pull(task["image_url"])
                for _ in range(args.k):
                    try:
                        rolls.append(rollout(policy, task, args.max_turns, args.reward))
                    except Exception as e:
                        print(f"warning: rollout failed {task['cve_id']}: {e!r}", flush=True)
            finally:
                remove_image(task["image_url"])
            if len(rolls) < 2:
                continue
            rs = torch.tensor([r["reward"] for r in rolls], dtype=torch.float)
            mean_r.append(rs.mean().item())
            n_res += sum(r["resolved"] for r in rolls)
            acts["valid_edit"] += sum(r["milestones"]["valid_edit"] for r in rolls)
            acts["submitted"] += sum(r["milestones"]["submitted"] for r in rolls)
            if rs.std() < 1e-5:
                continue
            adv = (rs - rs.mean()) / (rs.std() + 1e-4)
            for r, a in zip(rolls, adv):
                ids, labels = tokenize(policy.tok, r["messages"])
                if any(x != -100 for x in labels):
                    batch.append((torch.tensor([ids]), torch.tensor([labels]), a.item()))
        stats = grpo_step(policy, batch, args.beta, opt) if batch else {"pg": 0.0, "kl": 0.0}
        rec = {"step": step, "mean_reward": round(sum(mean_r) / max(len(mean_r), 1), 4), "resolved": n_res,
               "edits": acts["valid_edit"], "submits": acts["submitted"], "n_updates": len(batch), "reward_mode": args.reward,
               **{k: round(v, 4) for k, v in stats.items()}, "seconds": round(time.time() - t0)}
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n"); log.flush()
        policy.model.set_adapter("policy")
        policy.model.save_pretrained(out / "adapter", selected_adapters=["policy"])
    print(f"saved {out / 'adapter'}")


if __name__ == "__main__":
    main()
