"""LoRA SFT (Unsloth) of Qwen3-1.7B on the teacher's successful trajectories.

Behaviour cloning: next-token loss on the assistant turns only (tool observations and the task are
context, masked out), in the exact chat format pb/qwen.py uses at inference. Writes runs/<name>/adapter.

  python pb/sft.py --data runs/teacher_train/traces.jsonl --name sft_qwen3_1.7b --epochs 6
"""
import unsloth  # noqa: F401  (must precede transformers)

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from pb.qwen import CHAT_TOOLS, pick_dtype

ASSISTANT = "<|im_start|>assistant\n"
THINK = "<think>\n\n</think>\n\n"
END = "<|im_end|>"


def load_successful(path):
    return [json.loads(l)["messages"] for l in open(path) if json.loads(l)["resolved"]]


DECISIVE = {"replace_lines", "submit"}  # the rare, decisive actions the student under-emits


def tokenize(tok, messages, max_len, decisive_weight=4.0):
    """(input_ids, labels, weights). labels == -100 outside assistant turns; weights upweight the loss
    on turns that call a decisive action (replace_lines/submit) so the student learns to commit, not
    just explore. Empty think block re-inserted and context trimmed, matching inference."""
    from pb.agent import trim_context
    messages = trim_context(messages)
    turn_w = [decisive_weight if any(tc["function"]["name"] in DECISIVE for tc in m.get("tool_calls", [])) else 1.0
              for m in messages if m["role"] == "assistant"]
    text = tok.apply_chat_template(messages, tools=CHAT_TOOLS, tokenize=False, enable_thinking=False)
    head, *turns = text.split(ASSISTANT)
    text = head + "".join(ASSISTANT + (t if t.startswith(THINK) else THINK + t) for t in turns)
    spans, pos = [], 0
    while (a := text.find(ASSISTANT + THINK, pos)) != -1:
        a += len(ASSISTANT + THINK)
        b = text.index(END, a) + len(END)
        spans.append((a, b))
        pos = b
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    labels, weights, si = [], [], 0
    for tid, (s, _) in zip(enc.input_ids, enc.offset_mapping):
        while si < len(spans) and s >= spans[si][1]:
            si += 1
        if si < len(spans) and spans[si][0] <= s < spans[si][1]:
            labels.append(tid)
            weights.append(turn_w[si] if si < len(turn_w) else 1.0)
        else:
            labels.append(-100)
            weights.append(0.0)
    return enc.input_ids[:max_len], labels[:max_len], weights[:max_len]


def assistant_loss(model, ids, labels, weights, chunk=1024):
    """Weighted loss only where labels != -100, in checkpointed chunks to bound vocab-logit memory.
    weights upweight decisive-action turns; normalised by total weight so the scale stays comparable."""
    hidden = model.base_model.model.model(input_ids=ids).last_hidden_state[:, :-1]
    tgt, w = labels[:, 1:], weights[:, 1:]
    keep = tgt != -100
    h, y, wk = hidden[keep], tgt[keep], w[keep]
    head = model.base_model.model.lm_head
    ce = lambda a, b, wi: (F.cross_entropy(head(a).float(), b, reduction="none") * wi).sum()
    total = sum(checkpoint(ce, h[i:i + chunk], y[i:i + chunk], wk[i:i + chunk], use_reentrant=False)
                for i in range(0, len(y), chunk))
    return total / wk.sum().clamp(min=1.0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--max-len", type=int, default=20000)
    p.add_argument("--decisive-weight", type=float, default=4.0, help="loss weight on replace_lines/submit turns")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = Path("runs") / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    log = (out / "sft_log.jsonl").open("a")  # persist the training log (like grpo.py), not just stdout
    torch.manual_seed(args.seed)

    from unsloth import FastLanguageModel
    dtype = pick_dtype()
    model, tok = FastLanguageModel.from_pretrained(args.model, max_seq_length=args.max_len, dtype=dtype, load_in_4bit=False)
    model = FastLanguageModel.get_peft_model(
        model, r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=args.seed)

    data = [tokenize(tok, m, args.max_len, args.decisive_weight) for m in load_successful(args.data)]
    data = [d for d in data if any(x != -100 for x in d[1])]
    n_sup = sum(sum(x != -100 for x in l) for _, l, _ in data)
    n_dec = sum(sum(w > 1 for w in wt) for _, _, wt in data)
    print(f"{len(data)} trajectories, {sum(len(i) for i, _, _ in data)} tokens, {n_sup} supervised "
          f"({n_dec} on decisive turns, weight {args.decisive_weight})", flush=True)

    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)
    total = math.ceil(len(data) * args.epochs / args.grad_accum)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / max(1, int(0.1 * total)), 1.0))
    bf16 = dtype == torch.bfloat16
    scaler = torch.amp.GradScaler(enabled=not bf16)
    order = [i for _ in range(args.epochs) for i in random.Random(args.seed).sample(range(len(data)), len(data))]

    model.train()
    step, acc = 0, []
    for i, idx in enumerate(order):
        ids = torch.tensor([data[idx][0]], device="cuda")
        labels = torch.tensor([data[idx][1]], device="cuda")
        weights = torch.tensor([data[idx][2]], device="cuda", dtype=torch.float)
        with torch.autocast("cuda", dtype=dtype):
            loss = assistant_loss(model, ids, labels, weights) / args.grad_accum
        scaler.scale(loss).backward()
        acc.append(loss.item() * args.grad_accum)
        if (i + 1) % args.grad_accum and i + 1 != len(order):
            continue
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        step += 1
        rec = {"step": step, "loss": round(sum(acc) / len(acc), 4), "lr": round(sched.get_last_lr()[0], 7)}
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n"); log.flush()
        acc = []

    model.save_pretrained(out / "adapter")
    print(f"saved {out / 'adapter'}")


if __name__ == "__main__":
    main()
