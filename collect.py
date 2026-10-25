"""Run a policy on a PatchEval split and grade each attempt (PatchEval fix-run.sh). Sharded for parallel
workers; appends to one traces.jsonl under a lock. Used for teacher trace collection and the Qwen baseline.

  python collect.py --policy teacher --split train --limit 100 --attempts 4 --name teacher_train --shard i/N
  python collect.py --policy qwen   --split eval                             --name qwen_baseline
"""
import argparse
import fcntl
import json
import time
from collections import Counter
from pathlib import Path

from pb.agent import run_episode
from pb.data import split
from pb.env import PatchEnv
from pb.sandbox import grade, pull, remove_image


def build_policy(args):
    if args.policy == "teacher":
        from pb.teacher import Teacher
        return Teacher(args.teacher_model, reasoning_effort=args.reasoning)
    from pb.qwen import Qwen
    return Qwen(args.model, adapter=args.adapter)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", choices=["teacher", "qwen"], required=True)
    p.add_argument("--split", choices=["train", "eval"], default="train")
    p.add_argument("--name", required=True)
    p.add_argument("--limit", type=int, default=0, help="first N CVEs of the split (0 = all)")
    p.add_argument("--attempts", type=int, default=1, help="rollouts per CVE (K)")
    p.add_argument("--max-turns", type=int, default=40)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--adapter", default=None)
    p.add_argument("--teacher-model", default="gpt-5.6-terra")
    p.add_argument("--reasoning", default="medium", choices=["low","medium","high"])
    p.add_argument("--shard", default="0/1", help="i/N: this worker takes every N-th CVE from i")
    p.add_argument("--language", default=None, help="restrict to one programming_language (e.g. JavaScript)")
    args = p.parse_args()

    out = Path("runs") / args.name
    out.mkdir(parents=True, exist_ok=True)
    trace_path = out / "traces.jsonl"
    done = Counter(json.loads(l)["cve"] for l in trace_path.open()) if trace_path.exists() else Counter()

    tasks = split()[args.split]
    if args.language:
        tasks = [t for t in tasks if t["programming_language"] == args.language]
    tasks = tasks[:args.limit] if args.limit else tasks
    i, n = map(int, args.shard.split("/"))
    tasks = tasks[i::n]
    policy = build_policy(args)

    for t in tasks:
        cve, image = t["cve_id"], t["image_url"]
        todo = args.attempts - done[cve]
        if todo <= 0:
            continue
        print(f"[{cve} {t['programming_language']}] {todo} attempts", flush=True)
        try:
            pull(image)
            for k in range(todo):
                t0 = time.time()
                try:
                    with PatchEnv(t) as env:
                        ep = run_episode(policy, env, t, max_turns=args.max_turns)
                        patch = env.diff()
                    resolved, log = grade(image, patch, env.workdir)
                except Exception as e:
                    ep, patch, resolved, log = {"messages": [], "turns": [], "stop_reason": "error"}, "", False, repr(e)
                rec = {"cve": cve, "language": t["programming_language"], "resolved": resolved, "patch": patch,
                       "stop_reason": ep["stop_reason"], "n_turns": len(ep["turns"]), "messages": ep["messages"],
                       "log_tail": log[-2000:], "seconds": round(time.time() - t0, 1),
                       "usage": dict(getattr(policy, "usage", {}))}
                with trace_path.open("a") as f:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    f.write(json.dumps(rec) + "\n")
                print(f"    -> resolved={resolved} stop={ep['stop_reason']} turns={len(ep['turns'])} ({rec['seconds']}s)", flush=True)
        finally:
            remove_image(image)

    records = [json.loads(l) for l in trace_path.open()]
    summary = {"episodes": len(records), "cves": len({r["cve"] for r in records}),
               "resolved_episodes": sum(r["resolved"] for r in records),
               "cves_with_a_success": len({r["cve"] for r in records if r["resolved"]}),
               "by_language": {lang: {"episodes": sum(r["language"] == lang for r in records),
                                      "resolved": sum(r["resolved"] and r["language"] == lang for r in records)}
                               for lang in sorted({r["language"] for r in records})},
               "stop_reasons": dict(Counter(r["stop_reason"] for r in records))}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
