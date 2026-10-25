"""Shaped reward for GRPO. The objective is PatchEval's own check: the exploit no longer fires
(fix-run.sh) and the functional tests still pass (unit_test.sh), worth 1.0. Below that, dense milestone
credit gives signal from a near-zero policy and, by design, pays for the decisive actions the SFT model
under-emitted: landing a real edit and submitting.

Anti-reward-hacking:
  - network / repo-deletion / test-file edits are blocked at the tool level (pb/env.py), so they can
    never happen in a rollout;
  - test-file changes are stripped from the patch before anything is graded or credited;
  - the edit/submit milestones are GATED on a gradeable diff: no credit for an empty patch, a no-op edit
    (replace_lines writing identical text -> no diff), or a test-/docs-/config-only patch, so submitting
    with nothing real to show earns 0 (see milestones());
  - milestones are capped well below 1.0, so an actual fix always beats any amount of partial credit;
  - the vulnerable-file ground truth (task['vul_func']) is used for credit on TRAINING CVEs only and is
    never shown to the model.
"""
import re

TEST_FILE = re.compile(r"(^|/)(tests?|__tests__|spec|poc)/|_test\.(go|py)$|test_\w+\.py$|\.(test|spec)\.[jt]sx?$")
NONCODE = re.compile(r"\.(md|rst|txt|lock|json|ya?ml|toml|cfg|ini|sum|mod)$|(^|/)(LICENSE|CHANGELOG|README)", re.I)
# Weights upweight the decisive actions the SFT model under-emits (a landed edit and submit), so GRPO
# pushes toward committing a fix, not just exploring. Capped well below the 1.0 an actual fix earns.
CREDIT = {"read_vuln_file": 0.05, "valid_edit": 0.10, "edited_vuln_file": 0.10, "submitted": 0.15}  # sum 0.40


def strip_test_files(patch):
    kept = [s for s in re.split(r"(?m)^(?=diff --git )", patch)
            if not (s.startswith("diff --git") and TEST_FILE.search(s.split("\n", 1)[0].split(" b/", 1)[-1]))]
    return "".join(kept)


def patch_files(patch):
    return [l.split(" b/", 1)[-1] for l in patch.splitlines() if l.startswith("diff --git")]


def milestones(rollout, task):
    msgs = rollout["messages"]
    calls = [tc["function"] for m in msgs if m["role"] == "assistant" for tc in m.get("tool_calls", [])]
    outs = [m["content"] for m in msgs if m["role"] == "tool"]
    vfiles = {v["file_path"] for v in task["vul_func"]}
    vbases = {f.split("/")[-1] for f in vfiles}
    read = any(any(b in o for b in vbases) and re.search(r"\d+\t", o) for o in outs) \
        or any(any(b in o for b in vbases) and ":" in o for o in outs)
    # Gate edit/submit credit on a REAL diff: gradeable_patch is "" for an empty, no-op (old==new ->
    # no diff), test-only, or docs/config-only patch. So an "Edited" tool response or a submit call that
    # left no gradeable change earns nothing -- closing the "no-op edit + submit" and "empty submit" hacks.
    gp = gradeable_patch(rollout["patch"])
    valid_edit = bool(gp)
    files = patch_files(gp)
    edited_vuln = any(f in vfiles or f.split("/")[-1] in vbases for f in files)
    submitted = any(c["name"] == "submit" for c in calls) and bool(gp)
    return {"read_vuln_file": read, "valid_edit": valid_edit, "edited_vuln_file": edited_vuln, "submitted": submitted}


def reward(resolved, m):
    if resolved:
        return 1.0
    return round(sum(CREDIT[k] for k, v in m.items() if v), 4)


def gradeable_patch(patch):
    """The diff worth sending to the grader: non-empty, test-files stripped, touches at least one code
    file. Returns the stripped patch or '' (which scores by milestones only, no grading)."""
    p = strip_test_files(patch)
    files = patch_files(p)
    if not files or all(NONCODE.search(f) for f in files):
        return ""
    return p


# --- CodeBLEU-style dense reward (n-gram core; train-only, no execution) ---------------------------
import math as _math
from collections import Counter as _Counter

_TOK = re.compile(r"\w+|[^\s\w]")


def _added_code(patch):
    """The new code a diff introduces: '+' lines (excluding the +++ header), joined."""
    return " ".join(l[1:] for l in patch.splitlines() if l.startswith("+") and not l.startswith("+++"))


def _bleu(hyp, ref, max_n=4):
    """BLEU-4 with add-1 smoothing over code tokens; the dominant term of CodeBLEU."""
    h, r = _TOK.findall(hyp), _TOK.findall(ref)
    if len(h) < 1 or len(r) < 1:
        return 0.0
    logs = []
    for n in range(1, max_n + 1):
        hn = _Counter(tuple(h[i:i + n]) for i in range(len(h) - n + 1))
        rn = _Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
        overlap = sum((hn & rn).values())
        total = max(sum(hn.values()), 1)
        logs.append(_math.log((overlap + 1) / (total + 1)))  # smoothed precision
    bp = _math.exp(1 - len(r) / len(h)) if len(h) < len(r) else 1.0
    return bp * _math.exp(sum(logs) / max_n)


def codebleu_reward(patch, reference):
    """Dense [0,1] reward: CodeBLEU-style n-gram similarity between the model's added code and the gold
    patch's added code (per shared file, averaged). Test files stripped; empty patch -> 0. The reference
    is train-only, never shown to the model, never used at eval. (AST/data-flow terms of full CodeBLEU
    are omitted: the tree-sitter grammars' ABI did not match the installed runtime.)"""
    patch = strip_test_files(patch)
    if not patch.strip() or not reference.strip():
        return 0.0
    def by_file(p):
        out, cur = {}, None
        for ln in p.splitlines():
            if ln.startswith("diff --git"):
                cur = ln.split(" b/", 1)[-1]; out.setdefault(cur, [])
            elif cur is not None and ln.startswith("+") and not ln.startswith("+++"):
                out[cur].append(ln[1:])
        return {f: "\n".join(v) for f, v in out.items() if v}
    ref_f, hyp_f = by_file(reference), by_file(patch)
    hyp_base = {f.split("/")[-1]: c for f, c in hyp_f.items()}
    scores = [_bleu(hyp_f.get(f) or hyp_base.get(f.split("/")[-1], ""), c) for f, c in ref_f.items()]
    return round(sum(scores) / len(scores), 4) if scores else 0.0
