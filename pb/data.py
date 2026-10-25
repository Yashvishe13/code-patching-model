"""PatchEval-Verified CVEs and a fixed train/eval split."""
import json
import random
from collections import defaultdict
from pathlib import Path

DATASET = Path(__file__).resolve().parent.parent / "third_party/PatchEval/patcheval/datasets/patcheval_verified.json"


def load_tasks():
    return json.loads(DATASET.read_text())


def _stratified(tasks, per_lang, seed):
    by = defaultdict(list)
    for t in sorted(tasks, key=lambda t: t["cve_id"]):
        by[t["programming_language"]].append(t)
    rng = random.Random(seed)
    out = []
    for lang in sorted(by):
        pool = by[lang][:]
        rng.shuffle(pool)
        out.extend(pool[:per_lang])
    return out


def split(seed=0):
    """eval = 30 held-out (10/language); train = the other 200. Fixed across runs."""
    tasks = load_tasks()
    ev = _stratified(tasks, 10, seed)
    taken = {t["cve_id"] for t in ev}
    return {"eval": ev, "train": [t for t in tasks if t["cve_id"] not in taken]}
