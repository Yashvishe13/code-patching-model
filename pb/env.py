"""PatchEnv: the agent's view of one CVE. Tools run inside the container with the network off; the
submission is the agent's own git diff. Grading is delegated to sandbox.grade (PatchEval fix-run.sh)."""
import re
import shlex

from pb.sandbox import NET_OFF, Container

TOOLS = [
    {"name": "bash", "description": "Run a shell command in the repository root; returns combined stdout/stderr.",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "view_file", "description": "Show a file with line numbers (default 100 lines from start_line). Path is relative to the repo root.",
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string"}, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}},
         "required": ["path"]}},
    {"name": "replace_lines", "description": "Replace lines start_line..end_line (1-indexed and inclusive, exactly as numbered by view_file) in a file with new_str. This is the only edit tool: you specify which lines to replace, not their exact text.",
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string"}, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}, "new_str": {"type": "string"}},
         "required": ["path", "start_line", "end_line", "new_str"]}},
    {"name": "submit", "description": "Finish once the vulnerability is fixed; the current diff is submitted.",
     "parameters": {"type": "object", "properties": {}}},
]

# Cheap guardrails: block reaching for the answer or wrecking the task before the call runs.
_NET = re.compile(r"(?:^|[;&|(]|\$\()\s*(git\s+(clone|fetch|pull|ls-remote)|curl|wget|pip\s+install|npm\s+(install|view)|go\s+get)\b")
_RM = re.compile(r"(?:^|[;&|(]|\$\()\s*rm\s+(-\w+\s+)*-\w*r\w*\s+(-\w+\s+)*(/workspace\S*|\.|\*|/)(?=\s|$|;|&|\|)|shutil\.rmtree|rimraf")
_TEST = re.compile(r"(^|/)(tests?|__tests__|spec|poc)/|_test\.(go|py)$|test_\w+\.py$|\.(test|spec)\.[jt]sx?$")
MAX_OBS = 4000


class PatchEnv:
    def __init__(self, task, cmd_timeout=90):
        self.task, self.cmd_timeout = task, cmd_timeout
        self.box = self.workdir = None
        self.repo_present = True

    def __enter__(self):
        self.box = Container(self.task["image_url"])
        self.workdir = self._detect_workdir()
        ref = self.box.root / "workspace/fix.patch"
        self.reference_patch = ref.read_text(errors="replace") if ref.exists() else ""
        # unpacking rewrites every file's stat/mode; stop git seeing the whole tree as modified
        self._run("git config core.fileMode false && git update-index -q --refresh", timeout=600)
        self._hide_payload()
        self.base_tree = self._snapshot()
        return self

    def __exit__(self, *exc):
        if self.box:
            self.box.remove()

    def _run(self, cmd, timeout=None, network=True):
        return self.box.exec(cmd, workdir=self.workdir or "/", timeout=timeout or self.cmd_timeout, network=network)

    def _resolve(self, path):
        """Map a file path to one that exists relative to the repo root. Small models often prepend the
        repo name (e.g. 'docconv/pdf_ocr.go' when the workdir is already .../docconv) or use an absolute
        path; try the path as-is, with its leading component stripped, then a unique basename match."""
        p = str(path).strip()
        cands = [p, p.lstrip("/")]
        if "/" in p.strip("/"):
            cands.append(p.strip("/").split("/", 1)[1])  # drop a wrong leading dir (repo name)
        for c in dict.fromkeys(cands):
            if c and self._run(f"test -f {shlex.quote(c)}", network=False)[0] == 0:
                return c
        base = p.rstrip("/").split("/")[-1]
        _, out = self._run(f"find . -name {shlex.quote(base)} -type f -not -path '*/.git/*' | head -2", network=False)
        hits = [l[2:] if l.startswith("./") else l for l in out.splitlines() if l.strip()]
        return hits[0] if len(hits) == 1 else p

    def _detect_workdir(self):
        repo = self.task["repo"].rstrip("/").split("/")[-1].removesuffix(".git")
        for c in (f"/workspace/{repo}", f"/workspace/{repo.lower()}", "/workspace"):
            if (self.box.root / c.lstrip("/") / ".git").exists():
                return c
        _, out = self.box.exec("find /workspace -mindepth 2 -maxdepth 3 -type d -name .git | head -1", workdir="/")
        return out.strip().rsplit("/.git", 1)[0] if out.strip() else "/workspace"

    def _hide_payload(self):
        if self.workdir == "/workspace":
            self.box.exec("rm -f /workspace/fix.patch", workdir="/")
        else:
            top = self.workdir.removeprefix("/workspace/").split("/")[0]
            self.box.exec(f"find /workspace -mindepth 1 -maxdepth 1 ! -name {shlex.quote(top)} -exec rm -rf -- {{}} +", workdir="/")

    def _snapshot(self):
        cmd = ("{ [ -f /tmp/.idx ] || cp .git/index /tmp/.idx; } && export GIT_INDEX_FILE=/tmp/.idx && "
               "git add -A . >/dev/null && git write-tree")
        for _ in range(3):
            rc, out = self._run(cmd, timeout=600)
            if rc == 0 and out.strip():
                return out.strip().splitlines()[-1]
        raise RuntimeError(f"snapshot failed: {out[-200:]!r}")

    def _trunc(self, text):
        if len(text) <= MAX_OBS:
            return text
        head = text[:MAX_OBS].rsplit("\n", 1)[0]
        return f"{head}\n[+{len(text) - len(head)} chars truncated]"

    def call(self, name, args):
        if (name == "bash" and (_NET.search(str(args.get("command", ""))) or _RM.search(str(args.get("command", ""))))):
            return f"Blocked: {NET_OFF}." if _NET.search(str(args["command"])) else "Blocked: deleting the repo is not allowed."
        if name in ("str_replace", "replace_lines") and _TEST.search(_relpath(args.get("path", ""), self.workdir)):
            return "Blocked: editing test files is not allowed; fix the source code."
        if name == "bash":
            rc, out = self._run(str(args["command"]), network=False)
            return self._trunc(f"{out}\n[exit code {rc}]")
        if name == "view_file":
            s = int(args.get("start_line") or 1)
            e = int(args.get("end_line") or s + 99)
            rc, out = self._run(f"nl -ba {shlex.quote(self._resolve(args['path']))} | sed -n '{s},{e}p'", network=False)
            return self._trunc(out) if out.strip() else f"[no output; exit code {rc}]"
        if name == "str_replace":  # not an offered tool; steer the model to the line-number editor
            return "Error: str_replace is not available. Edit with replace_lines(path, start_line, end_line, new_str)."
        if name == "replace_lines":
            return self._replace_lines(args["path"], int(args["start_line"]), int(args["end_line"]), args["new_str"])
        raise KeyError(name)

    def _replace_lines(self, path, start, end, new):
        path = self._resolve(path)
        rc, content = self._run(f"cat {shlex.quote(path)}", network=False)
        if rc != 0:
            return f"Error: cannot read {path}."
        lines = content.split("\n")
        if not (1 <= start <= end <= len(lines)):
            return f"Error: lines {start}-{end} out of range; {path} has {len(lines)} lines."
        updated = "\n".join(lines[:start - 1] + new.split("\n") + lines[end:])
        rc, out = self.box.write(path, updated.encode(), workdir=self.workdir)
        return f"Edited {path} (lines {start}-{end})." if rc == 0 else f"Error writing {path}: {out[:150]}"

    def _str_replace(self, path, old, new):
        rc, content = self._run(f"cat {shlex.quote(path)}", network=False)
        if rc != 0:
            return f"Error: cannot read {path}."
        if content.count(old) == 1:  # exact match
            updated = content.replace(old, new, 1)
        else:  # whitespace-tolerant: match line-by-line ignoring per-line leading/trailing whitespace,
            # so an approximate old_str (common from small models) still lands. Must be uniquely locatable.
            clines = content.split("\n")
            olines = [l.strip() for l in old.strip("\n").split("\n")]
            hits = [i for i in range(len(clines) - len(olines) + 1)
                    if [clines[i + j].strip() for j in range(len(olines))] == olines]
            if len(hits) != 1:
                return (f"Error: old_str found {content.count(old)} times exactly and {len(hits)} times "
                        f"ignoring whitespace in {path}; it must locate exactly one place.")
            i = hits[0]
            indent = clines[i][:len(clines[i]) - len(clines[i].lstrip())]  # reuse the matched block's indent
            new_block = [(indent + l if l.strip() else l) for l in new.strip("\n").split("\n")]
            updated = "\n".join(clines[:i] + new_block + clines[i + len(olines):])
        rc, out = self.box.write(path, updated.encode(), workdir=self.workdir)
        return f"Edited {path}." if rc == 0 else f"Error writing {path}: {out[:150]}"

    def diff(self):
        self.repo_present = (self.box.root / self.workdir.lstrip("/") / ".git").is_dir()
        if not self.repo_present:
            return ""
        _, out = self._run(f"git diff -U3 {self.base_tree} {self._snapshot()}")
        return out if out.startswith("diff --git") else ""


def _relpath(path, workdir):
    path = str(path)
    return path[len(workdir):].lstrip("/") if workdir and path.startswith(workdir.rstrip("/") + "/") else path.lstrip("./")
