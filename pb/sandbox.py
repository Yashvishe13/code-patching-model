"""Rootless PatchEval sandbox, one CVE per container (udocker, no Docker daemon / no sudo).

PatchEnv runs the agent's tools inside the CVE image with the network off, collects its patch as a
git diff, and grades it with PatchEval's own `fix-run.sh` (success = the exploit no longer fires).
Each worker sets UDOCKER_DIR to its own directory so parallel workers never share image layers.
"""
import os
import shlex
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

UDOCKER = shutil.which("udocker") or str(Path(sys.executable).parent / "udocker")
NET_OFF = "network access is disabled in this sandbox; work with the files already in the repository"


def _udocker(*args, timeout=1800):
    r = subprocess.run([UDOCKER, "--quiet", *args], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"udocker {args[0]}: {(r.stderr or r.stdout)[-400:]}")
    return r.stdout.strip()


def pull(image, retries=3):
    for i in range(retries):
        try:
            return _udocker("pull", image)
        except RuntimeError:
            if i == retries - 1:
                raise
            try:
                _udocker("rmi", "-f", image)
            except RuntimeError:
                pass


def remove_image(image):
    try:
        _udocker("rmi", "-f", image)
    except RuntimeError:
        pass


class Container:
    """A udocker container is just a directory tree (self.root); command output is written to a file
    inside it and read back from the host, which also sidesteps udocker's lossy stdout relay."""

    def __init__(self, image):
        self.name = f"pb-{uuid.uuid4().hex[:12]}"
        _udocker("create", f"--name={self.name}", image)
        _udocker("setup", "--execmode=R1", self.name)  # rootless runc via user namespaces
        self.root = Path(_udocker("inspect", "-p", self.name))

    def exec(self, cmd, workdir="/", timeout=120, network=True):
        script = (f"cd {shlex.quote(workdir)} && timeout {timeout} bash -lc {shlex.quote(cmd)} "
                  f"> /tmp/.out 2>&1; echo $? > /tmp/.rc")
        run = [UDOCKER, "--quiet", "run", "--nobanner", self.name, "bash", "-c", script]
        if not network:  # udocker shares the host net; isolate it with an unprivileged net namespace
            run = ["unshare", "--user", "--map-root-user", "--net", "sh", "-c",
                   'ip link set lo up && exec "$@"', "sh", UDOCKER, "--allow-root", *run[1:]]
        (self.root / "tmp/.rc").unlink(missing_ok=True)
        try:
            subprocess.run(run, capture_output=True, timeout=timeout + 40)
        except subprocess.TimeoutExpired:
            return 124, f"[timed out after {timeout}s]"
        rc = int((self.root / "tmp/.rc").read_text().strip()) if (self.root / "tmp/.rc").exists() else 1
        out = (self.root / "tmp/.out").read_bytes().decode("utf-8", "replace") if (self.root / "tmp/.out").exists() else ""
        return rc, out

    def write(self, path, data, workdir="/"):
        # Write relative to the repo root, not '/', so a repo-relative path resolves the same way the
        # matching read (env._run, which cd's into workdir) did; otherwise the write silently misses.
        (self.root / "tmp/.in").write_bytes(data)
        return self.exec(f"cat /tmp/.in > {shlex.quote(path)} && rm /tmp/.in", workdir=workdir)

    def remove(self):
        try:
            _udocker("rm", "-f", self.name, timeout=600)
        except (RuntimeError, subprocess.TimeoutExpired):
            pass


def _run_script(image, patch, script, timeout):
    """Apply the model's patch (as /workspace/fix.patch) in a fresh container and run one PatchEval
    script. A fresh container per script avoids one script's git-apply colliding with the next's."""
    box = Container(image)
    try:
        box.write("/workspace/fix.patch", (patch if patch.endswith("\n") else patch + "\n").encode())
        return box.exec(f"cd /workspace && bash {script}", timeout=timeout)
    finally:
        box.remove()


def grade(image, patch, workdir, timeout=900):
    """PatchEval's own check: resolved == the exploit no longer fires (fix-run.sh) AND the functional
    tests still pass (unit_test.sh). Requiring both closes the obvious reward hack of breaking or
    deleting the vulnerable code to kill the exploit -- that passes fix-run.sh but fails unit_test.sh.
    The unit test only runs when the security check already passed, so it adds ~no cost while the policy
    is weak. Returns (resolved, log)."""
    if not patch.strip():
        return False, "empty patch"
    sec_rc, sec_log = _run_script(image, patch, "fix-run.sh", timeout)
    if sec_rc != 0:
        return False, sec_log  # exploit still fires (or patch did not apply)
    fn_rc, fn_log = _run_script(image, patch, "unit_test.sh", timeout)
    if fn_rc != 0:
        return False, "exploit blocked but functionality broken (unit_test.sh failed):\n" + fn_log
    return True, sec_log
