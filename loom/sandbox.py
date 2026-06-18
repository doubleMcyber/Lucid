"""Sandboxed execution of (model-generated) Python for the baseline eval.

Lucid is total, so its programs run in-process with no guards. Python is **not**
total — generated code can loop forever, recurse without bound, or touch the
filesystem — so every candidate runs in a *separate process* with:

  * a wall-clock timeout (parent-side `subprocess` timeout),
  * a CPU-time rlimit backstop (child-side, best effort), and
  * a restricted `__builtins__` (no `open`/`import`/`eval`/`exec`/`compile`/IO).

This asymmetry — "Lucid needs no sandbox, Python needs all of this" — is itself
part of the agent-target argument, so the harness makes it explicit rather than
hiding it.

`run_one(code, entry, inputs, timeout)` returns a dict:
  {"status": "ok", "output": <json value>} | {"status": "error", "msg": ...}
  | {"status": "timeout"}.
"""

from __future__ import annotations

import json
import subprocess
import sys

# Builtins the child is allowed to see. Permissive enough for ordinary
# arithmetic/list code, but no import/eval/exec/open/IO.
_SAFE_BUILTIN_NAMES = [
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter",
    "float", "int", "len", "list", "map", "max", "min", "pow", "range",
    "reversed", "round", "set", "sorted", "str", "sum", "tuple", "zip",
    "True", "False", "None", "isinstance", "ord", "chr",
    # exceptions a program might legitimately raise/catch
    "Exception", "ValueError", "IndexError", "KeyError", "TypeError",
    "ZeroDivisionError", "StopIteration",
]

TIMEOUT_DEFAULT = 5.0
CPU_LIMIT_SECONDS = 6


def _safe_builtins() -> dict:
    import builtins
    return {n: getattr(builtins, n) for n in _SAFE_BUILTIN_NAMES if hasattr(builtins, n)}


def _child_main() -> None:
    job = json.loads(sys.stdin.read())
    # best-effort CPU + address-space rlimit backstop (the wall-clock timeout in
    # the parent is the primary guard; rlimits are not portable).
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (CPU_LIMIT_SECONDS, CPU_LIMIT_SECONDS))
    except Exception:
        pass
    g = {"__builtins__": _safe_builtins()}
    try:
        exec(job["code"], g)
        fn = g[job["entry"]]
    except Exception as e:  # compile/exec error or missing entry
        print(json.dumps({"status": "error", "msg": f"load:{type(e).__name__}:{e}"}))
        return
    outputs = []
    for inp in job["inputs"]:
        try:
            out = fn(*inp)
            outputs.append({"ok": True, "output": json.loads(json.dumps(out))})
        except Exception as e:
            outputs.append({"ok": False, "msg": f"{type(e).__name__}:{e}"})
    print(json.dumps({"status": "ok", "outputs": outputs}))


def run_one(code: str, entry: str, inputs: list, timeout: float = TIMEOUT_DEFAULT) -> dict:
    """Run `entry(*inp)` for each inp in `inputs`, isolated. Returns per-input
    results so a single call covers all of a program's held-out IO."""
    job = json.dumps({"code": code, "entry": entry, "inputs": inputs})
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "loom.sandbox"],
            input=job, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}
    if proc.returncode != 0:
        return {"status": "error", "msg": f"exit{proc.returncode}:{proc.stderr[-200:]}"}
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return {"status": "error", "msg": f"badout:{proc.stdout[-200:]}"}


if __name__ == "__main__":
    _child_main()
