"""Constructor audit (static, ast) and OS-confined execution of `build_weights()`.

Two layers, both required. A restricted `__builtins__` alone is NOT a sandbox
(numpy and the stdlib keep real handles in their own globals), so the OS layer
(sandbox-exec on darwin, bwrap on linux) is the one that counts and the Python
layer in the runner is defence in depth. With no OS layer we fail closed.
"""
from __future__ import annotations

import ast
import os
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field

import numpy as np

from wsp import rnn

ALLOWED = frozenset({"numpy", "math", "random", "itertools", "functools", "collections", "typing"})
FORBIDDEN = frozenset({"torch", "jax", "tensorflow", "scipy.optimize", "sklearn", "autograd"})
BLOCKED_CALLS = frozenset({"open", "exec", "eval", "compile", "__import__",
                           "getattr", "setattr", "vars", "globals"})
BLOCKED_ATTRS = frozenset({"__builtins__", "__import__", "__globals__", "__subclasses__", "__loader__"})
BLOCKED_NAMES = frozenset({"os", "sys", "subprocess", "socket", "pathlib", "shutil", "ctypes",
                           "importlib", "builtins", "dataclasses"})
FAIL_CLOSED_REASON = "no process confinement available"
# Whole-token boundaries inside a string literal: whitespace, commas, quotes and brackets.
_LITERAL_TOKEN_SEP = re.compile(r"[\s,'\"\[\]\(\)\{\}]+")


@dataclass
class AuditReport:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    literal_matches: list[str] = field(default_factory=list)  # distinct observed strings found, sorted
    literal_match_fraction: float = 0.0  # matched distinct observed strings / observed strings


@dataclass
class SandboxResult:
    weights: dict | None
    audit: AuditReport
    stdout: str = ""
    stderr: str = ""
    returncode: int | None = None
    elapsed_s: float = 0.0
    confined: bool = False
    timed_out: bool = False  # killed at the wall timeout: no return code, stderr ends with the timeout line


def audit(src: str, observed_strings, line_cap: int = 300) -> AuditReport:
    reasons: list[str] = []
    n_lines = len(src.splitlines())
    if n_lines > line_cap:
        reasons.append(f"line cap exceeded: {n_lines} > {line_cap}")
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return AuditReport(False, reasons + [f"syntax error: {e}"])
    observed = set(observed_strings)
    literal_matches: set[str] = set()

    def is_forbidden(mod: str) -> bool:
        # `scipy.optimize` and anything below it; `torch`, `torch.nn`, ...
        return any(mod == f or mod.startswith(f + ".") for f in FORBIDDEN)

    def check_module(mod: str | None, names: tuple[str, ...] = ()) -> None:
        if not mod:
            reasons.append("relative import not allowed")
            return
        top = mod.split(".")[0]
        if top in ALLOWED:
            return
        # `from scipy import optimize` names the forbidden module via its alias.
        hits = [m for m in (mod, *(f"{mod}.{n}" for n in names)) if is_forbidden(m)]
        if hits:
            reasons.append(f"forbidden optimizer import: {hits[0]}")
        else:
            reasons.append(f"import not allowed: {mod}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                check_module(alias.name)
        elif isinstance(node, ast.ImportFrom):
            check_module(node.module if node.level == 0 else None,
                         tuple(a.name for a in node.names))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in BLOCKED_CALLS:
            reasons.append(f"call not allowed: {node.func.id}()")
        elif isinstance(node, ast.Attribute) and node.attr in BLOCKED_ATTRS:
            reasons.append(f"attribute not allowed: .{node.attr}")
        elif isinstance(node, ast.Name) and node.id in BLOCKED_NAMES:
            reasons.append(f"name not allowed: {node.id}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # Whole tokens inside the literal, so a dataset embedded as one multi-line string counts.
            literal_matches.update(observed.intersection(_LITERAL_TOKEN_SEP.split(node.value)))
    if not any(isinstance(n, ast.FunctionDef) and n.name == "build_weights" for n in tree.body):
        reasons.append("no top-level def build_weights()")
    return AuditReport(not reasons, sorted(set(reasons)), sorted(literal_matches),
                       len(literal_matches) / len(observed) if observed else 0.0)


_DARWIN_PROFILE = """(version 1)
(deny default)
(allow file-read*)
(deny file-read* (subpath "/Users"))
(allow file-read* (subpath "{base_prefix}") (subpath "{prefix}") (subpath "{tmp}"))
(allow file-read-metadata)
(allow process-exec*)
(allow process-fork)
(allow sysctl-read)
(allow mach-lookup)
(allow ipc-posix-shm)
(allow signal (target self))
(allow file-write* (subpath "{tmp}") (literal "/dev/null"))
"""


def confinement_command(python: str, tmp_dir: str) -> list[str] | None:
    """OS-level confinement prefix, or None when none is available (fail closed)."""
    tmp = os.path.realpath(tmp_dir)
    if sys.platform == "darwin":
        profile = os.path.join(tmp_dir, "constructor.sb")
        with open(profile, "w") as f:
            f.write(_DARWIN_PROFILE.format(base_prefix=sys.base_prefix, prefix=sys.prefix, tmp=tmp))
        return ["sandbox-exec", "-f", profile, python]
    if sys.platform.startswith("linux") and shutil.which("bwrap"):
        return ["bwrap", "--ro-bind", "/", "/", "--bind", tmp, tmp,
                "--unshare-net", "--die-with-parent", python]
    return None


# Runs inside the confined interpreter. Captures what it needs, then locks the
# Python layer down before the untrusted source is executed.
_RUNNER = r'''
import sys
import builtins
import numpy as _np

ALLOWED = %(allowed)r
REFUSED = {"_socket", "posix", "nt", "_io", "builtins", "dataclasses"} | %(blocked)r


def _main(out_path, src_path):
    # Captured handles live in this frame's locals, not in __main__'s globals,
    # so they are not one attribute chain away from the constructor.
    savez, real_import = _np.savez, builtins.__import__
    open_, exec_, eval_, compile_ = open, exec, eval, compile
    ndarray, dict_ = _np.ndarray, dict
    # numpy loads these lazily and the import machinery needs exec/compile: load them now.
    _np.linalg, _np.fft, _np.random, _np.polynomial, _np.dtypes, _np.exceptions
    with open_(src_path) as f:
        src = f.read()
    g = {"__name__": "constructor", "__builtins__": builtins}

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        # The constructor itself may import only ALLOWED top packages. Library
        # internals (numpy, and the import machinery loading numpy's lazy
        # submodules) may additionally re-import modules that are already
        # loaded, which grants nothing new; refused names stay refused for
        # everyone but the frozen import machinery, which needs _io itself.
        ns = globals or {}
        top = ((ns.get("__package__") or "") if level else name).split(".")[0]
        caller = ns.get("__name__") or ""
        if top in ALLOWED:
            ok = True
        elif ns is g or caller == "constructor":
            ok = False
        elif top in REFUSED:
            ok = caller.startswith("importlib._bootstrap") and name in sys.modules
        else:
            ok = level == 0 and name in sys.modules
        if not ok:
            raise ImportError("import of %%r is not allowed in the constructor sandbox" %% (name,))
        return real_import(name, globals, locals, fromlist, level)

    builtins.__import__ = guarded_import
    builtins.open = builtins.exec = builtins.eval = builtins.compile = None
    exec_(compile_(src, "constructor.py", "exec"), g)
    w = g["build_weights"]()
    builtins.__import__, builtins.open, builtins.exec, builtins.eval, builtins.compile = \
        real_import, open_, exec_, eval_, compile_
    if type(w) is not dict_:
        raise TypeError("build_weights() must return a dict, got %%s" %% type(w).__name__)
    for k, v in w.items():
        if type(k) is not str or type(v) is not ndarray or v.dtype.kind != "f":
            raise TypeError("weights[%%r] must be a float ndarray" %% (k,))
    savez(out_path, **w)


_main(sys.argv[1], sys.argv[2])
''' % {"allowed": set(ALLOWED), "blocked": set(BLOCKED_NAMES)}


def _limits(timeout_s: int, mem_mb: int):
    def preexec():
        resource.setrlimit(resource.RLIMIT_CPU, (timeout_s, timeout_s))
        try:
            lim = mem_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
        except (ValueError, OSError) as e:
            os.write(2, f"[sandbox] could not set RLIMIT_AS: {e}\n".encode())
    return preexec


def run_constructor(src: str, h: int, timeout_s: int = 10, mem_mb: int = 2048,
                    observed_strings=()) -> SandboxResult:
    report = audit(src, observed_strings)
    if not report.ok:
        return SandboxResult(None, report, stderr="audit failed: " + "; ".join(report.reasons))
    tmp = tempfile.mkdtemp(prefix="wsp-constructor-")
    t0 = time.monotonic()
    try:
        # Any failure returns weights=None with a reason; nothing propagates.
        try:
            prefix = confinement_command(sys.executable, tmp)
            if prefix is None:
                return SandboxResult(None, report, stderr=FAIL_CLOSED_REASON, confined=False)
            src_path, runner, out_path = (os.path.join(tmp, n)
                                          for n in ("constructor.py", "runner.py", "weights.npz"))
            with open(src_path, "w") as f:
                f.write(src)
            with open(runner, "w") as f:
                f.write(_RUNNER)
        except Exception as e:  # profile/runner could not be written
            return SandboxResult(None, report, stderr=f"sandbox setup failed: {e!r}",
                                 elapsed_s=time.monotonic() - t0, confined=False)
        try:
            proc = subprocess.run(prefix + ["-I", runner, out_path, src_path], cwd=tmp, env={},
                                  stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                  timeout=timeout_s, preexec_fn=_limits(timeout_s, mem_mb))
        except subprocess.TimeoutExpired as e:
            err = (e.stderr or b"").decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
            return SandboxResult(None, report, stderr=err + f"\nwall timeout after {timeout_s}s",
                                 elapsed_s=time.monotonic() - t0, confined=True, timed_out=True)
        except Exception as e:  # e.g. sandbox-exec/bwrap missing, preexec_fn (rlimit) failure
            return SandboxResult(None, report, stderr=f"sandbox launch failed: {e!r}",
                                 elapsed_s=time.monotonic() - t0, confined=False)
        elapsed = time.monotonic() - t0
        if proc.returncode != 0:
            return SandboxResult(None, report, proc.stdout, proc.stderr + f"\nconstructor exited {proc.returncode}",
                                 proc.returncode, elapsed, True)
        try:
            with np.load(out_path, allow_pickle=False) as z:
                weights = rnn.check_weights({k: z[k] for k in z.files}, h)
        except Exception as e:  # missing output, bad shapes, dtype, non-finite
            return SandboxResult(None, report, proc.stdout, proc.stderr + f"\ninvalid weights: {e!r}",
                                 proc.returncode, elapsed, True)
        return SandboxResult(weights, report, proc.stdout, proc.stderr, proc.returncode, elapsed, True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def mock_constructor_source(h: int, seed: int) -> str:
    """A valid constructor emitting seeded random weights; shared by D0 and the MockLLM."""
    return f'''import numpy as np

H = {int(h)}
SEED = {int(seed)}
SCALE = 0.5


def build_weights():
    rng = np.random.default_rng(SEED)
    return {{
        "Wx": rng.normal(0.0, SCALE, size=(2, H)),
        "Wh": rng.normal(0.0, SCALE, size=(H, H)),
        "bh": rng.normal(0.0, SCALE, size=(H,)),
        "w": rng.normal(0.0, SCALE, size=(H,)),
        "c": rng.normal(0.0, SCALE, size=()),
    }}
'''
