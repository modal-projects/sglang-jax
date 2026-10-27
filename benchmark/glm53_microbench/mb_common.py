"""Shared harness for the GLM-5.3 v6e Stage-0 microbenchmarks.

Modes (mutually exclusive):
  (default)    run on the real TPU and time.
  --cpu-smoke  tiny shapes, CPU backend, Pallas TPU interpret mode; checks
               numerics of every hand-written kernel against a pure-JAX
               reference. Cases that cannot be interpreted (sglang-jax gmm_v2)
               are AOT-compiled for v6e instead (see --aot).
  --aot        no TPU needed: lower + fully compile (Mosaic included) every
               case at the REAL shapes for a v6e topology using libtpu's
               compile-only client. Catches VMEM overflows, Mosaic NYI ops,
               semaphore/SMEM limits before burning TPU time.

Every measurement prints one JSON line (also appended to --out).
Timing: one jitted call = lax.fori_loop(inner) over the op with a real data
dependency between iterations (XLA can neither hoist nor CSE it);
us = median over --reps calls (after --warmup) / inner.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import socket
import statistics
import sys
import time
import traceback

V6E_1 = ("v6e:1x1", (1, 1, 1))
V6E_8 = ("v6e:2x4", (2, 4, 1))


def base_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--out", required=True, help="JSONL output file (appended)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--cpu-smoke", action="store_true",
                   help="tiny shapes + Pallas TPU interpret mode on CPU")
    g.add_argument("--aot", action="store_true",
                   help="compile-only for a v6e topology at real shapes (no TPU needed)")
    p.add_argument("--reps", type=int, default=None, help="timed calls (median)")
    p.add_argument("--warmup", type=int, default=None, help="untimed calls after compile")
    p.add_argument("--inner", type=int, default=None,
                   help="op repetitions inside one jitted fori_loop")
    p.add_argument("--only", default=None,
                   help="comma-separated substrings; run only cases whose name contains one")
    p.add_argument("--no-check", action="store_true", help="skip numerics checks on TPU")
    return p


def setup_env(args) -> None:
    """Call before `import jax`."""
    if args.cpu_smoke or args.aot:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        # compile-only libtpu client; allow it next to another such process
        os.environ.setdefault("ALLOW_MULTIPLE_LIBTPU_LOAD", "1")
    if args.reps is None:
        args.reps = 1 if args.cpu_smoke else 20
    if args.warmup is None:
        args.warmup = 0 if args.cpu_smoke else 3
    if args.inner is None:
        args.inner = 1 if args.cpu_smoke else 20


def mode_of(args) -> str:
    return "smoke" if args.cpu_smoke else ("aot" if args.aot else "tpu")


# ----------------------------------------------------------------------------
# CPU helpers
# ----------------------------------------------------------------------------

def patch_tpu_info_for_cpu():
    """On CPU, make get_tpu_info() report a v6e chip (gmm_v2 tiler, our kernels)."""
    import importlib
    from jax.experimental.pallas import tpu as pltpu
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_V6E, 1)
    fake = lambda: info  # noqa: E731
    pltpu.get_tpu_info = fake
    for modname in ("jax._src.tpu_info", "jax._src.pallas.mosaic.tpu_info"):
        try:
            importlib.import_module(modname).get_tpu_info = fake
        except Exception:  # noqa: BLE001
            pass
    try:  # sglang-jax tuned-tile lookup insists on a TPU device kind
        from sgl_jax.srt.kernels.gmm.megablox_gmm_kernel import tuned_tile_sizes as tts
        tts.get_device_name = lambda num_devices=None: "TPU v6e"  # noqa: E731
    except Exception:  # noqa: BLE001
        pass
    return info


def interpret_value(mode: str):
    """pallas_call(interpret=...) for our own kernels."""
    if mode != "smoke":
        return False
    from jax.experimental.pallas import tpu as pltpu
    return pltpu.InterpretParams(uninitialized_memory="zero")


_TOPO = {}


def aot_devices(topo=V6E_1):
    """Compile-only TPU devices for a v6e topology (works on a CPU-only host with libtpu)."""
    if topo not in _TOPO:
        from jax.experimental import topologies
        name, cphb = topo
        _TOPO[topo] = topologies.get_topology_desc(
            topology_name=name, platform="tpu", chip_config_name="default",
            chips_per_host_bounds=cphb).devices
    return _TOPO[topo]


def aot_compile(fn, arg_specs, mesh, in_shardings=None):
    """Lower+compile jitted `fn` for the abstract TPU mesh; returns info dict.

    arg_specs: pytree of ShapeDtypeStruct (no sharding). in_shardings: optional
    pytree of PartitionSpec matching arg_specs (default: replicated)."""
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    if in_shardings is None:
        in_shardings = jax.tree.map(lambda _: P(), arg_specs)
    specs = jax.tree.map(
        lambda s, p: jax.ShapeDtypeStruct(s.shape, s.dtype, sharding=NamedSharding(mesh, p)),
        arg_specs, in_shardings)
    t0 = time.perf_counter()
    with jax.sharding.use_abstract_mesh(mesh.abstract_mesh):
        compiled = fn.trace(*specs).lower().compile()
    info = {"aot": "ok", "compile_s": round(time.perf_counter() - t0, 3)}
    try:
        ma = compiled.memory_analysis()
        if ma is not None:
            info["aot_temp_bytes"] = int(getattr(ma, "temp_size_in_bytes", -1))
            info["aot_arg_bytes"] = int(getattr(ma, "argument_size_in_bytes", -1))
    except Exception:  # noqa: BLE001
        pass
    return info


# ----------------------------------------------------------------------------
# Recording
# ----------------------------------------------------------------------------

class Recorder:
    def __init__(self, args, script: str):
        self.args = args
        self.mode = mode_of(args)
        self.script = script
        self.f = open(args.out, "a")
        self._meta = None
        self.n_ok = 0
        self.n_err = 0
        self.only = [s for s in (args.only or "").split(",") if s]

    def want(self, name: str) -> bool:
        return not self.only or any(s in name for s in self.only)

    def meta(self):
        if self._meta is None:
            import jax
            d = jax.devices()[0]
            self._meta = {
                "mode": self.mode,
                "jax": jax.__version__,
                "backend": jax.default_backend(),
                "device_kind": "TPU v6 lite (aot)" if self.mode == "aot"
                else getattr(d, "device_kind", "?"),
                "n_devices": jax.device_count(),
                "host": socket.gethostname(),
            }
        return self._meta

    def emit(self, rec: dict) -> None:
        rec = {"script": self.script, **rec, **self.meta(), "t": round(time.time(), 3)}
        line = json.dumps(rec, default=_json_default)
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()
        if "error" in rec:
            self.n_err += 1
        else:
            self.n_ok += 1

    def error(self, case: dict, exc: BaseException) -> None:
        tb = traceback.format_exc(limit=8)
        msg = f"{type(exc).__name__}: {exc}"
        self.emit({**case, "error": msg[:3000], "traceback": tb[-4000:]})

    def close(self) -> int:
        self.f.close()
        print(f"# {self.script} [{self.mode}]: {self.n_ok} ok, {self.n_err} errors -> "
              f"{self.args.out}", file=sys.stderr, flush=True)
        return self.n_err


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:  # noqa: BLE001
        pass
    return str(o)


# ----------------------------------------------------------------------------
# Timing
# ----------------------------------------------------------------------------

def looped(step, inner: int):
    """jit((consts, state) -> fori_loop(inner, state -> step(consts, state)))."""
    import jax

    def run(consts, state):
        return jax.lax.fori_loop(0, inner, lambda i, s: step(consts, s), state)

    return jax.jit(run)


def time_call(fn, args_tuple, *, reps: int, warmup: int):
    import jax
    t0 = time.perf_counter()
    out = fn(*args_tuple)
    jax.block_until_ready(out)
    first_s = time.perf_counter() - t0
    for _ in range(warmup):
        jax.block_until_ready(fn(*args_tuple))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args_tuple))
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), min(ts), max(ts), first_s, out


_FN_CACHE = {}


def run_case(rec: Recorder, case: dict, consts, state, step, *, inner=None,
             derive=None, mesh=None, shardings=None, key=None, force_aot=False):
    """Time (mode tpu/smoke) or AOT-compile (mode aot) one case; emits one record.

    consts/state: pytrees of jax.Array (tpu/smoke) or ShapeDtypeStruct (aot).
    derive(us) -> dict of derived metrics (GB/s ...).
    In aot mode `mesh` is the abstract-device Mesh, `shardings` an optional
    (consts_specs, state_specs) pair of PartitionSpec pytrees.
    key: reuse the jitted loop across cases with identical shapes (e.g. routings).
    force_aot: compile-only even in smoke mode (kernels the interpreter cannot run).
    Returns the final state (tpu/smoke) or None.
    """
    inner = inner or rec.args.inner
    if key is not None:
        f = _FN_CACHE.get((key, inner))
        if f is None:
            f = _FN_CACHE[(key, inner)] = looped(step, inner)
    else:
        f = looped(step, inner)
    try:
        if rec.mode == "aot" or force_aot:
            if mesh is None:
                import numpy as np
                from jax.sharding import Mesh
                mesh = Mesh(np.array(aot_devices()[:1]), ("x",))
            info = aot_compile(f, (spec_like(consts), spec_like(state)), mesh, shardings)
            rec.emit({**case, **info, "inner": inner})
            return None
        med, mn, mx, first_s, out = time_call(
            f, (consts, state), reps=rec.args.reps, warmup=rec.args.warmup)
        us = med / inner * 1e6
        r = {**case, "us": round(us, 3), "us_min": round(mn / inner * 1e6, 3),
             "us_max": round(mx / inner * 1e6, 3), "first_call_s": round(first_s, 3),
             "inner": inner, "reps": rec.args.reps}
        if derive is not None:
            r.update(derive(us))
        rec.emit(r)
        return out
    except Exception as e:  # noqa: BLE001
        rec.error(case, e)
        return None


def fold(x, out, eps=1e-30):
    """x + eps*f(out): a real but numerically negligible dependency on `out`."""
    import jax.numpy as jnp
    s = jnp.sum(out.reshape(-1)[:8].astype(jnp.float32)) * eps
    return x + s.astype(x.dtype)


def opaque_zero(out):
    """int32 scalar that is always 0 at runtime but depends on `out`."""
    import jax.numpy as jnp
    v = out.reshape(-1)[0].astype(jnp.float32)
    return jnp.where(v == jnp.float32(1.2345e-37), 1, 0).astype(jnp.int32)


def rel_err(a, b) -> float:
    import jax.numpy as jnp
    a = jnp.asarray(a, jnp.float32)
    b = jnp.asarray(b, jnp.float32)
    den = float(jnp.max(jnp.abs(b))) or 1.0
    return float(jnp.max(jnp.abs(a - b))) / den


def spec_like(tree):
    """ShapeDtypeStruct pytree for arrays or ShapeDtypeStructs."""
    import jax
    return jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), tree)


@contextlib.contextmanager
def force_tpu_interpret():
    from jax.experimental.pallas import tpu as pltpu
    with pltpu.force_tpu_interpret_mode(pltpu.InterpretParams(uninitialized_memory="zero")):
        yield
