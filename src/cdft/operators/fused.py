"""Fused device kernels for the divergence stencil and the Chebyshev recurrence (D-65).

Device path only: on the CPU every function here returns exactly what the eager code returns (the
CPU float64 path is the audit reference, D-60); on CUDA arithmetic may be re-associated as long as
the GPU-vs-CPU fingerprint diff stays <= 1e-10 (D-65).

Three tiers for ``sum_d D_d^T [w_d D_d phi]`` (D-38), chosen once per CUDA device at first use and
recorded (G5.5, never a silent fallback): ``compile`` (``torch.compile`` of the eager stencil,
spatial shape static, batch dynamic, needs Triton); ``gemm`` (each 1-D staggered derivative and its
transpose as a dense ``(n + 2p) x n`` matrix, ~5 launches per axis instead of ~18, but dense flops
grow as ``n`` per point against 8 taps, so ``auto`` takes it only if a first-use probe beats
eager); ``eager`` (:func:`cdft.operators.divergence._weighted_divergence_eager`).

``CDFT_FUSED`` = ``auto`` (default) | ``compile`` | ``gemm`` | ``eager``. A forced tier that cannot
run raises; ``auto`` degrades in the order above and records every reason.
"""

from __future__ import annotations

import os
import statistics
import time
from functools import lru_cache
from typing import Callable

import torch

from .divergence import _weighted_divergence_eager, staggered_coefficients

__all__ = [
    "ENV_FUSED",
    "TIERS",
    "chebyshev_update",
    "device_weighted_divergence",
    "gemm_weighted_divergence",
    "kernel_record",
    "reset_registry",
    "resolve_tier",
    "staggered_matrices",
]

#: Environment variable selecting the device tier (``auto``, ``compile``, ``gemm``, ``eager``).
#: Read once per device at first use; :func:`reset_registry` makes it be read again.
ENV_FUSED = "CDFT_FUSED"

#: The tiers, in the order ``auto`` tries them.
TIERS = ("compile", "gemm", "eager")

#: Dynamo recompile limit raised to at least this: one code object serves every (grid, batch 1/>1)
#: pair and a profile visits ~16 grids, so the default 8 would silently drop to eager.
_RECOMPILE_LIMIT = 256

#: Timing probe of ``auto``: calls per candidate after one warm-up, median compared.
_PROBE_REPEATS = 3

#: Per-device record: ``{"tier", "reason", "divergence", "chebyshev", "compile_s", "probe"}``.
_REGISTRY: dict[str, dict[str, object]] = {}

#: Compiled callables, keyed by what Dynamo specialises on anyway.
_COMPILED: dict[tuple, Callable] = {}


# --- Tier selection ---


def _requested() -> str:
    """The tier requested by ``CDFT_FUSED`` (``auto`` if unset); an unknown value raises."""
    value = os.environ.get(ENV_FUSED, "").strip().lower() or "auto"
    if value not in ("auto", *TIERS):
        raise ValueError(f"{ENV_FUSED}={value!r}: expected one of auto, compile, gemm, eager")
    return value


def _describe(exc: BaseException) -> str:
    """Exception text for the record, first 400 characters."""
    return f"{type(exc).__name__}: {exc}"[:400]


def resolve_tier(
    requested: str,
    is_cuda: bool,
    probe_compile: Callable[[], str | None],
    probe_gemm: Callable[[], str | None],
) -> tuple[str, str]:
    """Return ``(tier, reason)`` for a device; pure logic, testable without a card.

    The probes return ``None`` when the tier may be used, else the reason it may not. On the CPU the
    answer is ``eager`` whatever is requested (the audit path, D-60); a forced tier whose probe
    refuses raises with that reason (G5.5).
    """
    if not is_cuda:
        return "eager", "CPU: the eager float64 path is the audit reference (D-60); fused tiers are device-only"
    if requested == "eager":
        return "eager", f"forced by {ENV_FUSED}=eager"
    if requested in ("compile", "gemm"):
        refusal = (probe_compile if requested == "compile" else probe_gemm)()
        if refusal is not None:
            raise RuntimeError(f"{ENV_FUSED}={requested} cannot run on this device: {refusal}")
        return requested, f"forced by {ENV_FUSED}={requested}"
    reasons = []
    refusal = probe_compile()
    if refusal is None:
        return "compile", "auto: torch.compile probe succeeded"
    reasons.append(f"compile unavailable ({refusal})")
    refusal = probe_gemm()
    if refusal is None:
        return "gemm", "auto: " + "; ".join(reasons) + "; gemm faster than eager in the probe"
    reasons.append(f"gemm not taken ({refusal})")
    return "eager", "auto: " + "; ".join(reasons)


def _probe_compile(device: torch.device) -> str | None:
    """Compile and run a tiny pointwise function on ``device``; the exception text if that fails."""
    try:
        _raise_recompile_limit()
        probe = torch.compile(lambda x: x * 2.0 + 1.0, dynamic=False)
        out = probe(torch.ones(4, device=device, dtype=torch.float64))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if not bool((out == 3.0).all()):
            return "compiled probe returned a wrong value"
    except Exception as exc:  # noqa: BLE001 - any failure is a recorded reason (G5.5)
        return _describe(exc)
    return None


def _time(fn: Callable[[], object], device: torch.device) -> float:
    """Median wall time of ``fn`` over :data:`_PROBE_REPEATS` synchronised calls after one warm-up."""
    fn()
    samples = []
    for _ in range(_PROBE_REPEATS):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def _raise_recompile_limit() -> None:
    """Raise Dynamo's per-code-object recompile limit to :data:`_RECOMPILE_LIMIT` (name varies by build)."""
    import torch._dynamo

    for name in ("recompile_limit", "cache_size_limit"):
        if hasattr(torch._dynamo.config, name):
            setattr(torch._dynamo.config, name, max(getattr(torch._dynamo.config, name), _RECOMPILE_LIMIT))


def _record_for(box: torch.Tensor, face_weights, spacing: float, accuracy: int, shape) -> dict[str, object]:
    """The device's record, selecting the tier on first use with ``box`` as the probe input."""
    device = box.device
    key = str(device)
    record = _REGISTRY.get(key)
    if record is not None:
        return record
    probe: dict[str, object] = {}

    def probe_gemm() -> str | None:
        """Refuse GEMM unless it runs and is faster than eager on this box."""
        try:
            eager_s = _time(lambda: _weighted_divergence_eager(box, face_weights, spacing, accuracy, shape), device)
            gemm_s = _time(lambda: gemm_weighted_divergence(box, face_weights, spacing, accuracy, shape), device)
        except Exception as exc:  # noqa: BLE001 - recorded
            return _describe(exc)
        probe.update({"shape": list(box.shape), "eager_s": eager_s, "gemm_s": gemm_s})
        if _requested() == "gemm" or gemm_s < eager_s:
            return None
        return f"probe on {list(box.shape)}: gemm {gemm_s * 1e3:.2f} ms >= eager {eager_s * 1e3:.2f} ms"

    tier, reason = resolve_tier(_requested(), device.type == "cuda", lambda: _probe_compile(device), probe_gemm)
    record = {
        "tier": tier,
        "reason": reason,
        "divergence": tier,
        "chebyshev": {"compile": "compile", "gemm": "alpha_chain", "eager": "eager"}[tier],
        "compile_s": 0.0,
        "n_compiled": 0,
        "probe": probe,
    }
    _REGISTRY[key] = record
    return record


def kernel_record(device: torch.device | str) -> dict[str, object]:
    """The tier record of ``device`` for ``measurements["fused_kernels"]``.

    ``"unselected"`` before first use on CUDA; always ``eager`` on the CPU.
    """
    device = torch.device(device)
    if device.type != "cuda":
        tier, reason = resolve_tier("eager", False, lambda: None, lambda: None)
        return {"tier": tier, "reason": reason}
    record = _REGISTRY.get(str(device))
    if record is None:
        return {"tier": "unselected", "reason": "no divergence apply on this device yet"}
    return {k: (dict(v) if isinstance(v, dict) else v) for k, v in record.items()}


def reset_registry() -> None:
    """Forget every tier choice and compiled callable (benchmarks, tests); ``CDFT_FUSED`` is re-read."""
    _REGISTRY.clear()
    _COMPILED.clear()


# --- Tier (b): dense staggered-derivative matrices contracted by GEMM ---


@lru_cache(maxsize=64)
def _staggered_matrices_cpu(extent: int, spacing: float, accuracy: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(D, D^T)`` for one axis in float64 on the CPU; see :func:`staggered_matrices`."""
    half = accuracy // 2
    scale = 1.0 / spacing
    forward = torch.zeros((extent + accuracy, extent), dtype=torch.float64)
    for row in range(extent + accuracy):
        face = row - half
        for index, coefficient in enumerate(staggered_coefficients(accuracy)):
            node = face + 1 + (index - half)
            if 0 <= node < extent:
                # The same float the eager stencil multiplies by.
                forward[row, node] = coefficient * scale
    return forward, forward.T.contiguous()


def staggered_matrices(
    extent: int, spacing: float, accuracy: int, device: torch.device | str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense ``D`` (``(extent + accuracy) x extent``) and its exact transpose, float64 on ``device``.

    The matrix of :func:`cdft.operators.divergence.staggered_derivative` on a zero-padded axis (row
    ``r`` is face ``r - p``, entries ``c_k / h`` at node ``face + 1 + k``, D-38), so the Dirichlet
    halo is built in. Cached per (extent, spacing, accuracy).
    """
    forward, transpose = _staggered_matrices_cpu(int(extent), float(spacing), int(accuracy))
    device = torch.device(device)
    if device.type == "cpu":
        return forward, transpose
    return _staggered_matrices_on(int(extent), float(spacing), int(accuracy), str(device))


@lru_cache(maxsize=64)
def _staggered_matrices_on(extent: int, spacing: float, accuracy: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Device copies of :func:`_staggered_matrices_cpu`, cached per device."""
    forward, transpose = _staggered_matrices_cpu(extent, spacing, accuracy)
    return forward.to(device), transpose.to(device)


def gemm_weighted_divergence(
    box: torch.Tensor,
    face_weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    spacing: float,
    accuracy: int,
    shape: tuple[int, int, int],
) -> torch.Tensor:
    """``sum_d D_d^T [w_d D_d phi]`` with each 1-D derivative as one dense matrix product (tier b).

    Per axis: move the axis last (a view), ``x @ D^T``, multiply by the moved weight in place,
    ``@ D``, move back and add. Equal to the eager stencil up to summation order only (BLAS blocks
    its inner products and the transpose sums faces in the opposite order), so ~1e-13 relative and
    never used on the CPU production path.
    """
    out: torch.Tensor | None = None
    first = box.dim() - 3
    for axis in range(3):
        forward, transpose = staggered_matrices(shape[axis], spacing, accuracy, box.device)
        moved = torch.movedim(box, first + axis, -1)
        faces = torch.matmul(moved, transpose)  # (..., n + accuracy): D applied along the axis
        faces.mul_(torch.movedim(face_weights[axis], axis, -1))
        term = torch.movedim(torch.matmul(faces, forward), -1, first + axis)
        out = term.contiguous() if out is None else out.add_(term)
    assert out is not None
    return out


# --- Tier (a): torch.compile of the eager stencil ---


def _stencil_taps(spacing: float, accuracy: int) -> tuple[tuple[tuple[int, float], ...], tuple[tuple[int, float], ...]]:
    """``((start, weight), ...)`` of the forward and transpose slices, exactly as the eager loops build them."""
    half = accuracy // 2
    scale = 1.0 / spacing
    coefficients = staggered_coefficients(accuracy)
    forward = tuple((accuracy - half + 1 + (i - half), c * scale) for i, c in enumerate(coefficients))
    transpose = tuple((half - 1 - (i - half), c * scale) for i, c in enumerate(coefficients))
    return forward, transpose


def _divergence_traceable(box, w0, w1, w2, forward_taps, transpose_taps, accuracy: int, shape):
    """The eager stencil of D-38 with its coefficients passed in, for Dynamo to trace (tier a).

    Same slices and term order as ``_weighted_divergence_eager``; the coefficients arrive as
    constants so Dynamo does not trace the exact-rational solve.
    """
    weights = (w0, w1, w2)
    out = None
    for axis in range(3):
        sizes = list(box.shape)
        sizes[axis + 1] += 2 * accuracy
        padded = torch.zeros(sizes, device=box.device, dtype=box.dtype)
        index = [slice(None)] * 4
        index[axis + 1] = slice(accuracy, accuracy + shape[axis])
        padded[tuple(index)] = box
        faces = None
        for start, weight in forward_taps:
            index = [slice(None)] * 4
            index[axis + 1] = slice(start, start + shape[axis] + accuracy)
            term = padded[tuple(index)]
            faces = term * weight if faces is None else faces + term * weight
        faces = faces * weights[axis]
        term_sum = None
        for start, weight in transpose_taps:
            index = [slice(None)] * 4
            index[axis + 1] = slice(start, start + shape[axis])
            term = faces[tuple(index)]
            term_sum = term * weight if term_sum is None else term_sum + term * weight
        out = term_sum if out is None else out + term_sum
    return out


def _compiled_divergence(record: dict[str, object], box: torch.Tensor, face_weights, spacing, accuracy, shape):
    """Run the compiled stencil for this (device, spatial shape, spacing, order); compile on first use.

    The batch is flattened to one leading dimension and marked dynamic, so one grid compiles at most
    twice (Dynamo specialises batch 1). First-call wall time goes into ``compile_s``.
    """
    import torch._dynamo

    batch = box.shape[: box.dim() - 3]
    flat = box.reshape(-1, *shape)
    key = (str(box.device), tuple(shape), float(spacing), int(accuracy), flat.shape[0] == 1)
    start = time.perf_counter()
    fn = _COMPILED.get(key)
    first_call = fn is None
    if fn is None:
        _raise_recompile_limit()
        fn = torch.compile(_divergence_traceable, dynamic=False)
        _COMPILED[key] = fn
    if flat.shape[0] > 1:
        torch._dynamo.mark_dynamic(flat, 0)
    forward_taps, transpose_taps = _stencil_taps(float(spacing), int(accuracy))
    out = fn(flat, *face_weights, forward_taps, transpose_taps, int(accuracy), tuple(int(s) for s in shape))
    if first_call:
        if box.device.type == "cuda":
            torch.cuda.synchronize(box.device)
        record["compile_s"] = float(record["compile_s"]) + (time.perf_counter() - start)  # type: ignore[arg-type]
        record["n_compiled"] = int(record["n_compiled"]) + 1  # type: ignore[call-overload]
    return out.reshape(*batch, *shape)


def device_weighted_divergence(
    box: torch.Tensor,
    face_weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    spacing: float,
    accuracy: int,
    shape: tuple[int, int, int],
) -> torch.Tensor:
    """The CUDA dispatch of :func:`cdft.operators.divergence.weighted_divergence`, tier per device.

    Under ``auto`` a compile failure at first real use demotes the device and records why; under a
    forced ``compile`` it raises (G5.5).
    """
    record = _record_for(box, face_weights, spacing, accuracy, shape)
    tier = record["divergence"]
    if tier == "compile":
        try:
            return _compiled_divergence(record, box, face_weights, spacing, accuracy, shape)
        except Exception as exc:  # noqa: BLE001 - recorded or re-raised
            if _requested() == "compile":
                raise RuntimeError(f"{ENV_FUSED}=compile: compiled stencil failed: {_describe(exc)}") from exc
            record["divergence"] = "eager"
            record["chebyshev"] = "alpha_chain"
            record["tier"] = "eager"
            record["reason"] = f"{record['reason']}; compiled stencil failed at first use ({_describe(exc)}), demoted to eager"
            tier = "eager"
    if tier == "gemm":
        return gemm_weighted_divergence(box, face_weights, spacing, accuracy, shape)
    return _weighted_divergence_eager(box, face_weights, spacing, accuracy, shape)


# --- The Chebyshev three-term recurrence ---


def _chebyshev_pointwise(applied: torch.Tensor, y: torch.Tensor, previous: torch.Tensor, coefficients: torch.Tensor):
    """``(applied - c y) a - b previous`` with ``(c, a, b)`` a device tensor (no value guards)."""
    return (applied - coefficients[0] * y) * coefficients[1] - coefficients[2] * previous


def chebyshev_update(
    applied: torch.Tensor, y: torch.Tensor, previous: torch.Tensor, c: float, a: float, b: float
) -> torch.Tensor:
    """One step of the scaled Chebyshev recurrence [F2]: ``(applied - c y) a - b previous``.

    The caller passes ``a = 2 sigma2 / e``, ``b = sigma sigma2``. On the CPU and under ``eager``
    the expression stands as written, so the bits do not move. On CUDA ``alpha_chain`` is
    ``add(y, alpha=-c).mul_(a).add_(previous, alpha=-b)`` (3 launches, no temporaries, ~1e-12
    relative, allowed by D-65) and ``compile`` is one fused kernel fed ``(c, a, b)`` by a pinned
    asynchronous copy, so no host sync and no recompilation per step.
    """
    if applied.device.type != "cuda":
        return (applied - c * y) * a - b * previous
    record = _REGISTRY.get(str(applied.device))
    mode = "eager" if record is None else record["chebyshev"]
    if mode == "compile":
        try:
            return _compiled_chebyshev(record, applied, y, previous, c, a, b)  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001 - recorded or re-raised
            if _requested() == "compile":
                raise RuntimeError(f"{ENV_FUSED}=compile: compiled recurrence failed: {_describe(exc)}") from exc
            record["chebyshev"] = "alpha_chain"  # type: ignore[index]
            record["reason"] = f"{record['reason']}; compiled recurrence failed ({_describe(exc)}), alpha chain used"  # type: ignore[index]
            mode = "alpha_chain"
    if mode == "alpha_chain":
        return applied.add(y, alpha=-c).mul_(a).add_(previous, alpha=-b)
    return (applied - c * y) * a - b * previous


def _compiled_chebyshev(record, applied, y, previous, c, a, b):
    """The compiled recurrence; both dimensions dynamic, so one compile serves every block."""
    import torch._dynamo

    key = ("chebyshev", str(applied.device))
    start = time.perf_counter()
    fn = _COMPILED.get(key)
    first_call = fn is None
    if fn is None:
        _raise_recompile_limit()
        fn = torch.compile(_chebyshev_pointwise, dynamic=False)
        _COMPILED[key] = fn
    coefficients = torch.tensor((c, a, b), dtype=applied.dtype).pin_memory().to(applied.device, non_blocking=True)
    for tensor in (applied, y, previous):
        for dim in range(tensor.dim()):
            if tensor.shape[dim] > 1:
                torch._dynamo.mark_dynamic(tensor, dim)
    out = fn(applied, y, previous, coefficients)
    if first_call:
        torch.cuda.synchronize(applied.device)
        record["compile_s"] = float(record["compile_s"]) + (time.perf_counter() - start)
        record["n_compiled"] = int(record["n_compiled"]) + 1
    return out
