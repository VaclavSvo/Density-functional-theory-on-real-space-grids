"""The precision policy and the execution-device seams, in one place.

Specified: float32 on the hot path (GPU only), float64 unconditionally for the Gram matrix,
orthonormalisation, the Rayleigh--Ritz subspace, every energy integral and the convergence test.
:class:`~contract.PrecisionConfig` refuses to relax the float64 stages. The float32 hot path
(D-60) is not wired yet: every stage runs in float64 on every device and :func:`hot_dtype` says so.
Also the device seams -- seeded random start, determinism, CSR index dtype, peak memory, the
provenance device block, the CUDA allocator setting (set at import) and the
:mod:`cdft.device_policy` hooks. Rationale: ``docs/03_METHOD.md (Part E)``.
"""

from __future__ import annotations

import functools
import json
import os
import subprocess
from typing import Sequence

import torch

from contract import Device, Precision, PrecisionConfig

__all__ = [
    "torch_dtype",
    "resolve_device",
    "hot_dtype",
    "accumulate_dtype",
    "subspace_dtype",
    "as_hot",
    "as_accumulate",
    "policy_record",
    "seeded_randn",
    "host_floats",
    "configure_device",
    "deterministic_mode",
    "csr_index_dtype",
    "solve_triangular_wide",
    "CUDA_TRSM_CHUNK",
    "reset_peak_memory",
    "peak_memory_bytes",
    "device_record",
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDA_CSR_INDEX_DTYPE",
    "ENV_CSR_INT64",
    "ENV_DETERMINISTIC_WARN_ONLY",
    "ENV_CUDA_GRAPHS",
    "ENV_LANCZOS_REUSE",
    "env_switch",
    "cuda_graphs_enabled",
    "set_cuda_graphs_allowed",
    "set_trsm_chunk",
    "release_device_memory",
    "cuda_alloc_conf",
    "CUDA_ALLOC_CONF",
    "ENV_CUDA_ALLOC_CONF",
    "ENV_ALLOC_CONF",
]

#: The CUDA caching-allocator setting this package asks for (D-69): expandable segments let a
#: segment grow in place instead of stranding freed blocks (the cause of a 1.34 GiB OOM).
CUDA_ALLOC_CONF = "expandable_segments:True"

#: Where torch's CUDA allocator reads its configuration, once, at the first CUDA allocation.
ENV_CUDA_ALLOC_CONF = "PYTORCH_CUDA_ALLOC_CONF"

#: The device-generic name newer torch builds also read; an explicit user value under either name
#: is left alone.
ENV_ALLOC_CONF = "PYTORCH_ALLOC_CONF"

# At import time, deliberately: the allocator reads the variable when CUDA is first used, so
# resolve_device/configure_device would be too late if torch had already initialised CUDA here.
# ``setdefault``: an explicit user value under either name wins. Harmless on a CPU-only machine.
if not os.environ.get(ENV_ALLOC_CONF):
    os.environ.setdefault(ENV_CUDA_ALLOC_CONF, CUDA_ALLOC_CONF)


def cuda_alloc_conf() -> str:
    """Return the requested allocator configuration from the environment, or ``"unset"``."""
    return os.environ.get(ENV_CUDA_ALLOC_CONF) or os.environ.get(ENV_ALLOC_CONF) or "unset"

#: cuBLAS workspace setting that makes cuBLAS deterministic (required by
#: ``torch.use_deterministic_algorithms(True)`` on CUDA >= 10.2). Set with ``setdefault``; the
#: value in force is recorded in provenance.
CUBLAS_WORKSPACE_CONFIG = ":4096:8"

#: Index dtype of the interpolation CSR operators on CUDA: cuSPARSE accepts int32 indices, which
#: halves the index memory. The CPU path keeps int64.
CUDA_CSR_INDEX_DTYPE = torch.int32

#: ``CDFT_CSR_INT64=1`` switches the CUDA CSR indices back to int64, for a torch/cuSPARSE build
#: that refuses int32 indices.
ENV_CSR_INT64 = "CDFT_CSR_INT64"

#: ``CDFT_DETERMINISTIC_WARN_ONLY=1`` turns deterministic-algorithm violations on CUDA into
#: warnings. The mode that ran is recorded in provenance (G5.5).
ENV_DETERMINISTIC_WARN_ONLY = "CDFT_DETERMINISTIC_WARN_ONLY"


#: ``CDFT_CUDA_GRAPHS=0`` switches off the CUDA-graph replay of the Chebyshev filter (D-67).
#: Default on for CUDA, never consulted on the CPU (always the eager recurrence). The outcome goes
#: into ``measurements["cuda_graphs"]`` (G5.5).
ENV_CUDA_GRAPHS = "CDFT_CUDA_GRAPHS"

#: ``CDFT_LANCZOS_REUSE=0`` makes every SCF iteration run a fresh Lanczos bound estimate instead
#: of reusing the previous one (D-68). Default on, every device; each iteration's choice goes into
#: ``measurements["lanczos_bound_history"]``.
ENV_LANCZOS_REUSE = "CDFT_LANCZOS_REUSE"

#: Two further switches live with their subjects: ``CDFT_GEOMETRY_CACHE`` (D-66) at
#: :data:`cdft.operators.geometry_cache.ENV_GEOMETRY_CACHE` and ``CDFT_FUSED`` (D-65) at
#: :data:`cdft.operators.fused.ENV_FUSED`.


def env_switch(name: str, default: bool) -> bool:
    """Read an on/off environment switch: ``1/true/yes/on`` -> True, ``0/false/no/off`` -> False.

    Unset or empty gives ``default``; anything else raises, so a typo in a switch that changes the
    solver's path is never read silently as either setting (G5.5).
    """
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name}={raw!r} is not an on/off value (use 1/0, true/false, yes/no, on/off)")


def cuda_graphs_enabled(device: torch.device) -> bool:
    """Whether the Chebyshev filter may be replayed as a CUDA graph on ``device``.

    False off CUDA whatever :data:`ENV_CUDA_GRAPHS` says; on CUDA, the switch (default on) and the
    device policy's permission (:func:`set_cuda_graphs_allowed`).
    """
    return device.type == "cuda" and _CUDA_GRAPHS_ALLOWED and env_switch(ENV_CUDA_GRAPHS, True)


#: The device policy's graph permission; True until :func:`set_cuda_graphs_allowed` says otherwise.
_CUDA_GRAPHS_ALLOWED = True


def set_cuda_graphs_allowed(allowed: bool) -> None:
    """Record whether the derived device policy allows CUDA graphs.

    The environment switch can turn graphs off but never on against the policy.
    """
    global _CUDA_GRAPHS_ALLOWED
    _CUDA_GRAPHS_ALLOWED = bool(allowed)


def set_trsm_chunk(chunk: int) -> None:
    """Set :data:`CUDA_TRSM_CHUNK` from the derived device policy; CUDA only.

    Chunking is bitwise identical to the unchunked solve (D-62), so the width changes time, never a
    number. A non-positive value is refused: the CPU policy's ``0`` is never applied.
    """
    global CUDA_TRSM_CHUNK
    if int(chunk) <= 0:
        raise ValueError(f"trsm chunk {chunk!r} must be positive")
    CUDA_TRSM_CHUNK = int(chunk)


def release_device_memory(device: torch.device) -> None:
    """Return cached device memory between solves; a no-op on the CPU. Not recorded."""
    if device.type != "cuda":
        return
    import gc

    gc.collect()
    torch.cuda.empty_cache()


_DTYPES: dict[Precision, torch.dtype] = {
    Precision.FLOAT32: torch.float32,
    Precision.FLOAT64: torch.float64,
    Precision.FLOAT16: torch.float16,
}


def torch_dtype(precision: Precision) -> torch.dtype:
    """Map a contract :class:`~contract.Precision` to a torch dtype."""
    return _DTYPES[precision]


def resolve_device(device: Device) -> torch.device:
    """Resolve a contract :class:`~contract.Device` to a concrete torch device.

    ``AUTO`` selects CUDA when available and falls back to CPU; an explicit ``CUDA`` on a machine
    without one raises rather than falling back silently (G5.5). For a CUDA result
    ``CUBLAS_WORKSPACE_CONFIG`` is set here, which is before the first cuBLAS handle exists.
    """
    if device is Device.CPU:
        return torch.device("cpu")
    if device is Device.CUDA:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "device=cuda requested but torch reports no CUDA device; refusing to fall back "
                "silently to CPU (gate G5.5). Use device=auto to accept whatever is present."
            )
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG)
        return torch.device("cuda")
    if torch.cuda.is_available():
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG)
        return torch.device("cuda")
    return torch.device("cpu")


def configure_device(device: torch.device, deterministic: bool = False) -> str:
    """Apply the determinism settings for ``device`` and return the mode now in force.

    Called by both solvers before any tensor is created. On CUDA, ``CUBLAS_WORKSPACE_CONFIG`` is
    always set (it must precede the first cuBLAS handle); ``deterministic`` then selects
    ``torch.use_deterministic_algorithms(True)`` -- ``warn_only`` under
    :data:`ENV_DETERMINISTIC_WARN_ONLY`, the audit mode of G5.2/G5.5 and roughly 2x slower -- or
    torch's default kernels, switching the flag back off if an earlier solve in this process turned
    it on. On the CPU nothing is changed whatever ``deterministic`` says: the CPU float64 path is
    the audit reference. Also the one place the derived device policy is applied
    (:func:`cdft.device_policy.apply_policy`: TF32 asserted off, graph and cache limits).
    """
    from .device_policy import apply_policy

    if device.type != "cuda" or torch.cuda.is_available():
        # A CUDA device on a machine without one only reaches here from a test that fakes the
        # switch; resolve_device refuses it for every real solve (G5.5).
        apply_policy(device)
    if device.type == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG)
        if deterministic:
            warn_only = env_switch(ENV_DETERMINISTIC_WARN_ONLY, False)
            torch.use_deterministic_algorithms(True, warn_only=warn_only)
        elif torch.are_deterministic_algorithms_enabled():
            torch.use_deterministic_algorithms(False)
    return deterministic_mode()


def deterministic_mode() -> str:
    """Return ``"strict"``, ``"warn_only"`` or ``"off"``: torch's deterministic-algorithm state now."""
    if not torch.are_deterministic_algorithms_enabled():
        return "off"
    return "warn_only" if torch.is_deterministic_algorithms_warn_only_enabled() else "strict"


def seeded_randn(
    shape: int | Sequence[int],
    generator: torch.Generator | None,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Draw a standard-normal tensor from ``generator`` and place it on ``device``.

    Every seeded random start in the solver goes through here. Draw on the generator's own device
    (the CPU for every generator this code creates), then ``.to(device)``: a CUDA generator draws a
    different stream from the same seed, so drawing on the CPU is what keeps a CPU run and a GPU
    run bit-identical (D-60). Do not replace this with ``torch.randn(..., device=...)``.
    """
    source = generator.device if generator is not None else torch.device("cpu")
    size = (shape,) if isinstance(shape, int) else tuple(shape)
    draw = torch.randn(size, device=source, dtype=dtype, generator=generator)
    return draw.to(device)


def host_floats(*values: torch.Tensor | float) -> list[float]:
    """Read several device scalars (and small vectors) to the host in one transfer.

    Arguments are tensors of any shape or Python numbers; the result is their elements flattened in
    argument order, cast to float64 and read with one ``tolist()`` -- a single host synchronisation
    on CUDA instead of one per ``float(x)``, and the same numbers.
    """
    tensors = [v for v in values if isinstance(v, torch.Tensor)]
    if not tensors:
        return [float(v) for v in values]
    device = tensors[0].device
    parts = [
        v.detach().reshape(-1).to(device=device, dtype=torch.float64)
        if isinstance(v, torch.Tensor)
        else torch.tensor([float(v)], dtype=torch.float64, device=device)
        for v in values
    ]
    return torch.cat(parts).tolist()


#: Largest number of right-hand sides per cuBLAS ``trsm`` call on CUDA (D-62). Above roughly 2^19
#: right-hand sides cuBLAS falls off a 5000x performance cliff; chunking at 2^18 stays below it and
#: is bitwise identical, each column being an independent forward substitution.
CUDA_TRSM_CHUNK = 2**18


def solve_triangular_wide(
    factor: torch.Tensor, block: torch.Tensor, *, upper: bool = False
) -> torch.Tensor:
    """``torch.linalg.solve_triangular(factor, block, upper=upper)`` for a wide ``block`` ``(k, n)``.

    On CUDA with more than :data:`CUDA_TRSM_CHUNK` columns the solve is applied in column chunks of
    at most that width, sidestepping a cuBLAS ``trsm`` cliff. The result is bitwise the unchunked
    one (columns are independent); the CPU path calls ``solve_triangular`` directly.
    """
    n = block.shape[-1]
    if block.device.type != "cuda" or n <= CUDA_TRSM_CHUNK:
        return torch.linalg.solve_triangular(factor, block, upper=upper)
    out = torch.empty_like(block)
    for start in range(0, n, CUDA_TRSM_CHUNK):
        stop = min(start + CUDA_TRSM_CHUNK, n)
        out[..., start:stop] = torch.linalg.solve_triangular(factor, block[..., start:stop], upper=upper)
    return out


def csr_index_dtype(device: torch.device) -> torch.dtype:
    """Index dtype for a sparse CSR operator on ``device``.

    int64 on the CPU (unchanged). On CUDA :data:`CUDA_CSR_INDEX_DTYPE` (int32), unless
    :data:`ENV_CSR_INT64` is set, in which case int64.
    """
    if device.type == "cpu":
        return torch.int64
    return torch.int64 if env_switch(ENV_CSR_INT64, False) else CUDA_CSR_INDEX_DTYPE


def reset_peak_memory(device: torch.device) -> None:
    """Reset the CUDA peak-memory counter at the start of a solve; nothing on the CPU."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_bytes(device: torch.device) -> int | None:
    """Return ``torch.cuda.max_memory_allocated`` since the last reset, or ``None`` off CUDA.

    ``None`` on the CPU rather than an RSS figure: the two are different quantities.
    """
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    return None


@functools.lru_cache(maxsize=1)
def _cuda_driver_version() -> str:
    """Return the NVIDIA driver version, or ``"unavailable"``; never raises.

    torch exposes the CUDA *toolkit* version but not the driver, so NVML is tried first and
    ``nvidia-smi`` second. Cached for the process: the driver cannot change under it and
    ``nvidia-smi`` is slow enough to matter per solve.
    """
    try:
        import pynvml  # type: ignore[import-not-found]

        pynvml.nvmlInit()
        try:
            version = pynvml.nvmlSystemGetDriverVersion()
        finally:
            pynvml.nvmlShutdown()
        return version.decode() if isinstance(version, bytes) else str(version)
    except Exception:  # noqa: BLE001 - optional dependency, any failure means "try the next source"
        pass
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        line = completed.stdout.strip().splitlines()[0].strip() if completed.stdout.strip() else ""
        return line or "unavailable"
    except (OSError, subprocess.SubprocessError, IndexError):
        return "unavailable"


def device_record(device: torch.device) -> dict[str, str]:
    """Return the device block of the provenance record (G5.1), every value a non-empty string.

    A GPU record is distinguishable from a CPU one by this block alone.
    """
    import platform

    record = {
        "torch_cuda_version": str(torch.version.cuda or "none"),
        "deterministic_algorithms": deterministic_mode(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG") or "unset",
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(index)
        record.update(
            device=f"cuda:{index}",
            device_name=str(torch.cuda.get_device_name(index)),
            device_capability=f"{major}.{minor}",
            cuda_driver_version=_cuda_driver_version(),
        )
    else:
        record.update(
            device=str(device),
            device_name=f"cpu:{platform.processor() or platform.machine()}",
            device_capability="n/a",
            cuda_driver_version="n/a",
        )
    return record


def hot_dtype(config: PrecisionConfig, device: torch.device) -> torch.dtype:
    """Return the dtype the hot path actually runs in: float64 on every device, for now.

    ``config.hot_path`` is not wired (D-60); returning it here would make G5.1 provenance claim
    float32 for a float64 run. The CPU stays float64 regardless -- it is the audit path of G2.10.
    """
    return torch.float64


def accumulate_dtype(config: PrecisionConfig) -> torch.dtype:
    """Return the dtype every integral and energy is accumulated in: always float64."""
    return torch_dtype(config.reduction)


def subspace_dtype(config: PrecisionConfig) -> torch.dtype:
    """Return the dtype of the Gram matrix, orthonormalisation and Rayleigh--Ritz: always float64."""
    return torch_dtype(config.subspace)


def as_hot(tensor: torch.Tensor, config: PrecisionConfig, device: torch.device) -> torch.Tensor:
    """Cast a tensor to the hot-path dtype and move it to the execution device."""
    return tensor.to(device=device, dtype=hot_dtype(config, device))


def as_accumulate(tensor: torch.Tensor, config: PrecisionConfig) -> torch.Tensor:
    """Cast a tensor to the accumulation dtype (float64), leaving its device alone."""
    return tensor.to(dtype=accumulate_dtype(config))


def policy_record(config: PrecisionConfig, device: torch.device) -> dict[str, str]:
    """Return the precision policy as flat strings for the provenance block of gate G5.1.

    Recorded per run, not per configuration: :func:`hot_dtype` resolves differently on CPU and GPU
    and the record must say what the run actually did. ``device_profile`` and ``derived_settings``
    are compact JSON of :mod:`cdft.device_policy`.
    """
    from .device_policy import capture_profile, derive_settings

    if device.type == "cpu" or (device.type == "cuda" and torch.cuda.is_available()):
        profile = capture_profile(device)
        compact = {"sort_keys": True, "separators": (",", ":")}
        profile_text = json.dumps(profile.as_dict(), **compact)
        settings_text = json.dumps(derive_settings(profile).as_dict(), **compact)
    else:
        # A device no solve can run on here (a CUDA name without CUDA, ``meta``): guess nothing.
        profile_text = settings_text = f"n/a (no {device.type} device profile in this process)"
    from .device_policy import _APPLIED

    # By the device's own key: canonicalising ``cuda`` needs a CUDA runtime this host may lack.
    applied = _APPLIED.get(str(device), {})
    return {
        "cuda_alloc_conf": cuda_alloc_conf(),
        "device_profile": profile_text,
        "derived_settings": settings_text,
        # What apply_policy actually wrote, when it ran in this process: an explicit cache budget
        # (``CDFT_GEOMETRY_CACHE_BYTES``) must be readable from the record (G5.5).
        "geometry_cache_applied": str(applied.get("geometry_cache_override", applied.get("geometry_cache", "not applied"))),
        "hot_path": str(hot_dtype(config, device)).removeprefix("torch."),
        "reduction": str(accumulate_dtype(config)).removeprefix("torch."),
        "subspace": str(subspace_dtype(config)).removeprefix("torch."),
        "orthonormalisation": config.orthonormalisation.value,
        "audit_in_float64": str(config.audit_in_float64),
        "audit_energy_tol": repr(config.audit_energy_tol),
        "audit_density_tol": repr(config.audit_density_tol),
        "audit_force_tol": repr(config.audit_force_tol),
        "device_type": device.type,
    }
