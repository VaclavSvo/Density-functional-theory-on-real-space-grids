"""Card-agnostic device policy: what the solver finds, and the settings derived from it (D-70).

Every device-dependent setting is derived from a :class:`DeviceProfile` captured once per process
from the device the run resolved to, never from a hard-coded card:

========================  ==========================================================================
setting                   rule (CUDA)
========================  ==========================================================================
geometry_cache_bytes      25 % of total VRAM, capped at 8 GiB (:data:`GEOMETRY_CACHE_FRACTION`)
graph_memory_fraction     ``min(0.25, 0.5 * free / total)``: a graph pool never takes more than half
                          the memory free at capture (:data:`GRAPH_FRACTION_CAP`)
graph_pool_blocks         10 -- buffers the captured recurrence keeps alive, a property of the code
trsm_chunk                2^18 -- the cuBLAS ``trsm`` cliff is a property of the cuBLAS build
                          (D-62) and chunking is bitwise, so the safe width is kept on every card
use_cuda_graphs           free VRAM >= 2 GiB at capture (:data:`GRAPHS_MIN_FREE_BYTES`)
max_concurrent_solves     ``clamp(floor((free - 0.5 GiB) / 2 GiB), 1, 4)``, 2 GiB being the atom peak
========================  ==========================================================================

On a CPU the policy degrades to the CPU path and never guesses a card: every device feature off,
probes skipped (``None``), geometry cache 25 % of host RAM capped at 4 GiB with ``psutil``, else
1 GiB.

TF32 is off on every card, asserted (:func:`enforce_tf32_off`): a correctness rule, not a
performance choice -- TF32 computes float32 matrix products in a 19-bit mantissa, which the
precision policy does not allow. A caller that switched it on gets an exception, not an override.

:func:`apply_policy` is called by :func:`cdft.precision.configure_device` before the first tensor,
idempotent per device; the profile and settings go into every record via
:func:`cdft.precision.policy_record`. No environment variable is read here.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import os
import platform
import time

import torch

__all__ = [
    "DeviceProfile",
    "DerivedSettings",
    "capture_profile",
    "derive_settings",
    "apply_policy",
    "enforce_tf32_off",
    "applied_record",
    "describe",
    "GIB",
    "GEOMETRY_CACHE_FRACTION",
    "ENV_GEOMETRY_CACHE_BYTES",
    "cgroup_memory_limit_bytes",
    "GEOMETRY_CACHE_CAP_BYTES",
    "HOST_CACHE_CAP_BYTES",
    "HOST_CACHE_FALLBACK_BYTES",
    "GRAPH_FRACTION_CAP",
    "GRAPHS_MIN_FREE_BYTES",
    "SOLVE_PEAK_BYTES",
    "SOLVE_RESERVE_BYTES",
    "MAX_CONCURRENT_CAP",
    "PROBE_GEMM_N",
    "PROBE_COPY_BYTES",
]

GIB = 2**30

#: Fraction of the device's total memory the geometry cache may hold (quadrature, ``L``/``Lᵀ``).
GEOMETRY_CACHE_FRACTION = 0.25
#: Explicit geometry-cache budget in bytes, overriding the derived one; recorded by ``apply_policy``.
ENV_GEOMETRY_CACHE_BYTES = "CDFT_GEOMETRY_CACHE_BYTES"
#: Absolute cap on the device geometry cache: a 48 GiB card does not need 12 GiB of cached geometry.
GEOMETRY_CACHE_CAP_BYTES = 8 * GIB
#: Cap on the host geometry cache (CPU path, from ``psutil``).
HOST_CACHE_CAP_BYTES = 4 * GIB
#: Host geometry cache when ``psutil`` is not installed (host RAM unknown; nothing is guessed).
HOST_CACHE_FALLBACK_BYTES = 1 * GIB
#: Largest graph-pool fraction of total memory.
GRAPH_FRACTION_CAP = 0.25
#: Free memory below which CUDA graphs are not attempted: a pool that cannot fit is a capture failure.
GRAPHS_MIN_FREE_BYTES = 2 * GIB
#: Per-solve peak assumed for concurrency: the measured atom peak rounded up.
SOLVE_PEAK_BYTES = 2 * GIB
#: Memory left for the CUDA context and the allocator's slack when counting concurrent solves.
SOLVE_RESERVE_BYTES = GIB // 2
#: Upper bound on concurrent solves on one card (launch contention dominates beyond a few).
MAX_CONCURRENT_CAP = 4
#: Edge of the square float64 GEMM of the throughput probe (3 timed products after one warm-up).
PROBE_GEMM_N = 2048
#: Size of the device-to-device copy of the bandwidth probe.
PROBE_COPY_BYTES = 128 * 2**20


@dataclasses.dataclass(frozen=True, slots=True)
class DeviceProfile:
    """What the process found on its device, captured once (:func:`capture_profile`).

    ``fp64_gflops`` is ``2 n^3 / t / 1e9`` for a float64 ``n x n`` GEMM, median of three;
    ``bandwidth_gbs`` is ``2 B / t / 1e9`` for a device-to-device copy of ``B`` bytes (read once,
    written once). Both ``None`` on the CPU and when the card is too full to probe. ``sm_count`` is
    the number of CUDA devices visible.
    """

    name: str
    device: str
    capability: str
    total_bytes: int
    free_bytes: int | None
    sm_count: int
    fp64_gflops: float | None
    bandwidth_gbs: float | None
    torch_version: str
    driver: str

    @property
    def is_cuda(self) -> bool:
        """Whether this is a CUDA device profile."""
        return self.device.startswith("cuda")

    def as_dict(self) -> dict[str, object]:
        """Plain-data form (the provenance and benchmark JSON)."""
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class DerivedSettings:
    """The device settings derived from a :class:`DeviceProfile` (rules in the module docstring).

    ``trsm_chunk = 0`` means no chunking (the CPU); ``graph_memory_fraction = 0`` means no graphs.
    """

    geometry_cache_bytes: int
    graph_memory_fraction: float
    graph_pool_blocks: int
    trsm_chunk: int
    use_cuda_graphs: bool
    max_concurrent_solves: int
    source: str

    def as_dict(self) -> dict[str, object]:
        """Plain-data form (the provenance and benchmark JSON)."""
        return dataclasses.asdict(self)


def _canonical(device: torch.device | str) -> torch.device:
    """``cuda`` -> ``cuda:<current>``; every other device unchanged (one cache entry per device)."""
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def cgroup_memory_limit_bytes() -> int | None:
    """The memory limit of this process's cgroup (v1 or v2), or ``None`` when there is none.

    A container's limit is what an OOM kill enforces, and it can be well below ``MemTotal``
    (5.8 GiB against 8 GiB on a cgroup-limited host); the CPU cache budget must see it.
    """
    import pathlib

    paths = ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]
    try:
        for line in pathlib.Path("/proc/self/cgroup").read_text().splitlines():
            parts = line.split(":")
            if len(parts) == 3 and parts[1] == "memory" and parts[2] != "/":
                paths.append(f"/sys/fs/cgroup/memory{parts[2]}/memory.limit_in_bytes")
            if len(parts) == 3 and parts[1] == "" and parts[2] != "/":
                paths.append(f"/sys/fs/cgroup{parts[2]}/memory.max")
    except OSError:
        pass
    limits = []
    for path in paths:
        try:
            text = pathlib.Path(path).read_text().strip()
        except OSError:
            continue
        if text.isdigit() and int(text) < 2**60:
            limits.append(int(text))
    return min(limits) if limits else None


def _host_ram_bytes() -> int | None:
    """Host RAM available to this process: ``psutil``'s total, capped by the cgroup limit if any."""
    try:
        import psutil  # type: ignore[import-untyped]
    except ImportError:
        return None
    total = int(psutil.virtual_memory().total)
    limit = cgroup_memory_limit_bytes()
    return min(total, limit) if limit else total


def _probe_gflops(device: torch.device) -> float:
    """Median float64 GEMM throughput on ``device`` in GFLOPS (:data:`PROBE_GEMM_N`, 1 warm-up + 3)."""
    n = PROBE_GEMM_N
    generator = torch.Generator(device="cpu").manual_seed(0)
    a = torch.rand((n, n), dtype=torch.float64, generator=generator).to(device)
    b = torch.rand((n, n), dtype=torch.float64, generator=generator).to(device)
    torch.matmul(a, b)
    torch.cuda.synchronize(device)
    times = []
    for _ in range(3):
        start = time.perf_counter()
        torch.matmul(a, b)
        torch.cuda.synchronize(device)
        times.append(time.perf_counter() - start)
    del a, b
    t = sorted(times)[1]
    return 2.0 * n**3 / t / 1e9


def _probe_bandwidth(device: torch.device) -> float:
    """Median device-to-device copy bandwidth on ``device`` in GB/s (:data:`PROBE_COPY_BYTES`)."""
    count = PROBE_COPY_BYTES // 8
    src = torch.ones(count, dtype=torch.float64, device=device)
    dst = torch.empty_like(src)
    dst.copy_(src)
    torch.cuda.synchronize(device)
    times = []
    for _ in range(3):
        start = time.perf_counter()
        dst.copy_(src)
        torch.cuda.synchronize(device)
        times.append(time.perf_counter() - start)
    del src, dst
    t = sorted(times)[1]
    return 2.0 * PROBE_COPY_BYTES / t / 1e9


@functools.cache
def _capture(device: torch.device) -> DeviceProfile:
    """Capture the profile of one canonical device (cached: once per process per device)."""
    from .precision import _cuda_driver_version

    if device.type != "cuda":
        ram = _host_ram_bytes()
        return DeviceProfile(
            name=f"cpu:{platform.processor() or platform.machine()}",
            device=str(device),
            capability="n/a",
            total_bytes=ram if ram is not None else 0,
            free_bytes=None,
            sm_count=torch.cuda.device_count() if torch.cuda.is_available() else 0,
            fp64_gflops=None,
            bandwidth_gbs=None,
            torch_version=torch.__version__,
            driver="n/a",
        )
    props = torch.cuda.get_device_properties(device)
    free, total = torch.cuda.mem_get_info(device)
    gflops = bandwidth = None
    # The probes need ~0.3 GiB transiently; on a nearly full card they are skipped, not forced.
    if free >= 3 * (8 * PROBE_GEMM_N**2) + 2 * PROBE_COPY_BYTES + GIB // 4:
        gflops = _probe_gflops(device)
        bandwidth = _probe_bandwidth(device)
        torch.cuda.empty_cache()
    return DeviceProfile(
        name=str(props.name),
        device=str(device),
        capability=f"{props.major}.{props.minor}",
        total_bytes=int(total),
        free_bytes=int(free),
        sm_count=int(torch.cuda.device_count()),
        fp64_gflops=None if gflops is None else round(gflops, 3),
        bandwidth_gbs=None if bandwidth is None else round(bandwidth, 3),
        torch_version=torch.__version__,
        driver=_cuda_driver_version(),
    )


def capture_profile(device: torch.device | str) -> DeviceProfile:
    """Return the :class:`DeviceProfile` of ``device``, captured on the first call in the process.

    ``free_bytes`` is the memory free at capture, before the first solve, which is what the derived
    settings are sized against. A CUDA device on a machine without CUDA raises: never guess a card.
    """
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("a CUDA device profile was requested and torch reports no CUDA device")
    return _capture(_canonical(device))


def derive_settings(profile: DeviceProfile, host_ram_bytes: int | None = None) -> DerivedSettings:
    """Derive the device settings from ``profile`` by the rules of the module docstring. Pure.

    ``host_ram_bytes`` is used on a CPU profile only; ``None`` falls back to ``profile.total_bytes``
    (host RAM at capture, 0 when ``psutil`` was absent), and 0 to :data:`HOST_CACHE_FALLBACK_BYTES`.
    """
    from .eigen import chefsi
    from .precision import CUDA_TRSM_CHUNK

    if not profile.is_cuda:
        ram = host_ram_bytes if host_ram_bytes is not None else profile.total_bytes
        cache = (
            min(int(GEOMETRY_CACHE_FRACTION * ram), HOST_CACHE_CAP_BYTES) if ram else HOST_CACHE_FALLBACK_BYTES
        )
        return DerivedSettings(
            geometry_cache_bytes=cache,
            graph_memory_fraction=0.0,
            graph_pool_blocks=chefsi.GRAPH_POOL_BLOCKS_DEFAULT,
            trsm_chunk=0,
            use_cuda_graphs=False,
            max_concurrent_solves=1,
            source="cpu: device features off; cache from host RAM"
            + ("" if ram else f" unknown (psutil absent), {HOST_CACHE_FALLBACK_BYTES // GIB} GiB"),
        )
    total = int(profile.total_bytes)
    free = int(profile.free_bytes if profile.free_bytes is not None else total)
    cache = min(int(GEOMETRY_CACHE_FRACTION * total), GEOMETRY_CACHE_CAP_BYTES)
    fraction = min(GRAPH_FRACTION_CAP, 0.5 * free / total) if total > 0 else 0.0
    graphs = free >= GRAPHS_MIN_FREE_BYTES
    concurrent = (free - SOLVE_RESERVE_BYTES) // SOLVE_PEAK_BYTES
    concurrent = int(max(1, min(MAX_CONCURRENT_CAP, concurrent)))
    return DerivedSettings(
        geometry_cache_bytes=cache,
        graph_memory_fraction=round(fraction if graphs else 0.0, 6),
        graph_pool_blocks=chefsi.GRAPH_POOL_BLOCKS_DEFAULT,
        trsm_chunk=CUDA_TRSM_CHUNK,
        use_cuda_graphs=graphs,
        max_concurrent_solves=concurrent,
        source=f"cuda: {profile.name} ({profile.capability}), {total / GIB:.2f} GiB total, "
        f"{free / GIB:.2f} GiB free at capture",
    )


def enforce_tf32_off() -> None:
    """Switch TF32 off for cuBLAS and cuDNN, raising if a caller had switched the matmul path on.

    Anything other than ``allow_tf32=False`` with ``float32_matmul_precision="highest"`` means
    somebody enabled TF32 matrix products, which is refused. ``cudnn.allow_tf32`` defaults to True
    and this package runs no convolutions, so it is set False silently.
    """
    matmul = bool(torch.backends.cuda.matmul.allow_tf32)
    precision = torch.get_float32_matmul_precision()
    if matmul or precision != "highest":
        raise RuntimeError(
            "TF32 matrix products are enabled (torch.backends.cuda.matmul.allow_tf32="
            f"{matmul}, float32_matmul_precision={precision!r}); this solver forbids TF32 on every "
            "card (D-70). Remove the setting from the calling code."
        )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


#: What :func:`apply_policy` did, per canonical device string (for the record and the print line).
_APPLIED: dict[str, dict[str, object]] = {}


def _apply_geometry_cache(limit: int, device: torch.device) -> str:
    """Hand the cache limit to ``cdft.operators.geometry_cache`` if that module exists.

    Imported lazily; its absence is recorded as ``"not available"``. The limit is kept per device
    type, so a later CPU apply never overwrites a limit a CUDA apply set.
    """
    try:
        from .operators import geometry_cache  # type: ignore[attr-defined]
    except ImportError:
        return "not available (cdft.operators.geometry_cache is not in this build)"
    setter = getattr(geometry_cache, "set_cache_limit_bytes", None)
    if setter is None:
        return "not available (geometry_cache has no set_cache_limit_bytes)"
    setter(device, limit)
    return f"applied {limit} bytes on {device.type}"


def apply_policy(device: torch.device | str) -> DerivedSettings:
    """Capture the profile of ``device``, derive its settings and write them where they are consumed.

    On CUDA: TF32 off (asserted), graph-pool limits into :mod:`cdft.eigen.chefsi`, graph permission
    and ``trsm`` chunk into :mod:`cdft.precision`. On every device: the geometry cache limit. On the
    CPU nothing else is touched, so the CPU path is unchanged. Idempotent; returns the settings.
    """
    device = _canonical(device)
    profile = capture_profile(device)
    settings = derive_settings(profile)
    key = str(device)
    if device.type == "cuda":
        enforce_tf32_off()
    if key in _APPLIED:
        return settings
    record: dict[str, object] = {}
    if device.type == "cuda":
        from . import precision
        from .eigen import chefsi

        chefsi.set_graph_memory_limits(settings.graph_memory_fraction, settings.graph_pool_blocks)
        precision.set_cuda_graphs_allowed(settings.use_cuda_graphs)
        precision.set_trsm_chunk(settings.trsm_chunk)
        record["chefsi"] = "graph limits applied"
        record["precision"] = "graph permission and trsm chunk applied"
        record["tf32"] = "off (asserted)"
    cache_bytes = settings.geometry_cache_bytes
    override = os.environ.get(ENV_GEOMETRY_CACHE_BYTES)
    if override is not None:
        # An explicit budget for a host the derivation cannot judge (a live diatomic solve beside a
        # full cache exceeds a 6 GiB cgroup); recorded, never silent (G5.5). A bad value raises.
        try:
            cache_bytes = int(override)
        except ValueError as exc:
            raise ValueError(f"{ENV_GEOMETRY_CACHE_BYTES}={override!r} is not an integer byte count") from exc
        if cache_bytes < 0:
            raise ValueError(f"{ENV_GEOMETRY_CACHE_BYTES}={override!r} is negative")
        record["geometry_cache_override"] = f"{ENV_GEOMETRY_CACHE_BYTES}={cache_bytes} (derived {settings.geometry_cache_bytes})"
    record["geometry_cache"] = _apply_geometry_cache(cache_bytes, device)
    _APPLIED[key] = record
    return settings


def applied_record(device: torch.device | str) -> dict[str, object]:
    """What :func:`apply_policy` wrote for ``device`` (empty before the first apply)."""
    return dict(_APPLIED.get(str(_canonical(device)), {}))


def describe(device: torch.device | str) -> str:
    """One printable line: the profile and the derived settings of ``device`` (the ``--device`` line)."""
    profile = capture_profile(device)
    settings = derive_settings(profile)
    return (
        f"device policy {json.dumps(profile.as_dict(), sort_keys=True)} -> "
        f"{json.dumps(settings.as_dict(), sort_keys=True)}"
    )
