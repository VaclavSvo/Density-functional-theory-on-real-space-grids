"""A process-wide cache of the geometry-only objects of a solve (D-66).

A scenario is solved 6--10 times per profile, each rebuilding the same density-independent
objects: the cusp factor, the cusp-aware quadrature (sphere rules, partition of unity, the
``L``/``L^T`` CSR operators, mass weights, bare operator terms), the Kohn--Sham step's
Hartree-split tensors and the Coulomb-cutoff kernel. This module keeps them between solves.

The key is a value, never an identity (:func:`geometry_key`): grid geometry (shape, spacing,
origin, both finite-difference orders, boundary, point count, a mask digest for a masked domain),
device string, dtype, the CSR index dtype in force there, the charges and positions as exact
float64 tuples, and :func:`constants_tag`. **The density never enters.** Nothing in a solve writes
into a cached object (D-66), so a hit reproduces a rebuild bit for bit.

One :class:`GeometryBundle` per key holds named parts built on first request; its size is recounted
whenever a part is added, because the objects fill lazy caches after they are built. Eviction is
least-recently-used per device type against :func:`cache_limit_bytes` (3 GiB CPU, 1.5 GiB CUDA,
overridable by a device profile). A diatomic bundle (~2.6 GiB) exceeds the CUDA default and is used
by its solve but not retained (``retained: False``) unless the budget is raised or ``L^T`` is the
CSC view of ``L`` (:data:`~cdft.operators.quadrature.TRANSPOSE_AS_CSC`).

``CDFT_GEOMETRY_CACHE=0`` builds every object afresh and stores nothing; :func:`bypass` does the
same inside a ``with`` block without disturbing stored entries. Every solve records
``measurements["geometry_cache"]``.
"""

from __future__ import annotations

import contextlib
import hashlib
import threading
import types
from collections import OrderedDict
from typing import Any, Callable, Hashable, Iterator

import torch

from ..precision import csr_index_dtype, env_switch

__all__ = [
    "CACHE_VERSION",
    "ENV_GEOMETRY_CACHE",
    "DEFAULT_CPU_LIMIT_BYTES",
    "DEFAULT_CUDA_LIMIT_BYTES",
    "GeometryBundle",
    "GeometryLease",
    "geometry_key",
    "constants_tag",
    "lease",
    "not_applicable_record",
    "enabled",
    "bypass",
    "clear",
    "stats",
    "set_cache_limit_bytes",
    "cache_limit_bytes",
    "tensor_bytes",
]

#: Version of what a cached geometry object contains. Bump it whenever a change to ``cusp.py``,
#: ``quadrature.py``, ``poisson.py`` or the geometry part of ``scf/step.py`` changes a cached
#: object's value without changing a class constant: the constants are in the key already, code is
#: not, so a process that reloads edited code would otherwise be served the old object.
CACHE_VERSION = 2

#: ``CDFT_GEOMETRY_CACHE=0`` switches the cache off (D-66): every solve builds its factor,
#: quadrature, step tensors and Poisson kernel afresh and stores nothing. Default on, every device.
ENV_GEOMETRY_CACHE = "CDFT_GEOMETRY_CACHE"

#: Default byte budget of the CPU entries: 3 GiB -- one diatomic bundle or four atomic ones.
DEFAULT_CPU_LIMIT_BYTES = 3 * 2**30

#: Default byte budget of the CUDA entries: 1.5 GiB (module docstring). Atomic bundles fit, a
#: diatomic one does not. A device profile overrides it (:func:`set_cache_limit_bytes`).
DEFAULT_CUDA_LIMIT_BYTES = 3 * 2**29


def constants_tag() -> tuple:
    """The class constants a cached object's value depends on, with :data:`CACHE_VERSION`.

    Read from the classes at call time, so a patched constant changes the key and cannot be served
    an object built under the old value.
    """
    from .cusp import CuspFactor
    from .quadrature import TRANSPOSE_AS_CSC, CuspQuadrature

    def public_constants(cls: type, names: tuple[str, ...]) -> tuple:
        return tuple((name, getattr(cls, name)) for name in names)

    return (
        ("version", CACHE_VERSION),
        public_constants(
            CuspQuadrature,
            (
                "degree", "taper_width_in_spacings", "taper_margin", "min_centre_in_spacings",
                "max_radius_in_spacings", "radial_order", "node_spacing", "node_spacing_overlapping",
                "cell_width_in_spacings", "far_refine", "far_edge_margin_in_spacings",
                "far_edge_width_in_spacings",
            ),
        ),
        public_constants(CuspFactor, ("_average_radius_cells", "_average_quadrature")),
        ("transpose_as_csc", bool(TRANSPOSE_AS_CSC)),
    )


def geometry_key(grid: Any, charges: tuple[float, ...], positions: torch.Tensor) -> tuple:
    """The value key of a grid and a set of nuclei (module docstring). Never the density.

    One host read of ``positions``; for a masked grid the mask's bytes are hashed per call.
    """
    geometry = grid.geometry
    mask_digest: str | None = None
    if grid.n_points != geometry.n_box_points:
        mask = grid.mask.detach().to("cpu").contiguous().numpy().tobytes()
        mask_digest = hashlib.sha256(mask).hexdigest()
    host_positions = positions.detach().to(device="cpu", dtype=torch.float64).reshape(-1).tolist()
    return (
        ("shape", tuple(int(n) for n in geometry.shape)),
        ("spacing", float(geometry.spacing)),
        ("origin", tuple(float(x) for x in geometry.origin)),
        ("fd_order", int(grid.fd_order)),
        ("gradient_order", int(grid.gradient_order)),
        ("boundary", str(getattr(grid.boundary, "value", grid.boundary))),
        ("n_points", int(grid.n_points)),
        ("mask", mask_digest),
        ("device", str(grid.device)),
        ("dtype", str(grid.dtype)),
        ("csr_index_dtype", str(csr_index_dtype(grid.device))),
        ("charges", tuple(float(z) for z in charges)),
        ("positions", tuple(float(x) for x in host_positions)),
        ("constants", constants_tag()),
    )


def short_hash(key: tuple) -> str:
    """A 12-hex-digit digest of a key, for the record (floats in ``float.hex`` form, so exact)."""

    def exact(value: Any) -> Any:
        if isinstance(value, float):
            return value.hex()
        if isinstance(value, tuple):
            return tuple(exact(v) for v in value)
        return value

    return hashlib.sha256(repr(exact(key)).encode("utf-8")).hexdigest()[:12]


# --- Byte accounting ---


def _tensor_storage_bytes(tensor: torch.Tensor, seen: set) -> int:
    """Bytes of a tensor's storage (or of a compressed sparse tensor's three arrays), counted once."""
    if tensor.layout in (torch.sparse_csr, torch.sparse_csc):
        if tensor.layout == torch.sparse_csr:
            parts = (tensor.crow_indices(), tensor.col_indices(), tensor.values())
        else:
            parts = (tensor.ccol_indices(), tensor.row_indices(), tensor.values())
        return sum(_tensor_storage_bytes(part, seen) for part in parts)
    if tensor.layout != torch.strided:
        return 0
    storage = tensor.untyped_storage()
    ident = (str(tensor.device), storage.data_ptr())
    if ident in seen:
        return 0
    seen.add(ident)
    return int(storage.nbytes())


#: Objects :func:`tensor_bytes` does not walk into.
_OPAQUE = (types.FunctionType, types.MethodType, types.BuiltinFunctionType, types.ModuleType, threading.Thread)


def tensor_bytes(obj: Any, seen: set | None = None, _depth: int = 0, _visited: set | None = None) -> int:
    """Sum of the bytes of the distinct tensors reachable from ``obj`` (attributes, containers).

    Shared storages count once. Walks ``__dict__``, ``__slots__``, lists, tuples, sets and dict
    values to a bounded depth; stops at modules, types and callables.
    """
    seen = set() if seen is None else seen
    visited = set() if _visited is None else _visited
    if isinstance(obj, torch.Tensor):
        return _tensor_storage_bytes(obj, seen)
    if _depth > 8 or obj is None or isinstance(obj, (str, bytes, int, float, bool, complex, type, torch.device, torch.dtype)):
        return 0
    if id(obj) in visited or isinstance(obj, _OPAQUE):
        return 0
    visited.add(id(obj))
    children: list[Any] = []
    if isinstance(obj, dict):
        children.extend(obj.values())
    elif isinstance(obj, (list, tuple, set, frozenset)):
        children.extend(obj)
    else:
        children.extend(getattr(obj, "__dict__", {}).values())
        for cls in type(obj).__mro__:
            for name in getattr(cls, "__slots__", ()):
                if hasattr(obj, name):
                    children.append(getattr(obj, name))
    return sum(tensor_bytes(child, seen, _depth + 1, visited) for child in children)


# --- The cache ---


class GeometryBundle:
    """The cached parts of one geometry key (module docstring)."""

    def __init__(self, key: tuple) -> None:
        """An empty bundle for ``key``."""
        self.key = key
        self.digest = short_hash(key)
        self.device_type = torch.device(dict(key)["device"]).type
        self.parts: dict[Hashable, Any] = {}
        self.nbytes = 0

    def recount(self) -> int:
        """Recount and return the bytes of every tensor the parts hold."""
        self.nbytes = tensor_bytes(list(self.parts.values()))
        return self.nbytes


class _Cache:
    """The process-wide LRU store; one instance, :data:`_CACHE`."""

    def __init__(self) -> None:
        """Empty, with the default budgets."""
        self.lock = threading.RLock()
        self.entries: OrderedDict[tuple, GeometryBundle] = OrderedDict()
        self.limits: dict[str, int] = {"cpu": DEFAULT_CPU_LIMIT_BYTES, "cuda": DEFAULT_CUDA_LIMIT_BYTES}
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.bypass_depth = 0

    def limit(self, device_type: str) -> int:
        """The byte budget of ``device_type`` (the CPU's for a type without its own)."""
        return self.limits.get(device_type, self.limits["cpu"])

    def evict(self, device_type: str, keep: tuple | None) -> None:
        """Drop least-recently-used bundles of ``device_type`` until its total fits the budget."""
        budget = self.limit(device_type)
        same = [k for k, b in self.entries.items() if b.device_type == device_type]
        total = sum(self.entries[k].nbytes for k in same)
        # Oldest first; the bundle in use (``keep``) goes last.
        order = [k for k in same if k != keep] + [k for k in same if k == keep]
        for k in order:
            if total <= budget:
                break
            total -= self.entries[k].nbytes
            del self.entries[k]
            self.evictions += 1


_CACHE = _Cache()


def enabled() -> bool:
    """Whether the cache is on: :data:`ENV_GEOMETRY_CACHE` (default on) and no :func:`bypass` active."""
    return env_switch(ENV_GEOMETRY_CACHE, True) and _CACHE.bypass_depth == 0


@contextlib.contextmanager
def bypass() -> Iterator[None]:
    """Force misses inside the block: objects are built afresh and not stored; entries are kept.

    For G5.2, whose repeat may want to rebuild every geometry object rather than test a cache hit
    (D-66); the record then says ``hit: False, bypassed: True``.
    """
    with _CACHE.lock:
        _CACHE.bypass_depth += 1
    try:
        yield
    finally:
        with _CACHE.lock:
            _CACHE.bypass_depth -= 1


def set_cache_limit_bytes(device: torch.device | str, n: int) -> None:
    """Set the byte budget for entries on ``device``'s type (``"cpu"``, ``"cuda"``) and evict to it."""
    device_type = torch.device(device).type
    if int(n) < 0:
        raise ValueError(f"cache limit must be non-negative, got {n}")
    with _CACHE.lock:
        _CACHE.limits[device_type] = int(n)
        _CACHE.evict(device_type, keep=None)


def cache_limit_bytes(device: torch.device | str) -> int:
    """The byte budget in force for ``device``'s type."""
    return _CACHE.limit(torch.device(device).type)


def clear() -> None:
    """Drop every entry and reset the hit/miss counters (the budgets are kept)."""
    with _CACHE.lock:
        _CACHE.entries.clear()
        _CACHE.hits = _CACHE.misses = _CACHE.evictions = 0


def reset_limits() -> None:
    """Restore the default budgets (for tests)."""
    with _CACHE.lock:
        _CACHE.limits = {"cpu": DEFAULT_CPU_LIMIT_BYTES, "cuda": DEFAULT_CUDA_LIMIT_BYTES}


def stats() -> dict[str, object]:
    """``hits``, ``misses``, ``evictions``, ``entries``, ``bytes`` (recounted, per device type too), budgets."""
    with _CACHE.lock:
        per_type: dict[str, int] = {}
        for bundle in _CACHE.entries.values():
            per_type[bundle.device_type] = per_type.get(bundle.device_type, 0) + bundle.recount()
        return {
            "hits": _CACHE.hits,
            "misses": _CACHE.misses,
            "evictions": _CACHE.evictions,
            "entries": len(_CACHE.entries),
            "bytes": sum(per_type.values()),
            "bytes_by_device": per_type,
            "limits": dict(_CACHE.limits),
            "enabled": enabled(),
        }


class GeometryLease:
    """One solve's access to the bundle of its key; ``part`` returns cached or freshly built objects.

    With the cache off (switch or :func:`bypass`) every part is built and nothing is stored.
    """

    def __init__(self, key: tuple) -> None:
        """Open the lease for ``key``; looks the bundle up (and marks it most recently used)."""
        self.key = key
        self.digest = short_hash(key)
        self.enabled = env_switch(ENV_GEOMETRY_CACHE, True)
        self.bypassed = _CACHE.bypass_depth > 0
        self.active = self.enabled and not self.bypassed
        self.parts: dict[str, bool] = {}
        self.local = GeometryBundle(key)
        self.bundle: GeometryBundle | None = None
        if self.active:
            with _CACHE.lock:
                self.bundle = _CACHE.entries.get(key)
                if self.bundle is not None:
                    _CACHE.entries.move_to_end(key)

    def part(self, name: Hashable, build: Callable[[], Any]) -> Any:
        """Return the part ``name`` of this key, building it with ``build()`` on a miss."""
        label = str(name)
        if self.active and self.bundle is not None and name in self.bundle.parts:
            with _CACHE.lock:
                _CACHE.hits += 1
            self.parts[label] = True
            return self.bundle.parts[name]
        value = build()
        self.parts[label] = False
        self.local.parts[name] = value
        if not self.active:
            return value
        with _CACHE.lock:
            _CACHE.misses += 1
            bundle = _CACHE.entries.get(self.key)
            if bundle is None:
                bundle = self.bundle if self.bundle is not None else GeometryBundle(self.key)
                _CACHE.entries[self.key] = bundle
            _CACHE.entries.move_to_end(self.key)
            bundle.parts[name] = value
            self.bundle = bundle
            if bundle.recount() > _CACHE.limit(bundle.device_type):
                # Larger than the whole budget: used by this solve, not retained.
                del _CACHE.entries[self.key]
                self.bundle = None
            else:
                _CACHE.evict(bundle.device_type, keep=self.key)
                if self.key not in _CACHE.entries:
                    self.bundle = None
        return value

    def record(self) -> dict[str, object]:
        """``measurements["geometry_cache"]`` of this solve.

        ``hit`` means every part came from the cache; ``retained`` means this key is still stored
        after the solve.
        """
        with _CACHE.lock:
            retained = self.bundle is not None and self.key in _CACHE.entries
            if retained:
                nbytes = self.bundle.recount()
            else:
                nbytes = self.local.recount() if self.local.parts else 0
            entries = len(_CACHE.entries)
        return {
            "hit": bool(self.parts) and all(self.parts.values()),
            "key": self.digest,
            "bytes": int(nbytes),
            "entries": entries,
            "retained": bool(retained),
            "parts": dict(self.parts),
            "enabled": self.enabled,
            "bypassed": self.bypassed,
        }


def not_applicable_record(reason: str) -> dict[str, object]:
    """``measurements["geometry_cache"]`` for a solve that builds no cached object (``reason`` says why)."""
    with _CACHE.lock:
        entries = len(_CACHE.entries)
    return {
        "hit": False,
        "key": None,
        "bytes": 0,
        "entries": entries,
        "retained": False,
        "parts": {},
        "enabled": env_switch(ENV_GEOMETRY_CACHE, True),
        "bypassed": _CACHE.bypass_depth > 0,
        "note": reason,
    }


def lease(grid: Any, charges: tuple[float, ...], positions: torch.Tensor) -> GeometryLease:
    """Open a :class:`GeometryLease` for ``grid`` and these nuclei (the value key of :func:`geometry_key`)."""
    return GeometryLease(geometry_key(grid, charges, positions))
