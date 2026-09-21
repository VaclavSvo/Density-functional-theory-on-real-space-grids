#!/usr/bin/env python3
"""Host-read ("sync") and operator census of one scenario solve.

    python scripts/sync_census.py harmonic_w1 --n-states 10
    python scripts/sync_census.py he_atom_lda --json reports/census_he_atom_lda.json
    python scripts/sync_census.py he_atom_lda --op-sites      # also rank launch sites

Every place a CUDA run would block on the device is a place the Python code reads a tensor value
back to the host; on the CPU those reads are free and so invisible in a profile. For one
``solve_scenario`` this counts, by innermost project frame, the host reads (nested ones once), the
``aten`` operators reaching the dispatcher (views and metadata apart, since they launch no kernel),
the operators that sync without a Python-level read, and each CUDA-graph replay as one launch.
``--device cuda`` runs the same census on the card; numerics mirror ``scripts/fingerprint.py``
unless ``--numerics`` says otherwise. The report goes to stdout and, with ``--json``, to that file.

:func:`census_solve` is the importable form, used by gate G5.9 (``cdft.gates.tier5``); the patches
live only inside :class:`SyncCensus`'s ``with`` block and are removed on exit, whatever happens.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import pathlib
import sys
import threading
import time
import weakref

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402

#: Tensor methods that read device memory from the host (each is one blocking sync on CUDA).
READ_METHODS = ("item", "__float__", "__int__", "__index__", "__bool__", "tolist", "cpu", "numpy")

#: Operators that launch no device kernel (views, metadata, allocation without fill).
NO_KERNEL_OPS = frozenset(
    {
        "view", "_unsafe_view", "as_strided", "slice", "select", "t", "transpose", "permute",
        "expand", "unsqueeze", "squeeze", "detach", "alias", "lift_fresh", "empty", "empty_like",
        "empty_strided", "reshape", "unbind", "split", "split_with_sizes", "chunk", "diagonal",
        "_reshape_alias", "resolve_conj", "resolve_neg", "is_same_size", "sym_size", "sym_stride",
        "sym_numel", "sym_storage_offset", "narrow", "movedim", "unfold", "view_as_real",
        "_local_scalar_dense", "set_", "_to_copy_noop",
    }
)

#: Operators that block on CUDA without a Python-level read (data-dependent shape or error check).
IMPLICIT_SYNC_OPS = frozenset(
    {
        "nonzero", "masked_select", "_linalg_check_errors", "linalg_cholesky", "cholesky",
        "linalg_eigh", "linalg_eigvalsh", "_linalg_eigh", "unique", "_unique2", "equal",
        "is_nonzero", "repeat_interleave", "masked_scatter", "argwhere", "nonzero_static",
    }
)

#: Reads and implicit-sync operators on tensors that live on the host by design on every device
#: (``device="cpu"`` in the source, or a numpy-bound computation): they cost nothing on CUDA and
#: are reported apart. ``(file suffix, function, token)``; a site matches when its source line
#: contains ``token``. The Gauss--Legendre rule is memoised per order and built on a CPU tensor
#: whatever the solve's device; a solve that needs many orders builds them once each at geometry
#: time, which the loop/setup split of :func:`loop_syncs` would otherwise read as a loop site
#: (O-31: 27 orders against a 7-iteration probe).
HOST_SIDE_READS = (
    ("src/cdft/operators/hamiltonian.py", "_lanczos_bounds", "ritz["),
    ("src/cdft/scf/mixing.py", "mix", "float(c)"),
    ("src/cdft/scf/mixing.py", "mix", "gram.diagonal()"),
    ("src/cdft/operators/gauss_legendre.py", "_rule", "eigvalsh"),
)

_STATE = threading.local()


def _depth() -> int:
    """Nesting depth of census wrappers on this thread (reads inside reads count once)."""
    return getattr(_STATE, "depth", 0)


class SyncCensus:
    """Count host reads and dispatched operators inside a ``with`` block.

    Frames under ``root`` (default: the repository root) are project frames and the innermost one
    names a site. ``op_sites`` also attributes every launching operator to its site, at the cost of
    one frame walk per operator.
    """

    def __init__(self, root: pathlib.Path | None = None, op_sites: bool = False) -> None:
        """Prepare an empty census; nothing is patched until ``__enter__``."""
        self.root = str((root or REPO_ROOT).resolve())
        self.self_file = str(pathlib.Path(__file__).resolve())
        self.op_sites = op_sites
        self.reads: collections.Counter = collections.Counter()
        self.read_methods: collections.Counter = collections.Counter()
        self.implicit: collections.Counter = collections.Counter()
        self.ops: collections.Counter = collections.Counter()
        self.launch_sites: collections.Counter = collections.Counter()
        self.n_ops = 0
        self.n_launch = 0
        self.n_graph_replays = 0
        self.live_bytes = 0
        self.peak_bytes = 0
        self._live: dict[int, list[int]] = {}
        self._saved: dict[str, object] = {}
        self._mode: TorchDispatchMode | None = None
        self.wall_s = 0.0
        self._t0 = 0.0

    # -- site attribution -----------------------------------------------------------------------

    def _site(self, frame) -> tuple[str, int, str]:
        """Return ``(relative file, line, function)`` of the innermost project frame above ``frame``."""
        while frame is not None:
            name = frame.f_code.co_filename
            if name.startswith(self.root) and name != self.self_file:
                # Forward slashes on every platform: the host-side list and the recorded budgets
                # name sites as ``src/cdft/...``; Windows frames carry backslashes (D-83).
                rel = name[len(self.root) + 1 :].replace("\\", "/")
                return rel, frame.f_lineno, frame.f_code.co_name
            frame = frame.f_back
        return "<outside>", 0, "?"

    # -- patches ----------------------------------------------------------------------------------

    def _wrap(self, method: str, original):
        """Return a counting wrapper around one ``torch.Tensor`` method."""
        census = self

        def wrapper(tensor, *args, **kwargs):
            if _depth() == 0:
                # Only a read of a *device-side* value would sync; meta/python scalars never reach here.
                census.reads[census._site(sys._getframe(1))] += 1
                census.read_methods[method] += 1
            _STATE.depth = _depth() + 1
            try:
                return original(tensor, *args, **kwargs)
            finally:
                _STATE.depth -= 1

        wrapper.__name__ = getattr(original, "__name__", method)
        wrapper.__doc__ = getattr(original, "__doc__", None)
        return wrapper

    def __enter__(self) -> "SyncCensus":
        """Install the method wrappers and the dispatch mode."""
        for method in READ_METHODS:
            original = torch.Tensor.__dict__.get(method, getattr(torch.Tensor, method))
            self._saved[method] = torch.Tensor.__dict__.get(method, _MISSING)
            setattr(torch.Tensor, method, self._wrap(method, original))
        census = self
        graph_class = getattr(torch.cuda, "CUDAGraph", None)
        if graph_class is not None:
            original_replay = graph_class.replay

            def replay(graph, *args, **kwargs):
                """A CUDA-graph replay bypasses the dispatcher: count it as one launch."""
                census.n_graph_replays += 1
                census.n_launch += 1
                census.ops["cuda_graph_replay"] += 1
                if census.op_sites:
                    census.launch_sites[census._site(sys._getframe(1))] += 1
                return original_replay(graph, *args, **kwargs)

            self._saved_replay = (graph_class, original_replay)
            graph_class.replay = replay

        class _Counter(TorchDispatchMode):
            """Counts every operator that reaches the dispatcher."""

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # noqa: D105
                name = func._schema.name.split("::", 1)[-1]
                census.n_ops += 1
                census.ops[name] += 1
                if name == "_local_scalar_dense":
                    if _depth() == 0:
                        census.implicit[(name,) + census._site(sys._getframe(1))] += 1
                elif name in IMPLICIT_SYNC_OPS or (name in ("index", "index_put", "index_put_") and _has_bool_index(args)):
                    census.implicit[(name,) + census._site(sys._getframe(1))] += 1
                if name not in NO_KERNEL_OPS:
                    census.n_launch += 1
                    if census.op_sites:
                        census.launch_sites[census._site(sys._getframe(1))] += 1
                out = func(*args, **(kwargs or {}))
                census._track(out)
                return out

        self._mode = _Counter()
        self._mode.__enter__()
        self._t0 = time.perf_counter()
        return self

    def _track(self, out) -> None:
        """Account the storages of an operator's tensor outputs toward the live/peak byte estimate.

        Keyed by storage pointer and reference-counted over the Python tensors holding it, so views
        and in-place results do not double count. Only an approximation of the caching allocator's
        ``max_memory_allocated``: storages held only by C++ are released early, sparse ones ignored.
        """
        items = out if isinstance(out, (list, tuple)) else (out,)
        for tensor in items:
            if not isinstance(tensor, torch.Tensor) or tensor.layout is not torch.strided:
                continue
            try:
                storage = tensor.untyped_storage()
                key, nbytes = storage.data_ptr(), storage.nbytes()
            except (RuntimeError, NotImplementedError):
                continue
            if nbytes == 0:
                continue
            entry = self._live.get(key)
            if entry is None or entry[0] != nbytes:
                if entry is not None:  # pointer reused after an early release: replace
                    self.live_bytes -= entry[0]
                entry = self._live[key] = [nbytes, 0]
                self.live_bytes += nbytes
                self.peak_bytes = max(self.peak_bytes, self.live_bytes)
            entry[1] += 1
            weakref.finalize(tensor, self._release, key, entry)

    def _release(self, key: int, entry: list[int]) -> None:
        """Drop one reference to a tracked storage; free its bytes at zero."""
        entry[1] -= 1
        if entry[1] == 0 and self._live.get(key) is entry:
            del self._live[key]
            self.live_bytes -= entry[0]

    def __exit__(self, *exc) -> None:
        """Remove every patch, whatever happened inside the block."""
        self.wall_s = time.perf_counter() - self._t0
        try:
            if self._mode is not None:
                self._mode.__exit__(*exc)
        finally:
            saved_replay = getattr(self, "_saved_replay", None)
            if saved_replay is not None:
                saved_replay[0].replay = saved_replay[1]
                self._saved_replay = None
            for method, saved in self._saved.items():
                if saved is _MISSING:
                    delattr(torch.Tensor, method)
                else:
                    setattr(torch.Tensor, method, saved)

    # -- reporting --------------------------------------------------------------------------------

    def is_host_side(self, site: tuple[str, int, str]) -> bool:
        """Whether a site reads or syncs a host-resident tensor by design (:data:`HOST_SIDE_READS`)."""
        import linecache

        rel, line, function = site
        source = linecache.getline(str(pathlib.Path(self.root) / rel), line)
        return any(
            rel.endswith(suffix) and function == fn and token in source
            for suffix, fn, token in HOST_SIDE_READS
        )

    @property
    def device_reads(self) -> collections.Counter:
        """Read sites that would synchronise on CUDA (host-side reads removed)."""
        return collections.Counter({k: v for k, v in self.reads.items() if not self.is_host_side(k)})

    @property
    def n_reads(self) -> int:
        """Total Python-level reads that would synchronise on CUDA."""
        return sum(self.device_reads.values())

    @property
    def n_host_reads(self) -> int:
        """Total Python-level reads of host-resident tensors (free on every device)."""
        return sum(self.reads.values()) - self.n_reads

    @property
    def device_implicit(self) -> collections.Counter:
        """Implicit-sync sites that would synchronise on CUDA (host-side sites removed)."""
        return collections.Counter(
            {k: v for k, v in self.implicit.items() if not self.is_host_side(k[1:])}
        )

    @property
    def n_implicit(self) -> int:
        """Total operator-level syncs that no Python read accounts for."""
        return sum(self.device_implicit.values())

    @property
    def n_host_implicit(self) -> int:
        """Total implicit syncs on host-resident tensors (free on every device)."""
        return sum(self.implicit.values()) - self.n_implicit

    def summary(self, rows: int = 30) -> dict:
        """Return the census as plain data (the JSON payload and the source of :meth:`table`)."""
        return {
            "reads_total": self.n_reads,
            "host_side_reads_total": self.n_host_reads,
            "implicit_total": self.n_implicit,
            "host_side_implicit_total": self.n_host_implicit,
            "syncs_total": self.n_reads + self.n_implicit,
            "ops_total": self.n_ops,
            "launch_ops_total": self.n_launch,
            "cuda_graph_replays": self.n_graph_replays,
            "read_methods": dict(self.read_methods.most_common()),
            "read_sites": [
                {"site": f"{f}:{ln}", "function": fn, "count": c}
                for (f, ln, fn), c in self.device_reads.most_common(rows)
            ],
            "implicit_sites": [
                {"op": op, "site": f"{f}:{ln}", "function": fn, "count": c}
                for (op, f, ln, fn), c in self.device_implicit.most_common(rows)
            ],
            "top_ops": dict(self.ops.most_common(rows)),
            "launch_sites": [
                {"site": f"{f}:{ln}", "function": fn, "count": c}
                for (f, ln, fn), c in self.launch_sites.most_common(rows)
            ],
            "peak_live_bytes_estimate": self.peak_bytes,
            "census_wall_s": round(self.wall_s, 2),
        }

    def table(self, rows: int = 30) -> str:
        """Return the census as printable text: read sites, implicit syncs, totals, top operators."""
        s = self.summary(rows)
        lines = [f"--- top {rows} host-read sites (python-level) ---", f"{'count':>9}  site  function"]
        lines += [f"{r['count']:>9}  {r['site']}  {r['function']}" for r in s["read_sites"]]
        lines.append(f"{s['reads_total']:>9}  TOTAL python-level reads that sync on CUDA   (+{s['host_side_reads_total']} reads "
            f"of host-resident tensors, free)   by method incl. host-side: {s['read_methods']}")
        lines.append("--- implicit syncs (operators that block on CUDA without a python read) ---")
        lines += [f"{r['count']:>9}  {r['op']:<20} {r['site']}  {r['function']}" for r in s["implicit_sites"]]
        lines.append(f"{s['implicit_total']:>9}  TOTAL implicit")
        lines.append(
            f"syncs total {s['syncs_total']}   aten ops {s['ops_total']}   kernel-launching ops "
            f"{s['launch_ops_total']} (incl. {s['cuda_graph_replays']} CUDA-graph replays at one launch each)   "
            f"peak live tensor bytes ~{s['peak_live_bytes_estimate'] / 2**20:.0f} MiB   "
            f"(census wall {s['census_wall_s']} s)"
        )
        lines.append("--- top operators ---")
        lines.append("  " + "  ".join(f"{k}:{v}" for k, v in s["top_ops"].items()))
        if s["launch_sites"]:
            lines.append(f"--- top {rows} launch sites ---")
            lines += [f"{r['count']:>9}  {r['site']}  {r['function']}" for r in s["launch_sites"]]
        return "\n".join(lines)


_MISSING = object()


def _has_bool_index(args) -> bool:
    """Whether an ``index``/``index_put`` call carries a boolean mask (data-dependent: syncs on CUDA)."""
    if len(args) < 2 or not isinstance(args[1], (list, tuple)):
        return False
    return any(isinstance(i, torch.Tensor) and i.dtype == torch.bool for i in args[1])


def census_numerics(scenario, preset: str = "fingerprint", device: str = "cpu"):
    """Return the numerics a census solve uses: the fingerprint's choice unless ``preset`` overrides.

    ``fingerprint``: model preset for atom-free scenarios, all-electron otherwise;
    ``all-electron`` / ``model``: that preset for every scenario. Cross-check off, orbitals stored.
    """
    from cdft.config import PRESETS, numerics_for_spec
    from contract import Device

    if preset == "fingerprint":
        base, _ = numerics_for_spec(scenario)
    elif preset in PRESETS:
        base = PRESETS[preset]
    else:
        raise ValueError(f"unknown numerics preset {preset!r}")
    return dataclasses.replace(
        base,
        eigen=dataclasses.replace(base.eigen, cross_check=False),
        output=dataclasses.replace(base.output, store_orbitals=True),
        device=Device(device),
    )


#: Measurement keys worth printing beside the census (iteration and application counts).
ITERATION_KEYS = (
    "eigen_iterations", "scf_iterations", "hamiltonian_applications", "scf_filter_steps_per_iteration",
    "eigen_stop_reason", "scf_stop_reason", "lanczos_reestimates", "lanczos_reuses",
    "scf_hamiltonian_applications", "cuda_graphs",
)


def _census_run(
    scenario,
    numerics,
    n_states: int | None,
    *,
    op_sites: bool = False,
    derive_grid: bool = True,
    root: pathlib.Path | None = None,
    preset: str = "custom",
):
    """Solve ``scenario`` at ``numerics`` under a :class:`SyncCensus`; return ``(census, artifact, plain)``.

    The one implementation behind :func:`run_census` and :func:`census_solve`; ``plain`` holds the
    grid size and the iteration counts.
    """
    from cdft.scf.solve import solve_scenario

    with SyncCensus(root=root, op_sites=op_sites) as census:
        artifact = solve_scenario(scenario, numerics, n_states=n_states, derive_grid=derive_grid)
    m = artifact.measurements
    grid = m.get("grid") or {}
    plain = {
        "scenario": scenario.scenario_id,
        "preset": preset,
        "n_states_requested": n_states,
        "n_states_solved": int(artifact.result.eigenvalues.shape[-1]) if artifact.result is not None else None,
        "status": artifact.status.value,
        "error": artifact.error_message,
        "n_points": grid.get("n_points"),
        "shape": grid.get("shape"),
        **{k: m.get(k) for k in ITERATION_KEYS if k in m},
    }
    return census, artifact, plain


def run_census(
    scenario_id: str,
    n_states: int | None,
    preset: str = "fingerprint",
    op_sites: bool = False,
    device: str = "cpu",
):
    """Solve one registered scenario under a :class:`SyncCensus`; return ``(census, artifact, plain)``."""
    from cdft.physics_config import REGISTRY

    scenario = REGISTRY[scenario_id]
    numerics = census_numerics(scenario, preset, device)
    return _census_run(scenario, numerics, n_states, op_sites=op_sites, preset=preset)


def census_solve(
    scenario,
    numerics,
    device: str | None = None,
    *,
    n_states: int | None = None,
    derive_grid: bool = True,
    root: pathlib.Path | None = None,
) -> dict:
    """Census of one solve without the command line (the importable API of gate G5.9).

    ``numerics`` is used as given except that a non-``None`` ``device`` replaces ``numerics.device``;
    ``derive_grid=False`` keeps the grid of ``numerics`` (the D-42 bypass of the ladder gates).
    Returns ``{"run": plain, "census": summary, "loop": loop}``, ``loop`` being :func:`loop_syncs`
    over ``scf_iterations`` for a self-consistent solve and ``eigen_iterations`` otherwise.
    """
    from contract import Device

    if device is not None:
        numerics = dataclasses.replace(numerics, device=Device(device))
    census, artifact, plain = _census_run(scenario, numerics, n_states, derive_grid=derive_grid, root=root)
    summary = census.summary(rows=None)
    key = "scf_iterations" if plain.get("scf_iterations") else "eigen_iterations"
    loop = loop_syncs(summary, int(float(plain.get(key) or 0)))
    loop["iterations_key"] = key
    del artifact
    return {"run": plain, "census": summary, "loop": loop}


def loop_syncs(summary: dict, iterations: int) -> dict:
    """Split a census's synchronising sites into loop sites and setup sites (gate G5.9).

    A site -- a read or an implicit-sync operator at one ``file:line`` -- counting at least
    ``iterations`` fires every iteration and is a loop site; a smaller count is setup or teardown.
    ``per_iteration`` is the loop sites' total over ``iterations``, so it grows by exactly one when
    an ``.item()`` enters the loop body, where a solve-level ratio would dilute it with setup reads.
    With ``iterations == 0`` every site is setup.
    """
    sites = [(r["site"], r["function"], int(r["count"])) for r in summary["read_sites"]]
    sites += [
        (f"{r['site']} [{r['op']}]", r["function"], int(r["count"])) for r in summary["implicit_sites"]
    ]
    loop = [s for s in sites if iterations > 0 and s[2] >= iterations]
    loop_total = sum(s[2] for s in loop)
    return {
        "iterations": iterations,
        "loop_syncs_total": loop_total,
        "setup_syncs_total": sum(s[2] for s in sites) - loop_total,
        "per_iteration": (loop_total / iterations) if iterations > 0 else 0.0,
        "loop_sites": [{"site": s[0], "function": s[1], "count": s[2]} for s in loop],
    }


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenario", help="registered scenario id")
    parser.add_argument("--n-states", type=int, default=None, help="states to solve (default: the solver's own count)")
    parser.add_argument("--numerics", choices=("fingerprint", "all-electron", "model"), default="fingerprint")
    parser.add_argument("--rows", type=int, default=30)
    parser.add_argument("--op-sites", action="store_true", help="also rank kernel-launch sites (slower)")
    parser.add_argument("--sync-ms", type=float, default=1.3, help="cost of one sync for the prediction line")
    parser.add_argument("--json", default=None, help="write the census to this JSON file")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="solve device (default cpu)")
    args = parser.parse_args(argv)

    torch.set_num_threads(max(1, torch.get_num_threads()))
    census, artifact, plain = run_census(args.scenario, args.n_states, args.numerics, args.op_sites, args.device)
    print(
        f"=== census {args.scenario}  preset {args.numerics}  n_states {args.n_states}  device {args.device} ===",
        flush=True,
    )
    print("  " + "  ".join(f"{k}={v}" for k, v in plain.items()), flush=True)
    print(census.table(args.rows), flush=True)
    s = census.summary(args.rows)
    print(
        f"predicted sync cost on the card: {s['syncs_total']} x {args.sync_ms} ms = "
        f"{s['syncs_total'] * args.sync_ms / 1000.0:.1f} s",
        flush=True,
    )
    if args.json:
        target = pathlib.Path(args.json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"run": plain, "census": s}, indent=2, default=str), encoding="utf-8")
    return 0 if artifact.result is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
