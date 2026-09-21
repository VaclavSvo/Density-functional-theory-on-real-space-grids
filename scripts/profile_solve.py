#!/usr/bin/env python3
"""Profile one scenario solve on a device: where the wall time goes.

    python scripts/profile_solve.py harmonic_w1 --device cuda --needed      # the fingerprint's solve
    python scripts/profile_solve.py h_atom_lda --device cuda --no-profile   # wall time only
    python scripts/profile_solve.py he_atom_lda --device cuda --census      # + host-read / launch census
    python scripts/profile_solve.py he_atom_lda --device cuda --repeat 2    # 2nd solve hits the cache

Runs ``solve_scenario`` under ``torch.profiler`` and prints the operators ranked by device and by
host time, the host<->device synchronisations the profiler saw, and the wall time. The LOBPCG
cross-check is off, as in the fingerprint. A diagnostic, not a benchmark -- ``scripts/benchmark.py``
is the benchmark; nothing is written to disk.

``--numerics fingerprint`` (the default) solves exactly what ``scripts/fingerprint.py`` solves: the
model-system preset for atom-free scenarios, the all-electron preset otherwise. States: the gates'
``states_required`` by default, the solver's own count with ``--needed``, or ``--n-states N``.
``--deterministic`` sets ``NumericsConfig.deterministic``, and the mode that ran is printed from the
provenance.

The summary also carries, where they apply, the CUDA allocator's peaks, retries, OOMs and
``cudaMalloc``/``cudaFree`` counts; the filter's CUDA-graph record (``CDFT_CUDA_GRAPHS=0`` turns
graphs off); and the Lanczos-bound re-estimates and reuses (``CDFT_LANCZOS_REUSE=0``).

``--repeat N`` solves ``N`` times in this process, splitting each wall time into build (whatever the
geometry cache did not serve) and solve, and prints the cache record; the profiler and the census
wrap the last repetition only. ``CDFT_GEOMETRY_CACHE=0`` turns the cache off for the comparison.
``--census`` runs the solve under :class:`sync_census.SyncCensus` as well; it slows the host side,
so a wall time taken with it is not a measurement.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import pathlib
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenario", help="registered scenario id (see physics_config.REGISTRY)")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    states = parser.add_mutually_exclusive_group()
    states.add_argument("--n-states", type=int, default=None, help="states to solve (default: the gates' states_required)")
    states.add_argument("--needed", action="store_true", help="the solver's own state count, as scripts/fingerprint.py")
    parser.add_argument(
        "--numerics", choices=("fingerprint", "all-electron", "model"), default="fingerprint",
        help="numerics preset (default: the fingerprint's choice per scenario)",
    )
    parser.add_argument("--deterministic", action="store_true", help="NumericsConfig.deterministic=True (strict kernels on CUDA)")
    parser.add_argument("--no-deterministic", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-profile", action="store_true", help="wall time only, no torch.profiler")
    parser.add_argument("--census", action="store_true", help="also count host reads and kernel launches (scripts/sync_census.py)")
    parser.add_argument("--repeat", type=int, default=1, help="solve this many times in one process (build vs solve wall per repetition)")
    parser.add_argument("--rows", type=int, default=25, help="rows per profiler table")
    parser.add_argument("--trace", default=None, help="also write a Chrome trace JSON to this path")
    args = parser.parse_args(argv)
    if args.deterministic and args.no_deterministic:
        parser.error("--deterministic and --no-deterministic contradict each other")
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")

    import torch

    import cdft.precision as precision
    from cdft.gates.runner import states_required
    from cdft.operators import geometry_cache
    from cdft.scf import loop as loop_module
    from cdft.scf import noninteracting as noninteracting_module
    from cdft.physics_config import REGISTRY
    from cdft.scf.solve import solve_scenario
    from contract import Device
    from sync_census import SyncCensus, census_numerics

    scenario = REGISTRY[args.scenario]
    numerics = dataclasses.replace(
        census_numerics(scenario, args.numerics),
        device=Device(args.device),
        deterministic=bool(args.deterministic),
    )
    device = precision.resolve_device(numerics.device)
    if args.needed:
        n_states = None
    elif args.n_states is not None:
        n_states = args.n_states
    else:
        n_states = states_required(scenario) or None

    print(f"scenario {args.scenario}  device {device}  torch {torch.__version__} (CUDA {torch.version.cuda})", flush=True)
    grid = numerics.grid
    print(
        f"  numerics {args.numerics}: preset spacing {grid.spacing} (the D-42 derivation may override it; "
        f"see the grid line below)  n_states {'solver default (needed)' if n_states is None else n_states}  "
        f"deterministic requested {numerics.deterministic}",
        flush=True,
    )
    if device.type == "cuda":
        print(f"  {torch.cuda.get_device_name(device)}  capability {torch.cuda.get_device_capability(device)}", flush=True)
        free, total = torch.cuda.mem_get_info(device)
        print(f"  device memory before the solve: {free / 2**30:.2f} GiB free of {total / 2**30:.2f} GiB", flush=True)
        torch.cuda.reset_peak_memory_stats(device)

    census = SyncCensus() if args.census else None

    # The solvers look the two builders up as module globals, so wrapping them here times exactly
    # the geometry construction of each solve.
    build_seconds = [0.0]

    def timed_builder(builder):
        """Wrap a geometry builder so its wall time (synchronised on CUDA) adds to ``build_seconds``."""

        def wrapper(*a, **kw):
            """Call the builder and account its time."""
            begin = time.perf_counter()
            try:
                return builder(*a, **kw)
            finally:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                build_seconds[0] += time.perf_counter() - begin
        return wrapper

    noninteracting_module.build_hamiltonian = timed_builder(noninteracting_module.build_hamiltonian)
    loop_module.build_interacting_step = timed_builder(loop_module.build_interacting_step)

    def run(with_census: bool):
        """One solve, under the census when asked for."""
        with census if (census is not None and with_census) else contextlib.nullcontext():
            return solve_scenario(scenario, numerics, n_states=n_states, cross_check=False)

    def cache_line(record) -> str:
        """The geometry-cache line of one solve plus the process-wide counters."""
        if not record:
            return "geometry cache: no record in the artifact"
        stats = geometry_cache.stats()
        return (
            f"geometry cache: {'hit' if record.get('hit') else 'miss'}  key {record.get('key')}  "
            f"bytes {record.get('bytes', 0) / 2**20:.1f} MiB  entries {record.get('entries')}  "
            f"parts {record.get('parts')}  enabled {record.get('enabled')}  "
            f"(process: hits {stats['hits']} misses {stats['misses']} evictions {stats['evictions']} "
            f"total {stats['bytes'] / 2**20:.1f} MiB, limit {geometry_cache.cache_limit_bytes(device) / 2**30:.2f} GiB)"
        )

    # CUDA context warm-up and kernel loading are deliberately not separated: the fingerprint pays
    # them too.
    prof = None
    for repetition in range(1, args.repeat + 1):
        last = repetition == args.repeat
        build_seconds[0] = 0.0
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        if args.no_profile or not last:
            artifact = run(last)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
        else:
            from torch.profiler import ProfilerActivity, profile

            activities = [ProfilerActivity.CPU]
            if device.type == "cuda":
                activities.append(ProfilerActivity.CUDA)
            with profile(activities=activities, record_shapes=False, profile_memory=False) as prof:
                artifact = run(last)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
        # One line per repetition, also for --repeat 1.
        print(
            f"solve {repetition}/{args.repeat}: wall {elapsed:.2f} s = build {build_seconds[0]:.2f} s "
            f"+ solve {elapsed - build_seconds[0]:.2f} s  status {artifact.status.value}",
            flush=True,
        )
        print("  " + cache_line(artifact.measurements.get("geometry_cache")), flush=True)

    prov = artifact.provenance
    m = artifact.measurements
    solved = int(artifact.result.eigenvalues.shape[-1]) if artifact.result is not None else None
    print(
        f"\nstatus {artifact.status.value}  wall {elapsed:.1f} s  deterministic {prov.deterministic_algorithms}  "
        f"gpu_peak {prov.gpu_peak_bytes}  n_states {solved}  grid {(m.get('grid') or {}).get('shape')}  "
        f"scf_iterations {m.get('scf_iterations')}  eigen_iterations {m.get('eigen_iterations')}  "
        f"hamiltonian_applications {m.get('hamiltonian_applications')}",
        flush=True,
    )
    if artifact.error_message:
        print(f"error: {artifact.error_message}")
    # The CUDA-graph record (absent on the CPU) and the Lanczos-bound choices.
    graphs = m.get("cuda_graphs")
    if graphs is not None:
        line = (
            f"cuda graphs: used {graphs.get('used')}  captures {graphs.get('captures')}  "
            f"replays {graphs.get('replays')}  eager calls {graphs.get('eager_calls')}  "
            f"warm-up applications {graphs.get('warmup_applications')}  "
            f"validation max rel diff {graphs.get('validation_max_rel_diff')}"
        )
        if graphs.get("reason"):
            line += f"  reason: {graphs['reason']}"
        print(line, flush=True)
    elif device.type == "cuda":
        print("cuda graphs: no record in the artifact", flush=True)
    if "lanczos_reestimates" in m:
        applications = m.get("scf_hamiltonian_applications") or []
        mean = sum(applications) / len(applications) if applications else float("nan")
        print(
            f"lanczos bounds: reuse {m.get('lanczos_bound_reuse')}  re-estimates {m.get('lanczos_reestimates')}  "
            f"reuses {m.get('lanczos_reuses')}  operator applications per SCF iteration {mean:.1f} "
            f"(per iteration: {applications})",
            flush=True,
        )
    if device.type == "cuda":
        # Alloc retries mean the device ran out of memory and the cache was flushed and refilled
        # (slow under Windows WDDM); a reserved peak near the card's size means the driver's sysmem
        # fallback may be paging device memory to host RAM.
        stats = torch.cuda.memory_stats(device)
        free, total = torch.cuda.mem_get_info(device)
        print(
            "allocator: "
            f"allocated peak {stats.get('allocated_bytes.all.peak', 0) / 2**30:.2f} GiB  "
            f"reserved peak {stats.get('reserved_bytes.all.peak', 0) / 2**30:.2f} GiB  "
            f"alloc retries {stats.get('num_alloc_retries', 'n/a')}  "
            f"OOMs {stats.get('num_ooms', 'n/a')}  "
            f"cudaMalloc calls {stats.get('num_device_alloc', 'n/a')}  "
            f"cudaFree calls {stats.get('num_device_free', 'n/a')}  "
            f"free now {free / 2**30:.2f} of {total / 2**30:.2f} GiB",
            flush=True,
        )
    if census is not None:
        print("\n=== census (host reads = synchronisations on CUDA; launches ~ cudaLaunchKernel) ===")
        print(census.table(args.rows))
        if prof is not None or device.type == "cpu":
            print("(the census wall time includes the census's own overhead)")
    if prof is None:
        return 0

    sort_keys = ["self_cpu_time_total"]
    if device.type == "cuda":
        sort_keys.insert(0, "self_device_time_total")
    for key in sort_keys:
        print(f"\n=== top {args.rows} operators by {key} ===")
        try:
            print(prof.key_averages().table(sort_by=key, row_limit=args.rows))
        except Exception as exc:  # noqa: BLE001 - profiler table names differ across torch versions
            print(f"(table by {key} unavailable: {exc})")
    if device.type == "cuda":
        sync_names = ("cudaDeviceSynchronize", "cudaStreamSynchronize", "cudaMemcpyAsync", "cudaMemcpy", "aten::item", "aten::_local_scalar_dense")
        print("\n=== synchronisation / copy calls ===")
        for row in prof.key_averages():
            if row.key in sync_names:
                print(f"  {row.key:<28} count {row.count:>8}  cpu total {row.cpu_time_total / 1e6:.2f} s")
    if args.trace:
        prof.export_chrome_trace(args.trace)
        print(f"\ntrace written to {args.trace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
