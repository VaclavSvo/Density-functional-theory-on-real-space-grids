#!/usr/bin/env python3
"""Throughput of the fractional-occupation scan shape: runs per hour, per-run wall, peak VRAM.

    python scripts/scan_throughput.py --dry-run                        # script check, CPU
    python scripts/scan_throughput.py --device cuda --workers 1 2 3    # k spawned processes
    python scripts/scan_throughput.py --device cuda --streams 2        # k CUDA streams, one process

The shape is one geometry with the electron number swept: the scenario (default ``h2plus_R2_lda``)
is replaced per ``N`` in ``--n-electrons``, ``--repeat`` times, and every run goes through
``solve_scenario`` and the gate suite at ``max_solves=0`` and the production numerics, so a run
counted here is a gate-judged record. Each mode is written to its own
``reports/throughput/<host>_<device>_<mode><k>.json``, rewritten after every run, and a Markdown
summary of every mode is printed at the end.

``--workers k`` spawns ``k`` processes (never fork: Windows-safe, and a CUDA context cannot be
forked); ``--streams k`` runs ``k`` threads on their own ``torch.cuda.Stream``, still serialising
Python-level work on the GIL, which is part of what is measured. ``k`` workers need ``k`` times one
solve's host peak, because each builds its quadrature and ``L``/``L^T`` on the CPU before moving
them (D-60); a worker lost to the OOM killer is recorded as an ``error`` run, never a hang. Per-run
``gpu_peak_bytes`` is shared between streams of one process, so an upper bound there.

``--dry-run`` checks the script and measures nothing: its coarse grid fails DERIVED gates by
construction, and it writes to ``reports/throughput/dryrun_*``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import dataclasses
import functools
import json
import multiprocessing
import os
import pathlib
import platform
import sys
import threading
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
for _entry in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

print = functools.partial(print, flush=True)  # noqa: A001

#: Default electron numbers of the scan (one geometry, N swept).
DEFAULT_N = (0.2, 0.4, 0.6, 0.8, 1.0)

#: Coarse grid of the dry run (a script check; see the module docstring).
DRY_SPACING = 0.5
DRY_BOX = 8.0  # a nucleus needs 3.25 bohr to the edge (cusp quadrature)
DRY_SCENARIO = "h_atom_lda"
DRY_N = (0.5, 1.0)
DRY_MAX_SCF = 15  # a fractional N on the coarse grid need not converge; the dry run only needs a record


def scan_scenario(scenario_id: str, n_electrons: float):
    """The registered scenario with ``N = n_electrons`` (``dataclasses.replace`` of its ElectronSpec)."""
    from cdft.physics_config import REGISTRY

    base = REGISTRY[scenario_id]
    electrons = dataclasses.replace(base.electrons, n_electrons=float(n_electrons))
    return dataclasses.replace(base, electrons=electrons, scenario_id=f"{scenario_id}_N{n_electrons:g}")


def scan_numerics(device: str, dry_run: bool):
    """Production numerics for the scan (quick-profile axes), or the dry run's coarse grid."""
    from cdft.config import ALL_ELECTRON_NUMERICS
    from contract import Device

    numerics = dataclasses.replace(
        ALL_ELECTRON_NUMERICS,
        eigen=dataclasses.replace(ALL_ELECTRON_NUMERICS.eigen, cross_check=False),
        device=Device(device),
        deterministic=False,
    )
    if dry_run:
        numerics = dataclasses.replace(
            numerics,
            grid=dataclasses.replace(numerics.grid, spacing=DRY_SPACING, box_lengths=(DRY_BOX,) * 3),
            scf=dataclasses.replace(numerics.scf, max_iterations=DRY_MAX_SCF),
        )
    return numerics


def solve_one(task: dict) -> dict:
    """Solve and gate one scan point in a worker or thread; return its plain-data record.

    ``task``: ``scenario``, ``n_electrons``, ``device``, ``dry_run``, ``index``, optional ``stream``.
    """
    import torch

    from cdft.gates.runner import GateSuite, states_required
    from cdft.scf.solve import solve_scenario

    scenario = scan_scenario(task["scenario"], task["n_electrons"])
    numerics = scan_numerics(task["device"], task["dry_run"])
    suite = GateSuite(max_solves=0)
    required = states_required(scenario, suite.artifact_gates)
    stream = task.get("stream")
    started = time.perf_counter()
    context = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
    with context:
        artifact = solve_scenario(
            scenario, numerics, n_states=required or None, derive_grid=not task["dry_run"]
        )
        artifact = suite.run_on_artifact(artifact)
        if stream is not None:
            stream.synchronize()
    wall = time.perf_counter() - started
    verdicts: dict[str, int] = {}
    for result in artifact.gates.results:
        verdicts[result.verdict.value] = verdicts.get(result.verdict.value, 0) + 1
    record = {
        "index": task["index"],
        "scenario": scenario.scenario_id,
        "n_electrons": task["n_electrons"],
        "status": artifact.status.value,
        "error": artifact.error_message,
        "wall_s": round(wall, 3),
        "scf_iterations": artifact.measurements.get("scf_iterations"),
        "total_energy": None if artifact.result is None else artifact.result.energies.total,
        "gpu_peak_bytes": artifact.provenance.gpu_peak_bytes,
        "verdicts": verdicts,
        "failed_gates": [r.gate_id for r in artifact.gates.results if r.verdict.value == "fail"],
        "pid": os.getpid(),
        "thread": threading.current_thread().name,
    }
    del artifact
    from cdft.precision import release_device_memory, resolve_device
    from contract import Device

    release_device_memory(resolve_device(Device(task["device"])))
    return record


def _worker_init(threads: int) -> None:
    """Spawned worker start-up: the torch thread count for this worker's share of the cores."""
    import torch

    torch.set_num_threads(max(1, threads))


class MemorySampler:
    """Polls ``torch.cuda.mem_get_info`` every 0.25 s and keeps the lowest free value (CUDA only)."""

    def __init__(self, device) -> None:
        """Prepare a sampler for ``device`` (inactive off CUDA)."""
        self.device = device
        self.min_free: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> MemorySampler:
        """Start sampling."""
        if self.device.type == "cuda":
            self._thread = threading.Thread(target=self._run, name="mem-sampler", daemon=True)
            self._thread.start()
        return self

    def _run(self) -> None:
        """Sampling loop."""
        import torch

        while not self._stop.is_set():
            free, _ = torch.cuda.mem_get_info(self.device)
            self.min_free = free if self.min_free is None else min(self.min_free, free)
            self._stop.wait(0.25)

    def __exit__(self, *exc) -> None:
        """Stop sampling."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()


class ModeLog:
    """The JSON file of one mode, rewritten after every run."""

    def __init__(self, path: pathlib.Path, header: dict) -> None:
        """Start the file with ``header``."""
        self.path = path
        self.data: dict = {"header": header, "runs": [], "summary": None}
        self.write()

    def add(self, record: dict) -> None:
        """Append one run and write."""
        self.data["runs"].append(record)
        self.write()

    def write(self) -> None:
        """Write atomically."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.partial")
        temporary.write_text(json.dumps(self.data, indent=2, default=str), encoding="utf-8")
        temporary.replace(self.path)


def _mem_info(device) -> dict | None:
    """``{"free", "total"}`` from ``torch.cuda.mem_get_info``, or ``None`` off CUDA."""
    import torch

    if device.type != "cuda":
        return None
    free, total = torch.cuda.mem_get_info(device)
    return {"free": int(free), "total": int(total)}


def run_mode(mode: str, k: int, tasks: list[dict], device, out_dir: pathlib.Path, prefix: str) -> dict:
    """Run ``tasks`` in one mode (``workers`` or ``streams``) with ``k`` and return its summary."""
    import torch

    from cdft.device_policy import capture_profile, derive_settings

    profile = capture_profile(device)
    path = out_dir / f"{prefix}{platform.node() or 'host'}_{device.type}_{mode}{k}.json"
    header = {
        "mode": mode,
        "k": k,
        "device": str(device),
        "device_profile": profile.as_dict(),
        "derived_settings": derive_settings(profile).as_dict(),
        "torch": torch.__version__,
        "hostname": platform.node(),
        "cores": os.cpu_count(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tasks": [{"scenario": t["scenario"], "n_electrons": t["n_electrons"]} for t in tasks],
        "dry_run": bool(tasks and tasks[0]["dry_run"]),
    }
    log = ModeLog(path, header)
    print(f"[{mode} {k}] {len(tasks)} runs on {device} -> {path}")
    before = _mem_info(device)
    started = time.perf_counter()
    with MemorySampler(device) as sampler:
        if mode == "workers":
            context = multiprocessing.get_context("spawn")
            threads = max(1, (os.cpu_count() or 1) // k) if device.type == "cpu" else 1
            # ProcessPoolExecutor, not Pool: a worker killed from outside (OOM killer, driver reset)
            # raises BrokenProcessPool here instead of hanging the measurement forever.
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=k, mp_context=context, initializer=_worker_init, initargs=(threads,)
            ) as pool:
                futures = {pool.submit(solve_one, task): task for task in tasks}
                for future in concurrent.futures.as_completed(futures):
                    try:
                        record = future.result()
                    except Exception as exc:  # noqa: BLE001 - a lost worker is recorded, not hidden
                        task = futures[future]
                        record = {
                            "index": task["index"],
                            "scenario": f"{task['scenario']}_N{task['n_electrons']:g}",
                            "n_electrons": task["n_electrons"],
                            "status": "error",
                            "error": f"worker lost: {type(exc).__name__}: {exc}",
                            "wall_s": 0.0,
                            "gpu_peak_bytes": None,
                            "pid": None,
                        }
                    log.add(record)
                    print(f"[{mode} {k}] {record['scenario']}: {record['status']} in {record['wall_s']:.1f} s (pid {record['pid']}) {record['error']}")
        elif mode == "streams":
            # Import here first: two threads importing the cdft.physics_config shim at once can see
            # it half initialised.
            import cdft.gates.runner  # noqa: F401
            import cdft.scf.loop  # noqa: F401
            import cdft.scf.solve  # noqa: F401

            for task in tasks[:1]:  # resolves the cdft.config / cdft.physics_config shims
                scan_scenario(task["scenario"], task["n_electrons"])
                scan_numerics(task["device"], task["dry_run"])

            streamed = [dict(t) for t in tasks]
            streams = [torch.cuda.Stream(device=device) for _ in range(k)] if device.type == "cuda" else [None] * k
            for i, task in enumerate(streamed):
                task["stream"] = streams[i % k]
            with concurrent.futures.ThreadPoolExecutor(max_workers=k, thread_name_prefix="stream") as pool:
                for future in concurrent.futures.as_completed([pool.submit(solve_one, t) for t in streamed]):
                    record = future.result()
                    log.add(record)
                    print(f"[{mode} {k}] {record['scenario']}: {record['status']} in {record['wall_s']:.1f} s ({record['thread']})")
        else:
            raise ValueError(f"unknown mode {mode!r}")
    wall = time.perf_counter() - started
    after = _mem_info(device)
    runs = log.data["runs"]
    completed = [r for r in runs if r["status"] not in ("error",)]
    valid = [r for r in runs if r["status"] == "valid"]
    peaks = [r["gpu_peak_bytes"] for r in runs if r["gpu_peak_bytes"] is not None]
    summary = {
        "mode": mode,
        "k": k,
        "runs": len(runs),
        "completed": len(completed),
        "valid": len(valid),
        "wall_s": round(wall, 3),
        "runs_per_hour": round(3600.0 * len(completed) / wall, 2) if wall > 0 else None,
        "valid_runs_per_hour": round(3600.0 * len(valid) / wall, 2) if wall > 0 else None,
        "mean_run_wall_s": round(sum(r["wall_s"] for r in runs) / len(runs), 3) if runs else None,
        "max_gpu_peak_bytes": max(peaks) if peaks else None,
        "mem_before": before,
        "mem_after": after,
        "min_free_bytes_sampled": sampler.min_free,
        "device_peak_bytes_sampled": (
            None if before is None or sampler.min_free is None else before["total"] - sampler.min_free
        ),
    }
    log.data["summary"] = summary
    log.data["header"]["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    log.write()
    return summary


def markdown(summaries: list[dict], device) -> str:
    """The Markdown table of every mode measured."""
    gib = 2**30
    lines = [
        f"### Scan throughput -- {platform.node()} / {device}",
        "",
        "| mode | k | runs | valid | wall s | runs/h | valid runs/h | mean run s | max run peak GiB | device peak GiB (sampled) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        peak = "-" if s["max_gpu_peak_bytes"] is None else f"{s['max_gpu_peak_bytes'] / gib:.2f}"
        device_peak = "-" if s["device_peak_bytes_sampled"] is None else f"{s['device_peak_bytes_sampled'] / gib:.2f}"
        lines.append(
            f"| {s['mode']} | {s['k']} | {s['runs']} | {s['valid']} | {s['wall_s']:.1f} | "
            f"{s['runs_per_hour']} | {s['valid_runs_per_hour']} | {s['mean_run_wall_s']} | {peak} | {device_peak} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run every requested mode, print the Markdown summary."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--scenario", default=None,
        help=f"registered scenario to sweep (default h2plus_R2_lda; {DRY_SCENARIO} for --dry-run)",
    )
    parser.add_argument("--n-electrons", nargs="+", type=float, default=list(DEFAULT_N))
    parser.add_argument("--repeat", type=int, default=1, help="repeat the N list this many times")
    parser.add_argument("--workers", nargs="*", type=int, default=[], help="process counts to measure")
    parser.add_argument("--streams", nargs="*", type=int, default=[], help="stream (thread) counts to measure")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "reports" / "throughput"))
    parser.add_argument("--dry-run", action="store_true", help="CPU script check: two tiny runs per mode (not a measurement)")
    args = parser.parse_args(argv)

    from cdft.precision import resolve_device
    from contract import Device

    prefix = ""
    if args.dry_run:
        args.device = "cpu"
        args.n_electrons = list(DRY_N)
        args.repeat = 1
        args.workers = args.workers or [2]
        args.streams = args.streams or [2]
        args.scenario = args.scenario or DRY_SCENARIO
        prefix = "dryrun_"
        print("DRY RUN: coarse grid, derive_grid=False -- a script check, not a measurement")
    args.scenario = args.scenario or "h2plus_R2_lda"
    if not args.workers and not args.streams:
        args.workers = [1]
    try:
        device = resolve_device(Device(args.device))
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    from cdft.device_policy import apply_policy, describe

    apply_policy(device)
    print(describe(device))
    tasks = [
        {"scenario": args.scenario, "n_electrons": n, "device": args.device, "dry_run": args.dry_run, "index": i}
        for i, n in enumerate([n for _ in range(max(1, args.repeat)) for n in args.n_electrons])
    ]
    out_dir = pathlib.Path(args.output_dir)
    summaries = []
    for k in args.workers:
        summaries.append(run_mode("workers", k, tasks, device, out_dir, prefix))
    for k in args.streams:
        summaries.append(run_mode("streams", k, tasks, device, out_dir, prefix))
    print()
    print(markdown(summaries, device))
    return 0 if all(s["completed"] == s["runs"] for s in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
