#!/usr/bin/env python3
"""One entry point that runs everything: unit tests, component gates and scenario gates.

    python test_suite.py                    # --profile quick is the default
    python test_suite.py --profile quick    # inner loop, CPU, target under 2 min
    python test_suite.py --profile full     # every gate and every pinned value, CPU
    python test_suite.py --profile gpu      # as full, on the CUDA device (error if there is none)
    python test_suite.py --device cuda      # any profile, on another device
    python test_suite.py --unit-only        # skip the gates
    python test_suite.py --gates-only       # skip pytest
    python test_suite.py --corpus out.h5    # also write the records to HDF5

The exit code is 0 only if everything passed (gate G5.6). The three layers are reported separately:
unit tests prove the code does what it was written to do; component gates verify an operator at a
configuration; scenario gates are evaluated on a completed run and travel with the record into the
corpus (D-13). A gate a scenario names and this build cannot evaluate is a failure, not an omission.

A profile decides which checks run, never the settings they run at: several gate thresholds are
DERIVED from the production grid, so a coarser grid would make them fail correctly. Profiles differ
in pytest selection, gate solve budget, LOBPCG cross-check, which scenarios are solved and the
default device (``gpu`` is ``full`` on CUDA); anything left out is printed with its reason (G5.3,
G5.5). Strict kernels live only inside gate G5.2 (D-61). ``--device`` also reaches the pytest
subprocess (``CDFT_DEVICE``); ``cuda`` without a device exits 2 and nothing falls back (G5.5).
"""

from __future__ import annotations

import argparse
import builtins
import dataclasses
import functools
import json
import os
import pathlib
import platform
import re
import subprocess
import sys
import time

#: Flush every progress line: block buffering hides the last scenario when piped to a log.
print = functools.partial(builtins.print, flush=True)  # noqa: A001

REPO_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

#: What each profile changes. Physics settings are deliberately absent from this table.
PROFILES: dict[str, dict[str, object]] = {
    "quick": {
        "pytest_args": ["-m", "fast"],
        "max_solves": 0,
        "cross_check": False,
        "device": "cpu",
        "deterministic": False,
        # Measured on two CPU cores: 126 s for the harmonic well (G1.2 needs h = 0.20) plus two
        # self-consistent scenarios of minutes each (D-53).
        "budget_seconds": 900,
        "scenarios": "quick",
        "description": "inner loop: fast tests, no ladder-building gates, no cross-check, two interacting scenarios",
    },
    "full": {
        "pytest_args": ["--runslow"],
        "max_solves": None,
        "cross_check": True,
        "device": "cpu",
        "deterministic": False,
        "budget_seconds": None,
        "scenarios": None,
        "description": "everything, on the CPU float64 path",
    },
    "gpu": {
        "pytest_args": ["--runslow"],
        "max_solves": None,
        "cross_check": True,
        "device": "cuda",
        # Strict kernels only inside the G5.2 pair; the primary solves run off.
        "deterministic": False,
        # Target on a GTX 1660 Ti; reported, never enforced.
        "budget_seconds": 1800,
        "scenarios": None,
        "description": "everything, on the CUDA device (= full on CUDA); an error if there is none (G5.5)",
    },
}

#: Scenarios a profile does not solve **when it runs on CUDA**, their measured peak not fitting the
#: card. Printed with :data:`CPU_ONLY_REASON` and recorded in the JSON; empty until measured.
CPU_ONLY_SCENARIOS: dict[str, frozenset[str]] = {
    "quick": frozenset(),
    "full": frozenset(),
    "gpu": frozenset(),
}

#: Why a scenario in :data:`CPU_ONLY_SCENARIOS` is not solved on the card.
CPU_ONLY_REASON = (
    "its measured peak does not fit this card (O-22); it runs in the same profile on the CPU"
)

#: Host memory a CPU solve of a scenario needs, where it exceeds a small host: the measured peak
#: plus headroom. `h2plus_R8_lda` on its derived 116^3 box reached 6.1 GB RSS and was killed by a
#: 5.8 GiB cgroup on 2026-09-18 (O-22); a host below the figure skips it visibly (G5.5).
HOST_MEMORY_NEEDED_BYTES: dict[str, int] = {"h2plus_R8_lda": 8 * 2**30}

#: Why a scenario in :data:`HOST_MEMORY_NEEDED_BYTES` is not solved on a small host.
HOST_MEMORY_REASON = "its measured CPU peak exceeds this host's memory (O-22); run it on a larger host"

#: Card memory a CUDA solve of a scenario needs, where it exceeds a small card: `h2plus_R8_lda`
#: ran 631 s on a 6 GiB GTX 1660 Ti and died asking for 434 MiB more at 4.58 GiB
#: allocated (2026-09-18 and 09-19, the ladder's rungs inside the 200-iteration limit), as it did
#: in the throughput work (3.55 + 1.34 GiB). A card below the figure skips it visibly (G5.5, D-83) instead of
#: spending ten minutes on an error row.
DEVICE_MEMORY_NEEDED_BYTES: dict[str, int] = {"h2plus_R8_lda": 8 * 2**30}

#: Why a scenario in :data:`DEVICE_MEMORY_NEEDED_BYTES` is not solved on a small card.
DEVICE_MEMORY_REASON = "its measured CUDA peak exceeds this card's memory (O-22); run it on a larger card"


def device_memory_bytes(device: "torch.device") -> int | None:
    """Total memory of the CUDA device, or ``None`` off the card."""
    import torch

    if device.type != "cuda":
        return None
    return int(torch.cuda.get_device_properties(device).total_memory)


def host_memory_bytes() -> int | None:
    """Host memory available to this process: RAM capped by the cgroup limit (``cdft.device_policy``)."""
    from cdft.device_policy import cgroup_memory_limit_bytes

    limits = []
    limit = cgroup_memory_limit_bytes()
    if limit:
        limits.append(limit)
    try:
        for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                limits.append(int(line.split()[1]) * 1024)
    except OSError:
        pass
    if not limits:
        try:
            import psutil  # type: ignore

            limits.append(int(psutil.virtual_memory().total))
        except Exception:
            return None
    return min(limits)


def _rule(title: str = "", width: int = 78) -> str:
    """Return a section rule, optionally with a title."""
    if not title:
        return "=" * width
    return f"\n{'=' * width}\n{title}\n{'=' * width}"


def resolve_suite_device(name: str):
    """Resolve ``cpu``/``cuda``/``auto`` to a torch device, or raise; there is no fallback."""
    from cdft.precision import resolve_device
    from contract import Device

    return resolve_device(Device(name))


class SuiteLog:
    """The JSON summary of one suite run, rewritten atomically after every update."""

    def __init__(self, path: pathlib.Path | None, header: dict[str, object]) -> None:
        """Start a log at ``path`` (``None`` disables writing) with a header block."""
        self.path = path
        self.data: dict[str, object] = {"header": header, "unit_tests": None, "component_gates": None, "scenarios": []}
        self.flush()

    def update(self, key: str, value: object) -> None:
        """Set a top-level entry and write."""
        self.data[key] = value
        self.flush()

    def add_scenario(self, entry: dict[str, object]) -> None:
        """Append one scenario's summary and write."""
        self.data["scenarios"].append(entry)  # type: ignore[union-attr]
        self.flush()

    def flush(self) -> None:
        """Write the whole log to a temporary file and move it into place."""
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".partial")
        temporary.write_text(json.dumps(self.data, indent=2, default=str), encoding="utf-8")
        temporary.replace(self.path)


def _gate_rows(results) -> list[dict[str, object]]:
    """Compact JSON form of a gate report; a row that did not pass carries the gate's detail.

    The 2026-09-18 CUDA full run recorded G5.9 as ``measured 4.0`` and nothing else, so which
    budgets the card's host exceeded could not be read back from the file (O-31).
    """
    rows: list[dict[str, object]] = []
    for r in results:
        row: dict[str, object] = {
            "gate_id": r.gate_id,
            "verdict": r.verdict.value,
            "measured": r.measured,
            "threshold": r.threshold,
            "reason": r.reason,
        }
        if r.verdict.value != "pass" and r.detail:
            row["detail"] = dict(r.detail)
        rows.append(row)
    return rows


#: Below this much host memory the unit tests run one file per pytest process (D-81): the whole
#: ``--runslow`` tier in one process was OOM-killed at 5.8 GB on a 2-core, 8 GiB host with a 5.8 GiB cgroup limit.
UNIT_TESTS_ONE_PROCESS_MIN_BYTES = 8 * 2**30


def _pytest_batches(per_file: bool) -> list[list[str]]:
    """The pytest invocations: one for the whole tree, or one per test file plus the golden pair."""
    if not per_file:
        return [[str(REPO_ROOT / "tests")]]
    batches: list[list[str]] = []
    for path in sorted((REPO_ROOT / "tests").glob("test_*.py")):
        if path.name == "test_golden_regression.py":
            # Its value tests and its two ladder gate tests exceed a small host together.
            batches.append([str(path), "-k", "not gate"])
            batches.append([str(path), "-k", "gate"])
        else:
            batches.append([str(path)])
    return batches


def run_unit_tests(profile: str, device: str = "cpu", verbose: bool = False) -> tuple[int, int]:
    """Run pytest for this profile on ``device`` (``CDFT_DEVICE``); return exit code and count.

    On a host below :data:`UNIT_TESTS_ONE_PROCESS_MIN_BYTES` the tree runs one file per process,
    printed as such (G5.5); the counts are summed and the exit code is the worst.
    """
    print(_rule("UNIT TESTS"))
    extra = list(PROFILES[profile]["pytest_args"])  # type: ignore[arg-type]
    from cdft.reference.oracles import oracles_mode

    host_bytes = host_memory_bytes()
    per_file = host_bytes is not None and host_bytes < UNIT_TESTS_ONE_PROCESS_MIN_BYTES
    print(f"  selection: {' '.join(extra) or 'all'}   CDFT_DEVICE={device}   oracles: {oracles_mode()}")
    if per_file:
        print(f"  one pytest process per test file: host memory {host_bytes / 2**30:.1f} GiB is below "
              f"{UNIT_TESTS_ONE_PROCESS_MIN_BYTES / 2**30:.0f} GiB (D-81)")
    environment = dict(os.environ, CDFT_DEVICE=device)  # carries CDFT_ORACLES to the pytest child
    started = time.perf_counter()
    worst = 0
    count = 0
    for batch in _pytest_batches(per_file):
        # No "-q": pyproject supplies one, and a second silences the summary line parsed below.
        command = [sys.executable, "-m", "pytest", *batch, *extra]
        if verbose:
            command.append("-v")
        completed = subprocess.run(
            command, cwd=REPO_ROOT, capture_output=True, text=True, check=False, env=environment
        )
        output = completed.stdout.strip().splitlines()
        if per_file:
            label = pathlib.Path(batch[0]).name + (f" {' '.join(batch[1:])}" if len(batch) > 1 else "")
            print(f"  {label}: {output[-1] if output else '(no output)'}")
            if completed.returncode:
                print("\n".join(output[-12:]))
        else:
            print("\n".join(output[-12:] if completed.returncode else output[-3:]))
        if completed.returncode and completed.stderr.strip():
            print(completed.stderr.strip()[-2000:], file=sys.stderr)
        # pytest exit 5 is "no tests collected" (a -k that matches nothing); not a failure here.
        if completed.returncode not in (0, 5):
            worst = max(worst, completed.returncode)
        for line in reversed(output):
            if " passed" in line or " failed" in line:
                count += sum(int(n) for n in re.findall(r"(\d+) (?:passed|failed)", line))
                break
    elapsed = time.perf_counter() - started
    print(f"\nunit tests: {'PASS' if worst == 0 else 'FAIL'} ({count} tests, {elapsed:.0f} s)")
    return worst, count


def run_gates(
    profile: str,
    corpus: pathlib.Path | None,
    device_name: str = "cpu",
    log: SuiteLog | None = None,
    only: tuple[str, ...] | None = None,
) -> int:
    """Run the component gates and the selected scenarios on ``device_name``; return exit code.

    ``only`` narrows the profile's scenarios to the ids given (``--scenarios``): the way a small host
    runs the ``full`` profile one process per scenario (D-81). Printed and recorded, never silent.
    """
    from cdft.config import ALL_ELECTRON_NUMERICS, MODEL_SYSTEM_NUMERICS
    from cdft.gates.known_open import COMPONENT, partition_failures
    from cdft.gates.runner import GateSuite, format_report, states_required
    from cdft.io.hdf5 import CorpusWriter
    from cdft.operators import geometry_cache
    from cdft.physics_config import QUICK_PROFILE_SCENARIOS, REGISTRY
    from cdft.scf.solve import solve_scenario
    from contract import Device, GateVerdict, RunStatus

    settings = dict(PROFILES[profile])
    settings["scenarios"] = QUICK_PROFILE_SCENARIOS if settings["scenarios"] == "quick" else None
    device = Device(device_name)

    def tune(numerics):
        """Apply the profile's non-physics axes: cross-check, device, deterministic kernels."""
        return dataclasses.replace(
            numerics,
            eigen=dataclasses.replace(numerics.eigen, cross_check=bool(settings["cross_check"])),
            device=device,
            deterministic=bool(settings["deterministic"]),
        )

    model = tune(MODEL_SYSTEM_NUMERICS)
    all_electron = tune(ALL_ELECTRON_NUMERICS)

    suite = GateSuite(max_solves=settings["max_solves"])  # type: ignore[arg-type]
    failures = 0
    known = 0
    writer = CorpusWriter(corpus) if corpus else None

    # Component gates build operators without a solver, so apply the kernel mode here first.
    from cdft.precision import configure_device, release_device_memory, resolve_device

    resolved = resolve_device(device)
    configure_device(resolved, bool(settings["deterministic"]))

    print(_rule("COMPONENT GATES"))
    started = time.perf_counter()
    component = suite.run_components(model)
    print(format_report(component))
    print(f"({time.perf_counter() - started:.0f} s)")
    if log is not None:
        log.update("component_gates", _gate_rows(component))
    component_split = partition_failures(COMPONENT, component)
    failures += component_split.new
    known += component_split.known
    known_rows: list[str] = list(component_split.rows)

    print(_rule("SCENARIO GATES"))
    summary: list[tuple] = []
    selected = [
        scenario for scenario in REGISTRY
        if settings["scenarios"] is None or scenario.scenario_id in settings["scenarios"]
    ]
    if only is not None:
        unknown = sorted(set(only) - {scenario.scenario_id for scenario in REGISTRY})
        if unknown:
            raise SystemExit(f"--scenarios names unknown scenario id(s): {', '.join(unknown)}")
        # The profile keeps its settings (budget, cross-check, device); which scenarios it solves
        # is the one axis --scenarios replaces, printed and recorded below.
        selected = [scenario for scenario in REGISTRY if scenario.scenario_id in only]
        print(f"  --scenarios: this process solves only {', '.join(only)} at the {profile!r} profile's "
              f"settings; the profile's own scenario list does not apply\n")
    if log is not None:
        log.update("scenario_selection", {"only": list(only) if only is not None else None})
    not_run = [scenario.scenario_id for scenario in REGISTRY if scenario not in selected]
    if not_run and only is None:
        # A profile may choose which scenarios it solves; it may not do so invisibly (G5.3, G5.5).
        print(
            f"  not solved in the {profile!r} profile (self-consistent scenarios cost minutes at the "
            f"production interacting grid, D-53; run --profile full): {', '.join(not_run)}\n"
        )
    cpu_only = CPU_ONLY_SCENARIOS.get(profile, frozenset()) if resolved.type == "cuda" else frozenset()
    excluded = [scenario.scenario_id for scenario in selected if scenario.scenario_id in cpu_only]
    if excluded:
        print(f"  not solved on {resolved} in the {profile!r} profile: {', '.join(excluded)} -- {CPU_ONLY_REASON}\n")
        selected = [scenario for scenario in selected if scenario.scenario_id not in cpu_only]
    if log is not None:
        log.update("cpu_only_excluded", {"scenarios": excluded, "reason": CPU_ONLY_REASON if excluded else ""})
    memory_excluded: list[str] = []
    if resolved.type == "cpu":
        table, reason, where = HOST_MEMORY_NEEDED_BYTES, HOST_MEMORY_REASON, "host"
        available = host_memory_bytes()
    else:
        table, reason, where = DEVICE_MEMORY_NEEDED_BYTES, DEVICE_MEMORY_REASON, "card"
        available = device_memory_bytes(resolved)
    for scenario in selected:
        needed = table.get(scenario.scenario_id)
        if needed is not None and available is not None and available < needed:
            memory_excluded.append(scenario.scenario_id)
    if memory_excluded:
        print(
            f"  not solved on this {where} in the {profile!r} profile: {', '.join(memory_excluded)} -- "
            f"{reason} (needs {max(table[s] for s in memory_excluded) / 2**30:.0f} GiB, "
            f"{where} {available / 2**30:.1f} GiB)\n"
        )
        selected = [scenario for scenario in selected if scenario.scenario_id not in memory_excluded]
    if log is not None:
        log.update("host_memory_excluded", {"scenarios": memory_excluded, "reason": reason if memory_excluded else ""})
    for scenario in selected:
        numerics = model if scenario.structure.n_atoms == 0 else all_electron
        if device.value == "cuda":
            # Consecutive scenarios never share a geometry key, so only card memory is released.
            geometry_cache.clear()
        started = time.perf_counter()
        required = states_required(scenario, suite.artifact_gates)
        artifact = solve_scenario(scenario, numerics, n_states=required or None)
        artifact = suite.run_on_artifact(artifact)
        elapsed = time.perf_counter() - started
        if writer is not None:
            writer.write(artifact)

        counts = {v.value: 0 for v in GateVerdict}
        for result in artifact.gates.results:
            counts[result.verdict.value] += 1

        # INVALID has three causes, only the last a defect: a gate this build does not implement
        # (INCOMPLETE), a known-open failure (which fails the build only when it worsens), or else.
        split = partition_failures(scenario.scenario_id, artifact.gates.results)
        unimplemented = split.unimplemented
        known_here = split.known
        new_failures = split.new
        known_rows.extend(split.rows)

        incomplete = new_failures == 0 and unimplemented > 0 and known_here == 0
        label = "incomplete" if incomplete else artifact.status.value
        peak_bytes = artifact.provenance.gpu_peak_bytes
        summary.append((
            scenario.scenario_id, label, counts["pass"], counts["marginal"],
            new_failures, known_here, unimplemented, counts["skipped"], elapsed,
            "-" if peak_bytes is None else f"{peak_bytes / 2**20:.0f}",
        ))
        title = f"{scenario.scenario_id}  [status={artifact.status.value}]"
        if artifact.error_message:
            title += f"  error={artifact.error_message}"
        print(format_report(artifact.gates.results, title))
        prov = artifact.provenance
        peak = prov.gpu_peak_bytes
        print(
            f"  device {prov.device} ({prov.device_name}, capability {prov.device_capability}, "
            f"deterministic {prov.deterministic_algorithms})   gpu_peak_bytes "
            f"{'n/a' if peak is None else f'{peak} ({peak / 2**30:.2f} GiB)'}"
        )
        print(f"  precision policy {json.dumps(dict(prov.precision_policy), sort_keys=True)}")
        print(f"({elapsed:.0f} s)\n")
        if log is not None:
            log.add_scenario(
                {
                    "scenario_id": scenario.scenario_id,
                    "status": artifact.status.value,
                    "label": label,
                    "error": artifact.error_message,
                    "elapsed_s": round(elapsed, 3),
                    "device": prov.device,
                    "device_name": prov.device_name,
                    "deterministic_algorithms": prov.deterministic_algorithms,
                    "gpu_peak_bytes": peak,
                    "precision_policy": dict(prov.precision_policy),
                    "scf_iterations": artifact.measurements.get("scf_iterations"),
                    "new_failures": new_failures,
                    "known_open": known_here,
                    "gates": _gate_rows(artifact.gates.results),
                }
            )
        known += known_here
        if new_failures or artifact.status in (RunStatus.ERROR, RunStatus.UNCONVERGED):
            failures += 1
        # The next scenario starts from an empty CUDA cache; a no-op on the CPU.
        del artifact
        release_device_memory(resolved)

    print(_rule("SUMMARY"))
    header = (
        f"{'scenario':<16}{'status':<12}{'pass':>6}{'marg':>6}{'new':>6}"
        f"{'known':>7}{'notyet':>8}{'skip':>6}{'s':>8}{'peak MiB':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in summary:
        print(f"{row[0]:<16}{row[1]:<12}{row[2]:>6}{row[3]:>6}{row[4]:>6}"
              f"{row[5]:>7}{row[6]:>8}{row[7]:>6}{row[8]:>8.0f}{row[9]:>10}")

    incomplete = [row[0] for row in summary if row[1] == "incomplete"]
    if incomplete:
        print(
            f"\nincomplete (expected at Increment 1): {', '.join(incomplete)}\n"
            f"  these scenarios name gates that need the self-consistent loop of Increment 3.\n"
            f"  Their records are written INVALID and excluded by the corpus reader, which is\n"
            f"  correct; they do not set the exit code."
        )
    if known_rows:
        print(f"\nknown open failures ({known}), see docs/02_STATUS.md (Part B):")
        for row in known_rows:
            print(row)
        print(
            "  These keep their FAIL verdict and their records stay INVALID. What they do not do\n"
            "  is set the exit code, for exactly as long as they do not get worse."
        )
    if corpus:
        print(f"\ncorpus written to {corpus}")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the requested layers, and return the process exit code."""
    parser = argparse.ArgumentParser(
        prog="test_suite.py",
        description="Run the cdft unit tests and the physics gate suite.",
        epilog="Exits non-zero if anything failed (gate G5.6).",
    )
    parser.add_argument(
        "--profile", choices=sorted(PROFILES), default="quick",
        help="; ".join(f"{name}: {cfg['description']}" for name, cfg in PROFILES.items()),
    )
    parser.add_argument("--unit-only", action="store_true", help="run only pytest")
    parser.add_argument("--gates-only", action="store_true", help="run only the gate suite")
    parser.add_argument(
        "--scenarios", nargs="+", default=None, metavar="ID",
        help="solve only these scenario ids in this process (one process per scenario on a small host, D-81)",
    )
    parser.add_argument("--corpus", default=None, help="also write the records to this HDF5 file")
    parser.add_argument(
        "--device", choices=("cpu", "cuda", "auto"), default=None,
        help="override the profile's device (quick/full: cpu, gpu: cuda); cuda without a device is an error",
    )
    parser.add_argument(
        "--json", default=None,
        help="JSON summary, rewritten after every scenario "
        "(default reports/suite_runs/<profile>_<device>_<host>.json)",
    )
    parser.add_argument("--no-json", action="store_true", help="do not write the JSON summary")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose pytest output")
    parser.add_argument(
        "--oracles", action="store_true",
        help="run the PySCF/libxc oracle tests and gates (G0.6) too; same as CDFT_ORACLES=1. "
        "Off by default, and the header, the JSON summary and every record say which (O-24)",
    )
    args = parser.parse_args(argv)

    settings = PROFILES[args.profile]
    print(_rule())
    print("cdft test suite")
    try:
        import cdft
        from contract import CONTRACT_VERSION

        print(f"contract {CONTRACT_VERSION}   package {cdft.__version__}   "
              f"profile {args.profile}  ({settings['description']})")
    except ImportError as exc:  # pragma: no cover - a broken install should say so plainly
        print(f"cannot import the package: {exc}", file=sys.stderr)
        return 2
    from cdft.reference.oracles import enable_oracles, oracles_mode

    if args.oracles:
        enable_oracles()  # before pytest is spawned and before any GateSuite reads the switch
    try:
        oracles = oracles_mode()
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    requested_device = args.device or str(settings["device"])
    try:
        import torch

        from cdft.precision import policy_record
        from contract import PrecisionConfig

        resolved = resolve_suite_device(requested_device)
    except RuntimeError as exc:
        print(
            f"ERROR: --profile {args.profile} needs device {requested_device!r}: {exc}",
            file=sys.stderr,
        )
        print(_rule())
        return 2
    device_name = resolved.type
    if resolved.type == "cuda":
        capability = ".".join(str(v) for v in torch.cuda.get_device_capability(resolved))
        described = f"{torch.cuda.get_device_name(resolved)} (capability {capability})"
    else:
        described = platform.processor() or platform.machine()
    print(
        f"device {resolved} = {described}   requested {requested_device}   torch {torch.__version__} "
        f"(CUDA build {torch.version.cuda})"
    )
    from cdft.device_policy import apply_policy, capture_profile, describe

    # Once per process, before any solve: the profile the settings are derived from.
    apply_policy(resolved)
    print(describe(resolved))
    print(f"precision policy {json.dumps(policy_record(PrecisionConfig(), resolved), sort_keys=True)}")
    print(
        f"deterministic kernels requested: {bool(settings['deterministic'])} "
        "(NumericsConfig.deterministic for the primary solves; the G5.2 pair always runs strict)"
    )
    print(f"oracles: {oracles} (PySCF/libxc tests and G0.6; --oracles or CDFT_ORACLES=1 turns them on, O-24)")
    print(_rule())

    json_path = None
    if not args.no_json:
        json_path = pathlib.Path(args.json) if args.json else (
            REPO_ROOT / "reports" / "suite_runs"
            / f"{args.profile}_{device_name}_{platform.node() or 'host'}.json"
        )
    log = SuiteLog(
        json_path,
        {
            "profile": args.profile,
            "device_requested": requested_device,
            "device": str(resolved),
            "device_description": described,
            "torch": torch.__version__,
            "torch_cuda": str(torch.version.cuda),
            "hostname": platform.node(),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "device_profile": capture_profile(resolved).as_dict(),
            "cpu_only_scenarios": sorted(CPU_ONLY_SCENARIOS.get(args.profile, frozenset())),
            "oracles": oracles,
        },
    )
    if json_path is not None:
        print(f"JSON summary: {json_path}")

    overall = 0
    total_started = time.perf_counter()

    if not args.gates_only:
        code, count = run_unit_tests(args.profile, device=device_name, verbose=args.verbose)
        overall |= code
        log.update("unit_tests", {"exit_code": code, "count": count})

    if not args.unit_only:
        corpus = pathlib.Path(args.corpus) if args.corpus else None
        if corpus and corpus.exists():
            # Appending is legal, but mixing two suite runs into one file is not useful.
            print(f"\nrefusing to append to the existing corpus {corpus}; move it or choose "
                  f"another path", file=sys.stderr)
            return 2
        overall |= run_gates(
            args.profile, corpus, device_name=device_name, log=log,
            only=tuple(args.scenarios) if args.scenarios else None,
        )

    elapsed = time.perf_counter() - total_started
    print(_rule())
    verdict = "ALL PASS" if overall == 0 else "FAILURES PRESENT"
    print(f"{verdict}   profile {args.profile}   device {resolved}   total {elapsed:.0f} s")
    log.update("result", {"exit_code": overall, "verdict": verdict, "total_s": round(elapsed, 3)})
    budget = settings["budget_seconds"]
    if budget is not None:
        # Reported, never enforced: a suite that failed on wall time would fail on a loaded machine.
        state = "within" if elapsed <= float(budget) else "OVER"
        print(f"  {state} the {budget} s target for this profile")
        if args.profile == "quick":
            print(
                f"  (target measured on 2 CPU cores; this machine reports "
                f"{os.cpu_count()} -- expect proportionally less on a wider one)"
            )
    if overall:
        print(
            "\nRecords that failed a gate are written with status INVALID and are excluded by the "
            "default corpus reader (gate G5.4).",
            file=sys.stderr,
        )
    print(_rule())
    return overall


if __name__ == "__main__":
    raise SystemExit(main())
