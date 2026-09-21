#!/usr/bin/env python3
"""Publication figures for every registered scenario, one ``figures.py`` process per scenario (D-72).

One process per scenario returns the card's memory between runs and isolates failures; each logs to
``<root>/<scenario>/figures.log`` and the suite writes ``suite_summary.{json,txt}`` under ``--out``.
Names, options, figure scope and CUDA availability (G5.5) are checked before the first solve;
``--help`` has the exit codes. Rationale: ``docs/03_METHOD.md`` Part E (``figures.py``).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import os
import pathlib
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
for _entry in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

#: Scenarios left out of the default plan, with the reason; naming one in ``--scenarios`` runs it.
DEFAULT_EXCLUDED: dict[str, str] = {
    "h2plus_R8_lda": (
        "O-22: no SCF convergence within 200 iterations on its derived 28.75-bohr box, and an "
        "out-of-memory stop on the 6-GB card; name it in --scenarios to run it anyway"
    ),
}

DEFAULT_ROOT = pathlib.Path("reports") / "figures"

#: Exit code of an interrupted suite (the shell convention for SIGINT).
INTERRUPTED = 130


@dataclass(frozen=True)
class PlannedRun:
    """One ``figures.py`` invocation of the suite."""

    scenario_id: str
    outdir: pathlib.Path
    #: The arguments after ``figures.py``.
    argv: tuple[str, ...]

    def command(self, python: str) -> list[str]:
        """Return the full command line of this run."""
        return [python, str(REPO_ROOT / "figures.py"), *self.argv]


@dataclass
class RunResult:
    """What one run left behind, read from its exit code and its ``manifest.json``."""

    scenario_id: str
    exit_code: int
    seconds: float
    outdir: str
    run_id: str | None = None
    status: str | None = None
    #: ``""`` for a trusted record, else its drawing mark (invalid, known-open, deferred).
    mark: str | None = None
    drawable: bool | None = None
    reason: str = ""
    drawn: list[str] = field(default_factory=list)
    not_drawn: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def outcome(self) -> str:
        """Return the one-word reading of the run for the summary table."""
        if self.exit_code == INTERRUPTED:
            return "interrupted"
        if self.status is None:
            return "no manifest"
        if self.drawable is False:
            return "refused"
        if self.errors:
            return "figure errors"
        return "ok" if self.exit_code == 0 else "drawn, exit 1"


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser of the suite runner."""
    parser = argparse.ArgumentParser(
        prog="figures_suite.py",
        description="Run figures.py for every registered scenario, one process each, and summarise.",
        epilog="Exit code: 0 every run exited 0; 1 a run exited non-zero (see the summary); 2 a usage "
               "error found before the first run; 130 interrupted.",
    )
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cuda",
                        help="device of every solve (default cuda; cuda without CUDA is an error, G5.5)")
    parser.add_argument("--gates", choices=["quick", "full"], default="quick",
                        help="gate budget of every run (default quick; full for publication)")
    parser.add_argument("--res", choices=("draft", "standard", "fine", "reference"), default="standard",
                        help="resolution level passed to figures.py (registered scenarios only)")
    parser.add_argument("--scenarios", nargs="+", default=None, metavar="ID",
                        help="run exactly these registered scenarios (default: all but the excluded ones)")
    parser.add_argument("--exclude", nargs="+", default=(), metavar="ID",
                        help="leave these scenarios out of the default plan")
    parser.add_argument("--scenario-file", default=None,
                        help="YAML file of scenario definitions to use instead of the registry")
    parser.add_argument("--out", default=None,
                        help="root folder of the suite (default reports/figures/suite_<UTC stamp>)")
    parser.add_argument("--allow-invalid", action="store_true",
                        help="also draw records outside TRUSTED_STATUSES, badged 'not for publication'")
    parser.add_argument("--only", nargs="+", default=None, metavar="FIGURE", help="draw only these figures")
    parser.add_argument("--skip", nargs="+", default=None, metavar="FIGURE", help="do not draw these figures")
    parser.add_argument("--formats", nargs="+", default=None, choices=["pdf", "png", "svg"],
                        help="file formats (figures.py default: pdf png)")
    parser.add_argument("--font", choices=["sans", "serif"], default=None, help="figure typeface family")
    parser.add_argument("--usetex", action="store_true", help="typeset with LaTeX (TeX, cm-super, dvipng)")
    parser.add_argument("--dpi", type=int, default=None, help="PNG resolution (figures.py default 600)")
    parser.add_argument("--map-points", type=int, default=None, help="points per axis of the density maps")
    parser.add_argument("--states", type=int, default=None, help="converge at least this many eigenstates")
    parser.add_argument("--no-oracle", action="store_true",
                        help="skip the radial Kohn-Sham and two-centre oracles (faster, fewer references)")
    parser.add_argument("--timeout-min", type=float, default=None,
                        help="stop a run that takes longer than this many minutes (default: no limit)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and the commands, run nothing")
    return parser


def load_registry(scenario_file: str | None):
    """Return the scenario registry the suite draws from (``physics_config`` or a YAML file)."""
    import physics_config

    return physics_config.load_scenarios(scenario_file) if scenario_file else physics_config.REGISTRY


def select_scenarios(registry, names: Sequence[str] | None, exclude: Sequence[str]) -> tuple[list[str], list[str]]:
    """Return ``(scenario ids to run, problems)`` for the requested names and exclusions."""
    known = [scenario.scenario_id for scenario in registry]
    problems = [f"unknown scenario {name!r}" for name in list(names or []) + list(exclude) if name not in known]
    if names:
        chosen = [name for name in dict.fromkeys(names) if name in known]
    else:
        chosen = [name for name in known if name not in DEFAULT_EXCLUDED]
    chosen = [name for name in chosen if name not in set(exclude)]
    if not chosen and not problems:
        problems.append("no scenario left to run")
    return chosen, problems


def figure_arguments(args: argparse.Namespace) -> list[str]:
    """Return the figures.py options every run of the suite shares, device and gates included."""
    argv = ["--device", args.device, "--gates", args.gates]
    if args.res != "standard":
        argv += ["--res", args.res]
    if args.scenario_file:
        argv += ["--scenario-file", str(args.scenario_file)]
    if args.allow_invalid:
        argv.append("--allow-invalid")
    for option, value in (("--only", args.only), ("--skip", args.skip), ("--formats", args.formats)):
        if value:
            argv += [option, *value]
    for option, value in (("--font", args.font), ("--dpi", args.dpi), ("--map-points", args.map_points),
                          ("--states", args.states)):
        if value is not None:
            argv += [option, str(value)]
    if args.usetex:
        argv.append("--usetex")
    if args.no_oracle:
        argv.append("--no-oracle")
    return argv


def default_root(now: datetime.datetime | None = None) -> pathlib.Path:
    """Return ``reports/figures/suite_<UTC stamp>`` under the repository root."""
    now = now or datetime.datetime.now(datetime.UTC)
    return REPO_ROOT / DEFAULT_ROOT / f"suite_{now.strftime('%Y%m%dT%H%M%SZ')}"


def plan_runs(args: argparse.Namespace, scenario_ids: Sequence[str], root: pathlib.Path) -> list[PlannedRun]:
    """Return one planned run per scenario, in the given order."""
    shared = figure_arguments(args)
    return [
        PlannedRun(scenario_id, root / scenario_id, (scenario_id, *shared, "--out", str(root / scenario_id)))
        for scenario_id in scenario_ids
    ]


def validate_plan(plan: Sequence[PlannedRun], registry) -> list[str]:
    """Every usage problem of the plan, found with figures.py's own checks, without solving."""
    import figures

    problems: list[str] = []
    parser = figures.build_parser()
    for run in plan:
        try:
            namespace = parser.parse_args(list(run.argv))
        except SystemExit:
            problems.append(f"{run.scenario_id}: figures.py rejects the arguments {' '.join(run.argv)}")
            continue
        for problem in figures._usage_problems(namespace):
            problems.append(f"{run.scenario_id}: {problem}")
        try:
            scenario, _ = figures.select_scenario(run.scenario_id, registry)
            numerics = figures.select_numerics(scenario, "auto", figures.GATE_PROFILES[namespace.gates],
                                               namespace.device)
        except (KeyError, ValueError, OSError) as exc:
            problems.append(f"{run.scenario_id}: {exc}")
            continue
        scope = figures.figure_scope_problem(scenario, numerics)
        if scope is not None:
            problems.append(f"{run.scenario_id}: {scope}")
    # The same figure-name problem is reported once, not once per scenario.
    return list(dict.fromkeys(problems))


def cuda_problem(device: str) -> str | None:
    """Return why ``--device cuda`` cannot be honoured here, or ``None``."""
    if device != "cuda":
        return None
    import torch

    if not torch.cuda.is_available():
        return ("--device cuda: torch reports no CUDA device (torch "
                f"{torch.__version__}); the suite does not fall back to the CPU (G5.5) -- install the CUDA "
                "wheel, or pass --device cpu")
    return None


def launch(run: PlannedRun, python: str, log_path: pathlib.Path, timeout_s: float | None = None) -> int:
    """Run one figures.py process, echoing its output to the console and to ``log_path``.

    Returns the exit code; a run stopped by ``timeout_s`` returns 1. The wait polls once a second,
    so Ctrl-C is honoured on Windows too, and ``KeyboardInterrupt`` stops the child and propagates.
    """
    env = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    deadline = None if timeout_s is None else time.monotonic() + timeout_s
    lock = threading.Lock()
    with open(log_path, "w", encoding="utf-8", newline="\n") as log:
        log.write("$ " + " ".join(run.command(python)) + "\n")
        log.flush()
        process = subprocess.Popen(run.command(python), cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                   bufsize=1)

        def pump() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                with lock:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log.write(line)

        reader = threading.Thread(target=pump, name=f"figures-{run.scenario_id}", daemon=True)
        reader.start()

        def note(message: str) -> None:
            with lock:
                sys.stdout.write(message)
                log.write(message)

        try:
            while True:
                try:
                    code = process.wait(timeout=1.0)
                    break
                except subprocess.TimeoutExpired:
                    if deadline is not None and time.monotonic() > deadline:
                        _stop(process)
                        reader.join(timeout=10)
                        note(f"stopped by the suite after {timeout_s / 60:.1f} min (--timeout-min)\n")
                        return 1
            reader.join()
            return code
        except KeyboardInterrupt:
            _stop(process)
            reader.join(timeout=10)
            note("interrupted\n")
            raise


def _stop(process: subprocess.Popen) -> None:
    """Terminate a child process, killing it if it does not exit within ten seconds."""
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def summarise(run: PlannedRun, exit_code: int, seconds: float) -> RunResult:
    """Return the result of a finished run, reading its ``manifest.json`` when it wrote one."""
    result = RunResult(run.scenario_id, exit_code, round(seconds, 1), str(run.outdir))
    path = run.outdir / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return result
    verdict = manifest.get("verdict", {})
    result.run_id = manifest.get("run_id")
    result.status = manifest.get("status")
    result.mark = verdict.get("mark", "")
    result.drawable = verdict.get("drawable")
    result.reason = verdict.get("reason", "") or ""
    result.drawn = [entry.get("name") for entry in manifest.get("figures", [])]
    result.not_drawn = [entry.get("name") for entry in manifest.get("not_drawn", [])]
    result.errors = [f"{entry.get('name')}: {entry.get('error')}" for entry in manifest.get("errors", [])]
    return result


def format_table(results: Sequence[RunResult], root: pathlib.Path) -> str:
    """Return the plain-text summary table of a suite."""
    header = ("scenario", "exit", "outcome", "status", "mark", "drawn", "seconds", "folder")
    rows = [header]
    for result in results:
        try:
            folder = str(pathlib.Path(result.outdir).relative_to(root))
        except ValueError:
            folder = result.outdir
        rows.append((result.scenario_id, str(result.exit_code), result.outcome(), result.status or "-",
                     result.mark or "-", str(len(result.drawn)), f"{result.seconds:.0f}", folder))
    widths = [max(len(row[k]) for row in rows) for k in range(len(header))]
    lines = ["  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip()
             for row in rows]
    lines.insert(1, "  ".join("-" * width for width in widths))
    notes = []
    for result in results:
        if result.reason and (result.mark or result.drawable is False):
            notes.append(f"  {result.scenario_id}: {result.reason}")
        for error in result.errors:
            notes.append(f"  {result.scenario_id}: figure error {error}")
    if notes:
        lines += ["", "notes:", *notes]
    if any(result.exit_code != 0 for result in results):
        lines += [
            "",
            "'drawn, exit 1': a gate failed outside the known-open registry (gates.tex in the run's folder); the",
            "figures carry a 'not for publication' badge. 'refused': the record is outside TRUSTED_STATUSES and",
            "--allow-invalid was not given. 'no manifest': the run stopped before drawing (see figures.log).",
        ]
    return "\n".join(lines) + "\n"


def write_summary(root: pathlib.Path, results: Sequence[RunResult], meta: dict[str, Any]) -> None:
    """Write ``suite_summary.json`` and ``suite_summary.txt`` into the suite root."""
    root.mkdir(parents=True, exist_ok=True)
    payload = {"figures_suite": meta, "runs": [dataclasses.asdict(result) for result in results]}
    with open(root / "suite_summary.json", "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    with open(root / "suite_summary.txt", "w", encoding="utf-8", newline="\n") as handle:
        handle.write(format_table(results, root))


def main(argv: Sequence[str] | None = None,
         runner: Callable[[PlannedRun, str, pathlib.Path, float | None], int] = launch) -> int:
    """Run the suite and return its exit code; ``runner`` is the launcher seam the tests replace."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        registry = load_registry(args.scenario_file)
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: cannot load {args.scenario_file}: {exc}", file=sys.stderr)
        return 2
    scenario_ids, problems = select_scenarios(registry, args.scenarios, args.exclude)
    if args.timeout_min is not None and args.timeout_min <= 0:
        problems.append("--timeout-min must be positive")
    root = pathlib.Path(args.out).resolve() if args.out else default_root()
    plan = plan_runs(args, scenario_ids, root)
    problems += validate_plan(plan, registry)
    device_issue = cuda_problem(args.device)
    if device_issue is not None and not args.dry_run:
        problems.append(device_issue)
    if problems:
        for problem in problems:
            print(f"ERROR: {problem}", file=sys.stderr)
        return 2

    import figures

    print(f"figures suite: {len(plan)} scenario(s), device {args.device}, gates {args.gates}, "
          f"figures.py {figures.FIGURES_VERSION}")
    print(f"root: {root}")
    for name, reason in DEFAULT_EXCLUDED.items():
        if name not in scenario_ids and args.scenarios is None:
            print(f"  left out: {name} ({reason})")
    if device_issue is not None:
        print(f"  note: {device_issue}")
    if args.dry_run:
        for run in plan:
            print("  " + " ".join(run.command(sys.executable)))
        return 0

    meta: dict[str, Any] = {
        "started_utc": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "root": str(root),
        "device": args.device,
        "gates": args.gates,
        "figures_py_version": figures.FIGURES_VERSION,
        "python": sys.executable,
        "argv": list(argv) if argv is not None else sys.argv[1:],
        "left_out": {name: reason for name, reason in DEFAULT_EXCLUDED.items() if name not in scenario_ids},
    }
    timeout_s = None if args.timeout_min is None else 60.0 * args.timeout_min
    results: list[RunResult] = []
    interrupted = False
    for index, run in enumerate(plan, start=1):
        print(f"\n[{index}/{len(plan)}] {run.scenario_id}", flush=True)
        run.outdir.mkdir(parents=True, exist_ok=True)
        if (run.outdir / "record.h5").exists():
            print(f"  note: {run.outdir / 'record.h5'} exists; this record is appended to it", flush=True)
        started = time.perf_counter()
        try:
            exit_code = runner(run, sys.executable, run.outdir / "figures.log", timeout_s)
        except KeyboardInterrupt:
            exit_code = INTERRUPTED
            interrupted = True
        results.append(summarise(run, exit_code, time.perf_counter() - started))
        write_summary(root, results, meta)
        if interrupted:
            break
    meta["finished_utc"] = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta["interrupted"] = interrupted
    write_summary(root, results, meta)
    print("\n" + format_table(results, root))
    print(f"summary: {root / 'suite_summary.txt'}")
    if interrupted:
        return INTERRUPTED
    return 0 if all(result.exit_code == 0 for result in results) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
