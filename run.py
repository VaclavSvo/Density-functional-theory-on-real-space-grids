"""Command-line entry point: ``python run.py`` at the repository root, ``python -m cdft.run`` or ``cdft``.

Runs scenarios against a numerics configuration, evaluates the gates, writes the HDF5 corpus, prints
the tables, and **exits non-zero if anything failed** (gate G5.6) -- which is what makes an
unattended run supervisable.
"""

from __future__ import annotations

# Path bootstrap: this file is canonical at the repository root next to ``contract.py`` (D-57), so
# the three "what and how" modules import without the package being installed.
import pathlib as _pathlib
import sys as _sys

_REPO_ROOT = _pathlib.Path(__file__).resolve().parent
for _entry in (_REPO_ROOT, _REPO_ROOT / "src"):
    if _entry.is_dir() and str(_entry) not in _sys.path:
        _sys.path.insert(0, str(_entry))
del _entry

import argparse
import dataclasses
import pathlib
import sys
from collections.abc import Callable

from contract import Device, NumericsConfig, RunStatus

from config import (
    LEVELS,
    MODEL_SYSTEM_NUMERICS,
    PRESETS,
    describe_setups,
    load_numerics,
    numerics_for,
    numerics_for_spec,
)
from cdft.gates.base import artifact_device
from cdft.gates.known_open import COMPONENT, partition_failures
from cdft.grid import grid_rule
from cdft.precision import release_device_memory
from cdft.gates.runner import GateSuite, format_report, states_required
from cdft.reference.oracles import ENV_ORACLES, enable_oracles, oracles_mode
from cdft.io.hdf5 import CorpusWriter
from physics_config import REGISTRY, load_scenarios
from cdft.scf.solve import solve_scenario

__all__ = ["main", "run_scenarios"]

#: Per-field resolution overrides: the ``config.Resolution`` fields as command-line options.
_OVERRIDES: tuple[tuple[str, type, str], ...] = (
    ("spacing", float, "bohr; on a Coulomb system the derivation target, not the grid itself"),
    ("half_box", float, "bohr of vacuum beyond the outermost nucleus (Coulomb systems)"),
    ("box", float, "bohr, full box edge (model systems)"),
    ("points_per_edge", int, "odd count along each edge (single bare nucleus, D-42)"),
    ("fd_order", int, "finite-difference order of the Laplacian"),
    ("eigen_tol", float, "eigensolver residual tolerance"),
    ("chebyshev_degree", int, "CheFSI filter degree"),
    ("scf_energy_tol", float, "SCF energy tolerance in Hartree"),
    ("scf_density_tol", float, "SCF density tolerance"),
    ("max_scf_iterations", int, "SCF iteration cap"),
)


def _announcing(
    plans: dict[str, tuple[NumericsConfig, dict[str, float]]], level: str
) -> Callable[[str], tuple[NumericsConfig, dict[str, float]]]:
    """Wrap resolved plans so each scenario announces the non-production level it runs at."""

    def announce(scenario_id: str) -> tuple[NumericsConfig, dict[str, float]]:
        numerics, targets = plans[scenario_id]
        detail = ", ".join(f"{name}={value:g}" for name, value in targets.items())
        if not detail:  # a model system carries its resolution in the numerics, not in the rule
            detail = f"spacing={numerics.grid.spacing:g}"
        print(
            f"resolution {level!r} ({detail}): exploratory -- not comparable with golden values "
            f"or the known-open registry",
            flush=True,
        )
        return numerics, targets

    return announce


def oracles_line(mode: str) -> str:
    """The one line every report starts with: whether the PySCF/libxc oracles were on (O-24)."""
    if mode == "on":
        return f"oracles: on ({ENV_ORACLES}=1 or --oracles): G0.6 and the libxc tests are evaluated"
    return (
        f"oracles: off (default): G0.6 and the libxc tests are off the plan and the record; "
        f"--oracles or {ENV_ORACLES}=1 evaluates them"
    )


def run_scenarios(
    scenario_ids: list[str],
    numerics: NumericsConfig,
    corpus: pathlib.Path | None = None,
    registry=REGISTRY,
    components: bool = True,
    numerics_for_scenario: Callable[[str], tuple[NumericsConfig, dict[str, float]]] | None = None,
) -> tuple[int, str]:
    """Run scenarios and their gates, returning ``(exit_code, report_text)``.

    Separate from :func:`main` so the increment logs and the test suite call exactly what the command
    line calls, rather than something that resembles it.

    ``numerics`` is what the component gates run at, and what every scenario runs at unless
    ``numerics_for_scenario`` gives one its own ``(numerics, grid rule targets)``
    (``config.numerics_for``). Solve and gates then run inside :func:`~cdft.grid.grid_rule`, so the
    gates that re-solve see the same rule.
    """
    suite = GateSuite()
    sections: list[str] = [oracles_line(suite.oracles_mode)]
    failures = 0
    known_rows: list[str] = []

    if components:
        component_results = suite.run_components(numerics)
        sections.append(format_report(component_results, "Component gates"))
        split = partition_failures(COMPONENT, component_results)
        failures += split.new
        known_rows.extend(split.rows)

    writer = CorpusWriter(corpus) if corpus else None
    for scenario_id in scenario_ids:
        scenario = registry[scenario_id]
        scenario_numerics, targets = (
            (numerics, {}) if numerics_for_scenario is None else numerics_for_scenario(scenario_id)
        )
        with grid_rule(**targets):
            required = states_required(scenario, suite.artifact_gates)
            artifact = solve_scenario(scenario, scenario_numerics, n_states=required or None)
            artifact = suite.run_on_artifact(artifact)
        if writer is not None:
            writer.write(artifact)
        title = f"{scenario_id}  [status={artifact.status.value}]"
        if artifact.error_message:
            title += f"  error={artifact.error_message}"
        sections.append(format_report(artifact.gates.results, title))

        # A record is INVALID for three different reasons and only one is a defect; treating INVALID
        # alone as the exit condition once reported FAILED on runs where every gate passed (D-45).
        split = partition_failures(scenario_id, artifact.gates.results)
        known_rows.extend(split.rows)
        if split.new or artifact.status in (RunStatus.ERROR, RunStatus.UNCONVERGED):
            failures += 1
        device = artifact_device(artifact)
        del artifact
        release_device_memory(device)  # the next scenario starts from an empty CUDA cache

    if known_rows:
        sections.append(
            "known open failures, see docs/02_STATUS.md (Part B):\n"
            + "\n".join(known_rows)
            + "\n  These keep their FAIL verdict and their records stay INVALID. What they do not\n"
            "  do is set the exit code, for exactly as long as they do not get worse."
        )

    return (1 if failures else 0), "\n\n".join(sections)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run, print the report, and return the process exit code."""
    parser = argparse.ArgumentParser(
        prog="cdft",
        description="Classical DFT solver: run scenarios, evaluate physics gates, write a corpus.",
    )
    parser.add_argument(
        "scenarios",
        nargs="*",
        help="scenario identifiers to run; default is every registered scenario",
    )
    parser.add_argument(
        "--numerics",
        default="auto",
        help="auto (each scenario's setup: its base preset at --res), a YAML numerics file, or one "
        "of the presets: " + ", ".join(sorted(PRESETS)) + ". A preset or file also fixes the "
        "spacing the component gates run at; all-electron's placeholder h = 1 fails G0.2 by "
        "construction (config.ALL_ELECTRON_NUMERICS), auto runs them at the model preset",
    )
    parser.add_argument(
        "--res",
        choices=LEVELS,
        default="standard",
        help="resolution level from the scenario's setup; standard is the production rule and the "
        "only level comparable with the golden values (D-73)",
    )
    parser.add_argument("--scenario-file", default=None, help="YAML file of scenario definitions")
    parser.add_argument("--corpus", default=None, help="HDF5 file to append records to")
    parser.add_argument(
        "--no-components",
        action="store_true",
        help="skip the component gates (Laplacian order, Poisson, hydrogenic ladder)",
    )
    parser.add_argument(
        "--oracles",
        action="store_true",
        help=f"evaluate the PySCF/libxc oracle gates (G0.6) too; same as {ENV_ORACLES}=1. "
        "Off by default, and the report and every record say which (O-24)",
    )
    parser.add_argument(
        "--list", action="store_true", help="list the registered scenarios and exit"
    )
    parser.add_argument(
        "--setups", action="store_true", help="print the per-system setups table and exit"
    )
    parser.add_argument(
        "--device",
        choices=[d.value for d in Device],
        default=None,
        help="override the numerics device (cpu, cuda, auto); cuda without a CUDA device is an "
        "error, never a fallback (G5.5). Default: whatever the numerics preset or file says",
    )
    group = parser.add_argument_group(
        "resolution overrides", "one field each, refused for a system the field means nothing for"
    )
    for name, kind, help_text in _OVERRIDES:
        group.add_argument(f"--{name.replace('_', '-')}", type=kind, default=None, help=help_text)
    args = parser.parse_args(argv)

    if args.setups:
        print(describe_setups())
        return 0

    if args.oracles:
        enable_oracles()  # before any GateSuite is built: the suite reads the switch once
    try:
        oracles_mode()
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    registry = load_scenarios(args.scenario_file) if args.scenario_file else REGISTRY
    if args.list:
        for scenario in registry:
            flag = "all-electron" if scenario.all_electron else "pseudopotential"
            print(f"{scenario.scenario_id:<16} {scenario.external.kind.value:<18} {flag}")
        return 0

    overrides = {
        name: getattr(args, name) for name, _, _ in _OVERRIDES if getattr(args, name) is not None
    }
    if args.numerics != "auto" and (args.res != "standard" or overrides):
        print(
            "ERROR: --numerics names a preset or a file, which fixes every numerics value; drop it "
            "to use --res or a field override, rather than mixing the two silently (D-73).",
            file=sys.stderr,
        )
        return 2
    if args.scenario_file is not None and args.res != "standard":
        print(
            "ERROR: scenarios from --scenario-file have no setup, so --res is undefined for "
            "them; the field overrides still apply.",
            file=sys.stderr,
        )
        return 2

    device = None
    if args.device is not None:
        # Not a physics setting (the CPU and CUDA paths agree to 1e-10), so overriding it does not
        # change what a preset means. Resolved now, so a missing CUDA device stops the run before
        # any work rather than inside the first solve.
        from cdft.precision import resolve_device

        device = Device(args.device)
        try:
            resolved = resolve_device(device)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        from cdft.device_policy import apply_policy, describe

        apply_policy(resolved)  # once per process: the card's profile and what follows from it
        print(f"device: {resolved}", flush=True)
        print(describe(resolved), flush=True)

    scenario_ids = args.scenarios or list(registry.ids())
    corpus = pathlib.Path(args.corpus) if args.corpus else None
    plan = None

    if args.numerics != "auto":
        numerics = (
            PRESETS[args.numerics] if args.numerics in PRESETS else load_numerics(args.numerics)
        )
        if device is not None:
            numerics = dataclasses.replace(numerics, device=device)
    else:
        # The component gates are scenario-independent, so they keep the model preset whatever the
        # scenarios run at; every scenario gets its own base preset and level instead.
        numerics = MODEL_SYSTEM_NUMERICS
        if device is not None:
            numerics = dataclasses.replace(numerics, device=device)
        try:
            plans = {
                scenario_id: (
                    numerics_for(scenario_id, args.res, device=device, **overrides)
                    if registry is REGISTRY
                    else numerics_for_spec(registry[scenario_id], device=device, **overrides)
                )
                for scenario_id in scenario_ids
            }
        except (KeyError, ValueError) as exc:
            print(f"ERROR: {exc.args[0] if exc.args else exc}", file=sys.stderr)
            return 2
        production = args.res == "standard" and not overrides
        plan = plans.__getitem__ if production else _announcing(plans, args.res)

    code, report = run_scenarios(
        scenario_ids,
        numerics,
        corpus=corpus,
        registry=registry,
        components=not args.no_components,
        numerics_for_scenario=plan,
    )
    print(report)
    if code:
        print(
            "\nFAILED: at least one gate failed or one run did not converge. "
            "Records are written with status INVALID and excluded by the default corpus reader.",
            file=sys.stderr,
        )
    return code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
