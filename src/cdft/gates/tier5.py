"""Tier 5 -- anti-silent-failure: process gates checking the record around a number, not physics."""

from __future__ import annotations

import ast
import pathlib
from dataclasses import dataclass

from contract import CONTRACT_VERSION, GateKind, GateResult, NumericsConfig, RunArtifact

from ..operators import geometry_cache
from .base import ArtifactGate, ComponentGate, GateSpec, make_result

__all__ = [
    "ProvenanceGate",
    "GateReportGate",
    "ScopeBoundaryGate",
    "DeterminismGate",
    "RecordedFallbackGate",
    "CatalogueConsistencyGate",
    "SyncBudgetGate",
    "SyncProbeBudget",
    "SYNC_BUDGETS",
    "measure_sync_budget",
]

#: Imports that would mean this package had started training. ``torch.nn`` counts: a trained
#: functional is a ``Module`` with parameters, while evaluating one needs none of this (G5.7).
_FORBIDDEN_IMPORTS = (
    "torch.optim",
    "torch.nn",
    "sklearn",
    "optax",
    "lightning",
    "pytorch_lightning",
    "keras",
    "tensorflow",
    "jax.example_libraries",
)

#: Function names that would mean a training loop had appeared.
_FORBIDDEN_FUNCTIONS = ("fit", "train", "train_step", "backward_pass", "training_step")


class ProvenanceGate(ArtifactGate):
    """G5.1 -- count of empty provenance fields; 0 missing fields (exact).

    The device block is required on every record: the CPU writes ``n/a``/``unset``, so an empty
    value always means "not captured". A completed CUDA run must also carry ``gpu_peak_bytes``. A
    dirty tree is reported in the detail rather than failed.
    """

    spec = GateSpec(
        gate_id="G5.1",
        name="Provenance completeness",
        threshold=0.0,
        kind=GateKind.EXACT,
        units="missing fields",
    )

    def applicable(self, artifact: RunArtifact) -> bool:
        """Applies to every record, including one that failed: provenance is why it is readable."""
        return True

    def evaluate(self, artifact: RunArtifact) -> GateResult:
        """Count the provenance fields that are empty."""
        prov = artifact.provenance
        missing = [
            name
            for name, value in (
                ("contract_version", prov.contract_version),
                ("config_hash", prov.config_hash),
                ("device_name", prov.device_name),
                ("hostname", prov.hostname),
                ("timestamp_utc", prov.timestamp_utc),
                ("device", prov.device),
                ("device_capability", prov.device_capability),
                ("torch_cuda_version", prov.torch_cuda_version),
                ("cuda_driver_version", prov.cuda_driver_version),
                ("deterministic_algorithms", prov.deterministic_algorithms),
                ("cublas_workspace_config", prov.cublas_workspace_config),
            )
            if not value
        ]
        if not prov.package_versions:
            missing.append("package_versions")
        if not prov.precision_policy:
            missing.append("precision_policy")
        on_cuda = str(prov.device).startswith("cuda")
        if on_cuda and artifact.result is not None and prov.gpu_peak_bytes is None:
            missing.append("gpu_peak_bytes")
        return make_result(
            self.spec,
            float(len(missing)),
            {
                "missing": missing,
                "git_sha": prov.git_sha or "unavailable",
                "git_dirty": prov.git_dirty,
                "contract_version": prov.contract_version,
                "contract_version_matches": prov.contract_version == CONTRACT_VERSION,
                "device": prov.device,
                "device_name": prov.device_name,
                "device_capability": prov.device_capability,
                "torch_cuda_version": prov.torch_cuda_version,
                "cuda_driver_version": prov.cuda_driver_version,
                "deterministic_algorithms": prov.deterministic_algorithms,
                "cublas_workspace_config": prov.cublas_workspace_config,
                "gpu_peak_bytes": prov.gpu_peak_bytes,
            },
        )


class GateReportGate(ArtifactGate):
    """G5.3 -- gate rows carrying neither a finite measurement nor a reason; 0 entries (exact).

    Evaluated last, over the report the other gates have written. ``SKIPPED`` and ``DEFERRED``
    (D-37) carry a reason instead of a measurement by definition, so they are checked for that.
    """

    spec = GateSpec(
        gate_id="G5.3",
        name="Gate report completeness",
        threshold=0.0,
        kind=GateKind.EXACT,
        units="incomplete entries",
    )

    def applicable(self, artifact: RunArtifact) -> bool:
        """Applies to every record."""
        return True

    def evaluate(self, artifact: RunArtifact) -> GateResult:
        """Count gate entries that carry neither a measurement nor a reason."""
        import math as _math

        offenders: list[str] = []
        marginal: list[str] = []
        for result in artifact.gates.results:
            if result.verdict.value in ("skipped", "deferred"):
                if not result.reason:
                    offenders.append(
                        f"{result.gate_id}: {result.verdict.value} without a reason"
                    )
            elif not _math.isfinite(result.measured):
                offenders.append(f"{result.gate_id}: non-finite measured value")
            elif result.verdict.value == "marginal":
                marginal.append(result.gate_id)
        return make_result(
            self.spec,
            float(len(offenders)),
            {"offenders": offenders, "marginal": marginal, "n_gates": len(artifact.gates.results)},
        )


class ScopeBoundaryGate(ComponentGate):
    """G5.7 -- forbidden ML imports and training functions in the solver source; 0 violations.

    The scope boundary of ``docs/01_PROJECT.md (Part B)`` 3.4 and D-24: evaluating somebody
    else's trained functional is in scope, producing one is not. The scan parses the source with
    ``ast``, so a name in a comment or string does not trip it and an aliased import does.
    """

    spec = GateSpec(
        gate_id="G5.7",
        name="Scope boundary: no machine learning",
        threshold=0.0,
        kind=GateKind.EXACT,
        citation="01_PROJECT.md (Part B) 3.4; D-24",
        units="violations",
    )

    def __init__(self, package_root: pathlib.Path | None = None) -> None:
        """Scan ``src/cdft`` by default, or an explicit directory.

        The default root also scans the solver modules at the repository top level (D-57): moving a
        file out of the package must not move it out of scope.
        """
        self.package_root = package_root or pathlib.Path(__file__).resolve().parents[1]
        self.extra_files: tuple[pathlib.Path, ...] = ()
        if package_root is None:
            repo_root = self.package_root.parents[1]
            self.extra_files = tuple(sorted(repo_root.glob("*.py")))

    def _modules(self) -> list[pathlib.Path]:
        return sorted(self.package_root.rglob("*.py")) + list(self.extra_files)

    def evaluate(self, numerics: NumericsConfig | None = None) -> GateResult:
        """Walk every module under the package root and report each violation with its location."""
        violations: list[str] = []
        scanned = 0
        for path in self._modules():
            scanned += 1
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except SyntaxError as exc:  # pragma: no cover - a parse failure is itself a violation
                violations.append(f"{path.name}: unparseable ({exc})")
                continue
            relative = (
                path.relative_to(self.package_root)
                if path.is_relative_to(self.package_root)
                else pathlib.Path("<repo>") / path.name
            )
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if _is_forbidden(alias.name):
                            violations.append(f"{relative}:{node.lineno}: imports {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if _is_forbidden(module):
                        violations.append(f"{relative}:{node.lineno}: imports from {module}")
                    for alias in node.names:
                        if _is_forbidden(f"{module}.{alias.name}"):
                            violations.append(
                                f"{relative}:{node.lineno}: imports {module}.{alias.name}"
                            )
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name in _FORBIDDEN_FUNCTIONS:
                        violations.append(f"{relative}:{node.lineno}: defines {node.name}()")
        return make_result(
            self.spec,
            float(len(violations)),
            {"violations": violations, "modules_scanned": scanned, "root": str(self.package_root)},
        )


def _is_forbidden(dotted: str) -> bool:
    """Whether a dotted import path is inside one of the forbidden namespaces."""
    return any(dotted == bad or dotted.startswith(bad + ".") for bad in _FORBIDDEN_IMPORTS)


class DeterminismGate(ArtifactGate):
    """G5.2 -- ``max |eps_A - eps_B|`` over two strict re-solves of the scenario; 1e-9 Ha.

    A determinism threshold, not an accuracy one: bit-identity is not required because BLAS
    reductions are not associative across thread counts, while a logical non-determinism lands
    orders above 1e-9. EXACT, not DERIVED: the reference is the identity and the tolerance is
    float64 reduction noise, tightening with no convergence parameter (B3). Both solves run on the
    artifact's own device with ``deterministic=True`` and the cross-check off (D-61, D-69);
    ``primary_vs_strict`` measures the kernel choice and is recorded, never thresholded. SKIPs when
    a strict solve fails.
    """

    spec = GateSpec(
        gate_id="G5.2",
        name="Determinism",
        threshold=1.0e-9,
        kind=GateKind.EXACT,
        citation="G5.2; D13; D-61",
        units="Ha",
        solves=2,
    )

    def applicable(self, artifact: RunArtifact) -> bool:
        """Applies to any completed run."""
        return artifact.result is not None

    def evaluate(self, artifact: RunArtifact) -> GateResult:
        """Solve the artifact's configuration twice in strict mode and compare every eigenvalue."""
        import dataclasses
        import time

        import torch

        from ..precision import configure_device, release_device_memory
        from ..operators import geometry_cache
        from ..scf.solve import solve_scenario
        from .base import artifact_device, skipped

        device = artifact_device(artifact)
        primary = torch.as_tensor(artifact.result.eigenvalues[0], dtype=torch.float64).cpu()
        n_states = int(primary.shape[-1])
        numerics = dataclasses.replace(
            artifact.numerics,
            eigen=dataclasses.replace(artifact.numerics.eigen, cross_check=False),
            deterministic=True,
        )

        runs: list[tuple[torch.Tensor, str, str, float]] = []
        try:
            for label in ("A", "B"):
                started = time.perf_counter()
                # Through the one door (D-53): re-solving an interacting scenario on the bare path
                # would compare two different problems and report their gap as non-determinism.
                # Cache bypassed (D-66): the claim covers the geometry build as well as the solve,
                # so neither run may reuse a cached bundle.
                with geometry_cache.bypass():
                    repeat = solve_scenario(artifact.scenario, numerics, n_states=n_states, device=device)
                wall = time.perf_counter() - started
                if repeat.result is None:
                    return skipped(
                        self.spec, f"strict solve {label} produced no result: {repeat.error_message}"
                    )
                runs.append(
                    (
                        torch.as_tensor(repeat.result.eigenvalues[0], dtype=torch.float64).cpu().clone(),
                        str(getattr(repeat.provenance, "deterministic_algorithms", "")),
                        str(getattr(repeat.provenance, "device", "")),
                        wall,
                    )
                )
                del repeat
                release_device_memory(device)
        except Exception as exc:  # pragma: no cover - a failed strict solve is reported, not hidden
            return skipped(self.spec, f"a strict solve failed: {type(exc).__name__}: {exc}")
        finally:
            # The pair switched strict mode on; restore the artifact's own mode.
            configure_device(device, bool(artifact.numerics.deterministic))

        (first, first_mode, first_device, first_wall), (second, repeat_mode, repeat_device, second_wall) = runs
        n = min(n_states, int(first.shape[-1]), int(second.shape[-1]))
        delta = (first[:n] - second[:n]).abs()
        primary_delta = (primary[:n] - first[:n]).abs()
        gpu_leg = device.type == "cuda"
        primary_mode = str(getattr(artifact.provenance, "deterministic_algorithms", ""))
        return make_result(
            self.spec,
            float(delta.max()),
            {
                "first_run": first[:n].tolist(),
                "repeat_run": second[:n].tolist(),
                "primary_run": primary[:n].tolist(),
                "per_state_difference": delta.tolist(),
                "primary_vs_strict": float(primary_delta.max()),
                "primary_vs_strict_per_state": primary_delta.tolist(),
                "seed": artifact.numerics.seed,
                "device": str(device),
                "repeat_device": repeat_device,
                "first_device": first_device,
                "threads": int(torch.get_num_threads()),
                "gpu_leg_exercised": gpu_leg,
                "solves": 2,
                "deterministic_algorithms": {
                    "primary": primary_mode,
                    "first_run": first_mode,
                    "repeat_run": repeat_mode,
                },
                "deterministic_requested": {
                    "primary": bool(artifact.numerics.deterministic),
                    "first_run": True,
                    "repeat_run": True,
                },
                "modes_differ": first_mode != repeat_mode,
                "primary_mode_differs": primary_mode != first_mode,
                "bitwise_identical": bool((delta == 0.0).all()),
                "strict_pair_wall_s": [round(first_wall, 3), round(second_wall, 3)],
                "primary_wall_s": float(getattr(artifact.provenance, "wall_time_s", 0.0) or 0.0),
                "note": (
                    ("GPU leg: all three solves on the artifact's CUDA device. " if gpu_leg else "CPU leg. ")
                    + "Verdict: strict solve A against strict solve B (D-61). primary_vs_strict "
                    "compares the artifact's own run (deterministic "
                    f"{primary_mode!r}) with A and is recorded, never thresholded. Bit-identity is "
                    "not required: BLAS reductions are not associative across thread counts, which "
                    "is hardware and not a defect"
                ),
            },
        )


class RecordedFallbackGate(ArtifactGate):
    """G5.5 -- settings the run resolved differently from the request with no recorded reason; 0.

    Covers the grid derivation (D-42), the physics-determined overrides (D-33) and an eigensolver
    that stops on its iteration limit: each must leave a readable sentence in
    ``measurements["grid_overrides"]``. The presence of the dictionary is not the check.
    """

    spec = GateSpec(
        gate_id="G5.5",
        name="No default-silent fallbacks",
        threshold=0.0,
        kind=GateKind.EXACT,
        citation="G5.5; D33; D42",
        units="unexplained changes",
    )

    def applicable(self, artifact: RunArtifact) -> bool:
        """Applies to any completed run."""
        return artifact.result is not None

    def evaluate(self, artifact: RunArtifact) -> GateResult:
        """Compare the resolved grid against the requested one and demand a reason for each change."""
        requested = artifact.numerics.grid
        overrides = dict(artifact.measurements.get("grid_overrides") or {})
        described = dict(artifact.measurements.get("grid") or {})

        unexplained: list[str] = []
        checks: dict[str, object] = {}

        resolved_spacing = described.get("spacing_bohr", described.get("spacing"))
        if resolved_spacing is not None:
            changed = abs(float(resolved_spacing) - requested.spacing) > 1e-12
            checks["spacing"] = {
                "requested": requested.spacing,
                "resolved": float(resolved_spacing),
                "changed": changed,
                "explained": "spacing" in overrides,
            }
            if changed and "spacing" not in overrides:
                unexplained.append("spacing")

        # The grid describes itself by shape, spacing and origin, not by edge length; reading a key
        # the description does not have would make this check silently inert.
        shape = described.get("shape")
        resolved_lengths = (
            tuple((int(n) - 1) * float(resolved_spacing) for n in shape)
            if shape is not None and resolved_spacing is not None
            else None
        )
        if resolved_lengths is not None and requested.box_lengths is not None:
            # One part in 1e6 of an edge, not 1e-9 absolute: the box is snapped so a nucleus lands
            # on a grid point, and that rounding of L/h is not a fallback to report.
            changed = any(
                abs(float(a) - float(b)) > 1e-6 * max(abs(float(b)), 1.0)
                for a, b in zip(resolved_lengths, requested.box_lengths)
            )
            checks["box_lengths"] = {
                "requested": list(requested.box_lengths),
                "resolved": [float(x) for x in resolved_lengths],
                "changed": changed,
                "explained": "box_lengths" in overrides,
            }
            if changed and "box_lengths" not in overrides:
                unexplained.append("box_lengths")

        stop_reason = str(artifact.measurements.get("eigen_stop_reason", "unrecorded"))
        checks["eigen_stop_reason"] = stop_reason
        if stop_reason == "unrecorded":
            unexplained.append("eigen_stop_reason")

        iterations = artifact.measurements.get("eigen_iterations")
        limit = artifact.numerics.eigen.max_iterations
        hit_limit = iterations is not None and int(float(iterations)) >= int(limit)
        checks["eigensolver"] = {
            "iterations": iterations,
            "max_iterations": limit,
            "stopped_on_limit": hit_limit,
            "explained": artifact.status.value != "valid" if hit_limit else True,
        }
        if hit_limit and artifact.status.value == "valid":
            # Stopping on the iteration limit while still reporting VALID changed the answer's
            # meaning without changing its label.
            unexplained.append("eigensolver_iteration_limit")

        empty_but_changed = bool(unexplained) and not overrides
        return make_result(
            self.spec,
            float(len(unexplained)),
            {
                "unexplained_changes": unexplained,
                "recorded_overrides": sorted(overrides),
                "override_reasons": overrides,
                "checks": checks,
                "overrides_empty_while_grid_changed": empty_but_changed,
                "note": (
                    "the presence of an overrides dictionary is not the check -- an empty "
                    "dictionary beside a rewritten grid passes a presence check and fails this one"
                ),
            },
        )


class CatalogueConsistencyGate(ComponentGate):
    """G5.8 -- disagreements between the catalogue, the wired evaluators and scenarios; 0 (D-40).

    Counted: a gate marked ``IMPLEMENTED`` with no evaluator, an evaluator for a gate marked
    ``DEFERRED`` or ``OUT_OF_PHASE_1``, an evaluator or a scenario naming a gate absent from the
    catalogue, a ``DEFERRED`` entry with no owner (D-37), and an :data:`ENFORCED_ELSEWHERE` location
    that no longer resolves -- otherwise that list becomes a place to hide an unimplemented gate.
    """

    spec = GateSpec(
        gate_id="G5.8",
        name="Gate catalogue consistency",
        threshold=0.0,
        kind=GateKind.EXACT,
        citation="D40; D37",
        units="disagreements",
    )

    #: Gates implemented somewhere other than the evaluator tuples, and where. A ``None`` location
    #: is not importable from the installed package, so only the entry itself is checked.
    ENFORCED_ELSEWHERE: dict[str, tuple[str | None, str]] = {
        "G5.3": (
            "cdft.gates.tier5:GateReportGate",
            "appended by GateSuite.run_on_artifact after the other gates have written their rows, "
            "because it audits that report and so cannot be one of the rows it audits",
        ),
        "G5.4": (
            "cdft.io.hdf5:CorpusWriter",
            "a property of CorpusWriter.run_ids and .iterate: both filter on "
            "contract.TRUSTED_STATUSES by default, and reading anything else requires passing "
            "include_invalid=True explicitly",
        ),
        "G5.6": (
            None,
            "the exit code of test_suite.py, which lives at the repository root and is therefore "
            "not importable from the installed package; covered by the suite's own contract tests",
        ),
    }

    def evaluate(self, numerics: NumericsConfig) -> GateResult:
        """Cross-check the catalogue against the wired evaluators and the scenario registry."""
        import importlib

        from ..gates.catalogue import CATALOGUE, Lifecycle
        from ..gates.runner import default_artifact_gates, default_component_gates
        from ..physics_config import REGISTRY

        entries = {entry.gate_id: entry for entry in CATALOGUE}
        wired = {
            gate.spec.gate_id
            for gate in (*default_artifact_gates(), *default_component_gates())
        }

        broken_locations: list[str] = []
        for gate_id, (location, _why) in self.ENFORCED_ELSEWHERE.items():
            if location is None:
                continue
            module_name, _, attribute = location.partition(":")
            try:
                module = importlib.import_module(module_name)
            except ImportError as exc:
                broken_locations.append(f"{gate_id}: cannot import {module_name} ({exc})")
                continue
            if attribute and not hasattr(module, attribute):
                broken_locations.append(f"{gate_id}: {module_name} has no {attribute}")

        accounted = wired | set(self.ENFORCED_ELSEWHERE)
        claimed_not_wired = sorted(
            gid
            for gid, entry in entries.items()
            if entry.lifecycle is Lifecycle.IMPLEMENTED and gid not in accounted
        )
        wired_not_claimed = sorted(
            gid
            for gid in wired
            if gid in entries and entries[gid].lifecycle is not Lifecycle.IMPLEMENTED
        )
        wired_unknown = sorted(gid for gid in wired if gid not in entries)
        scenario_unknown = sorted(
            {gid for scenario in REGISTRY for gid in scenario.gates} - set(entries)
        )
        deferred_without_owner = sorted(
            gid
            for gid, entry in entries.items()
            if entry.lifecycle is Lifecycle.DEFERRED and not entry.owner
        )

        disagreements = (
            len(claimed_not_wired)
            + len(wired_not_claimed)
            + len(wired_unknown)
            + len(scenario_unknown)
            + len(deferred_without_owner)
            + len(broken_locations)
        )
        return make_result(
            self.spec,
            float(disagreements),
            {
                "claimed_implemented_but_not_wired": claimed_not_wired,
                "wired_but_not_marked_implemented": wired_not_claimed,
                "wired_but_absent_from_catalogue": wired_unknown,
                "named_by_scenario_but_absent_from_catalogue": scenario_unknown,
                "deferred_without_owner": deferred_without_owner,
                "enforced_elsewhere": {
                    gate_id: location for gate_id, (location, _) in self.ENFORCED_ELSEWHERE.items()
                },
                "enforcement_locations_broken": broken_locations,
                "catalogue_size": len(entries),
                "wired_evaluators": len(wired),
                "note": (
                    "a gate in neither the implemented set nor the deferral table is a hard FAIL "
                    "elsewhere, so forgetting to implement something cannot be laundered into a "
                    "deferral by forgetting twice"
                ),
            },
        )


# G5.9 -- the sync/launch budget


@dataclass(frozen=True, slots=True)
class SyncProbeBudget:
    """The recorded census of one G5.9 probe solve, and the budget derived from it.

    ``reads_per_iteration`` and ``implicit_per_iteration`` are the loop sites' Python reads and
    implicit-sync operators per iteration (``scripts/sync_census.py``); ``launches`` counts the
    kernel-launching operators of the whole solve.
    """

    probe: str
    iterations: int
    reads_per_iteration: float
    implicit_per_iteration: float
    launches: int
    measured_on: str

    def budget(self, quantity: str) -> float:
        """Return the recorded ``quantity`` plus headroom, capped for the per-iteration counts."""
        recorded = float(getattr(self, quantity))
        headroom = BUDGET_HEADROOM * recorded
        if quantity != "launches":
            headroom = min(headroom, MAX_ITERATION_HEADROOM)
        return recorded + headroom


#: Relative headroom of every G5.9 budget over its recorded value.
BUDGET_HEADROOM = 0.10

#: Largest headroom on a per-iteration sync count: below one, so one new read per iteration fails.
MAX_ITERATION_HEADROOM = 0.5

#: Recorded census of the two G5.9 probes (:func:`measure_sync_budget`, CPU float64;
#: device-independent because the census counts dispatcher calls). Re-measure and record the reason
#: after any change that moves an iteration or apply count; never raise a budget to pass.
SYNC_BUDGETS: dict[str, SyncProbeBudget] = {
    "h_atom": SyncProbeBudget(
        probe="h_atom",
        iterations=7,
        reads_per_iteration=9 / 7,
        implicit_per_iteration=16 / 7,
        launches=34841,
        measured_on="cpu, 2026-09-16",
    ),
    "he_scf_probe": SyncProbeBudget(
        probe="he_scf_probe",
        iterations=12,
        reads_per_iteration=80 / 12,
        implicit_per_iteration=118 / 12,
        launches=127429,
        measured_on="cpu, 2026-09-16",
    ),
}

#: Grid of the self-consistent probe: coarse on purpose (it measures the code path, not physics).
HE_PROBE_SPACING = 0.5
HE_PROBE_BOX = 8.0


def _load_census_module():
    """Load ``scripts/sync_census.py`` by path (the scripts directory is not a package)."""
    import importlib.util
    import sys

    path = pathlib.Path(__file__).resolve().parents[3] / "scripts" / "sync_census.py"
    if not path.exists():
        return None, path
    name = "cdft_scripts_sync_census"
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module, path


def _probes():
    """Return the two probe solves as ``(name, scenario, numerics, derive_grid)``.

    ``h_atom`` is the registered scenario at the all-electron preset; ``he_scf_probe`` is neutral
    helium at VWN LDA on a deliberately coarse grid. A component gate measures the code path, so it
    chooses its own numerics and no physics number is read from it.
    """
    import dataclasses

    from contract import Device

    from ..config import ALL_ELECTRON_NUMERICS
    from ..physics_config import REGISTRY, helium, interacting, lda_functional

    base = dataclasses.replace(
        ALL_ELECTRON_NUMERICS,
        eigen=dataclasses.replace(ALL_ELECTRON_NUMERICS.eigen, cross_check=False),
        device=Device.CPU,
        deterministic=False,
    )
    probe = interacting(helium(), lda_functional("vwn"), "g59_he_scf_probe")
    coarse = dataclasses.replace(
        base,
        grid=dataclasses.replace(
            base.grid, spacing=HE_PROBE_SPACING, box_lengths=(HE_PROBE_BOX,) * 3
        ),
    )
    return (
        ("h_atom", REGISTRY["h_atom"], base, True),
        ("he_scf_probe", probe, coarse, False),
    )


def measure_sync_budget() -> dict[str, dict[str, object]]:
    """Run the G5.9 census on both probes (CPU) and return the numbers :data:`SYNC_BUDGETS` records.

    Raises ``FileNotFoundError`` when ``scripts/sync_census.py`` is absent.
    """
    import cdft

    module, path = _load_census_module()
    if module is None:
        raise FileNotFoundError(f"the census script is not present at {path}")
    root = pathlib.Path(cdft.__file__).resolve().parents[2]
    out: dict[str, dict[str, object]] = {}
    for name, scenario, numerics, derive in _probes():
        # Cache bypassed: a probe's launch count must not depend on whether an earlier solve in the
        # process left its geometry bundle behind.
        with geometry_cache.bypass():
            record = module.census_solve(scenario, numerics, "cpu", derive_grid=derive, root=root)
        loop = record["loop"]
        iterations = int(loop["iterations"])
        reads = sum(s["count"] for s in loop["loop_sites"] if not s["site"].endswith("]"))
        implicit = int(loop["loop_syncs_total"]) - reads
        out[name] = {
            "status": record["run"]["status"],
            "iterations": iterations,
            "iterations_key": loop["iterations_key"],
            "loop_reads": reads,
            "loop_implicit": implicit,
            "reads_per_iteration": reads / iterations if iterations else float("inf"),
            "implicit_per_iteration": implicit / iterations if iterations else float("inf"),
            "launches": int(record["census"]["launch_ops_total"]),
            "setup_syncs": int(loop["setup_syncs_total"]),
            "syncs_total": int(record["census"]["syncs_total"]),
            "loop_sites": loop["loop_sites"],
        }
    return out


class SyncBudgetGate(ComponentGate):
    """G5.9 -- census of two probe solves against :data:`SYNC_BUDGETS`; 0 budgets exceeded (D-60).

    Compares loop-site reads and implicit syncs per iteration and launching operators per solve, the
    gains a single ``.item()`` back in a loop would silently undo. Always on the CPU: the census
    counts dispatcher calls, and a CPU census cannot be perturbed by CUDA-graph capture. A probe
    that fails to solve FAILs; a missing census script SKIPs.
    """

    spec = GateSpec(
        gate_id="G5.9",
        name="Sync/launch budget",
        threshold=0.0,
        kind=GateKind.EXACT,
        citation="D-60; D-70",
        units="budgets exceeded",
    )

    def evaluate(self, numerics: NumericsConfig | None = None) -> GateResult:
        """Measure both probes on the CPU and count the budgets they exceed."""
        from .base import skipped

        try:
            measured = measure_sync_budget()
        except FileNotFoundError as exc:
            return skipped(self.spec, f"{exc}; the gate needs the repository's scripts directory")
        exceeded: list[str] = []
        probes: dict[str, object] = {}
        for name, record in measured.items():
            budget = SYNC_BUDGETS[name]
            rows = {}
            if record["status"] in ("error", "unconverged"):
                exceeded.append(f"{name}: probe solve ended {record['status']}")
            for quantity in ("reads_per_iteration", "implicit_per_iteration", "launches"):
                value = float(record[quantity])  # type: ignore[arg-type]
                limit = budget.budget(quantity)
                rows[quantity] = {
                    "measured": value,
                    "recorded": float(getattr(budget, quantity)),
                    "budget": limit,
                    "within": value <= limit,
                }
                if not value <= limit:
                    exceeded.append(f"{name}: {quantity} {value:.4g} > budget {limit:.4g}")
            probes[name] = {
                **rows,
                "iterations": record["iterations"],
                "iterations_key": record["iterations_key"],
                "recorded_iterations": budget.iterations,
                "setup_syncs": record["setup_syncs"],
                "syncs_total": record["syncs_total"],
                "loop_sites": record["loop_sites"],
                "status": record["status"],
            }
        return make_result(
            self.spec,
            float(len(exceeded)),
            {
                "exceeded": exceeded,
                "probes": probes,
                "census_device": "cpu",
                "headroom": {"relative": BUDGET_HEADROOM, "per_iteration_cap": MAX_ITERATION_HEADROOM},
                "note": (
                    "loop sites fire at least once per iteration; a new .item() in a loop adds at "
                    "least one read per iteration and exceeds the capped headroom. Re-measure with "
                    "cdft.gates.tier5.measure_sync_budget() after a change that moves an iteration "
                    "or apply count, and record the reason; never raise a budget to pass"
                ),
            },
        )
