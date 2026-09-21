"""The gate runner: evaluates the suite, attaches the report, and sets the record's status.

Gates run at solve time and travel with the data (D-13). A gate a scenario names and the suite does
not evaluate is a failure, not an omission. Gates that call an oracle (``GateSpec.requires_oracle``)
are off the plan and the record unless the oracles are requested; the record says which (O-24).
"""

from __future__ import annotations

from contract import (
    GateKind,
    GateReport,
    GateResult,
    GateVerdict,
    NumericsConfig,
    RunArtifact,
    RunStatus,
)

from ..precision import release_device_memory
from ..reference.oracles import oracles_enabled
from .base import ArtifactGate, ComponentGate, GateSpec, artifact_device, deferred, skipped
from .catalogue import by_id, deferral_owner, specified_ids
from .functional import (
    ExchangeScalingGate,
    FunctionalDerivativeGate,
    LibxcAgreementGate,
    UniformGasGate,
)
from .tier0 import (
    FiniteValuesGate,
    GradientOrderGate,
    HermiticityGate,
    LaplacianOrderGate,
    OrthonormalityGate,
    PoissonAnalyticGate,
    WeightedSelfAdjointnessGate,
)
from .tier1 import (
    ChargeNormalisationGate,
    GridIsotropyGate,
    HarmonicOscillatorGate,
    CuspWeightQuadratureGate,
    KatoCuspGate,
    NonInteractingLimitGate,
    KineticScalingGate,
    LiebOxfordGate,
    ParticleInBoxGate,
    RadialReferenceGate,
    TwoCentreReferenceGate,
    VirialTheoremGate,
)
from .tier2 import (
    EggBoxGate,
    EigenvalueResidualGate,
    EnergyClosureGate,
    HarrisFoulkesGate,
    JanakGate,
    TwoSolverAgreementGate,
    VariationalTailGate,
)
from .tier3 import (
    DomainSizeConvergenceGate,
    GridSpacingConvergenceGate,
    InitialGuessGate,
    SCFConvergenceGate,
    StencilOrderGate,
)
from .tier4 import PublishedValueGate, SelfInteractionGate
from .tier5 import (
    CatalogueConsistencyGate,
    DeterminismGate,
    GateReportGate,
    ProvenanceGate,
    RecordedFallbackGate,
    ScopeBoundaryGate,
    SyncBudgetGate,
)

__all__ = [
    "GateSuite",
    "UNEVALUABLE_GATE_NAME",
    "default_artifact_gates",
    "default_component_gates",
    "format_report",
    "planned",
    "states_required",
]

#: Result name of a FAIL for a gate a scenario names that this build cannot evaluate; the one
#: string ``known_open.partition_failures`` matches, so ``unimplemented`` is reachable (O-26).
UNEVALUABLE_GATE_NAME = "gate named by scenario cannot be evaluated"


def states_required(scenario, gates: tuple[ArtifactGate, ...] | None = None) -> int:
    """Return the number of eigenstates the gates a scenario names require it to converge.

    Zero when no named gate needs a spectrum; the caller then falls back to the electron count.
    """
    available = {gate.gate_id: gate for gate in (gates or default_artifact_gates())}
    return max(
        (available[gate_id].spec.min_states for gate_id in scenario.gates if gate_id in available),
        default=0,
    )


def default_artifact_gates() -> tuple[ArtifactGate, ...]:
    """Return the gates evaluated on every completed run."""
    return (
        OrthonormalityGate(),
        HermiticityGate(),
        WeightedSelfAdjointnessGate(),
        FiniteValuesGate(),
        HarmonicOscillatorGate(),
        ParticleInBoxGate(),
        GridIsotropyGate(),
        ChargeNormalisationGate(),
        CuspWeightQuadratureGate(),
        KatoCuspGate(),
        RadialReferenceGate(),
        TwoCentreReferenceGate(),
        KineticScalingGate(),
        VirialTheoremGate(),
        LiebOxfordGate(),
        EigenvalueResidualGate(),
        TwoSolverAgreementGate(),
        EnergyClosureGate(),
        HarrisFoulkesGate(),
        JanakGate(),
        VariationalTailGate(),
        SCFConvergenceGate(),
        InitialGuessGate(),
        PublishedValueGate(),
        SelfInteractionGate(),
        EggBoxGate(),
        GridSpacingConvergenceGate(),
        DomainSizeConvergenceGate(),
        StencilOrderGate(),
        DeterminismGate(),
        RecordedFallbackGate(),
        ProvenanceGate(),
    )


def default_component_gates() -> tuple[ComponentGate, ...]:
    """Return the gates that verify operators rather than runs."""
    return (
        LaplacianOrderGate(),
        GradientOrderGate(),
        PoissonAnalyticGate(),
        FunctionalDerivativeGate(),
        LibxcAgreementGate(),
        UniformGasGate(),
        ExchangeScalingGate(),
        NonInteractingLimitGate(),
        ScopeBoundaryGate(),
        CatalogueConsistencyGate(),
        SyncBudgetGate(),
    )


def planned(gates, oracles: bool):
    """Return ``gates`` without the oracle-backed ones when ``oracles`` is off (O-24)."""
    return tuple(gate for gate in gates if oracles or not gate.spec.requires_oracle)


class GateSuite:
    """Runs gates and attaches their verdicts to a record."""

    def __init__(
        self,
        artifact_gates: tuple[ArtifactGate, ...] | None = None,
        component_gates: tuple[ComponentGate, ...] | None = None,
        *,
        max_solves: int | None = None,
        oracles: bool | None = None,
    ) -> None:
        """Build a suite, defaulting to the standard gate set.

        Parameters
        ----------
        max_solves:
            Budget in extra solver runs for any single gate; ``None`` is unlimited. A gate over
            budget is reported SKIPPED naming the budget, never dropped.
        oracles:
            Whether gates with ``requires_oracle`` are on the plan; ``None`` reads
            ``CDFT_ORACLES`` (:func:`cdft.reference.oracles_enabled`, default off). Off, they
            are dropped from the plan and the record, not reported SKIPPED (O-24).
        """
        self.oracles = oracles_enabled() if oracles is None else bool(oracles)
        artifact_gates = artifact_gates if artifact_gates is not None else default_artifact_gates()
        if component_gates is None:
            component_gates = default_component_gates()
        self.artifact_gates = planned(artifact_gates, self.oracles)
        self.component_gates = planned(component_gates, self.oracles)
        # Off the plan; a scenario that names one still gets a SKIPPED row saying why (G5.3).
        self.dropped_gates = {
            gate.gate_id: gate
            for gate in (*artifact_gates, *component_gates)
            if gate not in self.artifact_gates and gate not in self.component_gates
        }
        self.max_solves = max_solves

    @property
    def oracles_mode(self) -> str:
        """``"on"`` or ``"off"``: the switch as it goes into every record this suite writes."""
        return "on" if self.oracles else "off"

    def run_on_artifact(self, artifact: RunArtifact) -> RunArtifact:
        """Evaluate every applicable gate, attach the report, and set the record's status.

        A gate the scenario names but the suite does not implement produces a ``FAIL`` naming it, so
        the record is invalid and the reason is in the file.
        """
        results: list[GateResult] = []
        implemented = {gate.gate_id for gate in self.artifact_gates} | {
            gate.gate_id for gate in self.component_gates
        }

        for gate in self.artifact_gates:
            if self.max_solves is not None and gate.spec.solves > self.max_solves:
                results.append(
                    skipped(
                        gate.spec,
                        f"costs {gate.spec.solves} extra solves and this profile allows "
                        f"{self.max_solves}; run with --profile full to evaluate it. The gate is "
                        f"implemented and this is a budget exclusion, not a deferral -- which is "
                        f"why it appears here as a row rather than not at all",
                    )
                )
            elif gate.applicable(artifact):
                results.append(gate.evaluate(artifact))
                if gate.spec.solves > 0:
                    # A ladder gate leaves its solves' blocks in the CUDA cache; hand them back
                    # before the next gate re-solves. No-op on the CPU.
                    release_device_memory(artifact_device(artifact))
            else:
                results.append(
                    skipped(gate.spec, f"not applicable to scenario {artifact.scenario.scenario_id}")
                )

        for gate_id in artifact.scenario.gates:
            if gate_id in implemented:
                continue
            if gate_id in self.dropped_gates:
                results.append(
                    skipped(
                        self.dropped_gates[gate_id].spec,
                        "calls an oracle and the oracles are off (CDFT_ORACLES unset); run with "
                        "--oracles or CDFT_ORACLES=1 to evaluate it (O-24)",
                    )
                )
                continue
            # Three distinct cases, kept apart: collapsing them into one FAIL leaves a reader unable
            # to tell an expected red row from a real one (D-37).
            owner = deferral_owner(gate_id)
            if owner is not None:
                entry = by_id(gate_id)
                results.append(
                    deferred(
                        GateSpec(
                            gate_id=gate_id,
                            name=entry.name,
                            threshold=0.0,
                            kind=entry.kind,
                        ),
                        owner,
                    )
                )
                continue
            reason = (
                f"scenario {artifact.scenario.scenario_id} names {gate_id}, which the specification "
                f"places outside Phase 1 -- it can never be evaluated on this system, so naming it "
                f"is a specification error rather than a deferral"
                if gate_id in specified_ids()
                else
                f"scenario {artifact.scenario.scenario_id} names {gate_id}, which the gate "
                f"catalogue does not define at all: either a typo, or docs/03_METHOD.md (Part C) "
                f"and cdft.gates.catalogue have drifted apart"
            )
            results.append(
                GateResult(
                    gate_id=gate_id,
                    name=UNEVALUABLE_GATE_NAME,
                    verdict=GateVerdict.FAIL,
                    measured=1.0,
                    threshold=0.0,
                    kind=GateKind.EXACT,
                    reason=reason,
                )
            )

        artifact.gates = GateReport(results=results)
        report_gate = GateReportGate()
        artifact.gates.results.append(report_gate.evaluate(artifact))
        artifact.measurements = {**dict(artifact.measurements), "oracles": self.oracles_mode}

        derived = artifact.gates.status
        if artifact.status in (RunStatus.ERROR, RunStatus.UNCONVERGED):
            return artifact
        artifact.status = derived
        return artifact

    def run_components(self, numerics: NumericsConfig) -> list[GateResult]:
        """Evaluate every component gate at one numerics configuration."""
        return [gate.evaluate(numerics) for gate in self.component_gates]


def format_report(results: list[GateResult], title: str = "") -> str:
    """Render a gate report as a fixed-width table.

    Measured values and thresholds are printed, never verdicts alone: the number is what says
    whether a pass was comfortable.
    """
    lines: list[str] = []
    if title:
        lines.append(title)
        lines.append("=" * len(title))
    header = f"{'gate':<8} {'verdict':<9} {'measured':>13} {'threshold':>13}  name"
    lines.append(header)
    lines.append("-" * max(len(header), 78))
    for result in results:
        measured = "n/a" if result.verdict is GateVerdict.SKIPPED else f"{result.measured:13.6e}"
        threshold = f"{result.threshold:13.6e}"
        lines.append(
            f"{result.gate_id:<8} {result.verdict.value:<9} {measured:>13} {threshold:>13}  "
            f"{result.name}"
        )
        if result.reason:
            lines.append(f"{'':<8} reason: {result.reason}")
    counts: dict[str, int] = {}
    for result in results:
        counts[result.verdict.value] = counts.get(result.verdict.value, 0) + 1
    lines.append("-" * max(len(header), 78))
    lines.append("  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return "\n".join(lines)

