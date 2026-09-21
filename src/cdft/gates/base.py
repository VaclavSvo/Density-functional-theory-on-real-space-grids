"""The gate framework: verdicts, thresholds, and the rule that nothing escapes unjudged.

A gate maps a run, or a component, to a :class:`~contract.GateResult` carrying a verdict, the
measured value, its threshold and kind, and a citation; gates are evaluated at solve time and travel
with every record (D-13, G5.3). Conventions: every gate reduces to one non-negative number for which
``measured <= threshold`` is the universal reading; MARGINAL is a pass above
:data:`MARGINAL_FRACTION` of the threshold, which flags a record rather than invalidating it; and
SKIPPED always carries a reason, since a gate that silently does not run looks like one that passed.
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass

from contract import (
    BoundaryMode,
    GateKind,
    GateResult,
    GateVerdict,
    NumericsConfig,
    RunArtifact,
)

__all__ = [
    "GateSpec",
    "ArtifactGate",
    "ComponentGate",
    "make_result",
    "skipped",
    "deferred",
    "artifact_device",
    "artifact_grid",
]

#: Fraction of the threshold above which a passing gate is reported ``MARGINAL`` (gate G5.3).
MARGINAL_FRACTION = 0.9


@dataclass(frozen=True, slots=True)
class GateSpec:
    """The fixed description of a gate: what it measures and what it is compared against."""

    gate_id: str
    name: str
    threshold: float
    kind: GateKind
    citation: str = ""
    units: str = ""
    min_states: int = 0
    """Eigenstates this gate needs the run to have converged.

    A property of the check, not of the system, so it lives here and the runner takes the maximum
    over the gates a scenario names; from the electron count alone a one-electron harmonic well
    would converge one state and silently lose both spectral gates.
    """

    solves: int = 0
    """Extra solver runs this gate performs, beyond reading the artifact it is handed.

    The ladder gates cost two to eight solves each; declaring the cost lets the runner exclude them
    by a stated budget rather than by a list that would drift. An excluded gate reports SKIPPED with
    its reason, never nothing. A cost switch only: a profile chooses which checks run and never the
    settings they run at, since a coarsened grid makes a DERIVED gate fail correctly.
    """

    requires_oracle: bool = False
    """Whether evaluating this gate calls PySCF/libxc (O-24).

    Off by default the runner drops such a gate from the plan and the record, rather than
    reporting SKIPPED on every surface; ``CDFT_ORACLES=1`` or ``--oracles`` restores it. Stored
    reference tables (:mod:`cdft.reference.computed`, G4.7) are data and do not set this.
    """


def artifact_device(artifact: RunArtifact):
    """Return the torch device the artifact was solved on, for gates that rebuild or re-solve.

    Read from ``Provenance.device`` so a re-solve lands on the device of the run it judges; a record
    without the field falls back to resolving ``numerics.device``.
    """
    import torch

    from ..precision import resolve_device

    recorded = str(getattr(artifact.provenance, "device", "") or "")
    if recorded:
        return torch.device(recorded)
    return resolve_device(artifact.numerics.device)


def artifact_grid(artifact: RunArtifact, device=None):
    """Rebuild a run's recorded grid -- shape, spacing, origin -- from ``measurements["grid"]``.

    A gate that reads a density point by point must place its points where the solver did.
    Re-deriving from the scenario reproduces the shape and spacing but not the origin, since a
    derived lattice is anchored on the first nucleus (D-64), which shifts every point by up to half
    a cell. A record without the description falls back to the derived grid.
    """
    import torch

    from ..grid import GridGeometry, UniformGrid, grid_config_for_scenario
    from ..scf.solve import is_interacting

    dev = device if device is not None else artifact_device(artifact)
    described = artifact.measurements.get("grid") if artifact.measurements else None
    if isinstance(described, dict) and {"shape", "spacing_bohr", "origin_bohr"} <= set(described):
        shape = tuple(int(n) for n in described["shape"])
        geometry = GridGeometry(
            shape=shape,  # type: ignore[arg-type]
            spacing=float(described["spacing_bohr"]),
            origin=tuple(float(x) for x in described["origin_bohr"]),  # type: ignore[arg-type]
        )
        boundary = BoundaryMode(str(described.get("boundary", artifact.numerics.grid.boundary.value)))
        return UniformGrid(
            geometry,
            torch.ones(shape, dtype=torch.bool, device=dev),
            fd_order=int(described.get("fd_order", artifact.numerics.grid.fd_order)),
            gradient_order=int(described.get("gradient_order", artifact.numerics.grid.fd_gradient_order)),
            boundary=boundary,
            device=dev,
            dtype=torch.float64,
        )
    config, _ = grid_config_for_scenario(
        artifact.scenario, artifact.numerics.grid, interacting=is_interacting(artifact.scenario)
    )
    structure = artifact.scenario.structure if artifact.scenario.structure.n_atoms else None
    return UniformGrid.from_config(config, structure=structure, device=dev, dtype=torch.float64)


def make_result(
    spec: GateSpec, measured: float, detail: dict[str, object] | None = None
) -> GateResult:
    """Build a :class:`~contract.GateResult` from a measured value, applying the verdict rule.

    A non-finite measurement is a ``FAIL``: ``nan > x`` is ``False``, so a naive threshold test
    would report a NaN as passing.
    """
    if not math.isfinite(measured):
        verdict = GateVerdict.FAIL
    elif abs(measured) > spec.threshold:
        verdict = GateVerdict.FAIL
    elif spec.threshold > 0.0 and abs(measured) > MARGINAL_FRACTION * spec.threshold:
        verdict = GateVerdict.MARGINAL
    else:
        verdict = GateVerdict.PASS
    return GateResult(
        gate_id=spec.gate_id,
        name=spec.name,
        verdict=verdict,
        measured=float(measured),
        threshold=float(spec.threshold),
        kind=spec.kind,
        citation=spec.citation,
        units=spec.units,
        detail=detail or {},
    )


def skipped(spec: GateSpec, reason: str) -> GateResult:
    """Build a ``SKIPPED`` result. The reason is mandatory and is written into the record."""
    if not reason:
        raise ValueError("a SKIPPED gate must state why; an unexplained skip is a silent failure")
    return GateResult(
        gate_id=spec.gate_id,
        name=spec.name,
        verdict=GateVerdict.SKIPPED,
        measured=float("nan"),
        threshold=float(spec.threshold),
        kind=spec.kind,
        citation=spec.citation,
        units=spec.units,
        reason=reason,
    )


def deferred(spec: GateSpec, owner: str) -> GateResult:
    """Build a ``DEFERRED`` result: the gate applies, and this build cannot evaluate it.

    Distinct from ``SKIPPED`` (does not apply) and ``FAIL`` (evaluated, physics wrong). The owning
    increment is mandatory, so the corpus carries its own list of what is still unchecked.
    """
    if not owner:
        raise ValueError("a DEFERRED gate must name the increment that will supply it")
    return GateResult(
        gate_id=spec.gate_id,
        name=spec.name,
        verdict=GateVerdict.DEFERRED,
        measured=float("nan"),
        threshold=float(spec.threshold),
        kind=spec.kind,
        citation=spec.citation,
        units=spec.units,
        reason=owner,
    )


class ArtifactGate(abc.ABC):
    """A gate evaluated on a completed run, per :class:`~contract.GateProtocol`."""

    spec: GateSpec

    @property
    def gate_id(self) -> str:
        """Identifier such as ``"G2.4"``, matching ``docs/03_METHOD.md (Part C)``."""
        return self.spec.gate_id

    def applicable(self, artifact: RunArtifact) -> bool:
        """Whether this gate applies to this run; by default, whenever the run produced a result."""
        return artifact.result is not None

    @abc.abstractmethod
    def evaluate(self, artifact: RunArtifact) -> GateResult:
        """Measure the quantity and return a verdict with the measured value attached."""


class ComponentGate(abc.ABC):
    """A gate verifying an operator at a configuration (:class:`~contract.ComponentGateProtocol`).

    A property of the code rather than of any particular solve, so running it through the artifact
    interface would mean fabricating a run to hold it.
    """

    spec: GateSpec

    @property
    def gate_id(self) -> str:
        """Identifier such as ``"G0.1"``, matching ``docs/03_METHOD.md (Part C)``."""
        return self.spec.gate_id

    @abc.abstractmethod
    def evaluate(self, numerics: NumericsConfig) -> GateResult:
        """Measure the component property at this configuration and return a verdict."""
