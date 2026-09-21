"""The gate catalogue: one machine-readable list of every gate the specification defines.

The source of truth for gate identity, against which ``tests/test_gate_catalogue.py`` checks the
prose table of ``docs/03_METHOD.md (Part C)`` and G5.8 checks the wired evaluators and the
scenario gate lists. The lifecycle field separates a gate that fails from one this build cannot
evaluate and from one no Phase 1 system can reach.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from contract import GateKind

__all__ = ["Lifecycle", "GateEntry", "CATALOGUE", "by_id", "deferral_owner", "specified_ids"]


class Lifecycle(str, Enum):
    """Where a gate stands in this build.

    ``IMPLEMENTED`` -- evaluated, verdict load-bearing. ``DEFERRED`` -- specified, not evaluable;
    :attr:`GateEntry.owner` names the increment that owes it and a scenario naming it reports
    ``DEFERRED``. ``OUT_OF_PHASE_1`` -- unreachable by any Phase 1 system, so a scenario naming one
    is a specification error and G5.8 fails rather than deferring.
    """

    IMPLEMENTED = "implemented"
    DEFERRED = "deferred"
    OUT_OF_PHASE_1 = "out_of_phase_1"


@dataclass(frozen=True, slots=True)
class GateEntry:
    """One row of ``docs/03_METHOD.md (Part C)``, in a form code can check."""

    gate_id: str
    name: str
    tier: int
    kind: GateKind
    lifecycle: Lifecycle
    owner: str = ""
    """For ``DEFERRED`` and ``OUT_OF_PHASE_1``: which increment or phase supplies this gate."""

    note: str = ""

    def __post_init__(self) -> None:
        """A deferral without an owner is a to-do pretending to be a plan."""
        if self.lifecycle is not Lifecycle.IMPLEMENTED and not self.owner:
            raise ValueError(f"{self.gate_id}: a non-implemented gate must name its owner")
        if self.lifecycle is Lifecycle.IMPLEMENTED and self.owner:
            raise ValueError(f"{self.gate_id}: an implemented gate has no owner to wait for")


#: Every gate in the specification, in specification order. Thresholds are not duplicated here:
#: they live on the implementation's :class:`~cdft.gates.base.GateSpec`, and the catalogue owns a
#: gate's identity -- name, tier, threshold kind and lifecycle.
CATALOGUE: tuple[GateEntry, ...] = (
    # Tier 0: the code does arithmetic correctly
    GateEntry("G0.1", "Laplacian convergence order", 0, GateKind.DERIVED, Lifecycle.IMPLEMENTED),
    GateEntry("G0.2", "Poisson solver against an analytic charge", 0, GateKind.DERIVED, Lifecycle.IMPLEMENTED),
    GateEntry("G0.3", "Orthonormality of the orbital block", 0, GateKind.EXACT, Lifecycle.IMPLEMENTED),
    GateEntry("G0.4", "Hermiticity of H", 0, GateKind.EXACT, Lifecycle.IMPLEMENTED),
    GateEntry(
        "G0.5", "v_xc is the functional derivative of E_xc", 0, GateKind.EXACT, Lifecycle.IMPLEMENTED,
        note="implemented 2026-09-14 (I2): autograd potentials against central differences, every native functional, both spin cases",
    ),
    GateEntry(
        "G0.6", "libxc agreement, pointwise", 0, GateKind.EXACT, Lifecycle.IMPLEMENTED,
        note="implemented 2026-09-14 (I2): libxc through PySCF (O-10 closed); r2SCAN is DEFERRED inside the gate's detail, not silently passed",
    ),
    GateEntry(
        "G0.7", "Kleinman-Bylander projector normalisation and ghost screening", 0,
        GateKind.EMPIRICAL, Lifecycle.OUT_OF_PHASE_1,
        owner="Phase 2", note="Phase 1 is all-electron (D-23); there is no projector to normalise",
    ),
    GateEntry("G0.8", "No NaN, no Inf", 0, GateKind.EXACT, Lifecycle.IMPLEMENTED),
    GateEntry("G0.9", "Gradient convergence order", 0, GateKind.DERIVED, Lifecycle.IMPLEMENTED),
    GateEntry(
        "G0.12", "Weighted self-adjointness of the operator", 0, GateKind.EXACT,
        Lifecycle.IMPLEMENTED,
        note=(
            "added with D-35; supersedes G0.4 on the cusp-factorised path; measured on the operator "
            "that generates the search space, not on the subspace matrix, since D-38 (gate review of D-75)"
        ),
    ),
    # Tier 1: the code solves the equations it claims
    GateEntry(
        "G1.1", "Uniform electron gas exchange and correlation", 1, GateKind.EXACT,
        Lifecycle.IMPLEMENTED, note="implemented 2026-09-14 (I2): Dirac exchange, GGA -> LDA at zero gradient, LDA correlations against the Ceperley-Alder points of PW92 Table II",
    ),
    GateEntry("G1.2", "3D isotropic harmonic oscillator", 1, GateKind.DERIVED, Lifecycle.IMPLEMENTED),
    GateEntry("G1.3", "Particle in a cubic box", 1, GateKind.DERIVED, Lifecycle.IMPLEMENTED),
    GateEntry("G1.4", "Grid isotropy", 1, GateKind.DERIVED, Lifecycle.IMPLEMENTED),
    GateEntry("G1.5", "Charge normalisation", 1, GateKind.EXACT, Lifecycle.IMPLEMENTED),
    GateEntry("G1.6", "Non-interacting limit", 1, GateKind.EXACT, Lifecycle.IMPLEMENTED),
    GateEntry(
        'G1.7', 'Virial theorem', 1, GateKind.EMPIRICAL,
        Lifecycle.IMPLEMENTED,
        note='implemented 2026-09-13; single centre only -- a molecule away from equilibrium violates the plain relation correctly, so multi-centre reports SKIPPED',
    ),
    GateEntry(
        "G1.8", "Lieb-Oxford bound", 1, GateKind.EXACT, Lifecycle.IMPLEMENTED,
        note="implemented 2026-09-14 (I2/I3): integrated bound on the converged density of every self-consistent run; the local form is deliberately not asserted (B88 violates it at large s by design)",
    ),
    GateEntry(
        "G1.9", "Uniform coordinate scaling of exchange", 1, GateKind.EXACT, Lifecycle.IMPLEMENTED,
        note="implemented 2026-09-14 (I2): exact-array scaling as in G1.12; the analogous scaling of T_s is G1.12",
    ),
    GateEntry(
        "G1.11", "Kato cusp condition", 1, GateKind.EMPIRICAL, Lifecycle.IMPLEMENTED,
        note=(
            "was implemented and gated before it was ever written into the table; added by this audit. "
            "EMPIRICAL as in Part C and the evaluator (a fitted slope at 0.1 relative), corrected from "
            "EXACT by the gate review of D-75"
        ),
    ),
    GateEntry(
        'G1.12', 'Coordinate scaling of the non-interacting kinetic energy', 1, GateKind.EXACT,
        Lifecycle.IMPLEMENTED,
        note='implemented 2026-09-13; a property of the operator, so it survives Increment 2 removing the analytic Tier-1 answers',
    ),
    GateEntry(
        "G1.13", "Cusp-weight quadrature", 1, GateKind.EXACT, Lifecycle.IMPLEMENTED,
        note="added 2026-09-14 (A-9); replaces the retired G1.10, whose ladder on the un-transformed path is kept as scripts/probes/cusp_factorisation_probe.py",
    ),
    # Tier 2: the numerics are converged
    GateEntry(
        "G2.1", "Harris-Foulkes agreement", 2, GateKind.DERIVED, Lifecycle.IMPLEMENTED,
        note="degenerates to the band energy without a functional, and is gated as such",
    ),
    GateEntry(
        "G2.2", "Janak's theorem", 2, GateKind.EXACT, Lifecycle.IMPLEMENTED,
        note="implemented 2026-09-14 (I3): central difference of the total energy in f_HOMO, two extra SCFs, full profile only",
    ),
    GateEntry(
        "G2.3", "Variational tail of the SCF", 2, GateKind.EMPIRICAL, Lifecycle.IMPLEMENTED,
        note="implemented 2026-09-14 (I3): variational output-orbital energy per iteration; rises below energy_tol are not counted",
    ),
    GateEntry("G2.4", "Eigenvalue residual", 2, GateKind.DERIVED, Lifecycle.IMPLEMENTED),
    GateEntry("G2.5", "Two-solver agreement", 2, GateKind.EXACT, Lifecycle.IMPLEMENTED),
    GateEntry(
        "G2.6", "Forces against finite differences", 2, GateKind.DERIVED, Lifecycle.DEFERRED,
        owner="I7", note="the cusp factor moves with the nuclei; see 03_METHOD.md (Part B)",
    ),
    GateEntry(
        'G2.7', 'Egg-box error', 2, GateKind.EMPIRICAL,
        Lifecycle.IMPLEMENTED,
        note='implemented 2026-09-13; the gate that would have caught D-39 from inside, with no oracle',
    ),
    GateEntry(
        "G2.8", "Translational and rotational invariance of the forces", 2, GateKind.EXACT,
        Lifecycle.DEFERRED, owner="I7", note="a sum rule on forces, which do not exist yet",
    ),
    GateEntry("G2.9", "Energy term closure", 2, GateKind.EXACT, Lifecycle.IMPLEMENTED),
    GateEntry(
        'G2.10', 'Mixed-precision audit', 2, GateKind.DERIVED,
        Lifecycle.DEFERRED, owner='Increment 6, second half',
        note=(
            'mixed-precision audit; needs the float32 hot path, the card-bound second half of '
            'Increment 6 (D-63, D-74); the GPU leg itself is in place (O-9 closed 2026-09-16)'
        ),
    ),
    # Tier 3: convergence with respect to the discretisation
    GateEntry(
        'G3.1', 'Grid-spacing convergence', 3, GateKind.DERIVED,
        Lifecycle.IMPLEMENTED,
        note='implemented 2026-09-13; bypasses the D-42 derivation via derive_grid=False, and reports monotonicity separately from the error',
    ),
    GateEntry(
        "G3.2", "Domain-size convergence", 3, GateKind.DERIVED,
        Lifecycle.IMPLEMENTED,
        note=(
            "the binding constraint on the cusp-factorised path, so this matters more here than in "
            "an ordinary real-space code: accuracy is bought with box size, not with spacing"
        ),
    ),
    GateEntry(
        "G3.3", "SCF convergence criterion", 3, GateKind.DERIVED, Lifecycle.IMPLEMENTED,
        note="implemented 2026-09-14 (I3): both criteria as the fraction of their tolerance at the final iteration",
    ),
    GateEntry(
        "G3.4", "Initial-guess independence", 3, GateKind.EXACT, Lifecycle.IMPLEMENTED,
        note="implemented 2026-09-14 (I3): hydrogenic, diffuse and compact starts; two extra SCFs, full profile only",
    ),
    GateEntry(
        'G3.5', 'Stencil-order independence', 3, GateKind.DERIVED,
        Lifecycle.IMPLEMENTED,
        note='implemented 2026-09-13; order 8 against order 12 at the production spacing',
    ),
    GateEntry(
        "G4.1", "Atomic eigenvalues against the pseudopotential file", 4, GateKind.EMPIRICAL,
        Lifecycle.OUT_OF_PHASE_1, owner="Phase 2",
        note="all-electron Phase 1 (D-23) ships no pseudopotential file to be checked against",
    ),
    GateEntry(
        "G4.2", "Bond lengths against published values", 4, GateKind.EMPIRICAL,
        Lifecycle.OUT_OF_PHASE_1, owner="Phase 2", note="needs geometry optimisation and forces",
    ),
    GateEntry(
        "G4.3", "Binding energies, cross-code against Octopus", 4, GateKind.EMPIRICAL,
        Lifecycle.OUT_OF_PHASE_1, owner="Phase 2",
    ),
    GateEntry(
        "G4.4", "Dissociation curve, epsilon-measure", 4, GateKind.EMPIRICAL,
        Lifecycle.OUT_OF_PHASE_1, owner="Phase 2",
    ),
    GateEntry(
        "G4.5", "Self-interaction error, reproduced deliberately", 4, GateKind.EMPIRICAL,
        Lifecycle.IMPLEMENTED,
        note="implemented 2026-09-14 (I3): E(H2+, R = 8) must lie below E(H) at the same functional; the magnitude is diagnostic D1.8",
    ),
    GateEntry(
        "G4.6", "Atomization energies", 4, GateKind.EMPIRICAL, Lifecycle.OUT_OF_PHASE_1,
        owner="Phase 2", note="blocked additionally on spin (Increment 4, D-51)",
    ),
    GateEntry(
        'G4.7', 'All-electron agreement with published values', 4, GateKind.EMPIRICAL,
        Lifecycle.IMPLEMENTED,
        note='implemented 2026-09-14 (I3): NIST SRD 141 for H, He, He+ at VWN LDA; PySCF computed oracle (O-12) for the rest',
    ),
    GateEntry("G4.8", "Radial reference agreement", 4, GateKind.DERIVED, Lifecycle.IMPLEMENTED),
    GateEntry(
        'G4.9', 'Two-centre reference agreement', 4, GateKind.DERIVED,
        Lifecycle.IMPLEMENTED,
        note='implemented 2026-09-13; H2+ against cdft.reference.two_centre at any bond length; threshold 2.5e-3 sits above the known O-13 error by design',
    ),
    # Tier 5: the record is trustworthy
    GateEntry("G5.1", "Provenance", 5, GateKind.EXACT, Lifecycle.IMPLEMENTED),
    GateEntry(
        'G5.2', 'Determinism', 5, GateKind.EXACT,
        Lifecycle.IMPLEMENTED,
        note=(
            'implemented 2026-09-13; GPU leg 2026-09-16 (O-9 closed); since D-61 two strict solves '
            'of its own (A, B) compared at 1e-9, the primary run off and its difference to A '
            'recorded (primary_vs_strict), never thresholded'
        ),
    ),
    GateEntry("G5.3", "Gate report attached", 5, GateKind.EXACT, Lifecycle.IMPLEMENTED),
    GateEntry("G5.4", "Corpus reader enforces status", 5, GateKind.EXACT, Lifecycle.IMPLEMENTED),
    GateEntry(
        'G5.5', 'No default-silent fallbacks', 5, GateKind.EXACT,
        Lifecycle.IMPLEMENTED,
        note='implemented 2026-09-13; counts unexplained grid changes, not the presence of an overrides dict',
    ),
    GateEntry("G5.6", "Non-zero exit", 5, GateKind.EXACT, Lifecycle.IMPLEMENTED),
    GateEntry("G5.7", "Scope boundary: no machine learning in src/cdft", 5, GateKind.EXACT, Lifecycle.IMPLEMENTED),
    GateEntry(
        'G5.8', 'Gate catalogue consistency', 5, GateKind.EXACT,
        Lifecycle.IMPLEMENTED,
        note='implemented 2026-09-13; cross-checks catalogue against wired evaluators against scenario gate lists',
    ),
    GateEntry(
        'G5.9', 'Sync/launch budget', 5, GateKind.EXACT,
        Lifecycle.IMPLEMENTED,
        note=(
            'implemented 2026-09-16 (D-70); CPU census of scripts/sync_census.py on h_atom and a '
            'coarse SCF helium probe: loop reads and implicit syncs per iteration, launches per '
            'solve, against recorded budgets with 10 % headroom (per-iteration headroom capped at 0.5)'
        ),
    ),
    # Tier 6: forces and geometry
    GateEntry(
        "G6.1", "Hellmann-Feynman forces including the cusp-factor terms", 6, GateKind.DERIVED,
        Lifecycle.DEFERRED, owner="I7",
    ),
)

_BY_ID: dict[str, GateEntry] = {entry.gate_id: entry for entry in CATALOGUE}
if len(_BY_ID) != len(CATALOGUE):  # pragma: no cover - a duplicate is a typo, caught at import
    raise RuntimeError("duplicate gate identifier in CATALOGUE")


def by_id(gate_id: str) -> GateEntry:
    """Return one catalogue entry, raising with the full list if the identifier is unknown."""
    if gate_id not in _BY_ID:
        raise KeyError(
            f"unknown gate {gate_id!r}. A scenario may only name a gate the specification "
            f"defines. Known: {', '.join(sorted(_BY_ID))}"
        )
    return _BY_ID[gate_id]


def specified_ids() -> frozenset[str]:
    """Every gate identifier the specification defines."""
    return frozenset(_BY_ID)


def deferral_owner(gate_id: str) -> str | None:
    """Return the increment that owes this gate, or ``None`` if it is not a legitimate deferral.

    ``OUT_OF_PHASE_1`` deliberately returns ``None``: a Phase 1 scenario naming a Phase 2 gate is a
    specification error and must fail loudly, not wait quietly for a phase that will never evaluate
    it on this system.
    """
    entry = _BY_ID.get(gate_id)
    if entry is None or entry.lifecycle is not Lifecycle.DEFERRED:
        return None
    return f"{entry.owner}: {entry.note or entry.name}"
