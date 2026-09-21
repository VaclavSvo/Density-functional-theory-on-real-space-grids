"""Frozen API contract for ``cdft`` -- the Classical DFT Solver.

Defines *what* every component must look like: configuration schemas, structural protocols, result
types and the on-disk layout. No physics, no algorithms. Implementations live under ``src/cdft/``
and are written against this file; amending it requires a ``docs/05_DECISION_LOG.md`` entry
and a bump of :data:`CONTRACT_VERSION`, and older records stay readable (the corpus is
append-only). Every replaceable component is a :class:`typing.Protocol`, so a learned functional,
another eigensolver or another Poisson kernel can be substituted without touching the SCF loop [C3].

All fields live on the masked real-space domain, flattened to a single point index ``n_pts``;
:class:`GridProtocol` owns the mapping back to the 3-D box. The spin axis is always present, with
extent 1 when unpolarised; occupations are floating point so fractional-electron scenarios [C1] fit.

==================  ====================================  ==========================================
Quantity            Shape                                 Notes
==================  ====================================  ==========================================
density ``n``       ``(n_spin, n_pts)``                   ``n_spin == 1`` is spin-restricted
gradient            ``(n_spin, 3, n_pts)``                Cartesian components
kinetic density     ``(n_spin, n_pts)``                   tau; always stored, even for LDA
orbitals ``psi``    ``(n_spin, n_states, n_pts)``         real for the ground state
eigenvalues         ``(n_spin, n_states)``                Hartree
occupations         ``(n_spin, n_states)``                float, never int
potentials          ``(n_spin, n_pts)`` or ``(n_pts,)``   spin-dependent for xc, otherwise not
forces              ``(n_atoms, 3)``                      Hartree / bohr
==================  ====================================  ==========================================

Hartree atomic units throughout: energies in Hartree, lengths in bohr, forces in Hartree/bohr;
conversion happens only at the configuration and reporting boundaries, via :mod:`cdft.constants`.
Reference tags such as ``[C3]`` resolve in ``docs/00_LITERATURE_SURVEY.md``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Protocol, Sequence, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch import Tensor
else:  # torch is a hard runtime dependency of the solver, but the contract stays importable
    Tensor = Any  # type: ignore[misc,assignment]

# --- Scope boundary ---
#
# This package contains NO machine learning: no training loop, loss, optimiser over functional
# parameters or network architecture under src/cdft/; gate G5.7 enforces it by static import scan.
# Evaluating a functional somebody else trained is in scope; producing one is not. Three things
# cross to a downstream project: the corpus (CORPUS_LAYOUT, SCAN_LAYOUT), XCFunctionalProtocol (a
# pure evaluate, no fit/train/parameters) and NonlocalFeatureProtocol with its adjoint.
#
# Only one differentiability is owed (D-47): inside the functional, where v_xc comes from autograd
# (gate G0.5). Through the SCF is downstream work; this repository owes only SCFStepProtocol, the
# fixed-point map kept extractable so implicit differentiation can be added later.

__all__ = [
    "CONTRACT_VERSION",
    "TRUSTED_STATUSES",
    "Precision",
    "Device",
    "XCRung",
    "MixingScheme",
    "EigensolverKind",
    "OccupationScheme",
    "PoissonKind",
    "RunStatus",
    "GateVerdict",
    "GateKind",
    "DomainMode",
    "BoundaryMode",
    "ExternalPotentialKind",
    "ReferenceKind",
    "InversionKind",
    "InversionSpec",
    "InversionProtocol",
    "GridConfig",
    "PrecisionConfig",
    "EigenConfig",
    "MixingConfig",
    "SCFConfig",
    "OutputConfig",
    "NumericsConfig",
    "AtomicStructure",
    "ExternalPotentialSpec",
    "ElectronSpec",
    "ReferenceValue",
    "XCSpec",
    "PseudoSpec",
    "ScenarioSpec",
    "GridProtocol",
    "PseudopotentialProtocol",
    "XCFunctionalProtocol",
    "PoissonSolverProtocol",
    "HamiltonianProtocol",
    "EigensolverProtocol",
    "MixerProtocol",
    "SCFStepProtocol",
    "OccupationProtocol",
    "InitialGuessProtocol",
    "GateProtocol",
    "ComponentGateProtocol",
    "DiagnosticProtocol",
    "ScanDiagnosticProtocol",
    "ScanRunnerProtocol",
    "SolverProtocol",
    "CorpusWriterProtocol",
    "NonlocalFeatureKind",
    "NLDFSpec",
    "NonlocalFeatureProtocol",
    "ScanKind",
    "ScanSpec",
    "ScanArtifact",
    "Diagnostic",
    "DiagnosticReport",
    "XCOutput",
    "EigenResult",
    "EnergyBreakdown",
    "SCFTrajectory",
    "SCFResult",
    "GateResult",
    "GateReport",
    "Provenance",
    "RunArtifact",
    "CORPUS_LAYOUT",
    "SCAN_LAYOUT",
]

#: Version of this contract. Stamped into every corpus record (gate G5.1). Bump on any
#: backwards-incompatible change and record the reason in the decision log.
#:
#: 1.0.0  initial frozen API.
#: 1.1.0  D-19 non-local density features; D-20 diagnostics; D-21 scans.
#: 1.2.0  D-23/D-27 external potentials; D-28 ElectronSpec; D-29 ReferenceValue; D-30 diagnostic
#:        and scan protocols; RunArtifact.measurements; GridConfig domain modes; fingerprint.
#: 1.3.0  D-35 cusp factorisation: ``ExternalPotentialSpec.cusp_factorisation``, default True.
#: 1.4.0  D-37 ``GateVerdict.DEFERRED`` as a first-class verdict; D-42 derived grid fields.
#: 1.5.0  D-49 :class:`ComponentGateProtocol`; :class:`SCFStepProtocol` (D-47).
#: 1.6.0  :class:`Provenance` gains the device block (gate G5.1); additive, every field defaulted.
#: 1.6.1  O-29 :class:`InversionProtocol` loses the ``gate_id`` / ``evaluate`` pair copied from
#:        :class:`ComponentGateProtocol`; no record field changes.
CONTRACT_VERSION = "1.6.1"


# --- Enumerations ---


class Precision(str, Enum):
    """Precision of a tensor or a stage; ``FLOAT16`` is declared but unused (no tensor cores)."""

    FLOAT32 = "float32"
    FLOAT64 = "float64"
    FLOAT16 = "float16"


class Device(str, Enum):
    """Execution device. ``AUTO`` selects CUDA when available and falls back to CPU."""

    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"


class XCRung(int, Enum):
    """Rung of Jacob's ladder [A3]: the ingredients consumed, hence the columns the corpus needs."""

    LDA = 1
    GGA = 2
    META_GGA = 3
    HYBRID = 4
    DOUBLE_HYBRID = 5
    LEARNED = 6


class NonlocalFeatureKind(str, Enum):
    """Kind of non-local density feature consumed by the functional.

    Semi-local ingredients cannot remove one-electron self-interaction. ``CIDER_GAUSSIAN`` is the
    affordable alternative to exact exchange: the density convolved with exp[-(a+b) r^2] over a
    density- and tau-dependent length scale [C26], [C27], one FFT pair per fixed exponent (D-03).
    """

    NONE = "none"
    CIDER_GAUSSIAN = "cider_gaussian"
    CUSTOM = "custom"


class DomainMode(str, Enum):
    """How the computational domain is carved out of the enclosing box.

    ``MASKED_SPHERES`` keeps the union of spheres of radius :attr:`GridConfig.mask_radius` [D3],
    [D5], removing roughly 55 percent of the cube's points at no accuracy cost. ``BOX`` keeps every
    point and is required with no atoms or a potential that is not atom-centred (G1.1, G1.3).
    """

    MASKED_SPHERES = "masked_spheres"
    BOX = "box"


class BoundaryMode(str, Enum):
    """How the finite-difference stencil is closed at the edge of the computational domain.

    Both impose the same Dirichlet condition and differ only in the halo values, which costs several
    orders of accuracy. ``ZERO`` is the molecular default, correct for a state already decayed below
    the tolerance of gate G3.2. ``ODD_REFLECTION`` is the correct closure for a hard wall, whose
    eigenfunctions are the sine series; zero-filling there feeds the stencil a kink and destroys the
    nominal order beyond three points (gate G1.3).
    """

    ZERO = "zero"
    ODD_REFLECTION = "odd_reflection"


class ExternalPotentialKind(str, Enum):
    """Source of the external potential v_ext (D-23).

    ``NUCLEAR_COULOMB`` is the all-electron path: a bare -Z/r per nucleus, one-electron eigenvalues
    -Z^2 / (2 n^2) exactly (gate G4.8), energies comparable with published all-electron values (gate
    G4.7); its cost is the nuclear cusp, so convergence in the spacing is measured, never assumed.
    ``SOFT_COULOMB``, -Z / sqrt(r^2 + a^2), is a *different Hamiltonian*, never compared against
    Coulombic literature. The rest are analytic test systems that localise a bug in an operator.
    """

    NONE = "none"
    NUCLEAR_COULOMB = "nuclear_coulomb"
    SOFT_COULOMB = "soft_coulomb"
    PSEUDOPOTENTIAL = "pseudopotential"
    HARMONIC = "harmonic"
    PARTICLE_IN_BOX = "particle_in_box"
    GAUSSIAN_WELL = "gaussian_well"


class InversionKind(str, Enum):
    """Method used to recover the Kohn--Sham potential from a reference density (D-17).

    ``EXACT_ONE_ORBITAL`` is exact to discretisation error: with one occupied orbital
    psi = sqrt(n / N), so v follows in closed form. ``WU_YANG`` [G1] and ``ZMP`` [G2] are the
    many-electron methods, where the inversion is ill-posed and [G4] governs the result.
    """

    EXACT_ONE_ORBITAL = "exact_one_orbital"
    WU_YANG = "wu_yang"
    ZMP = "zmp"


class ReferenceKind(str, Enum):
    """Where a reference value came from; ``LITERATURE`` and ``CROSS_CODE`` carry their method."""

    ANALYTIC = "analytic"
    LITERATURE = "literature"
    CROSS_CODE = "cross_code"


class MixingScheme(str, Enum):
    """Density-mixing scheme; ``PERIODIC_PULAY`` [F9] is the default, ``KERKER`` [F11] is off."""

    LINEAR = "linear"
    ANDERSON = "anderson"
    PULAY = "pulay"
    PERIODIC_PULAY = "periodic_pulay"
    KERKER = "kerker"


class EigensolverKind(str, Enum):
    """Iterative eigensolver; ``CHEFSI`` [F1] is production, ``LOBPCG`` [F4] the G2.5 oracle."""

    CHEFSI = "chefsi"
    LOBPCG = "lobpcg"
    DAVIDSON = "davidson"
    DIRECT = "direct"


class OccupationScheme(str, Enum):
    """How occupations are assigned; ``FIXED`` is the default, smearing [F18]-[F20] awaits D-51."""

    FIXED = "fixed"
    FRACTIONAL = "fractional"
    FERMI = "fermi"
    GAUSSIAN = "gaussian"
    METHFESSEL_PAXTON = "methfessel_paxton"
    MARZARI_VANDERBILT = "marzari_vanderbilt"


class PoissonKind(str, Enum):
    """Open-boundary Hartree solver; ``COULOMB_CUTOFF`` [F15] ships first, ``ISF`` [D14] refines."""

    COULOMB_CUTOFF = "coulomb_cutoff"
    ISF = "isf"
    MARTYNA_TUCKERMAN = "martyna_tuckerman"
    MULTIGRID = "multigrid"


class ScanKind(str, Enum):
    """Kind of multi-run curve; errors such as the convexity of E(N) [A21] belong to curves."""

    FRACTIONAL_CHARGE = "fractional_charge"
    FRACTIONAL_SPIN = "fractional_spin"
    FLAT_PLANE = "flat_plane"
    DISSOCIATION = "dissociation"
    GEOMETRY = "geometry"
    CONVERGENCE = "convergence"


class RunStatus(str, Enum):
    """Verdict attached to every corpus record.

    A failed gate makes a record ``INVALID``, excluded by the default corpus iterator (gate G5.4).
    ``DEFERRED`` means a named gate this build cannot evaluate: excluded too, reported apart (D-37).
    """

    VALID = "valid"
    MARGINAL = "marginal"
    DEFERRED = "deferred"
    INVALID = "invalid"
    UNCONVERGED = "unconverged"
    ERROR = "error"


#: Statuses whose numbers may be consumed by an analysis, a plot or a training set; the single
#: definition behind gate G5.4. ``MARGINAL`` is in, as a pass that used most of its budget.
TRUSTED_STATUSES: frozenset[RunStatus] = frozenset({RunStatus.VALID, RunStatus.MARGINAL})


class GateVerdict(str, Enum):
    """Outcome of a single gate.

    ``MARGINAL`` passed within 10 percent of the threshold; ``SKIPPED`` does not apply and says why;
    ``DEFERRED`` applies but is unevaluable here, and names the increment that will supply it.
    """

    PASS = "pass"
    MARGINAL = "marginal"
    DEFERRED = "deferred"
    FAIL = "fail"
    SKIPPED = "skipped"


class GateKind(str, Enum):
    """Provenance of a threshold: analytic, derived, or empirical with a mandatory citation."""

    EXACT = "exact"
    DERIVED = "derived"
    EMPIRICAL = "empirical"


# --- Numerics configuration: "how hard to try". Never contains an atom. ---


@dataclass(frozen=True, slots=True)
class GridConfig:
    """Real-space grid and domain.

    ``spacing`` (bohr) is the single convergence parameter, at which gate G3.1 requires
    ``|E(h) - E(h/2)| < 1 meV/atom``; ``fd_order`` is the order ``2p`` of the Laplacian [D2],
    measured by gate G0.1; ``mask_radius`` (bohr) is the sphere per atom (gate G3.2);
    ``use_double_grid`` [D16] and ``fourier_filter_projectors`` [D18] tame sharp projectors, the
    first being needed for forces to pass egg-box gate G2.7.
    """

    spacing: float = 0.30
    fd_order: int = 8
    fd_gradient_order: int = 8
    mask_radius: float = 11.0
    domain: DomainMode = DomainMode.MASKED_SPHERES
    boundary: BoundaryMode = BoundaryMode.ZERO
    box_lengths: tuple[float, float, float] | None = None
    poisson_pad_factor: float = 2.5
    use_double_grid: bool = True
    fourier_filter_projectors: bool = True

    def __post_init__(self) -> None:
        """Validate the grid parameters eagerly, so a bad config fails before any compute."""
        if self.spacing <= 0.0:
            raise ValueError("spacing must be positive (bohr)")
        if self.fd_order % 2 or not 2 <= self.fd_order <= 16:
            raise ValueError("fd_order must be an even integer in [2, 16]")
        if self.fd_gradient_order % 2 or not 2 <= self.fd_gradient_order <= 16:
            raise ValueError("fd_gradient_order must be an even integer in [2, 16]")
        if self.mask_radius <= 0.0:
            raise ValueError("mask_radius must be positive (bohr)")
        if self.box_lengths is not None and any(length <= 0.0 for length in self.box_lengths):
            raise ValueError("box_lengths must all be positive (bohr)")
        if self.domain is DomainMode.BOX and self.box_lengths is None:
            raise ValueError("DomainMode.BOX requires explicit box_lengths (bohr)")
        if self.poisson_pad_factor < 2.0:
            raise ValueError(
                "poisson_pad_factor must be at least 2.0; a smaller pad lets the periodic images "
                "of the charge distribution overlap inside the cutoff radius, which makes the "
                "open-boundary Hartree potential wrong without making it look wrong (gate G0.2)"
            )
        # 2.0 is the floor, 2.5 the default: at 2.0 the Hartree energy of an analytic Gaussian is
        # exact to 5e-16 Ha but its far-field potential carries 1.1e-4 relative error (ringing from
        # the sharp cutoff), which corrupts the asymptotic-decay diagnostic D1.7. At 2.5 both are at
        # machine precision.


@dataclass(frozen=True, slots=True)
class PrecisionConfig:
    """Mixed-precision policy.

    float64 runs at 1/32 of float32 throughput and a full-cube float64 orbital block for 100 atoms
    needs 9.9 GB against 6 GB of VRAM, so mixed precision is a feasibility requirement, not an
    optimisation. ``reduction``, ``subspace`` and ``orthonormalisation`` stay float64: they are
    cheap, and lost block orthogonality is how mixed precision breaks eigensolvers [I8].
    ``audit_in_float64`` drives gate G2.10: re-run in float64 on CPU, or the run fails.
    """

    hot_path: Precision = Precision.FLOAT32
    reduction: Precision = Precision.FLOAT64
    subspace: Precision = Precision.FLOAT64
    orthonormalisation: Precision = Precision.FLOAT64
    audit_in_float64: bool = True
    audit_energy_tol: float = 1.0e-6
    audit_density_tol: float = 1.0e-5
    audit_force_tol: float = 1.0e-5

    def __post_init__(self) -> None:
        """Reject policies that would silently defeat the precision gates."""
        strict = (self.reduction, self.subspace, self.orthonormalisation)
        if any(p is not Precision.FLOAT64 for p in strict):
            raise ValueError(
                "reduction, subspace and orthonormalisation must stay float64; "
                "relaxing them invalidates gates G0.3 and G2.10"
            )


@dataclass(frozen=True, slots=True)
class EigenConfig:
    """Iterative eigensolver settings.

    ``n_extra_states`` unoccupied states are carried above the occupied manifold for robustness;
    ``chebyshev_degree`` [F2] is usually 8-20; ``residual_tol`` targets
    ``max_i ||H psi_i - eps_i psi_i||`` (gate G2.4); ``cross_check`` requires the second solver to
    agree (gate G2.5), which is expensive and so reserved for gate and reference runs.
    """

    kind: EigensolverKind = EigensolverKind.CHEFSI
    n_extra_states: int = 8
    extra_states_fraction: float = 0.20
    chebyshev_degree: int = 12
    lanczos_steps: int = 12
    bound_safety: float = 1.02
    max_iterations: int = 60
    residual_tol: float = 1.0e-6
    cross_check: bool = False
    cross_check_tol: float = 1.0e-8

    def __post_init__(self) -> None:
        """Reject settings that would make the Chebyshev filter amplify the wrong subspace."""
        if self.chebyshev_degree < 2:
            raise ValueError("chebyshev_degree must be at least 2")
        if self.lanczos_steps < 2:
            raise ValueError("lanczos_steps must be at least 2")
        if self.bound_safety < 1.0:
            raise ValueError(
                "bound_safety must be >= 1.0: an *underestimated* spectral upper bound makes the "
                "Chebyshev filter amplify the unwanted subspace, and the resulting failure looks "
                "like slow convergence rather than an error"
            )


@dataclass(frozen=True, slots=True)
class MixingConfig:
    """Density-mixing settings; ``kerker`` [F11] is off, charge sloshing being a metallic effect."""

    scheme: MixingScheme = MixingScheme.PERIODIC_PULAY
    alpha: float = 0.30
    history: int = 8
    pulay_period: int = 3
    kerker: bool = False
    kerker_q0: float = 1.5


@dataclass(frozen=True, slots=True)
class SCFConfig:
    """Self-consistent field loop settings.

    Convergence requires *both* ``energy_tol`` and ``density_tol`` (gate G3.3): either alone can be
    satisfied by a stalled iteration. ``fallback_ladder`` enables the degradation sequence of [F12];
    every step down it is recorded and downgrades the status, hidden fallbacks being barred (G5.5).
    """

    energy_tol: float = 1.0e-8
    density_tol: float = 1.0e-6
    max_iterations: int = 200
    min_iterations: int = 3
    occupation: OccupationScheme = OccupationScheme.FIXED
    smearing_width: float = 0.0
    fallback_ladder: bool = True
    compute_harris_foulkes: bool = True


@dataclass(frozen=True, slots=True)
class OutputConfig:
    """What to write to the corpus; only ``store_orbitals`` is off by default, on size grounds."""

    store_density: bool = True
    store_gradients: bool = True
    store_tau: bool = True
    store_laplacian: bool = False
    store_potentials: bool = True
    store_orbitals: bool = False
    store_scf_trajectory: bool = True
    compression: str = "gzip"
    compression_level: int = 4


@dataclass(frozen=True, slots=True)
class NumericsConfig:
    """The complete "how hard to try" configuration; atoms and chemistry live in ScenarioSpec."""

    grid: GridConfig = field(default_factory=GridConfig)
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    eigen: EigenConfig = field(default_factory=EigenConfig)
    mixing: MixingConfig = field(default_factory=MixingConfig)
    scf: SCFConfig = field(default_factory=SCFConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    device: Device = Device.AUTO
    seed: int = 0
    poisson: PoissonKind = PoissonKind.COULOMB_CUTOFF
    deterministic: bool = False

    def fingerprint(self) -> str:
        """Return a stable hash, stored as ``config_hash`` (gate G5.1): JSON with sorted keys."""
        canonical = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# --- Physics configuration: "what to solve". Never contains a grid spacing. ---


@dataclass(frozen=True, slots=True)
class AtomicStructure:
    """A molecule: ``numbers`` ``(n_atoms,)``, ``positions`` ``(n_atoms, 3)`` bohr, float charge."""

    numbers: Sequence[int] = ()
    positions: Any = None
    charge: float = 0.0
    multiplicity: int = 1
    label: str = ""

    @property
    def n_atoms(self) -> int:
        """Number of atoms. Zero is legal: the model scenarios of rungs L0, L1 and L3 have none."""
        return len(self.numbers)

    @property
    def is_model(self) -> bool:
        """Whether this is an atom-free model system (harmonic well, box, uniform gas)."""
        return self.n_atoms == 0


@dataclass(frozen=True, slots=True)
class ExternalPotentialSpec:
    """What generates the external potential for a scenario.

    Physics, not numerics: this says *which Hamiltonian*, while :class:`GridConfig` says how finely
    it is resolved. ``charges`` are the nuclear Z of the Coulombic kinds, empty meaning
    :attr:`AtomicStructure.numbers`; ``softening`` is the a of -Z / sqrt(r^2 + a^2) in bohr;
    ``omega`` is the ``HARMONIC`` frequency, exact spectrum (n + 3/2) omega (gate G1.2);
    ``box_length`` the ``PARTICLE_IN_BOX`` edge L in bohr, exact spectrum
    pi^2 (nx^2 + ny^2 + nz^2) / (2 L^2) (gate G1.3); ``depth`` (Ha) and ``width`` (bohr) shape the
    ``GAUSSIAN_WELL``. ``cusp_factorisation`` solves for phi in psi = exp(-sum_a Z_a |r - R_a|) phi
    (D-35), removing the nuclear cusp analytically on the Coulombic kinds and a no-op elsewhere; it
    is on by default, turning an O(h) error into one limited only by the box (hydrogen 1.05e-2 ->
    2.35e-6 Ha) for twice the stencil work (gate G1.12).
    """

    kind: ExternalPotentialKind = ExternalPotentialKind.NUCLEAR_COULOMB
    cusp_factorisation: bool = True
    charges: tuple[float, ...] = ()
    softening: float = 0.0
    omega: float = 1.0
    box_length: float = 0.0
    depth: float = 0.0
    width: float = 1.0

    def __post_init__(self) -> None:
        """Reject a specification whose parameters do not match its kind."""
        if self.kind is ExternalPotentialKind.SOFT_COULOMB and self.softening <= 0.0:
            raise ValueError("SOFT_COULOMB requires softening > 0 (bohr)")
        if self.kind is ExternalPotentialKind.HARMONIC and self.omega <= 0.0:
            raise ValueError("HARMONIC requires omega > 0")
        if self.kind is ExternalPotentialKind.PARTICLE_IN_BOX and self.box_length <= 0.0:
            raise ValueError("PARTICLE_IN_BOX requires box_length > 0 (bohr)")
        if self.kind is ExternalPotentialKind.GAUSSIAN_WELL and (
            self.depth <= 0.0 or self.width <= 0.0
        ):
            raise ValueError("GAUSSIAN_WELL requires depth > 0 (Ha) and width > 0 (bohr)")
        if any(z < 0.0 for z in self.charges):
            raise ValueError("nuclear charges must be non-negative")


@dataclass(frozen=True, slots=True)
class ElectronSpec:
    """How many electrons there are, and how they are distributed over the spin channels.

    A total charge plus an integer multiplicity cannot express the flat-plane surface E(N, M) of
    [A23]. ``n_electrons`` may be fractional, ``None`` deriving it from the atomic numbers or the
    valence charges minus :attr:`AtomicStructure.charge`; ``magnetisation`` is M = N_up - N_down and
    needs ``spin_polarised``, which sets the spin axis to extent 2 (v1 floor: 1, D-10).
    """

    n_electrons: float | None = None
    magnetisation: float = 0.0
    spin_polarised: bool = False
    fixed_occupations: tuple[tuple[float, ...], ...] | None = None

    @property
    def n_spin(self) -> int:
        """Extent of the spin axis implied by this specification: 2 if polarised, otherwise 1."""
        return 2 if self.spin_polarised else 1

    def __post_init__(self) -> None:
        """Reject a specification the solver cannot represent, before any compute happens."""
        if self.n_electrons is not None and self.n_electrons < 0.0:
            raise ValueError("n_electrons must be non-negative")
        if self.magnetisation != 0.0 and not self.spin_polarised:
            raise ValueError("a non-zero magnetisation requires spin_polarised=True")
        if self.n_electrons is not None and abs(self.magnetisation) > self.n_electrons + 1.0e-12:
            raise ValueError("|magnetisation| cannot exceed the electron number")


@dataclass(frozen=True, slots=True)
class InversionSpec:
    """How to recover v_KS and v_xc from a reference density (D-17).

    ``regularisation`` is the smoothness penalty of [G4]: ignored by ``EXACT_ONE_ORBITAL``, and
    mandatory above zero for the iterative methods, where an unregularised solution oscillates on
    the grid scale while reproducing the density perfectly. Below ``density_floor`` an inversion is
    only noise amplified by division and is not reported (diagnostic D1.7 fits above it).
    """

    kind: InversionKind = InversionKind.EXACT_ONE_ORBITAL
    regularisation: float = 0.0
    density_floor: float = 1.0e-8
    max_iterations: int = 200
    tolerance: float = 1.0e-8

    def __post_init__(self) -> None:
        """Reject an iterative inversion with no regularisation, which is ill-posed by construction."""
        if self.kind is not InversionKind.EXACT_ONE_ORBITAL and self.regularisation <= 0.0:
            raise ValueError(
                "iterative Kohn-Sham inversion requires regularisation > 0 [G4]: the unregularised "
                "problem admits grid-scale oscillations that reproduce the density exactly"
            )
        if self.density_floor <= 0.0:
            raise ValueError("density_floor must be positive")


@dataclass(frozen=True, slots=True)
class ReferenceValue:
    """One published or analytic number that a scenario is checked against.

    ``value`` is in ``units``, ``"Ha"`` unless stated; ``method`` (``"PBE/unc-aug-cc-pV5Z"``) is
    mandatory for ``LITERATURE`` and ``CROSS_CODE``; ``tolerance`` is the agreement required of the
    solver, from the reference's uncertainty and the discretisation budget, never from what the
    solver achieves (gate G4.7).
    """

    value: float
    units: str = "Ha"
    kind: ReferenceKind = ReferenceKind.LITERATURE
    method: str = ""
    citation: str = ""
    tolerance: float = 0.0

    def __post_init__(self) -> None:
        """Reject a reference that cannot be audited: a published number without its method."""
        if self.kind is not ReferenceKind.ANALYTIC and not self.method:
            raise ValueError(
                "a literature or cross-code reference must record the method and basis that "
                "produced it; a bare number is not a reference"
            )
        if self.tolerance < 0.0:
            raise ValueError("tolerance must be non-negative")


@dataclass(frozen=True, slots=True)
class NLDFSpec:
    """Non-local density features: ``n_exponents`` costs that many FFT pairs per SCF step."""

    kind: NonlocalFeatureKind = NonlocalFeatureKind.NONE
    n_exponents: int = 0
    l_max: int = 0
    exponent_min: float = 0.05
    exponent_max: float = 50.0
    module_path: str | None = None

    def __post_init__(self) -> None:
        """Reject a configuration that asks for features without the means to compute them."""
        if self.kind is not NonlocalFeatureKind.NONE and self.n_exponents < 1:
            raise ValueError("non-local features require n_exponents >= 1")
        if self.exponent_min <= 0.0 or self.exponent_max <= self.exponent_min:
            raise ValueError("require 0 < exponent_min < exponent_max")


@dataclass(frozen=True, slots=True)
class XCSpec:
    """Which functional: a native torch ``name``, gate G0.6's libxc oracle, a learned module."""

    name: str = "pbe"
    rung: XCRung = XCRung.GGA
    libxc_reference: tuple[str, ...] = ("gga_x_pbe", "gga_c_pbe")
    module_path: str | None = None
    dispersion: str | None = None
    nonlocal_features: NLDFSpec = field(default_factory=NLDFSpec)


@dataclass(frozen=True, slots=True)
class PseudoSpec:
    """Which pseudopotential table to draw from: norm-conserving ONCV [E1], KB form [E4]."""

    table: str = "sg15"
    directory: str = "pseudos/sg15"
    functional: str = "pbe"
    relativistic: str = "scalar"


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    """One system and what is measured on it: the unit of the corpus sweep and the gate suite."""

    scenario_id: str
    structure: AtomicStructure = field(default_factory=AtomicStructure)
    xc: XCSpec = field(default_factory=XCSpec)
    external: ExternalPotentialSpec = field(default_factory=ExternalPotentialSpec)
    electrons: ElectronSpec = field(default_factory=ElectronSpec)
    pseudo: PseudoSpec | None = None
    gates: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()
    reference: Mapping[str, ReferenceValue] = field(default_factory=dict)
    reference_citation: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        """Reject a scenario whose external potential and pseudopotential table disagree."""
        if self.external.kind is ExternalPotentialKind.PSEUDOPOTENTIAL and self.pseudo is None:
            raise ValueError("the pseudopotential path requires a PseudoSpec")
        if self.external.kind is not ExternalPotentialKind.PSEUDOPOTENTIAL and self.pseudo is not None:
            raise ValueError(
                "a PseudoSpec was supplied for a scenario whose external potential is not a "
                "pseudopotential; one of the two is a mistake and guessing which would be worse"
            )

    @property
    def all_electron(self) -> bool:
        """Whether no pseudopotential is present, so absolute energies are comparable (G4.7)."""
        return self.external.kind is not ExternalPotentialKind.PSEUDOPOTENTIAL


# --- Result types ---


@dataclass(slots=True)
class XCOutput:
    """Return value of an exchange--correlation functional evaluation.

    ``e_xc`` is the energy density per unit volume, ``(n_pts,)``, totalling ``sum(w * e_xc)``.
    ``v_xc`` ``(n_spin, n_pts)`` is its derivative with respect to the density, checked by gate G0.5
    against both a finite difference and autograd. ``v_sigma`` ``(n_sigma, n_pts)`` and ``v_tau``
    ``(n_spin, n_pts)`` follow from rungs 2 and 3, and are ``None`` below them.
    """

    e_xc: Tensor
    v_xc: Tensor
    v_sigma: Tensor | None = None
    v_tau: Tensor | None = None
    v_lapl: Tensor | None = None
    v_nldf: Tensor | None = None
    """Derivative per non-local feature, ``(n_feat, n_pts)``; adjoint: ``apply_potential``."""


@dataclass(slots=True)
class EigenResult:
    """Output of one eigensolver call."""

    eigenvalues: Tensor
    eigenvectors: Tensor
    residuals: Tensor
    n_iterations: int
    converged: bool


@dataclass(slots=True)
class EnergyBreakdown:
    """Term-by-term decomposition of the total energy, in Hartree.

    It closes to round-off: gate G2.9 thresholds ``|total - sum(terms)|`` at 1e-12 Ha.
    ``harris_foulkes`` [F22] differs from ``total`` at second order in the density error (G2.1).
    """

    total: float
    kinetic: float
    external: float
    hartree: float
    xc: float
    nonlocal_ps: float
    ion_ion: float
    dispersion: float = 0.0
    entropy: float = 0.0
    harris_foulkes: float | None = None

    def closure_error(self) -> float:
        """Return ``|total - sum(terms)|``, the quantity gate G2.9 thresholds at 1e-12 Ha."""
        parts = (
            self.kinetic
            + self.external
            + self.hartree
            + self.xc
            + self.nonlocal_ps
            + self.ion_ion
            + self.dispersion
            + self.entropy
        )
        return abs(self.total - parts)


@dataclass(slots=True)
class SCFTrajectory:
    """The full history of the SCF iteration; kept in every record, not only on failure (G2.3)."""

    energies: list[float] = field(default_factory=list)
    residual_norms: list[float] = field(default_factory=list)
    density_changes: list[float] = field(default_factory=list)
    eigenvalue_history: list[Any] = field(default_factory=list)
    mixing_events: list[str] = field(default_factory=list)
    fallback_events: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SCFResult:
    """Converged (or abandoned) self-consistent solution."""

    converged: bool
    n_iterations: int
    density: Tensor
    eigenvalues: Tensor
    occupations: Tensor
    energies: EnergyBreakdown
    trajectory: SCFTrajectory
    v_hartree: Tensor | None = None
    v_xc: Tensor | None = None
    orbitals: Tensor | None = None
    forces: Tensor | None = None


# --- Gates ---


@dataclass(slots=True)
class Diagnostic:
    """A measured physical quantity that is recorded but never thresholded (D-20).

    A gate asserts that the solver is correct; a diagnostic measures how a *functional* behaves,
    where the interesting values are the wrong ones. ``per_point`` holds the grid-resolved form,
    for one electron ``n(r) v_H(r)/2 + e_xc(r)``, which integrates to zero for the exact functional.
    """

    diagnostic_id: str
    name: str
    value: float
    units: str = ""
    exact_value: float | None = None
    citation: str = ""
    per_point: Tensor | None = None
    detail: Mapping[str, Any] = field(default_factory=dict)

    @property
    def deviation(self) -> float | None:
        """Signed deviation from the exact value, when the exact value is known analytically."""
        return None if self.exact_value is None else self.value - self.exact_value


@dataclass(slots=True)
class DiagnosticReport:
    """All diagnostics measured for one run or scan."""

    results: list[Diagnostic] = field(default_factory=list)

    def by_id(self, diagnostic_id: str) -> Diagnostic | None:
        """Return one diagnostic by identifier, or ``None`` if it was not measured."""
        return next((d for d in self.results if d.diagnostic_id == diagnostic_id), None)


@dataclass(slots=True)
class GateResult:
    """Outcome of one physics gate; measured values, not only verdicts, enter the record (G5.3)."""

    gate_id: str
    name: str
    verdict: GateVerdict
    measured: float
    threshold: float
    kind: GateKind
    citation: str = ""
    units: str = ""
    reason: str = ""
    detail: Mapping[str, Any] = field(default_factory=dict)

    @property
    def margin(self) -> float:
        """Fraction of the threshold consumed. Values above 0.9 are reported as ``MARGINAL``."""
        return float("inf") if self.threshold == 0.0 else abs(self.measured) / abs(self.threshold)


@dataclass(slots=True)
class GateReport:
    """The complete gate table for one run, and the status it implies."""

    results: list[GateResult] = field(default_factory=list)

    @property
    def status(self) -> RunStatus:
        """Derive the record status from the verdicts: ``FAIL > DEFERRED > MARGINAL > VALID``."""
        if any(r.verdict is GateVerdict.FAIL for r in self.results):
            return RunStatus.INVALID
        if any(r.verdict is GateVerdict.DEFERRED for r in self.results):
            return RunStatus.DEFERRED
        if any(r.verdict is GateVerdict.MARGINAL for r in self.results):
            return RunStatus.MARGINAL
        return RunStatus.VALID

    @property
    def deferred_gates(self) -> tuple[str, ...]:
        """Identifiers of the gates this build could not evaluate, in report order."""
        return tuple(r.gate_id for r in self.results if r.verdict is GateVerdict.DEFERRED)


@dataclass(slots=True)
class Provenance:
    """Everything needed to reproduce a record (G5.1); ``git_dirty`` is recorded, not barred."""

    contract_version: str = CONTRACT_VERSION
    git_sha: str = ""
    git_dirty: bool = False
    config_hash: str = ""
    package_versions: Mapping[str, str] = field(default_factory=dict)
    device_name: str = ""
    driver_version: str = ""
    precision_policy: Mapping[str, str] = field(default_factory=dict)
    hostname: str = ""
    timestamp_utc: str = ""
    wall_time_s: float = 0.0
    device: str = ""
    """The torch device the run executed on (``cpu``, ``cuda:0``)."""
    device_capability: str = ""
    """CUDA compute capability (``7.5``), or ``n/a`` on the CPU."""
    torch_cuda_version: str = ""
    """``torch.version.cuda`` of the build; ``none`` for a CPU-only build."""
    cuda_driver_version: str = ""
    """NVIDIA driver version; ``n/a`` on the CPU, ``unavailable`` if it cannot be read."""
    deterministic_algorithms: str = ""
    """torch's deterministic-algorithm mode during the run: ``strict``, ``warn_only`` or ``off``."""
    cublas_workspace_config: str = ""
    """``CUBLAS_WORKSPACE_CONFIG`` in force, or ``unset``."""
    gpu_peak_bytes: int | None = None
    """``torch.cuda.max_memory_allocated`` over the solve; ``None`` off CUDA."""


@dataclass(slots=True)
class RunArtifact:
    """The complete, self-describing record of one solve; its verdict travels with its numbers."""

    run_id: str
    scenario: ScenarioSpec
    numerics: NumericsConfig
    provenance: Provenance
    result: SCFResult | None = None
    gates: GateReport = field(default_factory=GateReport)
    diagnostics: DiagnosticReport = field(default_factory=DiagnosticReport)
    measurements: Mapping[str, Any] = field(default_factory=dict)
    """Readings taken *during* the solve (G0.3, G0.4, G0.8), unrecoverable from a final density."""

    status: RunStatus = RunStatus.ERROR
    error_message: str = ""


@dataclass(frozen=True, slots=True)
class ScanSpec:
    """A family of runs that only means something as a curve.

    ``parameter`` names the swept quantity and ``values`` lists the points; the exact curve comes
    from ``integer_endpoints``, E(N) being linear between integers [A21] and E(N, M) flat [A23].
    """

    scan_id: str
    kind: ScanKind
    base: ScenarioSpec
    parameter: str
    values: tuple[float, ...]
    integer_endpoints: tuple[float, ...] = ()
    notes: str = ""


@dataclass(slots=True)
class ScanArtifact:
    """A completed scan: its member runs plus the diagnostics that exist only for the curve."""

    scan_id: str
    spec: ScanSpec
    runs: list[RunArtifact] = field(default_factory=list)
    diagnostics: DiagnosticReport = field(default_factory=DiagnosticReport)
    status: RunStatus = RunStatus.ERROR

    @property
    def complete(self) -> bool:
        """Whether every requested point produced a usable run."""
        return len(self.runs) == len(self.spec.values) and all(
            r.status in (RunStatus.VALID, RunStatus.MARGINAL) for r in self.runs
        )


# --- Component protocols ---


@runtime_checkable
class GridProtocol(Protocol):
    """Masked real-space grid: flat-index mapping, quadrature weights and the FFT machinery."""

    @property
    def n_points(self) -> int:
        """Number of points inside the mask."""

    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape of the enclosing box in grid points."""

    @property
    def spacing(self) -> float:
        """Uniform spacing in bohr."""

    @property
    def weights(self) -> Tensor:
        """Quadrature weights, shape ``(n_pts,)``. For a uniform grid these are all ``h**3``."""

    def integrate(self, f: Tensor) -> Tensor:
        """Integrate a field over the domain: ``sum(w * f)``, accumulated in float64."""

    def gradient(self, f: Tensor) -> Tensor:
        """Return the Cartesian gradient, shape ``(..., 3, n_pts)``, by finite differences."""

    def laplacian(self, f: Tensor) -> Tensor:
        """Apply the order-``2p`` finite-difference Laplacian [D2]. Gate G0.1 checks its order."""

    def scatter_to_box(self, f: Tensor) -> Tensor:
        """Expand a masked field into the full box, zero-filled. Used for FFTs and for output."""

    def gather_from_box(self, f: Tensor) -> Tensor:
        """Restrict a full-box field to the masked domain."""


@runtime_checkable
class PseudopotentialProtocol(Protocol):
    """Norm-conserving ONCV pseudopotential in Kleinman-Bylander separable form [E1], [E4]."""

    @property
    def z_valence(self) -> float:
        """Valence charge carried by this species."""

    def local_potential(self, grid: GridProtocol, position: Tensor) -> Tensor:
        """Return the local part on the grid, shape ``(n_pts,)``, in Hartree."""

    def projectors(self, grid: GridProtocol, position: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(beta, d_ij)``: KB projectors and coupling, double-gridded [D16], filtered."""

    def reference_eigenvalues(self) -> Mapping[str, float]:
        """Return the reference valence eigenvalues shipped in the file; gate G4.1 checks them."""


@runtime_checkable
class NonlocalFeatureProtocol(Protocol):
    """Computes non-local density features [C26] and maps their derivatives to a potential."""

    @property
    def n_features(self) -> int:
        """Number of features produced per spin channel."""

    def compute(self, density: Tensor, tau: Tensor | None, grid: GridProtocol) -> Tensor:
        """Return the features, shape ``(n_feat, n_pts)``, as convolutions in fixed exponents."""

    def apply_potential(self, v_nldf: Tensor, grid: GridProtocol) -> Tensor:
        """Map feature derivatives back to a local potential, ``(n_pts,)``: adjoint of compute."""


@runtime_checkable
class XCFunctionalProtocol(Protocol):
    """An exchange--correlation functional: the interface the downstream project connects across.

    A learned functional -- DM21 [C1], Skala [C2], or one trained on this corpus -- implements it
    and is substituted with no change to the SCF loop. Implementations are torch and differentiable
    with respect to their inputs: with no closed-form ``v_xc``, the potentials in :class:`XCOutput`
    come from ``torch.autograd`` (gate G0.5). Differentiating *through* the SCF is downstream work.
    """

    @property
    def rung(self) -> XCRung:
        """Rung of Jacob's ladder [A3], which determines the required ingredients."""

    @property
    def name(self) -> str:
        """Identifier written into the record."""

    def evaluate(
        self,
        density: Tensor,
        sigma: Tensor | None = None,
        tau: Tensor | None = None,
        lapl: Tensor | None = None,
        nldf: Tensor | None = None,
    ) -> XCOutput:
        """Evaluate the energy density and its derivatives.

        ``density``, ``tau`` and ``lapl`` have shape ``(n_spin, n_pts)``; ``sigma`` has
        ``(n_sigma, n_pts)``, ``n_sigma`` being 1 for ``n_spin == 1`` and 3 for ``n_spin == 2``.
        ``sigma`` is required from rung 2 and ``tau`` from rung 3; ``nldf`` ``(n_feat, n_pts)``
        comes from a :class:`NonlocalFeatureProtocol`, ``None`` for semi-local functionals.
        """


@runtime_checkable
class PoissonSolverProtocol(Protocol):
    """Open-boundary solver for the Hartree potential."""

    def solve(self, density: Tensor, grid: GridProtocol) -> Tensor:
        """Return ``v_H`` with free boundary conditions, shape ``(n_pts,)``.

        Gate G0.2 checks it against a Gaussian charge: ``V(r) = Q erf(r / (sqrt(2) s)) / r`` with
        Hartree energy ``Q^2 / (2 sqrt(pi) s)``.
        """

    def energy(self, density: Tensor, v_hartree: Tensor, grid: GridProtocol) -> float:
        """Return the Hartree energy ``0.5 * integral(n * v_H)`` in Hartree."""


@runtime_checkable
class HamiltonianProtocol(Protocol):
    """The Kohn--Sham Hamiltonian, never materialised: only its action on a block of orbitals."""

    def apply(self, psi: Tensor) -> Tensor:
        """Return ``H psi`` for a block of orbitals, shape ``(n_spin, n_states, n_pts)``."""

    def update_potentials(self, density: Tensor) -> None:
        """Recompute the density-dependent terms ``v_H`` and ``v_xc`` for a new density."""

    def spectral_bounds(self) -> tuple[float, float]:
        """Return estimated ``(lambda_min, lambda_max)`` for the Chebyshev filter [F2].

        Use a few Lanczos steps, not a fixed guess: an underestimated upper bound makes the filter
        amplify the wrong subspace, which looks like slow convergence rather than an error.
        """

    def local_potential(self) -> Tensor:
        """Return the current total local potential, for diagnostics and for the corpus record."""


@runtime_checkable
class EigensolverProtocol(Protocol):
    """Iterative solver for the lowest eigenpairs of a matrix-free Hamiltonian."""

    def solve(
        self,
        hamiltonian: HamiltonianProtocol,
        n_states: int,
        initial: Tensor | None = None,
    ) -> EigenResult:
        """Return the lowest ``n_states`` eigenpairs.

        Rayleigh--Ritz and orthonormalisation run in float64 whatever the hot-path precision: loss
        of block orthogonality breaks eigensolvers [I8] and gate G0.3 holds at 1e-10 always.
        """


@runtime_checkable
class MixerProtocol(Protocol):
    """Density mixing for the SCF iteration [F8]-[F11]."""

    def mix(self, n_in: Tensor, n_out: Tensor, iteration: int) -> Tensor:
        """Return the input density for the next iteration."""

    def reset(self) -> None:
        """Clear the history, for example after a fallback event."""

    @property
    def history_size(self) -> int:
        """Number of previous iterations currently retained."""


@runtime_checkable
class SCFStepProtocol(Protocol):
    """One application of the self-consistency map, as a pure function (D-47).

    The density is a fixed point ``n* = F(n*)``; mixing, damping and the fallback ladder change only
    the *path* to it (D-06, D-16) and do not enter ``(I - dF/dn) dn*/dtheta = dF/dtheta``, so the
    map lives apart from the loop that iterates it: no mixer state, no in-place mutation.
    """

    def step(
        self,
        density: Tensor,
        functional: XCFunctionalProtocol,
    ) -> tuple[Tensor, EigenResult, EnergyBreakdown]:
        """Return ``(n_out, eigen, energies)`` for one pass of the Kohn--Sham map at ``density``.

        ``n_out``, ``(n_spin, n_pts)``, is built from the occupied eigenvectors of the Hamiltonian
        constructed at the *input* density. No side effects: nothing passed in is mutated.
        """

    def residual(self, density: Tensor, functional: XCFunctionalProtocol) -> Tensor:
        """Return ``F(density) - density``, the quantity the mixer drives to zero (gate G3.3)."""


@runtime_checkable
class OccupationProtocol(Protocol):
    """Assignment of occupation numbers to eigenvalues."""

    def occupy(self, eigenvalues: Tensor, n_electrons: float) -> tuple[Tensor, float, float]:
        """Return ``(occupations, fermi_level, entropy_term)``.

        Occupations are float in every scheme. Smearing needs a bracketing search for the Fermi
        level, non-monotonic smearing functions breaking bisection [F20].
        """


@runtime_checkable
class InitialGuessProtocol(Protocol):
    """Initial density for the SCF iteration."""

    def guess(self, structure: AtomicStructure, grid: GridProtocol) -> Tensor:
        """Return a starting density ``(n_spin, n_pts)`` normalised to N; default [F13] (G3.4)."""


@runtime_checkable
class GateProtocol(Protocol):
    """A single physics gate."""

    @property
    def gate_id(self) -> str:
        """Identifier such as ``"G2.7"``, matching ``docs/03_METHOD.md (Part C)``."""

    def applicable(self, artifact: RunArtifact) -> bool:
        """Whether this gate applies; if not, ``evaluate`` returns ``SKIPPED`` with a reason."""

    def evaluate(self, artifact: RunArtifact) -> GateResult:
        """Measure the quantity and return a verdict with the measured value and its threshold."""


@runtime_checkable
class ComponentGateProtocol(Protocol):
    """A gate on an operator at a numerics configuration rather than on a run (D-30, D-49)."""

    @property
    def gate_id(self) -> str:
        """Identifier such as ``"G0.1"``, matching ``docs/03_METHOD.md (Part C)``."""

    def evaluate(self, numerics: NumericsConfig) -> GateResult:
        """Measure the component property at this configuration and return a verdict."""


@runtime_checkable
class InversionProtocol(Protocol):
    """Recovers v_KS and v_xc from a given density (D-17); see ``potentials/v_xc_inverted``."""

    @property
    def kind(self) -> InversionKind:
        """Which inversion method this implementation provides."""

    def invert(
        self,
        density: Tensor,
        grid: GridProtocol,
        n_electrons: float,
        v_ext: Tensor,
        v_hartree: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return ``(v_ks, v_xc)`` for a target density, both of shape ``(n_spin, n_pts)``.

        ``v_xc`` is ``v_ks - v_ext - v_hartree``: pass the same v_ext and v_H the density came from,
        since a different one silently solves a different problem.
        """


@runtime_checkable
class DiagnosticProtocol(Protocol):
    """A measurement of how a *functional* behaves, recorded with no threshold (D-20)."""

    @property
    def diagnostic_id(self) -> str:
        """Identifier such as ``"D1.1"``, matching ``docs/01_PROJECT.md (Part C)``."""

    def applicable(self, artifact: RunArtifact) -> bool:
        """Whether this diagnostic is defined for this run."""

    def measure(self, artifact: RunArtifact) -> Diagnostic:
        """Measure the quantity, with its exact value attached where one is known analytically."""


@runtime_checkable
class ScanDiagnosticProtocol(Protocol):
    """A diagnostic that exists only for a curve: convexity of E(N) [A21], flat plane [A23]."""

    @property
    def diagnostic_id(self) -> str:
        """Identifier such as ``"D1.4"``, matching ``docs/01_PROJECT.md (Part C)``."""

    def measure(self, scan: ScanArtifact) -> Diagnostic:
        """Measure the curve-level quantity against the reference built from the integer endpoints."""


@runtime_checkable
class ScanRunnerProtocol(Protocol):
    """Executes a :class:`ScanSpec` and assembles the resulting :class:`ScanArtifact`."""

    def run(self, spec: ScanSpec, numerics: NumericsConfig) -> ScanArtifact:
        """Run every point of the scan, plus the integer endpoints, and attach curve diagnostics."""


@runtime_checkable
class SolverProtocol(Protocol):
    """The top-level entry point: scenario plus numerics in, gated artifact out."""

    def solve(self, scenario: ScenarioSpec, numerics: NumericsConfig) -> RunArtifact:
        """Run one calculation to convergence, evaluate the gates, and return the artifact.

        A physics failure never raises: an unconverged or gate-failing run returns an artifact with
        the right :class:`RunStatus` and a populated :class:`GateReport`.
        """


@runtime_checkable
class CorpusWriterProtocol(Protocol):
    """HDF5 corpus writer and reader."""

    def write(self, artifact: RunArtifact) -> str:
        """Append a record and return its group path."""

    def read(self, run_id: str) -> RunArtifact:
        """Read one record back."""

    def iterate(self, include_invalid: bool = False) -> Iterator[RunArtifact]:
        """Iterate over records, yielding only trusted ones unless asked otherwise (gate G5.4)."""


# --- On-disk layout ---

#: Diagnostic fields common to both layouts; the corpus adds ``per_point`` to them.
_DIAGNOSTIC_FIELDS: tuple[str, ...] = (
    "diagnostic_id",
    "name",
    "value",
    "exact_value",
    "deviation",
    "units",
    "citation",
)

#: Canonical HDF5 group layout of a corpus record, so that a reader written in another language can
#: be built from this file alone. ``docs/03_METHOD.md (Part A)`` section 7 gives the rationale
#: for each stored field.
CORPUS_LAYOUT: Mapping[str, tuple[str, ...]] = {
    "attrs": (
        "contract_version",
        "status",
        "scenario_id",
        "git_sha",
        "git_dirty",
        "config_hash",
        "package_versions",
        "device_name",
        "driver_version",
        "precision_policy",
        "hostname",
        "timestamp_utc",
        "wall_time_s",
        "device",
        "device_capability",
        "torch_cuda_version",
        "cuda_driver_version",
        "deterministic_algorithms",
        "cublas_workspace_config",
        "gpu_peak_bytes",
    ),
    "structure": ("numbers", "positions", "charge", "multiplicity"),
    "scenario": (
        "scenario_id",
        "xc_name",
        "xc_rung",
        "external_kind",
        "external_params",
        "n_electrons",
        "magnetisation",
        "n_spin",
        "pseudo_table",
        "all_electron",
    ),
    "reference": ("key", "value", "units", "kind", "method", "citation", "tolerance"),
    "measurements": ("key", "value"),
    "grid": ("origin", "spacing", "shape", "mask", "weights"),
    "density": ("n", "grad_n", "tau", "lapl_n"),
    "potentials": ("v_ext", "v_hartree", "v_xc", "v_nonlocal_diag", "v_ks_inverted", "v_xc_inverted"),
    "orbitals": ("eigenvalues", "occupations", "psi"),
    "energies": (
        "total",
        "kinetic",
        "external",
        "hartree",
        "xc",
        "nonlocal_ps",
        "ion_ion",
        "dispersion",
        "entropy",
        "harris_foulkes",
    ),
    "forces": ("forces",),
    "scf": (
        "energies",
        "residual_norms",
        "density_changes",
        "mixing_events",
        "fallback_events",
    ),
    "gates": (
        "gate_id",
        "name",
        "verdict",
        "measured",
        "threshold",
        "kind",
        "citation",
        "units",
        "reason",
    ),
    "diagnostics": _DIAGNOSTIC_FIELDS + ("per_point",),
}

#: Layout of a scan record, stored alongside run records in the same file under ``/scan_<id>/``.
SCAN_LAYOUT: Mapping[str, tuple[str, ...]] = {
    "attrs": ("contract_version", "scan_id", "kind", "parameter", "status", "complete"),
    "spec": ("values", "integer_endpoints", "base_scenario_id"),
    "members": ("run_ids",),
    "curve": ("parameter_values", "total_energies", "exact_reference"),
    "diagnostics": _DIAGNOSTIC_FIELDS,
}
