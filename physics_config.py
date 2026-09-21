"""The scenario registry -- "what to solve". Never contains a grid spacing (Q1.5).

Every reference here is analytic or cited; a wanted but untranscribed number is *absent* and the
gate that needs it reports SKIPPED with a reason (G5.3). Phase 1 is exactly ``Z <= 2`` and
``N <= 2``. Why each scenario is in the set: ``docs/03_METHOD.md (Part E)``.
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

import math
import pathlib
from typing import Iterator, Mapping

import yaml

from contract import (
    AtomicStructure,
    ElectronSpec,
    ExternalPotentialKind,
    ExternalPotentialSpec,
    ReferenceKind,
    ReferenceValue,
    ScenarioSpec,
    XCRung,
    XCSpec,
)

from cdft.structure import structure_from_symbols

__all__ = [
    "ScenarioRegistry",
    "REGISTRY",
    "load_scenarios",
    "hydrogenic",
    "harmonic_well",
    "NONINTERACTING_XC",
    "lda_functional",
    "pbe_functional",
    "interacting",
    "QUICK_PROFILE_SCENARIOS",
]

#: The marker for a bare-potential (non-interacting) scenario: no Hartree term, no functional
#: (D-53). Rung LDA is a placeholder; the enumeration has no rung for "none".
NONINTERACTING_XC = XCSpec(name="none", rung=XCRung.LDA, libxc_reference=())


def lda_functional(correlation: str = "vwn") -> XCSpec:
    """Slater exchange with one of the LDA correlation parametrisations (``vwn``, ``pw92``, ``pz81``).

    ``vwn`` by default: it is what NIST SRD 141 used, so G4.7 compares like with like.
    """
    libxc = {"vwn": "lda_c_vwn", "pw92": "lda_c_pw", "pz81": "lda_c_pz"}[correlation]
    return XCSpec(name=f"lda_{correlation}", rung=XCRung.LDA, libxc_reference=("lda_x", libxc))


def pbe_functional() -> XCSpec:
    """PBE exchange and correlation [A4]."""
    return XCSpec(name="pbe", rung=XCRung.GGA, libxc_reference=("gga_x_pbe", "gga_c_pbe"))


# ---- Builders --------------------------------------------------------------------------------


def harmonic_well(omega: float = 1.0, n_electrons: float = 1.0, scenario_id: str = "") -> ScenarioSpec:
    """Isotropic 3-D harmonic well, ``V = omega^2 r^2 / 2`` -- rung L0, gate G1.2.

    Exact spectrum ``(n + 3/2) omega``, degeneracy ``(n+1)(n+2)/2``. No electrostatics, so a failure
    here cannot be blamed on anything but the Laplacian, the boundary or the eigensolver.
    """
    spec_id = scenario_id or f"harmonic_w{omega:g}"
    return ScenarioSpec(
        scenario_id=spec_id,
        structure=AtomicStructure(label=spec_id),
        xc=NONINTERACTING_XC,
        external=ExternalPotentialSpec(kind=ExternalPotentialKind.HARMONIC, omega=omega),
        electrons=ElectronSpec(n_electrons=n_electrons),
        gates=("G1.2", "G1.4", "G2.4", "G2.5", "G0.3", "G0.4", "G0.8"),
        reference={
            "eigenvalue_0": ReferenceValue(
                value=1.5 * omega, kind=ReferenceKind.ANALYTIC, tolerance=1.0e-6
            ),
            "eigenvalue_1": ReferenceValue(
                value=2.5 * omega, kind=ReferenceKind.ANALYTIC, tolerance=1.0e-6
            ),
            "eigenvalue_4": ReferenceValue(
                value=3.5 * omega, kind=ReferenceKind.ANALYTIC, tolerance=1.0e-6
            ),
        },
        notes="Rung L0. Non-interacting; Hartree and XC are switched off by the gate harness.",
    )


def particle_in_box(length: float = 10.0, scenario_id: str = "") -> ScenarioSpec:
    """Particle in a cubic infinite well of edge ``length`` bohr -- rung L1, gate G1.3.

    Exact spectrum ``pi^2 (nx^2 + ny^2 + nz^2) / (2 L^2)``. Tests the boundary treatment in
    isolation, hence ``BoundaryMode.ODD_REFLECTION``: every sine eigenfunction is odd about a wall.
    """
    spec_id = scenario_id or f"box_L{length:g}"
    base = math.pi**2 / (2.0 * length**2)
    return ScenarioSpec(
        scenario_id=spec_id,
        structure=AtomicStructure(label=spec_id),
        xc=NONINTERACTING_XC,
        external=ExternalPotentialSpec(
            kind=ExternalPotentialKind.PARTICLE_IN_BOX, box_length=length
        ),
        electrons=ElectronSpec(n_electrons=1.0),
        gates=("G1.3", "G2.4", "G0.3", "G0.4", "G0.8"),
        reference={
            "eigenvalue_0": ReferenceValue(
                value=3.0 * base, kind=ReferenceKind.ANALYTIC, tolerance=1.0e-5
            ),
            "eigenvalue_1": ReferenceValue(
                value=6.0 * base, kind=ReferenceKind.ANALYTIC, tolerance=1.0e-5
            ),
        },
        notes="Rung L1. Tolerance on G1.3 is relative, per docs/03_METHOD.md (Part C).",
    )


def hydrogenic(
    charge: float = 1.0, symbol: str = "H", scenario_id: str = "", n_electrons: float = 1.0
) -> ScenarioSpec:
    """One-electron atom with a bare -Z/r nucleus: H (Z=1), He+ (Z=2).

    Exact spectrum ``-Z^2 / (2 n^2)``, so it needs no literature; the first test of the
    all-electron path (D-23). Cusp-factorised (D-35) the ground state is exact to round-off, so the
    tolerances below are gross-regression guards and G4.8 plus the golden layer carry the tight
    checks. With one electron the exact functional gives ``E_H + E_xc = 0`` (diagnostic D1.1).
    """
    spec_id = scenario_id or f"{symbol.lower()}_Z{charge:g}"
    return ScenarioSpec(
        scenario_id=spec_id,
        structure=structure_from_symbols([symbol], [(0.0, 0.0, 0.0)], units="bohr", label=spec_id),
        xc=NONINTERACTING_XC,
        external=ExternalPotentialSpec(
            kind=ExternalPotentialKind.NUCLEAR_COULOMB, charges=(charge,)
        ),
        electrons=ElectronSpec(n_electrons=n_electrons),
        gates=("G1.13", "G2.4", "G2.5", "G0.3", "G0.4", "G0.8"),
        diagnostics=("D1.1",),
        reference={
            "eigenvalue_0": ReferenceValue(
                value=-0.5 * charge**2,
                kind=ReferenceKind.ANALYTIC,
                citation="hydrogenic spectrum",
                tolerance=1.0e-4,
            ),
            "eigenvalue_1": ReferenceValue(
                value=-0.125 * charge**2,
                kind=ReferenceKind.ANALYTIC,
                citation="hydrogenic spectrum",
                tolerance=1.0e-3,
            ),
        },
        notes=(
            "All-electron path (D-23), point nucleus, cusp-factorised (D-35): exact to round-off, "
            "so the reference tolerances are regression guards rather than accuracy claims."
        ),
    )


def h2_plus(bond_length: float = 2.0, scenario_id: str = "") -> ScenarioSpec:
    """H2+ at a fixed bond length in bohr: one electron, two protons -- rung L7.

    The self-interaction reference system (``docs/01_PROJECT.md (Part C)``). G4.5 requires the
    LDA/PBE failure to be reproduced: getting stretched H2+ *right* means a bug in Hartree (D-12).
    """
    spec_id = scenario_id or f"h2plus_R{bond_length:g}"
    half = bond_length / 2.0
    structure = structure_from_symbols(
        ["H", "H"], [(0.0, 0.0, -half), (0.0, 0.0, half)], units="bohr", charge=1.0, label=spec_id
    )
    reference: dict[str, ReferenceValue] = {}
    if abs(bond_length - 2.0) < 1.0e-12:
        # Separable in prolate spheroidal coordinates: the root of a transcendental equation, not a
        # fitted number. The eighth decimal is deliberately not claimed.
        reference["eigenvalue_0"] = ReferenceValue(
            value=-1.1026342,
            kind=ReferenceKind.ANALYTIC,
            citation="two-centre Coulomb problem, prolate spheroidal separation",
            tolerance=1.0e-4,
        )
        reference["total_energy"] = ReferenceValue(
            value=-0.6026342,
            kind=ReferenceKind.ANALYTIC,
            citation="two-centre Coulomb problem; electronic term plus 1/R proton repulsion",
            tolerance=1.0e-4,
        )
    return ScenarioSpec(
        scenario_id=spec_id,
        structure=structure,
        xc=NONINTERACTING_XC,
        external=ExternalPotentialSpec(
            kind=ExternalPotentialKind.NUCLEAR_COULOMB, charges=(1.0, 1.0)
        ),
        electrons=ElectronSpec(n_electrons=1.0),
        gates=("G2.4", "G2.5", "G0.3", "G0.4", "G0.8"),
        diagnostics=("D1.1", "D1.8"),
        reference=reference,
        notes="Rung L7. The self-interaction reference system; see docs/01_PROJECT.md (Part C).",
    )


def helium(scenario_id: str = "he_atom") -> ScenarioSpec:
    """Neutral helium, all-electron: the two-electron endpoint of Phase 1.

    Its exact non-relativistic energy is known to more digits than any functional needs, so the
    *functional* error is read off directly. No Kohn--Sham reference is attached until a published
    (functional, basis) pair is transcribed with its source; G4.7 reports SKIPPED meanwhile.
    """
    return ScenarioSpec(
        scenario_id=scenario_id,
        structure=structure_from_symbols(["He"], [(0.0, 0.0, 0.0)], units="bohr", label="He"),
        xc=NONINTERACTING_XC,
        external=ExternalPotentialSpec(kind=ExternalPotentialKind.NUCLEAR_COULOMB, charges=(2.0,)),
        electrons=ElectronSpec(n_electrons=2.0),
        gates=("G1.5", "G2.1", "G2.4", "G2.9", "G3.3", "G3.4", "G0.3", "G0.4", "G0.8"),
        diagnostics=("D1.2", "D1.5", "D1.6", "D1.7"),
        reference={
            "exact_nonrelativistic": ReferenceValue(
                value=-2.903724377034,
                kind=ReferenceKind.LITERATURE,
                method="exact non-relativistic, infinite nuclear mass (variational, Hylleraas-type)",
                citation="standard helium reference energy; transcribe the exact source before use",
                tolerance=1.0e-9,
            )
        },
        notes=(
            "Two-electron closed shell: the single occupied spatial orbital makes the Kohn-Sham "
            "inversion exact in closed form (InversionKind.EXACT_ONE_ORBITAL, D-17)."
        ),
    )


def hydrogen_molecule(bond_length: float = 1.4, scenario_id: str = "") -> ScenarioSpec:
    """H2 at a fixed bond length in bohr -- rung L7, the flat-plane system of [A23].

    No energy reference: H2 at a given functional is the quantity being measured, and its exact
    energy is a correlated-method number this solver cannot produce (O-5). What it carries instead
    is scan membership -- by [A21] and [A23] the exact ``E(N, M)`` surface follows from the integer
    endpoints.
    """
    spec_id = scenario_id or f"h2_R{bond_length:g}"
    half = bond_length / 2.0
    return ScenarioSpec(
        scenario_id=spec_id,
        structure=structure_from_symbols(
            ["H", "H"], [(0.0, 0.0, -half), (0.0, 0.0, half)], units="bohr", label=spec_id
        ),
        xc=NONINTERACTING_XC,
        external=ExternalPotentialSpec(
            kind=ExternalPotentialKind.NUCLEAR_COULOMB, charges=(1.0, 1.0)
        ),
        electrons=ElectronSpec(n_electrons=2.0),
        gates=("G1.5", "G2.1", "G2.4", "G2.9", "G3.3", "G0.3", "G0.4", "G0.8"),
        diagnostics=("D1.2", "D1.3", "D1.4", "D1.8"),
        notes="Rung L7; the flat-plane demonstration system of [A23].",
    )


def interacting(
    base: ScenarioSpec,
    xc: XCSpec,
    scenario_id: str,
    *,
    n_electrons: float | None = None,
    gates: tuple[str, ...] | None = None,
    extra_gates: tuple[str, ...] = (),
    diagnostics: tuple[str, ...] | None = None,
    reference: Mapping[str, ReferenceValue] | None = None,
    notes: str = "",
) -> ScenarioSpec:
    """Return ``base`` solved self-consistently with Hartree and the functional ``xc``.

    Wrapping a bare-nucleus eigenproblem with a functional is what switches the solver to
    :mod:`cdft.scf.loop` (``cdft.scf.solve.is_interacting``). NIST SRD 141 values are read from
    :mod:`cdft.reference.literature` by gate G4.7, never copied here.
    """
    electrons = base.electrons if n_electrons is None else ElectronSpec(
        n_electrons=n_electrons,
        magnetisation=base.electrons.magnetisation,
        spin_polarised=base.electrons.spin_polarised,
    )
    default_gates = (
        "G1.5", "G1.7", "G1.8", "G1.11", "G1.13", "G2.1", "G2.2", "G2.3", "G2.4", "G2.5", "G2.9",
        "G3.3", "G3.4", "G0.3", "G0.4", "G0.8",
    )
    return ScenarioSpec(
        scenario_id=scenario_id,
        structure=base.structure,
        xc=xc,
        external=base.external,
        electrons=electrons,
        gates=(default_gates if gates is None else gates) + tuple(extra_gates),
        diagnostics=base.diagnostics if diagnostics is None else diagnostics,
        reference=dict(reference) if reference is not None else {},
        notes=notes or f"{base.scenario_id} solved self-consistently at {xc.name} (I3, D-53/D-54).",
    )


# ---- The registry ----------------------------------------------------------------------------


class ScenarioRegistry:
    """A named collection of scenarios, sweepable against any numerics configuration.

    Re-registering an identifier raises rather than replacing: the identifier is written into every
    record, and two systems sharing one would make the corpus unreadable after the fact.
    """

    def __init__(self) -> None:
        """Create an empty registry."""
        self._scenarios: dict[str, ScenarioSpec] = {}

    def register(self, scenario: ScenarioSpec) -> ScenarioSpec:
        """Add a scenario and return it."""
        if scenario.scenario_id in self._scenarios:
            raise ValueError(f"scenario_id {scenario.scenario_id!r} is already registered")
        self._scenarios[scenario.scenario_id] = scenario
        return scenario

    def __getitem__(self, scenario_id: str) -> ScenarioSpec:
        """Return one scenario by identifier."""
        if scenario_id not in self._scenarios:
            known = ", ".join(sorted(self._scenarios))
            raise KeyError(f"unknown scenario {scenario_id!r}. Registered: {known}")
        return self._scenarios[scenario_id]

    def __contains__(self, scenario_id: object) -> bool:
        """Whether an identifier is registered."""
        return scenario_id in self._scenarios

    def __iter__(self) -> Iterator[ScenarioSpec]:
        """Iterate over scenarios in registration order."""
        return iter(self._scenarios.values())

    def __len__(self) -> int:
        """Number of registered scenarios."""
        return len(self._scenarios)

    def ids(self) -> tuple[str, ...]:
        """Return the registered identifiers in registration order."""
        return tuple(self._scenarios)


def _default_registry() -> ScenarioRegistry:
    """Build the Phase 1 registry."""
    reg = ScenarioRegistry()
    # Analytic model systems: rungs L0 and L1.
    reg.register(harmonic_well(omega=1.0, scenario_id="harmonic_w1"))
    reg.register(particle_in_box(length=10.0, scenario_id="box_L10"))
    # All-electron one-electron systems: exact references, no functional involved.
    reg.register(hydrogenic(charge=1.0, symbol="H", scenario_id="h_atom"))
    reg.register(hydrogenic(charge=2.0, symbol="He", scenario_id="he_plus", n_electrons=1.0))
    reg.register(h2_plus(bond_length=2.0, scenario_id="h2plus_R2"))
    # Two-electron systems: the Phase 1 endpoint.
    reg.register(helium())
    reg.register(hydrogen_molecule(bond_length=1.4, scenario_id="h2_R1.4"))

    # The same systems with Hartree and a functional (D-53).
    lda = lda_functional("vwn")  # what NIST SRD 141 used, so G4.7 compares like with like
    pbe = pbe_functional()
    nist = ("G4.7",)
    reg.register(interacting(hydrogenic(1.0, "H"), lda, "h_atom_lda", extra_gates=nist, notes=(
        "Spin-restricted hydrogen at VWN LDA against NIST SRD 141 (-0.445671 Ha); the 0.054 Ha gap "
        "to -0.5 is the self-interaction error of unpolarised VWN (diagnostic D1.1). Box-limited "
        "at the production half-box of 7 bohr (eigenvalue -0.233 Ha); G3.2 measures it."
    )))
    reg.register(interacting(hydrogenic(2.0, "He", n_electrons=1.0), lda, "he_plus_lda", extra_gates=nist, notes=(
        "One electron at Z = 2, VWN LDA, against NIST SRD 141 (-1.861237 Ha)."
    )))
    reg.register(interacting(helium(), lda, "he_atom_lda", extra_gates=nist, notes=(
        "The Phase 1 endpoint at VWN LDA, against NIST SRD 141 (-2.834836 Ha, 1e-6). The "
        "refinement ladder to the NIST comparison is scripts/interacting_ladder.py."
    )))
    reg.register(interacting(helium(), pbe, "he_atom_pbe", extra_gates=("G4.7",), notes=(
        "Helium at PBE, against PySCF all-electron RKS at cc-pV5Z/cc-pV6Z computed by "
        "scripts/pyscf_reference.py (COMPUTED_ORACLE, O-12); G1.11 does not assert -2Z here (D-55)."
    )))
    reg.register(interacting(hydrogen_molecule(1.4), lda, "h2_R1.4_lda", extra_gates=("G4.7",), notes=(
        "H2 at R = 1.4 bohr, VWN LDA, against PySCF (scripts/pyscf_reference.py)."
    )))
    reg.register(interacting(hydrogen_molecule(1.4), pbe, "h2_R1.4_pbe", extra_gates=("G4.7",), notes=(
        "H2 at R = 1.4 bohr, PBE, against PySCF (scripts/pyscf_reference.py); the flat-plane system."
    )))
    reg.register(interacting(h2_plus(2.0), lda, "h2plus_R2_lda", notes=(
        "H2+ at R = 2, VWN LDA: the self-interaction reference (D-12). Its LDA energy is below the "
        "exact -0.6026 Ha by construction of the error it exists to show (G4.5, recorded at R = 8)."
    )))
    reg.register(interacting(h2_plus(8.0, scenario_id="h2plus_R8"), lda, "h2plus_R8_lda", gates=(
        "G1.5", "G1.8", "G2.1", "G2.3", "G2.4", "G2.9", "G3.3", "G4.5", "G0.3", "G0.4", "G0.8",
    ), notes=(
        "Stretched H2+: the textbook semi-local dissociation failure. G4.5 requires E(R = 8) to lie "
        "BELOW E(H) at the same functional -- a solver that gets this right has a bug in its "
        "Hartree term (D-12); the magnitude is diagnostic D1.8 and is never thresholded (D-20)."
    )))
    reg.register(interacting(hydrogenic(1.0, "H"), lda, "h_atom_N0.5_lda", n_electrons=0.5, gates=(
        "G1.5", "G2.1", "G2.3", "G2.4", "G2.9", "G3.3", "G0.3", "G0.4", "G0.8",
    ), notes=(
        "Hydrogen with half an electron at VWN LDA: the first point of diagnostic D1.2 (E(N) between "
        "the integers is exactly linear for the exact functional [A21]). Recorded, not compared."
    )))
    return reg


#: The default registry. Phase 1 in full: every system with ``Z <= 2`` and ``N <= 2``.
REGISTRY = _default_registry()

#: Scenarios the ``quick`` test-suite profile solves; the ``full`` profile solves every one. A
#: profile decides which checks run, never their settings. The inner loop takes the seven
#: non-interacting scenarios plus the two cheapest interacting ones that exercise every new code
#: path once (helium at LDA and at PBE, both on the 14-bohr box; the hydrogen-containing scenarios
#: need 21--29 bohr boxes under D-53 and cost 4--8x as much), and names the ones it skipped.
QUICK_PROFILE_SCENARIOS: tuple[str, ...] = (
    "harmonic_w1", "box_L10", "h_atom", "he_plus", "h2plus_R2", "he_atom", "h2_R1.4",
    "he_atom_lda", "he_atom_pbe",
)


def _reference_from_dict(key: str, data: Mapping[str, object]) -> ReferenceValue:
    """Build one :class:`~contract.ReferenceValue` from a YAML mapping, rejecting unknown keys."""
    allowed = {"value", "units", "kind", "method", "citation", "tolerance"}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"reference {key!r}: unknown key(s) {', '.join(sorted(unknown))}")
    return ReferenceValue(
        value=float(data["value"]),  # type: ignore[arg-type]
        units=str(data.get("units", "Ha")),
        kind=ReferenceKind(str(data.get("kind", "literature"))),
        method=str(data.get("method", "")),
        citation=str(data.get("citation", "")),
        tolerance=float(data.get("tolerance", 0.0)),  # type: ignore[arg-type]
    )


def load_scenarios(path: str | pathlib.Path) -> ScenarioRegistry:
    """Load scenarios from a YAML file into a fresh registry.

    The format mirrors :class:`~contract.ScenarioSpec` one-for-one; ``units: angstrom`` is
    converted on the way in and never seen again.
    """
    data = yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8")) or {}
    entries = data.get("scenarios", [])
    reg = ScenarioRegistry()
    for entry in entries:
        symbols = entry.get("symbols", [])
        positions = entry.get("positions", [])
        units = entry.get("units", "bohr")
        structure = (
            structure_from_symbols(
                symbols,
                positions,
                units=units,
                charge=float(entry.get("charge", 0.0)),
                multiplicity=int(entry.get("multiplicity", 1)),
                label=entry.get("label", entry["scenario_id"]),
            )
            if symbols
            else AtomicStructure(label=entry.get("label", entry["scenario_id"]))
        )
        ext = entry.get("external", {}) or {}
        external = ExternalPotentialSpec(
            kind=ExternalPotentialKind(ext.get("kind", "nuclear_coulomb")),
            charges=tuple(float(z) for z in ext.get("charges", ())),
            softening=float(ext.get("softening", 0.0)),
            omega=float(ext.get("omega", 1.0)),
            box_length=float(ext.get("box_length", 0.0)),
            depth=float(ext.get("depth", 0.0)),
            width=float(ext.get("width", 1.0)),
        )
        el = entry.get("electrons", {}) or {}
        electrons = ElectronSpec(
            n_electrons=None if el.get("n_electrons") is None else float(el["n_electrons"]),
            magnetisation=float(el.get("magnetisation", 0.0)),
            spin_polarised=bool(el.get("spin_polarised", False)),
        )
        xc_entry = entry.get("xc", {}) or {}
        xc = XCSpec(
            name=str(xc_entry.get("name", "pbe")),
            rung=XCRung[str(xc_entry.get("rung", "GGA")).upper()],
            libxc_reference=tuple(xc_entry.get("libxc_reference", ("gga_x_pbe", "gga_c_pbe"))),
        )
        reg.register(
            ScenarioSpec(
                scenario_id=entry["scenario_id"],
                structure=structure,
                xc=xc,
                external=external,
                electrons=electrons,
                gates=tuple(entry.get("gates", ())),
                diagnostics=tuple(entry.get("diagnostics", ())),
                reference={
                    key: _reference_from_dict(key, value)
                    for key, value in (entry.get("reference", {}) or {}).items()
                },
                reference_citation=str(entry.get("reference_citation", "")),
                notes=str(entry.get("notes", "")),
            )
        )
    return reg
