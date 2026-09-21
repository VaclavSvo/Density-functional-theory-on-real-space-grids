"""Transcribed published reference values, each with the provenance needed to check it.

A number in this project is analytic, computed by an oracle here, or transcribed here from a named
source; there is no "approximately known" category. Every entry carries its method, stated accuracy
and citation, and all were transcribed on :data:`TRANSCRIPTION_DATE`.

The NIST values answer a different problem -- Kohn--Sham with the VWN local functional -- so they
test whether we solve the same equations the same way, not whether the functional is right.
Gaussian-basis totals for He and H2 are deliberately absent (O-12). Sources and per-row notes:
``docs/03_METHOD.md (Part E)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

__all__ = [
    "LiteratureValue",
    "NIST_SRD141",
    "EXACT_NONRELATIVISTIC",
    "TWO_CENTRE_TABULATED",
    "TRANSCRIPTION_DATE",
    "lookup",
]

#: Date on which every value in this module was transcribed from its cited source.
TRANSCRIPTION_DATE = "2026-09-13"


@dataclass(frozen=True, slots=True)
class LiteratureValue:
    """One published number, with everything needed to decide whether it applies.

    ``value`` is exactly as printed in the source, never rounded here; ``units`` is ``"Ha"``
    throughout, explicit because a silent unit is how a Rydberg factor of two gets in; ``accuracy``
    is the source's own claim and is the floor on any tolerance built from the value, since a gate
    cannot demand agreement tighter than the reference claims.
    """

    value: float
    units: str
    quantity: str
    system: str
    method: str
    accuracy: float
    citation: str
    note: str = ""

    def __post_init__(self) -> None:
        """Reject an uncited entry, and any unit but Hartree."""
        if not self.citation:
            raise ValueError(f"{self.system}/{self.quantity}: a literature value needs a citation")
        if self.units != "Ha":
            raise ValueError(
                f"{self.system}/{self.quantity}: units are Hartree throughout this project; "
                f"convert at transcription time, not at comparison time"
            )


_NIST_CITE = (
    "S. Kotochigova, Z. H. Levine, E. L. Shirley, M. D. Stiles and C. W. Clark, "
    "'Atomic Reference Data for Electronic Structure Calculations', NIST Standard Reference "
    "Database 141, National Institute of Standards and Technology (2009), "
    "https://doi.org/10.18434/T4ZP4F"
)
_NIST_METHOD_LDA = (
    "Kohn-Sham LDA, exchange-correlation functional of Vosko, Wilk and Nusair (VWN), "
    "non-relativistic, spin-unpolarised, point nucleus, radial solver"
)
_NIST_METHOD_LSD = (
    "Kohn-Sham LSD, exchange-correlation functional of Vosko, Wilk and Nusair (VWN), "
    "non-relativistic, spin-POLARISED, point nucleus, radial solver"
)
#: The database's own stated absolute accuracy in the total energy.
_NIST_ACCURACY = 1.0e-6


def _nist(system: str, quantity: str, value: float, spin: bool, note: str = "") -> LiteratureValue:
    """Build one NIST SRD 141 entry, so the shared fields cannot drift between rows."""
    return LiteratureValue(
        value=value,
        units="Ha",
        quantity=quantity,
        system=system,
        method=_NIST_METHOD_LSD if spin else _NIST_METHOD_LDA,
        accuracy=_NIST_ACCURACY,
        citation=_NIST_CITE,
        note=note,
    )


#: Kohn-Sham LDA and LSD atomic reference data, transcribed from NIST SRD 141; internal
#: consistency is checked by :func:`decomposition_residual`.
NIST_SRD141: Mapping[str, LiteratureValue] = {
    # hydrogen, Z = 1, one electron
    "H/lda/total_energy": _nist(
        "H", "total_energy", -0.445671, spin=False,
        note="one electron: the exact functional would give -0.5 exactly. The 0.054 Ha gap is pure "
             "self-interaction error of spin-unpolarised VWN and is diagnostic D1.1's published value.",
    ),
    "H/lda/kinetic_energy": _nist("H", "kinetic_energy", 0.425027, spin=False),
    "H/lda/hartree_energy": _nist("H", "hartree_energy", 0.282827, spin=False),
    "H/lda/external_energy": _nist("H", "external_energy", -0.920999, spin=False),
    "H/lda/xc_energy": _nist("H", "xc_energy", -0.232525, spin=False),
    "H/lda/eigenvalue_1s": _nist("H", "eigenvalue_1s", -0.233471, spin=False),
    "H/lsd/total_energy": _nist(
        "H", "total_energy", -0.478671, spin=True,
        note="spin-polarised: closer to -0.5 than LDA but still 0.021 Ha above it. The residual is "
             "the self-interaction error that spin polarisation does not remove.",
    ),
    "H/lsd/kinetic_energy": _nist("H", "kinetic_energy", 0.466643, spin=True),
    "H/lsd/hartree_energy": _nist("H", "hartree_energy", 0.298377, spin=True),
    "H/lsd/external_energy": _nist("H", "external_energy", -0.965619, spin=True),
    "H/lsd/xc_energy": _nist("H", "xc_energy", -0.278072, spin=True),
    "H/lsd/eigenvalue_1s_majority": _nist("H", "eigenvalue_1s_majority", -0.268975, spin=True),
    "H/lsd/eigenvalue_1s_minority": _nist("H", "eigenvalue_1s_minority", -0.100175, spin=True),
    # helium, Z = 2, two electrons
    "He/lda/total_energy": _nist(
        "He", "total_energy", -2.834836, spin=False,
        note="closed shell, so LDA and LSD coincide identically -- which is itself a check that the "
             "spin machinery is right when it arrives. 0.069 Ha above the exact -2.903724.",
    ),
    "He/lda/kinetic_energy": _nist("He", "kinetic_energy", 2.767922, spin=False),
    "He/lda/hartree_energy": _nist("He", "hartree_energy", 1.996120, spin=False),
    "He/lda/external_energy": _nist("He", "external_energy", -6.625564, spin=False),
    "He/lda/xc_energy": _nist("He", "xc_energy", -0.973314, spin=False),
    "He/lda/eigenvalue_1s": _nist("He", "eigenvalue_1s", -0.570425, spin=False),
    "He/lsd/total_energy": _nist(
        "He", "total_energy", -2.834836, spin=True,
        note="closed shell: every LSD term equals its LDA term, as the table prints (defect D-5 closed)",
    ),
    "He/lsd/kinetic_energy": _nist("He", "kinetic_energy", 2.767922, spin=True),
    "He/lsd/hartree_energy": _nist("He", "hartree_energy", 1.996120, spin=True),
    "He/lsd/external_energy": _nist("He", "external_energy", -6.625564, spin=True),
    "He/lsd/xc_energy": _nist("He", "xc_energy", -0.973314, spin=True),
    "He/lsd/eigenvalue_1s": _nist("He", "eigenvalue_1s", -0.570425, spin=True),
    # He+, Z = 2, one electron
    "He+/lda/total_energy": _nist(
        "He+", "total_energy", -1.861237, spin=False,
        note="one electron: exact answer -2.0. The 0.139 Ha gap is the Z-scaled self-interaction "
             "error and is why He+ is in the diagnostic set alongside H.",
    ),
    "He+/lda/kinetic_energy": _nist("He+", "kinetic_energy", 1.828737, spin=False),
    "He+/lda/hartree_energy": _nist("He+", "hartree_energy", 0.592268, spin=False),
    "He+/lda/external_energy": _nist("He+", "external_energy", -3.823853, spin=False),
    "He+/lda/xc_energy": _nist("He+", "xc_energy", -0.458389, spin=False),
    "He+/lda/eigenvalue_1s": _nist("He+", "eigenvalue_1s", -1.410933, spin=False),
    "He+/lsd/total_energy": _nist("He+", "total_energy", -1.941703, spin=True),
    "He+/lsd/kinetic_energy": _nist("He+", "kinetic_energy", 1.923780, spin=True),
    "He+/lsd/hartree_energy": _nist("He+", "hartree_energy", 0.609461, spin=True),
    "He+/lsd/external_energy": _nist("He+", "external_energy", -3.922578, spin=True),
    "He+/lsd/xc_energy": _nist("He+", "xc_energy", -0.552365, spin=True),
    "He+/lsd/eigenvalue_1s_majority": _nist("He+", "eigenvalue_1s_majority", -1.510389, spin=True),
    "He+/lsd/eigenvalue_1s_minority": _nist("He+", "eigenvalue_1s_minority", -1.051640, spin=True),
}


#: Exact non-relativistic energies: the answers no functional can improve on.
EXACT_NONRELATIVISTIC: Mapping[str, LiteratureValue] = {
    "He/total_energy": LiteratureValue(
        value=-2.90372437703411959831,
        units="Ha",
        quantity="total_energy",
        system="He",
        method=(
            "exact non-relativistic Schrodinger, infinite nuclear mass, clamped point nucleus; "
            "variational Hylleraas-type expansion"
        ),
        accuracy=1.0e-19,
        citation=(
            "H. Nakashima and H. Nakatsuji, J. Chem. Phys. 127, 224104 (2007), free iterative "
            "complement interaction; confirmed to 20 digits by C. Schwartz, Int. J. Mod. Phys. E 15, "
            "877 (2006), and reproduced to -2.9037243770341195983110 by J. S. Sims, B. Padhy and "
            "M. B. Ruiz, Int. J. Quantum Chem. 121, e26470 (2021), doi:10.1002/qua.26470"
        ),
        note=(
            "The correlated two-electron answer. No Kohn-Sham functional reproduces it and none is "
            "expected to: the gap to any functional's number IS that functional's error, which is "
            "the measurement this project is built to make. Never use it as a gate on the solver."
        ),
    ),
    "H/total_energy": LiteratureValue(
        value=-0.5,
        units="Ha",
        quantity="total_energy",
        system="H",
        method="exact non-relativistic, infinite nuclear mass; closed form -Z^2/2",
        accuracy=0.0,
        citation="hydrogenic spectrum, closed form; no transcription involved",
    ),
    "He+/total_energy": LiteratureValue(
        value=-2.0,
        units="Ha",
        quantity="total_energy",
        system="He+",
        method="exact non-relativistic, infinite nuclear mass; closed form -Z^2/2",
        accuracy=0.0,
        citation="hydrogenic spectrum, closed form; no transcription involved",
    ),
}


#: The classical two-centre tabulation, one published geometry, kept as a cross-check on
#: :mod:`cdft.reference.two_centre`, which computes the same quantity at any bond length.
TWO_CENTRE_TABULATED: Mapping[str, LiteratureValue] = {
    "H2+/R2.0/total_energy": LiteratureValue(
        value=-0.6026342144949,
        units="Ha",
        quantity="total_energy",
        system="H2+ R=2.0 bohr",
        method=(
            "exact one-electron two-centre problem, separation in prolate spheroidal coordinates; "
            "clamped nuclei, includes the 1/R proton repulsion"
        ),
        accuracy=1.0e-13,
        citation="M. M. Madsen and J. M. Peek, Atomic Data 2, 171 (1971)",
        note=(
            "Electronic energy alone is this minus 1/R = -1.1026342144949 Ha. R = 2.0 bohr is very "
            "close to, but not exactly, the equilibrium bond length."
        ),
    ),
}


_ALL: dict[str, LiteratureValue] = {}
for _source in (NIST_SRD141, EXACT_NONRELATIVISTIC, TWO_CENTRE_TABULATED):
    for _key, _entry in _source.items():
        _ALL[_key] = _entry


def lookup(key: str) -> LiteratureValue:
    """Return one transcribed value, raising with the available keys if it is not present.

    Raising rather than returning ``None``: a caller that silently skips a missing reference
    produces a report with one check fewer and no sign of it.
    """
    if key not in _ALL:
        raise KeyError(
            f"no transcribed literature value {key!r}. "
            f"Available: {', '.join(sorted(_ALL))}"
        )
    return _ALL[key]


def decomposition_residual(system: str, functional: str) -> float:
    """Return ``|E_tot - (E_kin + E_coul + E_enuc + E_xc)|`` for one NIST row, in Hartree.

    A transcription check only: a non-zero result means a digit was mistyped here. No test calls it
    (``docs/02_STATUS.md (Part C)`` D-5).
    """
    parts = ("kinetic_energy", "hartree_energy", "external_energy", "xc_energy")
    keys = [f"{system}/{functional}/{part}" for part in parts]
    missing = [key for key in keys if key not in NIST_SRD141]
    if missing:
        raise KeyError(f"incomplete NIST row for {system}/{functional}: missing {missing}")
    total = NIST_SRD141[f"{system}/{functional}/total_energy"].value
    return abs(total - sum(NIST_SRD141[key].value for key in keys))
