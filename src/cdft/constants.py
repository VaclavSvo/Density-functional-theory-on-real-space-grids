"""Physical constants and unit conversions -- CODATA 2018, full published precision.

Hartree atomic units throughout the solver; conversions happen only at the configuration and
reporting boundaries. Rationale: ``docs/03_METHOD.md (Part E)``.
"""

from __future__ import annotations

import math

__all__ = [
    "BOHR_TO_ANGSTROM",
    "ANGSTROM_TO_BOHR",
    "HARTREE_TO_EV",
    "EV_TO_HARTREE",
    "HARTREE_TO_KCAL_PER_MOL",
    "KCAL_PER_MOL_TO_HARTREE",
    "HARTREE_TO_KJ_PER_MOL",
    "HARTREE_TO_WAVENUMBER",
    "CHEMICAL_ACCURACY_HA",
    "MEV_PER_ATOM_HA",
    "C_X",
    "LIEB_OXFORD_FACTOR",
    "ELEMENT_SYMBOLS",
    "atomic_number",
]

# -- unit conversions (CODATA 2018) --

#: Bohr radius in Angstrom: a_0 = 0.529177210903(80) A.
BOHR_TO_ANGSTROM: float = 0.529177210903
ANGSTROM_TO_BOHR: float = 1.0 / BOHR_TO_ANGSTROM

#: Hartree energy in electronvolt: E_h = 27.211386245988(53) eV.
HARTREE_TO_EV: float = 27.211386245988
EV_TO_HARTREE: float = 1.0 / HARTREE_TO_EV

#: Hartree in kJ/mol: E_h * N_A = 2625.4996394798(50) kJ/mol.
HARTREE_TO_KJ_PER_MOL: float = 2625.4996394798

#: Hartree in kcal/mol, with the thermochemical calorie defined as exactly 4.184 J.
HARTREE_TO_KCAL_PER_MOL: float = HARTREE_TO_KJ_PER_MOL / 4.184
KCAL_PER_MOL_TO_HARTREE: float = 1.0 / HARTREE_TO_KCAL_PER_MOL

#: Hartree in reciprocal centimetres: 219474.6313632(43) cm^-1.
HARTREE_TO_WAVENUMBER: float = 219474.6313632

#: 1 kcal/mol in Hartree -- "chemical accuracy". Several Tier-4 thresholds are fractions of it.
CHEMICAL_ACCURACY_HA: float = KCAL_PER_MOL_TO_HARTREE

#: 1 meV in Hartree. The unit of gates G3.1, G3.2, G3.5 and G2.7, all stated per atom.
MEV_PER_ATOM_HA: float = 1.0e-3 * EV_TO_HARTREE

# -- exact constants of density functional theory --

#: Dirac--Slater exchange constant C_x = (3/4)(3/pi)^(1/3); e_x = -C_x n^(4/3) [A8], [A10].
#: Computed rather than transcribed because gate G1.1 thresholds it at 1e-12 relative.
C_X: float = 0.75 * (3.0 / math.pi) ** (1.0 / 3.0)

#: Lieb--Oxford bound coefficient: E_xc[n] >= -2.273 E_x^LDA[n] [A10], [A16]. Gate G1.8 marks a
#: violating record INVALID.
LIEB_OXFORD_FACTOR: float = 2.273

# -- elements --

#: Symbols indexed by atomic number; index 0 is a placeholder so ``SYMBOLS[Z]`` works.
ELEMENT_SYMBOLS: tuple[str, ...] = (
    "X",
    "H", "He",
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
)

_SYMBOL_TO_Z: dict[str, int] = {s: i for i, s in enumerate(ELEMENT_SYMBOLS) if i > 0}


def atomic_number(symbol: str) -> int:
    """Return the atomic number of an element symbol, case-insensitively.

    Raises ``KeyError`` for a symbol outside :data:`ELEMENT_SYMBOLS`, never a default (Q8.1).
    """
    key = symbol.strip().capitalize()
    if key not in _SYMBOL_TO_Z:
        raise KeyError(f"unknown element symbol {symbol!r}")
    return _SYMBOL_TO_Z[key]
