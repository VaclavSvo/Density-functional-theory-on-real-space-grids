"""Geometry helpers for :class:`~contract.AtomicStructure`.

The contract stores positions as an opaque ``Any`` so it imports without torch; this module is where
that becomes a tensor in bohr, and the only place a length in Angstrom enters the solver.
Rationale: ``docs/03_METHOD.md (Part E)``.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch

from contract import AtomicStructure, ExternalPotentialKind, ExternalPotentialSpec

from .constants import ANGSTROM_TO_BOHR, atomic_number

__all__ = [
    "positions_tensor",
    "structure_from_symbols",
    "bounding_box",
    "nuclear_charges",
    "electron_count",
    "nuclear_repulsion",
    "from_ase",
]


def positions_tensor(
    structure: AtomicStructure,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return the atomic positions as a ``(n_atoms, 3)`` tensor in bohr.

    Float64 by default regardless of the hot-path precision: positions are inputs, not intermediates.
    """
    if structure.n_atoms == 0:
        return torch.zeros((0, 3), device=device, dtype=dtype)
    return torch.as_tensor(structure.positions, device=device, dtype=dtype).reshape(-1, 3)


def structure_from_symbols(
    symbols: Sequence[str],
    positions: Sequence[Sequence[float]],
    units: str = "bohr",
    charge: float = 0.0,
    multiplicity: int = 1,
    label: str = "",
) -> AtomicStructure:
    """Build a structure from element symbols and coordinates.

    ``units`` is ``"bohr"`` or ``"angstrom"``, explicit and mandatory: a geometry wrong by 1.889
    converges perfectly well and answers a different question.
    """
    if units not in ("bohr", "angstrom"):
        raise ValueError(f"units must be 'bohr' or 'angstrom', got {units!r}")
    scale = ANGSTROM_TO_BOHR if units == "angstrom" else 1.0
    coords = tuple(tuple(float(x) * scale for x in row) for row in positions)
    numbers = tuple(atomic_number(s) for s in symbols)
    if len(numbers) != len(coords):
        raise ValueError(f"{len(numbers)} symbols but {len(coords)} positions")
    return AtomicStructure(
        numbers=numbers,
        positions=coords,
        charge=charge,
        multiplicity=multiplicity,
        label=label or "".join(symbols),
    )


def bounding_box(
    structure: AtomicStructure, padding: float
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return ``(origin, lengths)`` in bohr of the axis-aligned box enclosing the atoms plus padding.

    The padding is the mask radius, so the box is the smallest one containing every sphere of the
    masked domain. An atom-free scenario raises: it must supply ``GridConfig.box_lengths``.
    """
    if structure.n_atoms == 0:
        raise ValueError(
            "an atom-free scenario has no bounding box; set GridConfig.box_lengths explicitly"
        )
    pos = positions_tensor(structure)
    lo = (pos.min(dim=0).values - padding).tolist()
    hi = (pos.max(dim=0).values + padding).tolist()
    origin = (float(lo[0]), float(lo[1]), float(lo[2]))
    lengths = (float(hi[0] - lo[0]), float(hi[1] - lo[1]), float(hi[2] - lo[2]))
    return origin, lengths


def nuclear_charges(
    structure: AtomicStructure, external: ExternalPotentialSpec
) -> tuple[float, ...]:
    """Return the charge Z carried by each nucleus under this external-potential specification.

    All-electron: the atomic numbers. Pseudopotential: refused, since valence charges come from the
    ONCV files and are not the atomic numbers.
    """
    if external.charges:
        if len(external.charges) != structure.n_atoms:
            raise ValueError(
                f"{len(external.charges)} explicit charges for {structure.n_atoms} atoms"
            )
        return tuple(float(z) for z in external.charges)
    if external.kind is ExternalPotentialKind.PSEUDOPOTENTIAL:
        raise ValueError(
            "valence charges on the pseudopotential path come from the ONCV files (Increment 5); "
            "they are not the atomic numbers and must not be guessed"
        )
    return tuple(float(z) for z in structure.numbers)


def electron_count(
    structure: AtomicStructure, external: ExternalPotentialSpec, n_electrons: float | None = None
) -> float:
    """Return the electron number: an explicit (possibly fractional) count wins, else Z_total - charge."""
    if n_electrons is not None:
        return float(n_electrons)
    if structure.n_atoms == 0:
        raise ValueError(
            "an atom-free scenario cannot derive its electron count; set ElectronSpec.n_electrons"
        )
    return float(sum(nuclear_charges(structure, external))) - float(structure.charge)


def nuclear_repulsion(structure: AtomicStructure, external: ExternalPotentialSpec) -> float:
    """Return the classical ion--ion repulsion in Hartree: point charges over unique pairs.

    The ``ion_ion`` term of :class:`~contract.EnergyBreakdown`, exact at every SCF iteration because
    it depends on no electronic quantity.
    """
    if structure.n_atoms < 2:
        return 0.0
    charges = torch.tensor(nuclear_charges(structure, external), dtype=torch.float64, device="cpu")
    pos = positions_tensor(structure)
    delta = pos[:, None, :] - pos[None, :, :]
    dist = delta.norm(dim=-1)
    iu = torch.triu_indices(structure.n_atoms, structure.n_atoms, offset=1, device="cpu")
    pair_dist = dist[iu[0], iu[1]]
    if bool((pair_dist < 1.0e-8).any()):
        raise ValueError("two nuclei occupy the same position; the repulsion energy is infinite")
    pair_charge = charges[iu[0]] * charges[iu[1]]
    return float((pair_charge / pair_dist).sum())


def from_ase(atoms: Any, charge: float = 0.0, multiplicity: int = 1) -> AtomicStructure:
    """Convert an ``ase.Atoms`` to a contract structure, Angstrom to bohr.

    ASE is optional, for geometry input only; nothing in the solver path imports it.
    """
    numbers = tuple(int(z) for z in atoms.get_atomic_numbers())
    coords = tuple(
        tuple(float(x) * ANGSTROM_TO_BOHR for x in row) for row in atoms.get_positions()
    )
    return AtomicStructure(
        numbers=numbers,
        positions=coords,
        charge=charge,
        multiplicity=multiplicity,
        label=atoms.get_chemical_formula(),
    )
