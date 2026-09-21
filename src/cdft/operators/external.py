"""External potentials: the all-electron path of D-23 and the analytic model systems.

``NUCLEAR_COULOMB`` is a bare ``-Z/r`` per nucleus, regularised at a coincident grid point by the
cell average of ``1/r``: a stated O(h) approximation that nothing runs by default since D-35.
``PSEUDOPOTENTIAL`` raises rather than substituting a bare Coulomb potential. Rationale:
``docs/03_METHOD.md (Part E)``.
"""

from __future__ import annotations

import math

import torch

from contract import AtomicStructure, ExternalPotentialKind, ExternalPotentialSpec

from ..grid import UniformGrid
from ..structure import nuclear_charges, positions_tensor

__all__ = ["external_potential", "coulomb_cell_average"]


def coulomb_cell_average(spacing: float) -> float:
    """Return ``<1/r> = 3/(2 r_c)`` over the cell-volume sphere ``r_c = (3/(4 pi))**(1/3) h``.

    The on-nucleus value of a bare Coulomb potential.
    """
    r_c = (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0) * spacing
    return 1.5 / r_c


def external_potential(
    spec: ExternalPotentialSpec,
    structure: AtomicStructure,
    grid: UniformGrid,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return ``v_ext`` on the grid domain, shape ``(n_points,)``, in Hartree.

    ``structure`` is ignored by the model kinds; ``grid.spacing`` is used only for the Coulomb
    regularisation. ``PSEUDOPOTENTIAL`` raises (D-09).
    """
    target_dtype = dtype or grid.dtype
    points = grid.points().to(target_dtype)
    kind = spec.kind

    if kind is ExternalPotentialKind.NONE:
        return torch.zeros(grid.n_points, device=grid.device, dtype=target_dtype)

    if kind is ExternalPotentialKind.HARMONIC:
        r_sq = (points * points).sum(dim=-1)
        return 0.5 * spec.omega**2 * r_sq

    if kind is ExternalPotentialKind.PARTICLE_IN_BOX:
        # The well is the domain itself: zero inside, walls from BoundaryMode.ODD_REFLECTION rather
        # than a large number in the potential.
        return torch.zeros(grid.n_points, device=grid.device, dtype=target_dtype)

    if kind is ExternalPotentialKind.GAUSSIAN_WELL:
        r_sq = (points * points).sum(dim=-1)
        return -spec.depth * torch.exp(-r_sq / (2.0 * spec.width**2))

    if kind is ExternalPotentialKind.PSEUDOPOTENTIAL:
        raise NotImplementedError(
            "the pseudopotential path arrives with Increment 5 (D-09); it is not approximated here"
        )

    if kind in (ExternalPotentialKind.NUCLEAR_COULOMB, ExternalPotentialKind.SOFT_COULOMB):
        charges = nuclear_charges(structure, spec)
        positions = positions_tensor(structure, device=grid.device, dtype=target_dtype)
        potential = torch.zeros(grid.n_points, device=grid.device, dtype=target_dtype)
        on_site = coulomb_cell_average(grid.spacing)
        for atom, z in enumerate(charges):
            delta = points - positions[atom]
            r_sq = (delta * delta).sum(dim=-1)
            if kind is ExternalPotentialKind.SOFT_COULOMB:
                potential -= z / torch.sqrt(r_sq + spec.softening**2)
            else:
                r = torch.sqrt(r_sq)
                # Half a spacing, not an epsilon: otherwise the treatment depends on whether a
                # nucleus lands exactly on a point, tying the energy to the grid origin.
                near = r < 0.5 * grid.spacing
                inv_r = torch.where(near, torch.full_like(r, on_site), 1.0 / r.clamp_min(1e-30))
                potential -= z * inv_r
        return potential

    raise ValueError(f"unhandled external potential kind {kind!r}")  # pragma: no cover
