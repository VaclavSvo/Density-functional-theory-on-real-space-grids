"""An independent radial reference solver -- the strongest oracle this project can build itself.

For a spherically symmetric ``v(r)`` the Kohn--Sham equation separates exactly, so this checks the
3-D solver with nothing in common with it: separated physics, a logarithmic mesh that resolves the
cusp directly (no factorisation, no regularisation), and direct banded diagonalisation.

On ``x = ln r`` with ``u = exp(x/2) w`` the equation becomes the symmetric generalised problem
``-1/2 w_xx + [1/8 + exp(2x) v_eff] w = eps exp(2x) w``, which scaling by ``B^(-1/2)`` makes
standard and banded. An oracle: never on the solver hot path (D-05). Rationale and measured
accuracies: ``docs/03_METHOD.md (Part E)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.linalg import eig_banded

from ..operators.laplacian import fd_coefficients, stencil_half_width

__all__ = ["RadialGrid", "RadialResult", "solve_radial", "hydrogenic_reference"]


@dataclass(frozen=True, slots=True)
class RadialGrid:
    """A logarithmic radial grid ``r_i = r_min exp(i dx)``.

    ``r_min`` (bohr) sets the accuracy: the Dirichlet condition truncates at order ``r_min``, so
    refining ``n_points`` does nothing until ``r_min`` is small enough, and costs cubically.
    ``r_max`` is where the orbital must have decayed. ``fd_order`` is the order of the
    second-derivative stencil in ``x = ln r``; the bandwidth is half of it.
    """

    r_min: float = 1.0e-12
    r_max: float = 60.0
    n_points: int = 1600
    fd_order: int = 8

    def __post_init__(self) -> None:
        """Reject a grid that cannot represent what it is being asked to."""
        if not 0.0 < self.r_min < self.r_max:
            raise ValueError("require 0 < r_min < r_max")
        if self.n_points < 4 * self.fd_order:
            raise ValueError("too few radial points for the requested stencil order")

    @property
    def dx(self) -> float:
        """Spacing of the logarithmic variable."""
        return (math.log(self.r_max) - math.log(self.r_min)) / (self.n_points - 1)

    def radii(self) -> np.ndarray:
        """Return the radial points in bohr, ascending."""
        x = math.log(self.r_min) + self.dx * np.arange(self.n_points)
        return np.exp(x)

    def points_inside(self, radius: float) -> int:
        """Number of grid points inside ``radius``: the measure of cusp resolution."""
        return int((self.radii() < radius).sum())


@dataclass(slots=True)
class RadialResult:
    """Eigenvalues and radial functions of one angular-momentum channel."""

    eigenvalues: np.ndarray
    radial_functions: np.ndarray
    """``u(r) = r R(r)``, shape ``(n_states, n_points)``, normalised so ``integral u^2 dr = 1``."""

    radii: np.ndarray
    angular_momentum: int
    grid: RadialGrid


def solve_radial(
    potential: Callable[[np.ndarray], np.ndarray],
    n_states: int = 1,
    angular_momentum: int = 0,
    grid: RadialGrid | None = None,
) -> RadialResult:
    """Solve the radial Kohn--Sham (or Schrodinger) equation for a spherical potential.

    ``potential`` is the full local ``v(r)`` in Hartree, vectorised over radii and finite away from
    the origin; the centrifugal ``l(l+1)/(2 r^2)`` is added here. Returns eigenvalues in Hartree,
    lowest first, and the normalised ``u(r)``.

    Dirichlet conditions come from omitting the boundary rows, which is exact here: ``u ~ r^(l+1)``
    vanishes at the origin for every ``l`` and a bound orbital has decayed by ``r_max``.
    """
    mesh = grid or RadialGrid()
    radii = mesh.radii()
    dx = mesh.dx
    size = mesh.n_points

    effective = potential(radii)
    if angular_momentum:
        effective = effective + angular_momentum * (angular_momentum + 1) / (2.0 * radii**2)

    # A = -1/2 d2/dx2 + diag(1/8 + r^2 v_eff);  B = diag(r^2).
    coefficients = np.asarray(fd_coefficients(2, mesh.fd_order), dtype=np.float64)
    half_width = stencil_half_width(2, mesh.fd_order)
    scale = -0.5 / (dx * dx)
    diagonal_shift = 0.125 + radii**2 * effective
    b_diagonal = radii**2

    # Upper-banded storage, scaled by B^(-1/2) on both sides to make the problem standard; the
    # bandwidth survives because B is diagonal.
    inverse_sqrt_b = 1.0 / np.sqrt(b_diagonal)
    banded = np.zeros((half_width + 1, size), dtype=np.float64)
    for offset in range(half_width + 1):
        coefficient = coefficients[half_width + offset] * scale
        if offset == 0:
            values = coefficient + diagonal_shift
            banded[half_width, :] = values * inverse_sqrt_b * inverse_sqrt_b
        else:
            columns = np.arange(offset, size)
            banded[half_width - offset, columns] = (
                coefficient * inverse_sqrt_b[columns - offset] * inverse_sqrt_b[columns]
            )

    values, vectors = eig_banded(
        banded, lower=False, select="i", select_range=(0, n_states - 1), overwrite_a_band=True
    )

    # Undo the scaling and normalise in r: integral u^2 dr = integral w^2 exp(2x) dx.
    functions = (vectors.T * inverse_sqrt_b) * np.sqrt(radii)
    norms = np.sqrt((functions**2 * radii * dx).sum(axis=1))
    functions = functions / norms[:, None]
    return RadialResult(
        eigenvalues=values,
        radial_functions=functions,
        radii=radii,
        angular_momentum=angular_momentum,
        grid=mesh,
    )


def hydrogenic_reference(charge: float, n_states: int = 1, grid: RadialGrid | None = None) -> RadialResult:
    """Solve the bare ``-Z/r`` problem, whose exact s eigenvalues are ``-Z^2/(2 n^2)``.

    A reference for the 3-D solver, and the closed form this module is itself validated against.
    """
    return solve_radial(lambda r: -charge / r, n_states=n_states, grid=grid)
