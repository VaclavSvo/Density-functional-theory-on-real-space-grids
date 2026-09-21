"""An independent two-centre reference: H2+ solved exactly by prolate-spheroidal separation (D-41).

Gives the multi-centre reference at any bond length, where the radial oracle covers only spherical
systems. In ``xi = (r_A + r_B)/R``, ``eta = (r_A - r_B)/R`` the equal-charge problem separates into
an angular and a radial equation sharing a separation constant ``A`` and the energy ``E``; fixing
``E`` makes each a linear eigenproblem for ``A``, and the physical ``E`` is the Brent root of
``A_radial(E) - A_angular(E)``. The coordinates remove the Coulomb singularities, so no cusp
treatment is needed.

An oracle: nothing in the solver path imports it. Rationale, the cell-centred flux subtlety and the
measured extrapolation orders: ``docs/03_METHOD.md (Part E)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq

__all__ = [
    "TwoCentreGrid",
    "TwoCentreResult",
    "solve_two_centre",
    "solve_two_centre_extrapolated",
    "h2_plus_energy",
    "united_atom_limit",
    "separated_atom_limit",
]


@dataclass(frozen=True, slots=True)
class TwoCentreGrid:
    """Discretisation of the two separated equations.

    ``n_eta`` points on ``(-1, 1)``, open at both ends: the endpoints are regular singular points
    and the flux form keeps the operator symmetric without imposing a boundary condition by hand.
    ``n_xi`` points on ``(1, xi_max)``; ``X`` decays like ``exp(-c xi)``, so ``c (xi_max - 1) >> 1``
    suffices (flat to under 1e-13 Ha between 40 and 60 at ``R = 2``).
    """

    n_eta: int = 800
    n_xi: int = 1600
    xi_max: float = 40.0

    def __post_init__(self) -> None:
        """Reject a grid too coarse to mean anything."""
        if self.n_eta < 50 or self.n_xi < 50:
            raise ValueError("too few points for a meaningful two-centre solve")
        if self.xi_max <= 1.0:
            raise ValueError("xi_max must exceed 1, which is the internuclear axis itself")


@dataclass(slots=True)
class TwoCentreResult:
    """The converged solution of the two-centre one-electron problem."""

    electronic_energy: float
    """``E_el`` in Hartree: the electron in the field of both fixed nuclei, no nuclear repulsion."""

    total_energy: float
    """``E_el + Z_a Z_b / R``, the quantity tabulated by Madsen and Peek (1971)."""

    separation_constant: float
    bond_length: float
    charges: tuple[float, float]
    angular_momentum: int
    residual: float
    """``|A_radial - A_angular|`` at the returned energy: the root find's own error estimate."""

    convergence_order: float = float("nan")
    """Refinement order measured from three nested grids; re-measured every call, since an
    extrapolation outside its asymptotic regime means nothing."""

    extrapolation_shift: float = 0.0
    """``E_extrapolated - E_finest``: a conservative error bar."""

    levels: tuple[float, ...] = ()
    """The energies on each nested grid, coarsest first, so the order can be re-derived."""


def _flux_operator(faces: np.ndarray, spacing: float, right_dirichlet: float) -> tuple[np.ndarray, np.ndarray]:
    """Assemble the symmetric finite-volume operator ``-d/dt[p(t) d/dt]`` on a cell-centred grid.

    ``p`` vanishes at the physical endpoints, which are regular singular points, not walls: the
    wavefunction is non-zero there. Cell-centred, the outer faces land on those endpoints, so zero
    flux is exact in the stencil; a vertex-centred grid would silently impose Dirichlet.

    ``faces`` is ``p`` at the ``n - 1`` interior faces; ``right_dirichlet`` is ``p`` at the far face
    when a genuine Dirichlet condition is wanted (the decaying radial tail at ``xi_max``), zero for
    a natural endpoint. Returns the symmetric tridiagonal ``(diagonal, off_diagonal)``, the latter
    already negated for the driver.
    """
    n = faces.size + 1
    inverse_square = 1.0 / (spacing * spacing)
    diagonal = np.zeros(n, dtype=np.float64)
    diagonal[:-1] += faces * inverse_square
    diagonal[1:] += faces * inverse_square
    # A Dirichlet face is half a cell away from the last centre, hence the factor two.
    diagonal[-1] += 2.0 * right_dirichlet * inverse_square
    return diagonal, -faces * inverse_square


def _angular_constant(c_squared: float, m: int, n_eta: int, node_index: int) -> float:
    """Return the separation constant ``A`` of the angular equation at fixed ``c^2``.

    ``d/d_eta[(1 - eta^2) Y'] + [A + c^2 eta^2 - m^2/(1 - eta^2)] Y = 0`` becomes the symmetric
    eigenproblem ``[K - c^2 eta^2 + m^2/(1 - eta^2)] Y = A Y`` with ``K`` the flux operator above.
    Both endpoints are natural, so no boundary condition is imposed. ``node_index`` selects the
    angular state: 0 is the nodeless sigma ground state.
    """
    spacing = 2.0 / n_eta
    centres = -1.0 + spacing * (np.arange(n_eta) + 0.5)
    faces = 1.0 - (-1.0 + spacing * np.arange(1, n_eta)) ** 2
    diagonal, off_diagonal = _flux_operator(faces, spacing, right_dirichlet=0.0)
    diagonal = diagonal - c_squared * centres**2
    if m:
        diagonal = diagonal + m * m / (1.0 - centres**2)
    values = eigh_tridiagonal(
        diagonal, off_diagonal, select="i", select_range=(node_index, node_index), eigvals_only=True
    )
    return float(values[0])


def _radial_constant(
    energy: float, bond_length: float, charge_sum: float, m: int, grid: TwoCentreGrid, node_index: int
) -> float:
    """Return the separation constant ``A`` demanded by the radial equation at this ``E``.

    ``d/d_xi[(xi^2 - 1) X'] + [-A + R (Z_a + Z_b) xi - c^2 xi^2 - m^2/(xi^2 - 1)] X = 0`` becomes
    ``[K + c^2 xi^2 - R Z_sum xi + m^2/(xi^2 - 1)] X = -A X``. The inner endpoint ``xi = 1`` is
    natural; the outer one is a true Dirichlet cut, placed where the exponential tail is negligible.
    """
    c_squared = -0.5 * energy * bond_length * bond_length
    n = grid.n_xi
    spacing = (grid.xi_max - 1.0) / n
    centres = 1.0 + spacing * (np.arange(n) + 0.5)
    faces = (1.0 + spacing * np.arange(1, n)) ** 2 - 1.0
    diagonal, off_diagonal = _flux_operator(
        faces, spacing, right_dirichlet=grid.xi_max**2 - 1.0
    )
    diagonal = diagonal + c_squared * centres**2 - bond_length * charge_sum * centres
    if m:
        diagonal = diagonal + m * m / (centres**2 - 1.0)
    values = eigh_tridiagonal(
        diagonal, off_diagonal, select="i", select_range=(node_index, node_index), eigvals_only=True
    )
    return -float(values[0])


def solve_two_centre(
    bond_length: float,
    charges: tuple[float, float] = (1.0, 1.0),
    angular_momentum: int = 0,
    grid: TwoCentreGrid | None = None,
    energy_bracket: tuple[float, float] | None = None,
    angular_node: int = 0,
    radial_node: int = 0,
) -> TwoCentreResult:
    """Solve the one-electron two-centre problem exactly, by separation of variables.

    ``bond_length`` in bohr; ``angular_momentum`` is ``|m|`` about the internuclear axis. Only the
    sum of ``charges`` enters the separated equations, so a heteronuclear problem, which adds a
    term in the difference, is refused rather than silently approximated. ``energy_bracket``
    defaults to the united-atom limit ``-(Z_a + Z_b)^2 / 2`` up to just below the separated-atom
    limit ``-max(Z)^2 / 2``, rigorous bounds on the bonding state; the mismatch is monotone there,
    so Brent reaches machine precision in about forty evaluations.
    """
    if bond_length <= 0.0:
        raise ValueError("bond_length must be positive")
    if abs(charges[0] - charges[1]) > 1.0e-12:
        raise NotImplementedError(
            "heteronuclear two-centre problems need the antisymmetric term in the angular equation; "
            "refusing rather than returning a homonuclear answer under a heteronuclear label"
        )
    mesh = grid or TwoCentreGrid()
    charge_sum = charges[0] + charges[1]
    m = abs(int(angular_momentum))

    def mismatch(energy: float) -> float:
        """``A_radial(E) - A_angular(E)``; zero at the physical energy."""
        c_squared = -0.5 * energy * bond_length * bond_length
        angular = _angular_constant(c_squared, m, mesh.n_eta, angular_node)
        radial = _radial_constant(energy, bond_length, charge_sum, m, mesh, radial_node)
        return radial - angular

    if energy_bracket is None:
        united = -0.5 * charge_sum**2
        separated = -0.5 * max(charges) ** 2
        # Open the bracket slightly: at either limit exactly the mismatch has a removable zero,
        # which Brent must not be handed as an endpoint.
        energy_bracket = (united * 1.001, separated * 0.999)

    lower, upper = energy_bracket
    f_low, f_high = mismatch(lower), mismatch(upper)
    if f_low * f_high > 0.0:
        raise RuntimeError(
            f"the energy bracket {energy_bracket} does not straddle a root "
            f"(mismatch {f_low:.6e} and {f_high:.6e}); widen it or check the state indices"
        )

    energy = brentq(mismatch, lower, upper, xtol=1.0e-14, rtol=8.9e-16, maxiter=200)
    c_squared = -0.5 * energy * bond_length * bond_length
    constant = _angular_constant(c_squared, m, mesh.n_eta, angular_node)
    return TwoCentreResult(
        electronic_energy=float(energy),
        total_energy=float(energy + charges[0] * charges[1] / bond_length),
        separation_constant=constant,
        bond_length=float(bond_length),
        charges=charges,
        angular_momentum=m,
        residual=abs(mismatch(energy)),
        levels=(float(energy),),
    )


def solve_two_centre_extrapolated(
    bond_length: float,
    charges: tuple[float, float] = (1.0, 1.0),
    angular_momentum: int = 0,
    grid: TwoCentreGrid | None = None,
    levels: int = 3,
    **kwargs: object,
) -> TwoCentreResult:
    """Solve on ``levels`` nested grids and Richardson-extrapolate to zero spacing.

    The discretisation is exactly second order, so three solves and a division buy what a
    thousand-fold finer grid would. The order is re-measured every call and the extrapolation is
    refused when it deviates from 2 by more than 10 percent: outside its asymptotic regime it is
    invention, not inference (D-48). Three levels is the measured optimum; a fourth is worse.
    """
    if levels < 2:
        raise ValueError("Richardson extrapolation needs at least two grids")
    base = grid or TwoCentreGrid()
    results: list[TwoCentreResult] = []
    for level in range(levels):
        factor = 2**level
        refined = TwoCentreGrid(
            n_eta=base.n_eta * factor, n_xi=base.n_xi * factor, xi_max=base.xi_max
        )
        results.append(
            solve_two_centre(
                bond_length,
                charges=charges,
                angular_momentum=angular_momentum,
                grid=refined,
                **kwargs,  # type: ignore[arg-type]
            )
        )

    energies = [r.electronic_energy for r in results]
    order = float("nan")
    if levels >= 3:
        first, second = energies[1] - energies[0], energies[2] - energies[1]
        if second != 0.0 and first / second > 0.0:
            order = math.log2(abs(first / second))

    finest = results[-1]
    extrapolated = energies[-1]
    if levels >= 3 and math.isfinite(order) and abs(order - 2.0) <= 0.2:
        # Repeated Richardson: each sweep removes the leading remaining power of h.
        working = list(energies)
        for sweep in range(1, levels):
            power = 2.0 ** (2 * sweep)
            working = [
                (power * working[i + 1] - working[i]) / (power - 1.0)
                for i in range(len(working) - 1)
            ]
        extrapolated = working[0]

    return TwoCentreResult(
        electronic_energy=float(extrapolated),
        total_energy=float(extrapolated + charges[0] * charges[1] / bond_length),
        separation_constant=finest.separation_constant,
        bond_length=float(bond_length),
        charges=charges,
        angular_momentum=abs(int(angular_momentum)),
        residual=finest.residual,
        convergence_order=order,
        extrapolation_shift=float(extrapolated - energies[-1]),
        levels=tuple(float(e) for e in energies),
    )


def h2_plus_energy(
    bond_length: float = 2.0, grid: TwoCentreGrid | None = None, levels: int = 3
) -> TwoCentreResult:
    """Ground state (1s sigma_g) of H2+ at a given bond length, Richardson-extrapolated.

    Reproduces Madsen and Peek, *Atomic Data* **2**, 171 (1971) -- ``-0.6026342144949`` Ha at
    ``R = 2.0`` -- to 2.2e-9.
    """
    return solve_two_centre_extrapolated(
        bond_length, charges=(1.0, 1.0), angular_momentum=0, grid=grid, levels=levels
    )


def united_atom_limit(charges: tuple[float, float] = (1.0, 1.0)) -> float:
    """The ``R -> 0`` electronic limit, ``-(Z_a + Z_b)^2 / 2``; a closed form for the tests."""
    return -0.5 * (charges[0] + charges[1]) ** 2


def separated_atom_limit(charges: tuple[float, float] = (1.0, 1.0)) -> float:
    """The ``R -> inf`` electronic limit, ``-max(Z)^2 / 2``: the electron localises on one centre."""
    return -0.5 * max(charges) ** 2
