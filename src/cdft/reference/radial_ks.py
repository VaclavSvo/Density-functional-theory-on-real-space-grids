"""A self-consistent radial Kohn--Sham oracle for spherical atoms (Increment 3, extends D-36).

For a spin-restricted atom the Kohn--Sham equations separate and the radial problem solves to
eleven digits on a logarithmic grid with none of the three-dimensional machinery: no Cartesian
stencil, no cusp factorisation, no sphere quadrature, no FFT Poisson solver. Iterated to
self-consistency it is the oracle for the interacting all-electron path (G4.8), and it covers the
PBE atoms NIST does not, without a basis-set truncation error.

Independent of the 3-D path: the discretisation, the Hartree solver, the quadrature. Shared: the
functional objects of :mod:`cdft.xc`, validated against libxc separately (G0.6), so an agreement
here says the solver is right for the functional it was given (D-47).

``v_H(r) = (4 pi / r) int_0^r n r'^2 dr' + 4 pi int_r^inf n r' dr'``; in radial form
``v_xc = e_n - (1/r^2) d/dr (r^2 2 e_sigma n')`` with ``sigma = n'^2``, derivatives by the same
order-8 stencil in ``x = ln r`` as the eigenproblem.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from contract import XCRung

from ..operators.laplacian import fd_coefficients, stencil_half_width
from .radial import RadialGrid, solve_radial

__all__ = ["RadialKSResult", "solve_radial_ks"]


@dataclass(slots=True)
class RadialKSResult:
    """Converged radial Kohn--Sham atom."""

    total: float
    kinetic: float
    external: float
    hartree: float
    xc: float
    eigenvalues: np.ndarray
    occupations: np.ndarray
    density: np.ndarray
    radii: np.ndarray
    iterations: int
    converged: bool
    harris_foulkes: float


def _radial_derivative(values: np.ndarray, grid: RadialGrid) -> np.ndarray:
    """``d/dr`` on the log grid: ``(1/r) d/dx``, centred order-``fd_order`` stencil."""
    coefficients = np.asarray(fd_coefficients(1, grid.fd_order), dtype=np.float64)
    half = stencil_half_width(1, grid.fd_order)
    padded = np.pad(values, half, mode="edge")
    out = np.zeros_like(values)
    for offset in range(-half, half + 1):
        out += coefficients[half + offset] * padded[half + offset : half + offset + values.size]
    return out / (grid.dx * grid.radii())


def _cumulative(values: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cumulative integral from the first point, Simpson's rule (fourth order in the log spacing)."""
    from scipy.integrate import cumulative_simpson

    return cumulative_simpson(values, x=x, initial=0.0)


def _hartree(density: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Radial Hartree potential of a spherical density.

    ``v_H(r) = (4 pi / r) int_0^r n r'^2 dr' + 4 pi int_r^inf n r' dr'``, both cumulative integrals
    taken in the log variable ``x`` (``dr = r dx``) with Simpson's rule -- a trapezoid leaves a
    3e-5 Ha error in the helium total. The charge inside ``r_min``, ``(4 pi / 3) n(0) r_min^3``, is
    added.
    """
    x = np.log(radii)
    inner = _cumulative(density * radii**3, x) + density[0] * radii[0] ** 3 / 3.0
    outer_cumulative = _cumulative(density * radii**2, x)
    outer = outer_cumulative[-1] - outer_cumulative
    return 4.0 * math.pi * (inner / radii + outer)


def _integrate(values: np.ndarray, radii: np.ndarray) -> float:
    """``4 pi int values r^2 dr`` in the log variable by Simpson's rule."""
    from scipy.integrate import simpson

    return float(simpson(4.0 * math.pi * values * radii**3, x=np.log(radii)))


def solve_radial_ks(
    charge: float,
    n_electrons: float,
    functional,
    grid: RadialGrid | None = None,
    *,
    max_iterations: int = 200,
    energy_tol: float = 1.0e-10,
    density_tol: float = 1.0e-8,
    alpha: float = 0.5,
) -> RadialKSResult:
    """Solve a spin-restricted spherical atom self-consistently in the s channel.

    ``n_electrons`` fills the 1s shell only (``<= 2``, fractional allowed, D-11), which covers every
    Phase 1 atom; ``functional`` is a :class:`~cdft.xc.base.SemiLocalFunctional` of rung 1 or 2.
    """
    if functional.rung.value > XCRung.GGA.value:
        raise NotImplementedError("the radial oracle covers LDA and GGA; meta-GGA needs the tau term")
    if n_electrons > 2.0 + 1.0e-12:
        raise NotImplementedError("the radial oracle occupies the 1s shell only (Phase 1: N <= 2)")
    mesh = grid or RadialGrid()
    radii = mesh.radii()
    needs_sigma = functional.rung.value >= XCRung.GGA.value

    def xc_terms(density: np.ndarray):
        n = torch.tensor(density[None, :], dtype=torch.float64)
        sigma = None
        dn = None
        if needs_sigma:
            dn = _radial_derivative(density, mesh)
            sigma = torch.tensor((dn * dn)[None, :], dtype=torch.float64)
        out = functional.evaluate(n, sigma)
        e_xc = out.e_xc.numpy()
        v_xc = out.v_xc[0].numpy().copy()
        double_counting = _integrate(density * out.v_xc[0].numpy(), radii)
        if needs_sigma:
            e_sigma = out.v_sigma[0].numpy()
            flux = 2.0 * e_sigma * dn
            v_xc = v_xc - _radial_derivative(radii**2 * flux, mesh) / radii**2
            double_counting += _integrate(2.0 * e_sigma * dn * dn, radii)
        return e_xc, v_xc, double_counting

    # Initial guess: the hydrogenic 1s density scaled to N.
    density = n_electrons * (charge**3 / math.pi) * np.exp(-2.0 * charge * radii)
    converged = False
    energy_previous = float("inf")
    total = harris = kinetic = external = hartree_energy = xc_energy = float("nan")
    eigenvalue = float("nan")
    for iteration in range(1, max_iterations + 1):
        v_h = _hartree(density, radii)
        e_xc, v_xc, double_counting = xc_terms(density)
        def potential(r, _vh=v_h, _vxc=v_xc):
            return -charge / r + np.interp(r, radii, _vh) + np.interp(r, radii, _vxc)

        solution = solve_radial(potential, n_states=1, grid=mesh)
        eigenvalue = float(solution.eigenvalues[0])
        u = solution.radial_functions[0]
        density_out = n_electrons * (u / radii) ** 2 / (4.0 * math.pi)
        hartree_energy = 0.5 * _integrate(density * v_h, radii)
        xc_energy = _integrate(e_xc, radii)
        # Harris-Foulkes form at the input density (second order in the residual).
        band = n_electrons * eigenvalue
        harris = band - hartree_energy - double_counting + xc_energy
        # Variational form at the output density.
        v_h_out = _hartree(density_out, radii)
        e_xc_out, _, _ = xc_terms(density_out)
        hartree_out = 0.5 * _integrate(density_out * v_h_out, radii)
        xc_out = _integrate(e_xc_out, radii)
        kinetic_plus_external = band - _integrate(density_out * (v_h + v_xc), radii)
        total = kinetic_plus_external + hartree_out + xc_out
        residual = _integrate(np.abs(density_out - density), radii)
        if abs(total - energy_previous) < energy_tol and residual < density_tol:
            converged = True
            density = density_out
            break
        energy_previous = total
        density = density + alpha * (density_out - density)
    # Decomposition at the converged density: T from the kinetic functional of the radial orbital.
    from scipy.integrate import simpson

    u = solution.radial_functions[0]
    du = _radial_derivative(u, mesh)
    kinetic = n_electrons * 0.5 * float(simpson(du * du * radii, x=np.log(radii)))
    external = _integrate(density * (-charge / radii), radii)
    return RadialKSResult(
        total=total,
        kinetic=kinetic,
        external=external,
        hartree=hartree_out,
        xc=xc_out,
        eigenvalues=np.array([eigenvalue]),
        occupations=np.array([n_electrons]),
        density=density,
        radii=radii,
        iterations=iteration,
        converged=converged,
        harris_foulkes=harris,
    )
