"""The one-step Kohn--Sham map as a pure function -- :class:`~contract.SCFStepProtocol` (D-49).

``step(density, functional)`` builds the Hamiltonian at the input density, diagonalises it and
returns the output density with the eigenpairs and an energy breakdown. It holds no mixer state and
mutates nothing, so the same inputs give the same outputs (the eigensolver is seeded from the
numerics) and the fixed point stays differentiable by implicit differentiation. Mixing and the
fallback ladder live in :mod:`cdft.scf.loop` and change only the path (D-16).

The interacting potential in the cusp-factorised frame (D-54): ``n = f^2 rho`` with ``rho`` smooth,
and every density integral and potential matrix element goes through
:class:`~cdft.operators.quadrature.CuspQuadrature`.

* Hartree: ``n`` splits as ``sum_a c_a e^{-2 Z_a s_a} + n_s`` with the ``c_a`` solved from
  ``n(R_a)``; the cusp part uses the closed form of
  :func:`~cdft.operators.quadrature.hydrogenic_1s_potential`, the ``C^1`` remainder the
  Coulomb-cutoff FFT solver [F15]. ``E_H = (1/2) Q[n v_H]``, and the operator sees the lumped
  ``omega~_H / omega``.
* XC: ``grad n = f^2 (grad rho - 2 rho grad u)`` with ``grad u`` analytic, so no stencil
  differences across the cusp. The GGA potential enters in weak form, so ``-div(2 e_sigma grad n)``
  is never formed and the ``-0.017/r`` divergence of PBE at a nucleus (D-52 item 3) is integrated
  rather than sampled.
* ``E = sum_i f_i eps_i - E_H - int n v_xc + E_xc + E_ii`` with ``int n v_xc = Q[n e_n + 2 sigma
  e_sigma]``; ``E_ext`` is reported as the complement (D-52 item 2), the bounded by-parts value
  beside it.

Spin-restricted (``n_spin == 1``) in this increment; the arrays already carry the spin axis (D-51).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Callable

import torch

from contract import (
    EigenResult,
    EnergyBreakdown,
    NumericsConfig,
    ScenarioSpec,
    XCFunctionalProtocol,
    XCRung,
)

from ..eigen.chefsi import ChebyshevFilteredSubspace, GraphedChebyshevFilter, SpectralBoundsHint
from ..grid import UniformGrid
from ..operators.cusp import CuspFactor
from ..operators.hamiltonian import CuspFactoredHamiltonian
from ..operators.poisson import CoulombCutoffPoisson
from ..operators.quadrature import CuspQuadrature, hydrogenic_1s_potential
from ..precision import host_floats
from ..structure import electron_count, nuclear_charges, nuclear_repulsion, positions_tensor
from .occupations import aufbau_occupations

__all__ = ["KohnShamStep", "Potentials", "StepGeometry"]


@dataclass(slots=True)
class Potentials:
    """Everything the Hamiltonian and the energy need at one density.

    Scalars stay 0-d tensors on the run's device; the SCF reads what it needs once per iteration
    (:func:`~cdft.precision.host_floats`) and record-only scalars are computed by :meth:`record`,
    once per solve on the final potentials.
    """

    v_hartree: torch.Tensor
    """``v_H`` at the grid points, ``(n_points,)``."""
    v_xc: torch.Tensor
    """Pointwise ``v_xc``, ``(n_spin, n_points)``; for the record, not the operator (GGA: by finite
    differences on the grid)."""
    effective: torch.Tensor
    """The lumped effective potential ``(omega~_H + omega~_xc) / omega`` the operator multiplies by."""
    hartree_energy: torch.Tensor
    """``E_H``, 0-d."""
    xc_energy: torch.Tensor
    """``E_xc``, 0-d."""
    xc_double_counting: torch.Tensor
    """``int n v_xc`` in weak form, ``Q[n e_n + 2 sigma e_sigma]``, 0-d."""
    hartree_coefficients: torch.Tensor
    """The ``c_a`` of the cusp split, ``(n_atoms,)``."""
    record_terms: Callable[[], dict[str, object]]
    """Deferred record-only scalars: ``lieb_oxford_integral`` (``Q[n^(4/3)]``, G1.8),
    ``smooth_charge_outside_cutoff`` (fraction of ``n_s`` beyond the Poisson kernel's validity) and
    ``virial_scaling_term`` (``-integral n r . grad v_xc`` about the first nucleus, G1.7; ``None``
    for a GGA)."""

    def record(self) -> dict[str, object]:
        """Return the record scalars as Python numbers, reading the device tensors in one transfer."""
        terms = self.record_terms()
        scalars: dict[str, object] = {
            "hartree_energy": self.hartree_energy,
            "xc_energy": self.xc_energy,
            "xc_double_counting": self.xc_double_counting,
            "lieb_oxford_integral": terms["lieb_oxford_integral"],
        }
        if terms["virial_scaling_term"] is not None:
            scalars["virial_scaling_term"] = terms["virial_scaling_term"]
        values = host_floats(*scalars.values(), self.hartree_coefficients)
        out: dict[str, object] = dict(zip(scalars, values))
        out["hartree_coefficients"] = tuple(values[len(scalars) :])
        out.setdefault("virial_scaling_term", None)
        out["smooth_charge_outside_cutoff"] = terms["smooth_charge_outside_cutoff"]
        return out


@dataclass(frozen=True, slots=True)
class StepGeometry:
    """The geometry-only tensors of the Hartree split; read-only, cached per geometry.

    ``positions`` ``(n_atoms, 3)``; ``overlap`` ``S_ab = exp(-2 Z_b R_ab)``; ``cusp_grid`` /
    ``v1s_grid`` ``(n_atoms, n_points)`` the unit cusp densities ``e^{-2 Z_a s_a}`` and their
    closed-form potentials on the grid; ``v1s_node`` ``(n_atoms, m)`` the same at the sphere nodes;
    ``nucleus_f2`` ``f^2(R_a)`` without nucleus ``a``'s own factor.
    """

    positions: torch.Tensor
    overlap: torch.Tensor
    cusp_grid: torch.Tensor
    v1s_grid: torch.Tensor
    v1s_node: torch.Tensor
    nucleus_f2: torch.Tensor

    @classmethod
    def build(
        cls,
        grid: UniformGrid,
        quadrature: CuspQuadrature,
        charges: tuple[float, ...],
        positions: torch.Tensor,
    ) -> "StepGeometry":
        """Compute the tensors for ``grid``, the quadrature's nodes and the nuclei (D-54 split).

        ``positions`` is :func:`~cdft.structure.positions_tensor` in float64 on the grid's device.
        """
        q = quadrature
        distances = [
            [float((positions[a] - positions[b]).norm()) for b in range(len(charges))]
            for a in range(len(charges))
        ]
        overlap = torch.tensor(
            [[math.exp(-2.0 * z_b * distances[a][b]) for b, z_b in enumerate(charges)] for a in range(len(charges))],
            dtype=torch.float64,
            device=grid.device,
        )
        # Validated once here so the per-call solve can skip its error check, which would
        # synchronise the host on CUDA.
        torch.linalg.solve(overlap, torch.ones(len(charges), dtype=torch.float64, device=grid.device))
        pts = grid.points().to(torch.float64)
        cusp_grid = torch.stack([torch.exp(-2.0 * z * (pts - positions[a]).norm(dim=-1)) for a, z in enumerate(charges)])
        v1s_grid = torch.stack([hydrogenic_1s_potential(z, (pts - positions[a]).norm(dim=-1)) for a, z in enumerate(charges)])
        v1s_node = torch.stack([hydrogenic_1s_potential(z, (q.points - positions[a]).norm(dim=-1)) for a, z in enumerate(charges)])
        nucleus_f2 = torch.tensor(
            [
                math.exp(-2.0 * sum(z_b * distances[a][b] for b, z_b in enumerate(charges) if b != a))
                for a in range(len(charges))
            ],
            dtype=torch.float64,
            device=grid.device,
        )
        return cls(positions, overlap, cusp_grid, v1s_grid, v1s_node, nucleus_f2)


class KohnShamStep:
    """One application of the Kohn--Sham map on the cusp-factorised all-electron path."""

    def __init__(
        self,
        scenario: ScenarioSpec,
        numerics: NumericsConfig,
        grid: UniformGrid,
        factor: CuspFactor,
        quadrature: CuspQuadrature,
        n_states: int,
        functional: XCFunctionalProtocol,
        *,
        geometry: StepGeometry | None = None,
        poisson: CoulombCutoffPoisson | None = None,
    ) -> None:
        """Bind the geometry-dependent pieces once; ``step`` then depends on the density alone.

        ``geometry`` and ``poisson`` come from the geometry cache and are read, never written, so a
        shared instance gives the same numbers as a fresh one; ``None`` builds them here.
        ``poisson`` must have been built with ``numerics.grid.poisson_pad_factor``.
        """
        self.scenario = scenario
        self.numerics = numerics
        self.grid = grid
        self.factor = factor
        self.quadrature = quadrature
        self.n_states = n_states
        self.functional = functional
        self.n_spin = scenario.electrons.n_spin
        self.n_electrons = electron_count(
            scenario.structure, scenario.external, scenario.electrons.n_electrons
        )
        self.charges = nuclear_charges(scenario.structure, scenario.external)
        self.ion_ion = nuclear_repulsion(scenario.structure, scenario.external)
        self.poisson = poisson if poisson is not None else CoulombCutoffPoisson(
            grid, pad_factor=numerics.grid.poisson_pad_factor
        )
        self._bare = CuspFactoredHamiltonian(grid, factor, quadrature=quadrature)
        # Geometry-only pieces of the Hartree split, usually served by the geometry cache.
        if geometry is None:
            geometry = StepGeometry.build(
                grid,
                quadrature,
                self.charges,
                positions_tensor(scenario.structure, device=grid.device, dtype=torch.float64),
            )
        self.geometry = geometry
        self.positions = geometry.positions
        self._overlap = geometry.overlap
        self._cusp_grid = geometry.cusp_grid
        self._v1s_grid = geometry.v1s_grid
        self._v1s_node = geometry.v1s_node
        self._nucleus_f2 = geometry.nucleus_f2
        self._f2_grid = factor.weight.to(torch.float64)

    def potentials(self, density: torch.Tensor, functional: XCFunctionalProtocol | None = None) -> Potentials:
        """Hartree and XC potentials, energies and lumped weights at ``density`` (module docstring)."""
        functional = functional or self.functional
        if density.shape[0] != 1:
            raise NotImplementedError(
                "the self-consistent path is spin-restricted in this increment (n_spin = 1); the "
                "spin-polarised leg is Increment 4 (D-51)"
            )
        q = self.quadrature
        grid = self.grid
        f2 = self._f2_grid
        n_grid = density.sum(dim=0).to(torch.float64)
        # ``rho = n / f^2``, summed over spin: the field the quadrature interpolates.
        rho = n_grid / f2

        # -- Hartree: cusp split + smooth FFT part --
        rho_at_nuclei = q.interpolate_at(self.positions, rho)
        n_at_nuclei = self._nucleus_f2 * rho_at_nuclei
        coefficients = torch.linalg.solve_ex(self._overlap, n_at_nuclei, check_errors=False).result
        n_cusp = (coefficients[:, None] * self._cusp_grid).sum(dim=0)
        n_smooth = n_grid - n_cusp
        v_smooth = self.poisson.solve(n_smooth)
        v_hartree_grid = (coefficients[:, None] * self._v1s_grid).sum(dim=0) + v_smooth
        v_hartree_node = (coefficients[:, None] * self._v1s_node).sum(dim=0) + q.interpolate(v_smooth)
        n_node, grad_n_node = q.density_gradient_at_nodes(rho)
        hartree_weights = q.lumped_weights(q.f2 * v_hartree_node, f2 * v_hartree_grid)
        hartree_energy = 0.5 * (hartree_weights * rho).sum()

        # -- Exchange-correlation --
        needs_sigma = functional.rung.value >= XCRung.GGA.value
        grad_n_grid = q.density_gradient_on_grid(rho) if needs_sigma else None
        sigma_grid = (grad_n_grid * grad_n_grid).sum(dim=0)[None, :] if needs_sigma else None
        sigma_node = (grad_n_node * grad_n_node).sum(dim=0)[None, :] if needs_sigma else None
        out_grid = functional.evaluate(n_grid[None, :], sigma_grid)
        out_node = functional.evaluate(n_node[None, :], sigma_node)
        xc_energy = q.integrate(out_node.e_xc, out_grid.e_xc)
        e_n_node, e_n_grid = out_node.v_xc[0], out_grid.v_xc[0]
        xc_weights = q.lumped_weights(q.f2 * e_n_node, f2 * e_n_grid)
        if needs_sigma:
            e_s_node, e_s_grid = out_node.v_sigma[0], out_grid.v_sigma[0]
            flux_node = 2.0 * e_s_node[None, :] * q.f2[None, :] * grad_n_node
            flux_grid = 2.0 * e_s_grid[None, :] * f2[None, :] * grad_n_grid
            xc_weights = xc_weights + q.lumped_gradient_weights(flux_node, flux_grid)
            cross_node = -2.0 * (flux_node * q.grad_u).sum(dim=0)
            cross_grid = -2.0 * (flux_grid * self.factor.grad_u.to(torch.float64)).sum(dim=0)
            xc_weights = xc_weights + q.lumped_weights(cross_node, cross_grid)
            # Pointwise v_xc for the record only: finite differences, wrong within a few cells of a
            # nucleus, where the operator never uses it.
            divergence = torch.zeros_like(e_n_grid)
            flux_plain = 2.0 * e_s_grid[None, :] * grad_n_grid
            for d in range(3):
                divergence = divergence + grid.partial_derivative(flux_plain[d], d)
            v_xc_grid = e_n_grid - divergence
        else:
            # Cloned, not aliased: the record may outlive ``out_grid.v_xc[0]``.
            v_xc_grid = e_n_grid.clone()
        xc_double_counting = (xc_weights * rho).sum()
        positions = self.positions
        poisson = self.poisson

        def record_terms() -> dict[str, object]:
            """The record-only scalars of this density, evaluated on demand."""
            lieb_oxford = q.integrate(n_node ** (4.0 / 3.0), n_grid ** (4.0 / 3.0))
            virial_scaling = None
            if not needs_sigma:
                # -int n r.grad v_xc = int v_xc (3 n + r . grad n), Levy-Perdew scaling about the
                # first nucleus, with grad n from the analytic-gradient form (D-54).
                grad_n_grid_lda = q.density_gradient_on_grid(rho)
                r_node = (q.points - positions[0]).transpose(0, 1)
                r_grid = (grid.points().to(torch.float64) - positions[0]).transpose(0, 1)
                node_term = e_n_node * (3.0 * n_node + (r_node * grad_n_node).sum(dim=0))
                grid_term = e_n_grid * (3.0 * n_grid + (r_grid * grad_n_grid_lda).sum(dim=0))
                virial_scaling = q.integrate(node_term, grid_term)
            return {
                "lieb_oxford_integral": lieb_oxford,
                "virial_scaling_term": virial_scaling,
                "smooth_charge_outside_cutoff": poisson.charge_outside_cutoff(n_smooth),
            }

        effective = (hartree_weights + xc_weights) / q.mass_weights
        return Potentials(
            v_hartree=v_hartree_grid,
            v_xc=v_xc_grid[None, :],
            effective=effective,
            hartree_energy=hartree_energy,
            xc_energy=xc_energy,
            xc_double_counting=xc_double_counting,
            hartree_coefficients=coefficients,
            record_terms=record_terms,
        )

    def hamiltonian(self, potentials: Potentials) -> CuspFactoredHamiltonian:
        """The transformed Hamiltonian ``A = f^-1 (T + v_ext + v_H + v_xc) f`` at these potentials."""
        return CuspFactoredHamiltonian(
            self.grid, self.factor, extra_potential=potentials.effective, quadrature=self.quadrature
        )

    def diagonalise(
        self,
        hamiltonian: CuspFactoredHamiltonian,
        initial: torch.Tensor | None = None,
        max_iterations: int | None = None,
        *,
        bounds: SpectralBoundsHint | None = None,
        graphed_filter: GraphedChebyshevFilter | None = None,
    ) -> tuple[EigenResult, dict[str, object]]:
        """Run the production eigensolver, seeded from the numerics so the map is deterministic.

        Returns the result and, in a separate mapping because the step keeps no state, the stop
        reason and Ritz drift (G5.5, D-44) plus what the solver filtered with
        (``eigen_spectral_bounds``, ``eigen_bounds_source``, ``eigen_bounds_shift``,
        ``eigen_top_ritz_fraction``). ``max_iterations``, ``bounds`` (a carried spectral interval)
        and ``graphed_filter`` are path options owned by the caller; ``step`` converges fully.
        """
        generator = torch.Generator(device="cpu").manual_seed(self.numerics.seed)
        config = self.numerics.eigen
        if max_iterations is not None:
            config = dataclasses.replace(config, max_iterations=max_iterations)
        # No Ritz-stability stop inside the SCF (D-55 item 8): the density needs converged
        # eigenvectors, and that criterion stops once the eigenvalues are stable -- from a warm
        # start, after two filter steps whatever the eigenvector residual.
        solver = ChebyshevFilteredSubspace(self.grid, config, ritz_stop=False, graphed_filter=graphed_filter)
        result = solver.solve(hamiltonian, self.n_states, initial=initial, generator=generator, bounds=bounds)
        info = {
            "eigen_stop_reason": str(getattr(solver, "stop_reason", "unrecorded")),
            "eigen_eigenvalue_drift": float(getattr(solver, "eigenvalue_drift", float("nan"))),
            "eigen_spectral_bounds": list(solver.spectral_bounds or ()),
            "eigen_bounds_source": solver.bounds_source,
            "eigen_bounds_shift": solver.bounds_shift,
            "eigen_top_ritz_fraction": solver.top_ritz_fraction,
        }
        return result, info

    def occupy(self, eigenvalues: torch.Tensor) -> torch.Tensor:
        """Aufbau occupations, floats (D-11); the Fermi-level machinery is Increment 4."""
        return aufbau_occupations(
            eigenvalues.reshape(1, -1).repeat(self.n_spin, 1),
            self.n_electrons,
            n_spin=self.n_spin,
            magnetisation=self.scenario.electrons.magnetisation,
        )

    def output_density(self, eigen: EigenResult, occupations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(n_out, rho_out)`` from the occupied eigenvectors, ``n_out`` shaped ``(n_spin, n_points)``."""
        phi = eigen.eigenvectors.to(torch.float64)
        rho_out = torch.stack([(occupations[s][:, None] * phi.pow(2)).sum(dim=0) for s in range(self.n_spin)])
        return rho_out * self._f2_grid, rho_out.sum(dim=0)

    def energies(
        self,
        eigen: EigenResult,
        occupations: torch.Tensor,
        rho_out: torch.Tensor,
        potentials_in: Potentials,
        potentials_out: Potentials | None,
    ) -> tuple[EnergyBreakdown, dict[str, float]]:
        """Assemble the energy breakdown and the consistency measurements.

        ``total`` is the variational Kohn--Sham energy of the output orbitals when
        ``potentials_out`` is given, else the Harris--Foulkes form at the input density;
        ``harris_foulkes`` is always the latter, ``sum f eps - E_H[n_in] - int n_in v_xc[n_in] +
        E_xc[n_in] + E_ii``, second order in the residual (G2.1). This is :meth:`energy_terms`, one
        host read and :meth:`energies_from_host`; the SCF loop calls those two itself so the read
        also carries its residual norms.
        """
        terms = self.energy_terms(eigen, occupations, rho_out, potentials_in, potentials_out)
        return self.energies_from_host(dict(zip(terms, host_floats(*terms.values()))))

    #: Keys of :meth:`energy_terms`, in order.
    ENERGY_TERMS = (
        "total", "kinetic", "external", "hartree", "xc", "harris_foulkes", "band",
        "external_direct", "kinetic_plus_external",
    )

    def energy_terms(
        self,
        eigen: EigenResult,
        occupations: torch.Tensor,
        rho_out: torch.Tensor,
        potentials_in: Potentials,
        potentials_out: Potentials | None,
    ) -> dict[str, torch.Tensor]:
        """The terms of :meth:`energies` as 0-d device tensors (keys :attr:`ENERGY_TERMS`), no host read."""
        band = (occupations * eigen.eigenvalues.reshape(1, -1).repeat(self.n_spin, 1)).sum()
        hamiltonian = self._bare
        phi = eigen.eigenvectors.to(torch.float64)
        kinetic = sum(hamiltonian.kinetic_energy_tensor(phi, occupations[s]) for s in range(self.n_spin))
        double_counting_out = (potentials_in.effective * self.quadrature.mass_weights * rho_out).sum()
        kinetic_plus_external = band - double_counting_out
        external_direct = hamiltonian.external_energy_direct_tensor(rho_out)
        harris_foulkes = (
            band
            - potentials_in.hartree_energy
            - potentials_in.xc_double_counting
            + potentials_in.xc_energy
            + self.ion_ion
        )
        if potentials_out is None:
            total = harris_foulkes
            hartree, xc = potentials_in.hartree_energy, potentials_in.xc_energy
        else:
            hartree, xc = potentials_out.hartree_energy, potentials_out.xc_energy
            total = kinetic_plus_external + hartree + xc + self.ion_ion
        external = total - kinetic - hartree - xc - self.ion_ion
        values = {
            "total": total,
            "kinetic": kinetic,
            "external": external,
            "hartree": hartree,
            "xc": xc,
            "harris_foulkes": harris_foulkes,
            "band": band,
            "external_direct": external_direct,
            "kinetic_plus_external": kinetic_plus_external,
        }
        return {key: values[key] for key in self.ENERGY_TERMS}

    def energies_from_host(self, values: dict[str, float]) -> tuple[EnergyBreakdown, dict[str, float]]:
        """Build the breakdown and the consistency measurements from host floats of :meth:`energy_terms`."""
        breakdown = EnergyBreakdown(
            total=values["total"],
            kinetic=values["kinetic"],
            external=values["external"],
            hartree=values["hartree"],
            xc=values["xc"],
            nonlocal_ps=0.0,
            ion_ion=self.ion_ion,
            harris_foulkes=values["harris_foulkes"],
        )
        measurements = {
            "band_energy": values["band"],
            "external_direct": values["external_direct"],
            "external_consistency": abs(values["external"] - values["external_direct"]),
            "kinetic_plus_external_from_band": values["kinetic_plus_external"],
        }
        return breakdown, measurements

    # -- the contract surface --

    def step(
        self, density: torch.Tensor, functional: XCFunctionalProtocol
    ) -> tuple[torch.Tensor, EigenResult, EnergyBreakdown]:
        """One pass of the Kohn--Sham map at ``density`` (:class:`~contract.SCFStepProtocol`).

        Pure: the eigensolver starts from the seeded random block, nothing is mutated or stored.
        Returns the Harris--Foulkes energies at the input density.
        """
        potentials = self.potentials(density, functional)
        eigen, _ = self.diagonalise(self.hamiltonian(potentials))
        occupations = self.occupy(eigen.eigenvalues)
        n_out, rho_out = self.output_density(eigen, occupations)
        breakdown, _ = self.energies(eigen, occupations, rho_out, potentials, None)
        return n_out, eigen, breakdown

    def residual(self, density: torch.Tensor, functional: XCFunctionalProtocol) -> torch.Tensor:
        """``F(density) - density``, the quantity the mixer drives to zero."""
        n_out, _, _ = self.step(density, functional)
        return n_out - density
