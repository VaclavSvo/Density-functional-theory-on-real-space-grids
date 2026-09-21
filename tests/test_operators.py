"""Stencil weights, grid construction, the Poisson solver and scenario grid reconciliation.

Unit tests, not gates: nothing here needs a converged solution. All ``fast``; the classes
comparing against a closed form also carry ``physics``. Seconds.
"""

from __future__ import annotations

import math

import pytest
import torch

from cdft.grid import GridGeometry, UniformGrid, grid_config_for_scenario
from cdft.operators.external import coulomb_cell_average, external_potential
from cdft.operators.hamiltonian import LocalHamiltonian
from cdft.operators.laplacian import fd_coefficients, stencil_half_width
from cdft.operators.poisson import (
    CoulombCutoffPoisson,
    analytic_gaussian_hartree_energy,
    analytic_gaussian_potential,
)
from cdft.physics_config import REGISTRY
from contract import (
    AtomicStructure,
    BoundaryMode,
    DomainMode,
    ExternalPotentialKind,
    ExternalPotentialSpec,
    GridConfig,
)

#: Markers for every test in this module (see ``tests/conftest.py``).
pytestmark = [pytest.mark.fast]


def _box_grid(n: int = 24, h: float = 0.4, order: int = 8, boundary=BoundaryMode.ZERO) -> UniformGrid:
    """A small full-box grid for operator tests."""
    geometry = GridGeometry(shape=(n, n, n), spacing=h, origin=(-0.5 * n * h,) * 3)
    mask = torch.ones((n, n, n), dtype=torch.bool)
    return UniformGrid(geometry, mask, fd_order=order, gradient_order=order, boundary=boundary)


class TestFiniteDifferenceWeights:
    """Finite-difference weights: known values, symmetry and the Taylor conditions."""

    def test_known_second_derivative_stencils(self) -> None:
        assert fd_coefficients(2, 2) == (1.0, -2.0, 1.0)
        assert fd_coefficients(2, 4) == pytest.approx(
            (-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12), abs=1e-15
        )

    def test_known_first_derivative_stencils(self) -> None:
        assert fd_coefficients(1, 2) == pytest.approx((-0.5, 0.0, 0.5), abs=1e-15)
        assert fd_coefficients(1, 4) == pytest.approx(
            (1 / 12, -2 / 3, 0.0, 2 / 3, -1 / 12), abs=1e-15
        )

    def test_weights_are_symmetric_and_sum_to_zero(self) -> None:
        for accuracy in (2, 4, 6, 8, 12):
            weights = fd_coefficients(2, accuracy)
            assert sum(weights) == pytest.approx(0.0, abs=1e-12)
            assert weights == tuple(reversed(weights))

    def test_taylor_conditions_hold_exactly(self) -> None:
        """The defining property ``sum_j c_j j**m = n! delta(m, n)``."""
        for derivative, accuracy in ((1, 6), (2, 8)):
            weights = fd_coefficients(derivative, accuracy)
            hw = stencil_half_width(derivative, accuracy)
            for moment in range(2 * hw + 1):
                value = sum(c * (j**moment) for c, j in zip(weights, range(-hw, hw + 1)))
                expected = math.factorial(derivative) if moment == derivative else 0.0
                assert value == pytest.approx(expected, abs=1e-9)

    def test_rejects_odd_accuracy(self) -> None:
        with pytest.raises(ValueError):
            fd_coefficients(2, 7)


class TestGrid:
    """The mapping between the flat point index, the box and the padded halo."""

    def test_scatter_gather_round_trip(self) -> None:
        grid = _box_grid(n=10)
        field = torch.randn(grid.n_points, dtype=torch.float64)
        assert torch.allclose(grid.gather_from_box(grid.scatter_to_box(field)), field)

    def test_batched_operations_preserve_leading_dimensions(self) -> None:
        grid = _box_grid(n=10)
        block = torch.randn((2, 5, grid.n_points), dtype=torch.float64)
        assert grid.laplacian(block).shape == block.shape
        assert grid.gradient(block).shape == (2, 5, 3, grid.n_points)

    def test_integrate_uses_the_quadrature_weight(self) -> None:
        grid = _box_grid(n=8, h=0.5)
        ones = torch.ones(grid.n_points, dtype=torch.float64)
        assert float(grid.integrate(ones)) == pytest.approx(grid.n_points * 0.5**3)

    def test_laplacian_of_a_gaussian_is_accurate_in_the_interior(self) -> None:
        grid = _box_grid(n=40, h=0.3)
        sigma = 1.5
        r_sq = (grid.points() ** 2).sum(dim=-1)
        field = torch.exp(-r_sq / (2 * sigma**2))
        exact = field * (r_sq / sigma**4 - 3.0 / sigma**2)
        interior = r_sq <= (3.0 * sigma) ** 2
        assert float((grid.laplacian(field) - exact)[interior].abs().max()) < 1e-6

    def test_odd_reflection_reproduces_the_sine_spectrum(self) -> None:
        """An odd-reflection halo gives the infinite well its exact sine eigenvalue."""
        length, n = 10.0, 29
        h = length / (n + 1)
        geometry = GridGeometry(shape=(n, n, n), spacing=h, origin=(-0.5 * length + h,) * 3)
        mask = torch.ones((n, n, n), dtype=torch.bool)
        grid = UniformGrid(geometry, mask, fd_order=8, boundary=BoundaryMode.ODD_REFLECTION)
        points = grid.points() + 0.5 * length
        mode = torch.ones(grid.n_points, dtype=torch.float64)
        for axis in range(3):
            mode = mode * torch.sin(math.pi * points[:, axis] / length)
        eigenvalue = -float(grid.inner(mode, grid.laplacian(mode)) / grid.inner(mode, mode)) / 2
        assert eigenvalue == pytest.approx(3 * math.pi**2 / (2 * length**2), rel=1e-9)

    def test_zero_halo_does_not_reproduce_it(self) -> None:
        """A zero-filled halo misses that eigenvalue by far more than the tolerance above."""
        length, n = 10.0, 29
        h = length / (n + 1)
        geometry = GridGeometry(shape=(n, n, n), spacing=h, origin=(-0.5 * length + h,) * 3)
        mask = torch.ones((n, n, n), dtype=torch.bool)
        grid = UniformGrid(geometry, mask, fd_order=8, boundary=BoundaryMode.ZERO)
        points = grid.points() + 0.5 * length
        mode = torch.ones(grid.n_points, dtype=torch.float64)
        for axis in range(3):
            mode = mode * torch.sin(math.pi * points[:, axis] / length)
        eigenvalue = -float(grid.inner(mode, grid.laplacian(mode)) / grid.inner(mode, mode)) / 2
        exact = 3 * math.pi**2 / (2 * length**2)
        assert abs(eigenvalue - exact) / exact > 1e-4

    def test_masked_domain_removes_points(self) -> None:
        structure = AtomicStructure(numbers=(1,), positions=((0.0, 0.0, 0.0),))
        config = GridConfig(spacing=0.5, mask_radius=4.0, domain=DomainMode.MASKED_SPHERES)
        grid = UniformGrid.from_config(config, structure=structure)
        assert grid.n_points < grid.geometry.n_box_points
        # Compare against the sphere's volume in cells, not pi/6 of the box points: the bounding
        # box holds (L/h + 1)**3 points for (L/h)**3 cells, so the point ratio sits below the
        # volume ratio by a factor that vanishes only as h -> 0 (0.429 against 0.437 here).
        cells_in_sphere = (4.0 / 3.0) * math.pi * 4.0**3 / 0.5**3
        assert grid.n_points / grid.geometry.n_box_points == pytest.approx(
            cells_in_sphere / grid.geometry.n_box_points, rel=0.03
        )

    def test_empty_domain_is_rejected(self) -> None:
        geometry = GridGeometry(shape=(4, 4, 4), spacing=0.5, origin=(0.0, 0.0, 0.0))
        with pytest.raises(ValueError):
            UniformGrid(geometry, torch.zeros((4, 4, 4), dtype=torch.bool))


@pytest.mark.physics
class TestPoisson:
    """The open-boundary Hartree solver against a charge whose potential is known in closed form."""

    def test_gaussian_energy_and_potential(self) -> None:
        # sigma = 4h is the resolution the gate threshold assumes and the box reaches 8 sigma: a
        # wider Gaussian leaves charge at the boundary and the truncation dominates the error.
        config = GridConfig(
            spacing=0.25,
            domain=DomainMode.BOX,
            box_lengths=(16.0, 16.0, 16.0),
            use_double_grid=False,
            fourier_filter_projectors=False,
        )
        grid = UniformGrid.from_config(config)
        radius = grid.points().norm(dim=-1)
        sigma = 1.0
        density = (2 * math.pi * sigma**2) ** -1.5 * torch.exp(-(radius**2) / (2 * sigma**2))
        density = density / float(grid.integrate(density))
        solver = CoulombCutoffPoisson(grid, pad_factor=config.poisson_pad_factor)
        potential = solver.solve(density)
        assert solver.energy(density, potential) == pytest.approx(
            analytic_gaussian_hartree_energy(sigma), abs=1e-9
        )
        exact = analytic_gaussian_potential(radius, sigma)
        assert float((potential - exact).abs().max() / exact.abs().max()) < 1e-6

    def test_insufficient_padding_is_refused_by_the_contract(self) -> None:
        with pytest.raises(ValueError):
            GridConfig(poisson_pad_factor=1.5)

    def test_spin_resolved_density_is_refused(self) -> None:
        grid = _box_grid(n=12, h=0.5)
        solver = CoulombCutoffPoisson(grid)
        with pytest.raises(ValueError, match="sum over the spin axis"):
            solver.solve(torch.zeros((1, grid.n_points), dtype=torch.float64))


@pytest.mark.physics
class TestExternalPotential:
    """The all-electron nuclear path (D-23) and the analytic model potentials."""

    def test_harmonic_well_is_exact(self) -> None:
        grid = _box_grid(n=12, h=0.5)
        spec = ExternalPotentialSpec(kind=ExternalPotentialKind.HARMONIC, omega=2.0)
        potential = external_potential(spec, AtomicStructure(), grid)
        expected = 0.5 * 4.0 * (grid.points() ** 2).sum(dim=-1)
        assert torch.allclose(potential, expected)

    def test_coulomb_is_regularised_at_the_nucleus_and_exact_away_from_it(self) -> None:
        grid = _box_grid(n=13, h=0.5)
        structure = AtomicStructure(numbers=(2,), positions=((0.0, 0.0, 0.0),))
        spec = ExternalPotentialSpec(
            kind=ExternalPotentialKind.NUCLEAR_COULOMB, charges=(2.0,)
        )
        potential = external_potential(spec, structure, grid)
        radius = grid.points().norm(dim=-1)
        assert torch.isfinite(potential).all()
        far = radius > 1.0
        assert torch.allclose(potential[far], -2.0 / radius[far])
        on_site = radius < 1e-12
        if bool(on_site.any()):
            assert float(potential[on_site][0]) == pytest.approx(
                -2.0 * coulomb_cell_average(grid.spacing)
            )

    def test_pseudopotential_path_refuses_rather_than_approximating(self) -> None:
        grid = _box_grid(n=8, h=0.5)
        spec = ExternalPotentialSpec(kind=ExternalPotentialKind.PSEUDOPOTENTIAL)
        with pytest.raises(NotImplementedError):
            external_potential(spec, AtomicStructure(numbers=(1,), positions=((0, 0, 0),)), grid)


class TestHamiltonian:
    """The Hamiltonian seam: hermiticity, spectral bounds and the refusals."""

    def test_apply_is_hermitian(self) -> None:
        grid = _box_grid(n=10, h=0.5)
        potential = torch.randn(grid.n_points, dtype=torch.float64)
        hamiltonian = LocalHamiltonian(grid, potential)
        block = torch.randn((6, grid.n_points), dtype=torch.float64)
        applied = hamiltonian.apply(block)
        matrix = (block @ applied.transpose(-1, -2)) * grid.volume_element
        assert float((matrix - matrix.transpose(-1, -2)).abs().max()) < 1e-10

    def test_spectral_bounds_enclose_the_spectrum(self) -> None:
        grid = _box_grid(n=8, h=0.6)
        potential = torch.zeros(grid.n_points, dtype=torch.float64)
        hamiltonian = LocalHamiltonian(grid, potential)
        identity = torch.eye(grid.n_points, dtype=torch.float64)
        dense = hamiltonian.apply(identity)
        exact = torch.linalg.eigvalsh(0.5 * (dense + dense.transpose(-1, -2)))
        lo, hi = hamiltonian.spectral_bounds(steps=20)
        assert lo <= float(exact[0]) and hi >= float(exact[-1])

    def test_update_potentials_refuses_rather_than_doing_nothing(self) -> None:
        grid = _box_grid(n=6, h=0.6)
        hamiltonian = LocalHamiltonian(grid, torch.zeros(grid.n_points, dtype=torch.float64))
        with pytest.raises(NotImplementedError):
            hamiltonian.update_potentials(torch.zeros(grid.n_points, dtype=torch.float64))


class TestScenarioGridReconciliation:
    """A scenario overrides the grid values its physics determines, and records why."""

    def test_infinite_well_overrides_box_and_boundary_and_says_so(self) -> None:
        scenario = REGISTRY["box_L10"]
        config = GridConfig(
            spacing=0.25,
            domain=DomainMode.BOX,
            boundary=BoundaryMode.ZERO,
            box_lengths=(16.0, 16.0, 16.0),
        )
        resolved, overrides = grid_config_for_scenario(scenario, config)
        assert resolved.box_lengths == (10.0, 10.0, 10.0)
        assert resolved.boundary is BoundaryMode.ODD_REFLECTION
        assert set(overrides) == {"box_lengths", "boundary"}
        assert all(len(reason) > 20 for reason in overrides.values())

    def test_vacuum_scenario_is_not_given_a_hard_wall(self) -> None:
        scenario = REGISTRY["h_atom"]
        config = GridConfig(
            spacing=0.25,
            domain=DomainMode.BOX,
            boundary=BoundaryMode.ODD_REFLECTION,
            box_lengths=(12.0, 12.0, 12.0),
        )
        resolved, overrides = grid_config_for_scenario(scenario, config)
        assert resolved.boundary is BoundaryMode.ZERO
        assert "boundary" in overrides

    def test_origin_snaps_so_a_nucleus_lands_on_a_grid_point(self) -> None:
        config = GridConfig(
            spacing=0.16,
            domain=DomainMode.BOX,
            box_lengths=(12.0, 12.0, 12.0),
            use_double_grid=False,
            fourier_filter_projectors=False,
        )
        grid = UniformGrid.from_config(config)
        # 12 / 0.16 is not an even integer, so an unsnapped box would straddle the origin. A point
        # exactly on the nucleus is what decides whether the on-site regularisation applies.
        assert float(grid.points().norm(dim=-1).min()) == pytest.approx(0.0, abs=1e-12)
        assert grid.spacing == pytest.approx(0.16)


@pytest.mark.fast
class TestInteractingBoxRule:
    """D-53: the vacuum margin follows the estimated decay, floored at 7 bohr."""

    def test_hydrogen_systems_get_the_wide_box_and_helium_the_floor(self) -> None:
        from cdft.grid import INTERACTING_HALF_BOX, interacting_box_edge, interacting_half_box
        from cdft.physics_config import REGISTRY
        from cdft.scf.solve import is_interacting
        from cdft.structure import electron_count

        margins = {}
        for scenario in REGISTRY:
            if not is_interacting(scenario):
                continue
            n = electron_count(scenario.structure, scenario.external, scenario.electrons.n_electrons)
            margins[scenario.scenario_id] = interacting_half_box(scenario.structure, n)
            assert interacting_box_edge(scenario.structure, n) >= interacting_box_edge(scenario.structure)
        assert margins["he_atom_lda"] == INTERACTING_HALF_BOX
        assert margins["he_plus_lda"] == INTERACTING_HALF_BOX
        # H at LDA is 1.24e-4 Ha too high in a 14-bohr box and 2.3e-6 in a 20-bohr one.
        assert 10.0 < margins["h_atom_lda"] < 11.0
        assert margins["h2_R1.4_lda"] == margins["h_atom_lda"] == margins["h2plus_R8_lda"]


class TestPartialDerivative:
    """One-axis derivative against the full gradient: the S1 shortcut must move no bit."""

    @pytest.mark.parametrize("boundary", [BoundaryMode.ZERO, BoundaryMode.ODD_REFLECTION])
    @pytest.mark.parametrize("order", [4, 8])
    def test_matches_the_gradient_component(self, order: int, boundary) -> None:
        """``partial_derivative(f, d) == gradient(f)[d]`` exactly, both halo fills, orders 4 and 8."""
        grid = _box_grid(n=12, h=0.3, order=order, boundary=boundary)
        field = torch.randn(grid.n_points, dtype=torch.float64)
        full = grid.gradient(field)
        for d in range(3):
            assert torch.equal(grid.partial_derivative(field, d), full[d])

    def test_matches_with_batched_leading_dimensions(self) -> None:
        """The equality holds for a batched field, where the component axis is the second to last."""
        grid = _box_grid(n=10, h=0.5)
        field = torch.randn((2, 5, grid.n_points), dtype=torch.float64)
        full = grid.gradient(field)
        for d in range(3):
            assert torch.equal(grid.partial_derivative(field, d), full[..., d, :])
