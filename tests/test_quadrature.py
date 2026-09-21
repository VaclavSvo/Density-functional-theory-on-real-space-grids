"""The cusp-aware quadrature (D-54): Lagrange stencils, the sphere rule and the Hartree split.

Lagrange and closed forms are ``fast``; the hydrogenic integrals are ``physics``. Seconds.
"""

from __future__ import annotations

import math

import pytest
import torch

from contract import BoundaryMode, DomainMode, GridConfig
from cdft.grid import UniformGrid
from cdft.operators.cusp import CuspFactor
from cdft.operators.quadrature import CuspQuadrature, hydrogenic_1s_potential, lagrange_weights


def _grid(h: float, edge: float) -> UniformGrid:
    config = GridConfig(
        spacing=h, fd_order=8, domain=DomainMode.BOX, boundary=BoundaryMode.ZERO,
        box_lengths=(edge, edge, edge), use_double_grid=False, fourier_filter_projectors=False,
    )
    return UniformGrid.from_config(config)


def _atom(z: float, h: float, edge: float) -> tuple[UniformGrid, CuspFactor, CuspQuadrature]:
    grid = _grid(h, edge)
    factor = CuspFactor(grid, (z,), torch.zeros((1, 3), dtype=torch.float64))
    return grid, factor, CuspQuadrature(grid, factor)


@pytest.mark.fast
class TestLagrange:
    def test_degree_seven_reproduces_polynomials_and_their_derivative(self) -> None:
        t = torch.tensor([0.0, 0.31, 0.77, 0.999], dtype=torch.float64)
        w, d = lagrange_weights(t, 7)
        xs = torch.arange(-3, 5, dtype=torch.float64)
        for j, tt in enumerate(t):
            f = xs**7 - 3.0 * xs**4 + 2.0 * xs
            assert abs(float((w[j] * f).sum()) - float(tt**7 - 3.0 * tt**4 + 2.0 * tt)) < 1e-12
            assert abs(float((d[j] * f).sum()) - float(7.0 * tt**6 - 12.0 * tt**3 + 2.0)) < 1e-11
        assert torch.allclose(w.sum(dim=1), torch.ones(4, dtype=torch.float64), atol=1e-14)
        assert torch.allclose(d.sum(dim=1), torch.zeros(4, dtype=torch.float64), atol=1e-13)


@pytest.mark.fast
class TestClosedForms:
    def test_1s_potential_limits_and_series_match_the_full_formula(self) -> None:
        for z in (1.0, 2.0):
            s = torch.tensor([1e-4, 5e-4, 9.9e-4, 1.01e-3, 2e-3, 0.5, 3.0], dtype=torch.float64)
            full = (math.pi / z**3) * (1.0 - torch.exp(-2.0 * z * s) * (1.0 + z * s)) / s
            assert torch.allclose(hydrogenic_1s_potential(z, s), full, rtol=1e-11, atol=1e-13)
            at_zero = float(hydrogenic_1s_potential(z, torch.zeros(1, dtype=torch.float64))[0])
            assert abs(at_zero - math.pi / z**2) < 1e-14
            far = float(hydrogenic_1s_potential(z, torch.tensor([40.0], dtype=torch.float64))[0])
            assert abs(far - math.pi / (z**3 * 40.0)) < 1e-14


@pytest.mark.physics
class TestHydrogenicIntegrals:
    """On the derived atomic grid (Z h = 1) everything about the 1s density is closed-form."""

    @pytest.mark.parametrize("z,h,edge", [(1.0, 1.0, 32.0), (2.0, 0.5, 16.0)])
    def test_weight_integral_and_hartree_energy(self, z: float, h: float, edge: float) -> None:
        grid, factor, q = _atom(z, h, edge)
        exact_weight = math.pi / z**3
        plain = float(grid.integrate(factor.weight))
        assert abs(plain / exact_weight - 1.0) > 0.1, "the plain rule must still be badly wrong (A-9)"
        assert abs(float(q.mass_weights.sum()) / exact_weight - 1.0) < 1e-8
        assert bool((q.mass_weights > 0).all())
        c = z**3 / math.pi
        r_node = q.points.norm(dim=-1)
        r_grid = grid.points().norm(dim=-1)
        v_node = c * hydrogenic_1s_potential(z, r_node)
        v_grid = c * hydrogenic_1s_potential(z, r_grid)
        hartree = 0.5 * float(q.integrate(c * q.f2 * v_node, c * factor.weight * v_grid))
        assert abs(hartree - 5.0 * z / 16.0) < 1e-8

    def test_smooth_field_interpolated_at_nodes_is_integrated_to_interpolation_accuracy(self) -> None:
        grid, factor, q = _atom(1.0, 0.5, 16.0)
        r = grid.points().norm(dim=-1)
        rho = torch.exp(-0.2 * r * r) * (1.0 + 0.1 * grid.points()[:, 0])
        rn = q.points.norm(dim=-1)
        exact_nodes = torch.exp(-0.2 * rn * rn) * (1.0 + 0.1 * q.points[:, 0])
        # Degree-7 interpolation at h = 0.5 of a field varying on the bohr scale: (h^8/8!) times
        # its eighth derivative is a few 1e-5, and the lumped rule inherits that.
        assert float((q.interpolate(rho) - exact_nodes).abs().max()) < 1e-4
        lumped = float((q.mass_weights * rho).sum())
        direct = float(q.integrate(q.f2 * exact_nodes, factor.weight * rho))
        assert abs(lumped - direct) < 5e-5

    def test_lift_is_the_transpose_of_interpolate(self) -> None:
        grid, factor, q = _atom(1.0, 1.0, 32.0)
        generator = torch.Generator().manual_seed(3)
        field = torch.randn(grid.n_points, dtype=torch.float64, generator=generator)
        node_values = torch.randn(q.n_nodes, dtype=torch.float64, generator=generator)
        left = float((q.interpolate(field) * node_values).sum())
        right = float((q.lift(node_values) * field).sum())
        assert abs(left - right) < 1e-10 * max(1.0, abs(left))

    def test_identity_factor_reduces_to_the_plain_sum(self) -> None:
        grid = _grid(0.5, 8.0)
        factor = CuspFactor(grid, (), torch.zeros((0, 3), dtype=torch.float64))
        q = CuspQuadrature(grid, factor)
        assert q.n_nodes == 0
        assert torch.allclose(q.mass_weights, grid.weights)

    def test_two_equal_centres_against_the_closed_form(self) -> None:
        """H2+ at R = 2 on its production grid: G1.13 under 1e-8 with the lumped far rule (D-75)."""
        grid = _grid(0.25, 14.0)
        positions = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
        factor = CuspFactor(grid, (1.0, 1.0), positions)
        q = CuspQuadrature(grid, factor)
        exact = factor.weight_integral_exact()
        in_box = exact - factor.weight_integral_outside_box()
        assert abs(float(grid.integrate(factor.weight)) / exact - 1.0) > 5e-4
        assert q.describe()["far_rule"].startswith("mass lumped-2x")
        near = float((q.weights * q.f2).sum())
        plain_far = float(q.plain_far.sum())
        lumped_far = float(q.lumped_far.sum())
        # The plain far sum aliases the taper shell at 2.7e-8 (A-11); the lumped one does not.
        assert abs(near + plain_far - in_box) / exact > 1.5e-8
        assert abs(near + lumped_far - in_box) / exact < 5e-9
        assert abs(float(q.mass_weights.sum()) - in_box) / exact < 5e-9
        assert bool((q.mass_weights > 0).all())

    def test_lumped_far_rule_sums_to_the_refined_midpoint_rule(self) -> None:
        """``sum_i omega_far,i`` is the h/2 midpoint sum of ``far f^2`` up to the edge blend (D-75).

        Nuclei 23 spacings from the faces, as on the production grids (24 or more): the blend back
        to plain weights within 6 h of a face then costs 7e-11 of the far integral.
        """
        grid = _grid(0.5, 24.0)
        positions = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
        factor = CuspFactor(grid, (1.0, 1.0), positions)
        q = CuspQuadrature(grid, factor)
        assert q.far_lumped and all(q.overlapping) and q.describe()["far_rule"].startswith("mass lumped-2x")
        h = grid.spacing
        axes = [grid.origin[d] - 0.5 * h + 0.5 * h * (torch.arange(2 * grid.shape[d], dtype=torch.float64) + 0.5) for d in range(3)]
        fine = torch.cartesian_prod(*axes)
        u = sum((fine - p).norm(dim=-1) for p in positions)
        refined = float(((0.5 * h) ** 3 * q._far_factor(fine) * torch.exp(-2.0 * u)).sum())
        lumped = float(q.lumped_far.sum())
        assert abs(lumped - refined) < 2e-10 * refined
        # The taper shell's aliasing: the plain sum differs from the refined one, the lumped does not,
        # and the difference is what the record carries as ``far_aliasing``.
        plain = float(q.plain_far.sum())
        assert abs(plain - refined) > 1e-9 * refined
        assert abs(q.far_aliasing - (plain - lumped) / lumped) < 1e-15
        # The far sums of grid fields keep the plain rule; a single centre keeps it everywhere.
        assert torch.equal(q.far_weights, grid.volume_element * q.far)
        _, _, atom = _atom(1.0, 1.0, 32.0)
        assert not atom.far_lumped and atom.describe()["far_rule"] == "plain"
        assert torch.equal(atom.lumped_far, atom.plain_far)

    def test_refuses_a_grid_too_coarse_for_the_charge(self) -> None:
        """At ``Z h > 1`` the lumped weights go negative and the rule refuses to continue."""
        with pytest.raises(ValueError, match="not all positive"):
            _atom(2.0, 1.0, 24.0)
