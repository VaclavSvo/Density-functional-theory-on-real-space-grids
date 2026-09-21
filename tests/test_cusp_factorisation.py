"""Structural properties of the cusp factorisation (D-35) that make it safe on by default.

Identity reduction, the transformed potential, subspace symmetry, the Kato cusp, the measure and
the LOBPCG preconditioner frame. Marked ``fast`` and ``physics``; seconds.
"""

from __future__ import annotations

import math

import pytest
import torch

from cdft.eigen.measure import Measure
from cdft.eigen.rayleigh_ritz import rayleigh_ritz
from cdft.grid import UniformGrid
from cdft.operators.cusp import CuspFactor
from cdft.operators.hamiltonian import CuspFactoredHamiltonian, LocalHamiltonian
from contract import BoundaryMode, DomainMode, GridConfig

#: Markers for every test in this module (see ``tests/conftest.py``).
pytestmark = [pytest.mark.fast, pytest.mark.physics]



def _grid(box: float = 8.0, h: float = 0.5) -> UniformGrid:
    """A small cubic grid with the origin exactly on a point."""
    return UniformGrid.from_config(
        GridConfig(
            spacing=h,
            domain=DomainMode.BOX,
            boundary=BoundaryMode.ZERO,
            box_lengths=(box, box, box),
            use_double_grid=False,
            fourier_filter_projectors=False,
        )
    )


class TestIdentityReduction:
    """With no nuclei the transform changes nothing, so no model scenario is perturbed."""

    def test_factor_is_identity(self) -> None:
        grid = _grid()
        factor = CuspFactor(grid, (), torch.zeros((0, 3), dtype=torch.float64))
        assert factor.is_identity
        assert torch.allclose(factor.weight, torch.ones(grid.n_points, dtype=torch.float64))
        assert float(factor.grad_u.abs().max()) == 0.0
        assert float(factor.transformed_potential.abs().max()) == 0.0

    def test_apply_matches_the_untransformed_operator_exactly(self) -> None:
        """``apply`` is the untransformed operator bit for bit, not approximately."""
        grid = _grid()
        potential = 0.5 * (grid.points() ** 2).sum(dim=-1)
        factor = CuspFactor(grid, (), torch.zeros((0, 3), dtype=torch.float64))
        transformed = CuspFactoredHamiltonian(grid, factor, extra_potential=potential)
        plain = LocalHamiltonian(grid, potential)
        block = torch.randn((4, grid.n_points), dtype=torch.float64)
        assert torch.equal(transformed.apply(block), plain.apply(block))

    def test_measure_stays_uniform(self) -> None:
        grid = _grid()
        factor = CuspFactor(grid, (), torch.zeros((0, 3), dtype=torch.float64))
        assert CuspFactoredHamiltonian(grid, factor).measure.is_uniform


class TestTransformedPotential:
    """The algebra: every 1/r cancels, and the value at the nucleus is the spherical average."""

    @pytest.mark.parametrize("charge", [1.0, 2.0, 3.5])
    def test_single_nucleus_gives_a_constant_potential(self, charge: float) -> None:
        """For one centre |grad u|^2 = Z^2 everywhere, so W = -Z^2/2 identically."""
        grid = _grid()
        factor = CuspFactor(grid, (charge,), torch.zeros((1, 3), dtype=torch.float64))
        expected = torch.full((grid.n_points,), -0.5 * charge**2, dtype=torch.float64)
        assert torch.allclose(factor.transformed_potential, expected, atol=1e-12)

    def test_value_at_the_nucleus_is_not_zero(self) -> None:
        """``W`` at the nucleus is ``-Z^2/2``, where the unit vector is undefined."""
        # Zeroing the undefined direction loses the magnitude too and puts W = 0 at the maximum of
        # the weight, injecting Z^2/2 * h^3 / integral(f^2): 0.14 Ha for He+ at h = 0.3.
        grid = _grid()
        factor = CuspFactor(grid, (2.0,), torch.zeros((1, 3), dtype=torch.float64))
        radius = grid.points().norm(dim=-1)
        at_nucleus = radius < 1e-12
        assert bool(at_nucleus.any()), "this test needs a grid point on the nucleus"
        assert float(factor.transformed_potential[at_nucleus][0]) == pytest.approx(-2.0)

    def test_two_nuclei_potential_is_bounded_and_correct_far_away(self) -> None:
        """Far from both centres the two unit vectors align, so W -> -(Z_A + Z_B)^2 / 2."""
        grid = _grid(box=12.0, h=0.5)
        positions = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
        factor = CuspFactor(grid, (1.0, 1.0), positions)
        potential = factor.transformed_potential
        assert torch.isfinite(potential).all()
        assert float(potential.min()) >= -0.5 * (1.0 + 1.0) ** 2 - 1e-12
        far = grid.points().norm(dim=-1) > 5.0
        assert float(potential[far].max()) < -1.9


class TestGalerkinSubspace:
    """The subspace matrix ``<basis|A|basis>_w`` is symmetric by construction (D-38).

    The class name predates the staggered divergence form; there is no separate weak-form
    ``energy_matrix`` any more, only the one operator.
    """

    def test_subspace_matrix_is_symmetric_to_round_off(self) -> None:
        grid = _grid()
        factor = CuspFactor(grid, (2.0,), torch.zeros((1, 3), dtype=torch.float64))
        hamiltonian = CuspFactoredHamiltonian(grid, factor)
        block = torch.randn((6, grid.n_points), dtype=torch.float64)
        matrix = hamiltonian.measure.cross(block, hamiltonian.apply(block))
        asymmetry = float((matrix - matrix.transpose(-1, -2)).abs().max() / matrix.abs().max())
        assert asymmetry < 1e-13

    def test_rayleigh_ritz_reaches_the_exact_constant_solution(self) -> None:
        """The exact solution phi = const gives -Z^2/2 through the weak form."""
        # Boundary-limited, not spacing-limited: phi tends to a constant, so the zero halo imposes
        # the wrong condition and the residue is of order exp(-2 Z R) / h**2.
        grid = _grid(box=12.0, h=0.5)
        charge = 2.0
        factor = CuspFactor(grid, (charge,), torch.zeros((1, 3), dtype=torch.float64))
        hamiltonian = CuspFactoredHamiltonian(grid, factor)
        constant = torch.ones((1, grid.n_points), dtype=torch.float64)
        evals, _, _ = rayleigh_ritz(hamiltonian, constant, hamiltonian.measure)
        assert float(evals[0]) == pytest.approx(-0.5 * charge**2, abs=1e-8)


class TestDensityAndCusp:
    """The density normalises and carries the exact Kato cusp; the kinetic energy is right."""

    def test_density_normalises_and_carries_the_exact_cusp(self) -> None:
        grid = _grid(box=16.0, h=0.4)
        charge = 1.0
        factor = CuspFactor(grid, (charge,), torch.zeros((1, 3), dtype=torch.float64))
        measure = Measure(grid, factor.weight)
        phi = torch.ones((1, grid.n_points), dtype=torch.float64)
        phi = phi / measure.norm(phi)[:, None]
        occupations = torch.tensor([1.0], dtype=torch.float64)
        density = factor.density(phi, occupations)
        assert float(grid.integrate(density)) == pytest.approx(1.0, abs=1e-10)

        radius = grid.points().norm(dim=-1)
        window = (radius > 0.5 * grid.spacing) & (radius < 3.0 * grid.spacing)
        log_n = torch.log(density[window])
        r = radius[window]
        centred = r - r.mean()
        slope = float((centred * (log_n - log_n.mean())).sum() / (centred * centred).sum())
        # Exact, because the cusp is carried analytically by f and phi is smooth.
        assert slope == pytest.approx(-2.0 * charge, abs=1e-9)

    def test_kinetic_energy_is_non_negative_and_correct_for_hydrogen(self) -> None:
        """``T = 1/2`` for the hydrogen ground state, from the gradient form of the factor."""
        grid = _grid(box=20.0, h=0.4)
        factor = CuspFactor(grid, (1.0,), torch.zeros((1, 3), dtype=torch.float64))
        measure = Measure(grid, factor.weight)
        phi = torch.ones((1, grid.n_points), dtype=torch.float64)
        phi = phi / measure.norm(phi)[:, None]
        kinetic = factor.kinetic_energy(phi, torch.tensor([1.0], dtype=torch.float64))
        assert kinetic > 0.0
        assert kinetic == pytest.approx(0.5, abs=1e-4)


class TestMeasure:
    """The weighted inner product the eigensolvers work in."""

    def test_uniform_measure_reproduces_the_plain_quadrature(self) -> None:
        grid = _grid()
        measure = Measure.uniform(grid)
        a = torch.randn(grid.n_points, dtype=torch.float64)
        b = torch.randn(grid.n_points, dtype=torch.float64)
        assert float(measure.inner(a, b)) == pytest.approx(float(grid.inner(a, b)), rel=1e-14)
        assert measure.dynamic_range == 1.0

    def test_weighted_measure_matches_an_explicit_sum(self) -> None:
        grid = _grid()
        weight = torch.rand(grid.n_points, dtype=torch.float64) + 0.5
        measure = Measure(grid, weight)
        a = torch.randn(grid.n_points, dtype=torch.float64)
        b = torch.randn(grid.n_points, dtype=torch.float64)
        expected = float((a * b * weight).sum() * grid.volume_element)
        assert float(measure.inner(a, b)) == pytest.approx(expected, rel=1e-14)

    def test_degenerate_weight_is_refused(self) -> None:
        grid = _grid()
        with pytest.raises(ValueError, match="strictly positive"):
            Measure(grid, torch.zeros(grid.n_points, dtype=torch.float64))

    def test_gram_of_an_orthonormal_block_is_the_identity(self) -> None:
        grid = _grid()
        weight = torch.exp(-grid.points().norm(dim=-1))
        measure = Measure(grid, weight)
        from cdft.eigen.rayleigh_ritz import orthonormalise

        block = orthonormalise(torch.randn((5, grid.n_points), dtype=torch.float64), measure)
        gram = measure.gram(block)
        identity = torch.eye(5, dtype=torch.float64)
        assert float((gram - identity).abs().max()) < 1e-12


def test_hydrogen_ground_state_beats_the_bare_operator_by_orders() -> None:
    """The transformed operator beats the bare one by over an order on the same grid and seed."""
    from cdft.eigen.chefsi import ChebyshevFilteredSubspace
    from cdft.operators.external import external_potential
    from contract import (
        AtomicStructure,
        EigenConfig,
        ExternalPotentialKind,
        ExternalPotentialSpec,
    )

    grid = _grid(box=14.0, h=0.5)
    config = EigenConfig(n_extra_states=2, chebyshev_degree=14, max_iterations=80, residual_tol=1e-8)
    positions = torch.zeros((1, 3), dtype=torch.float64)

    factor = CuspFactor(grid, (1.0,), positions)
    transformed = CuspFactoredHamiltonian(grid, factor)
    got = ChebyshevFilteredSubspace(grid, config).solve(
        transformed, 1, generator=torch.Generator().manual_seed(0)
    )

    structure = AtomicStructure(numbers=(1,), positions=((0.0, 0.0, 0.0),))
    spec = ExternalPotentialSpec(kind=ExternalPotentialKind.NUCLEAR_COULOMB, charges=(1.0,))
    plain = LocalHamiltonian(grid, external_potential(spec, structure, grid))
    bare = ChebyshevFilteredSubspace(grid, config).solve(
        plain, 1, generator=torch.Generator().manual_seed(0)
    )

    transformed_error = abs(float(got.eigenvalues[0]) + 0.5)
    bare_error = abs(float(bare.eigenvalues[0]) + 0.5)
    assert transformed_error < 1e-3
    assert bare_error > 1e-2
    assert bare_error / transformed_error > 20.0
    assert math.isfinite(transformed_error)


class TestLOBPCGPreconditionerFrame:
    """C5 (D-56): the kinetic preconditioner acts on ``psi = f phi``, mapped back to ``phi``."""

    def test_similarity_frame_is_symmetric_in_the_weighted_measure(self) -> None:
        """``<a, f^-1 K f b>_{f^2}`` is symmetric; the frame ``<a, K b>_{f^2}`` is not."""
        from cdft.eigen.lobpcg import LOBPCG
        from contract import EigenConfig

        grid = _grid(box=8.0, h=0.5)
        factor = CuspFactor(grid, (1.0,), torch.zeros((1, 3), dtype=torch.float64))
        hamiltonian = CuspFactoredHamiltonian(grid, factor)
        solver = LOBPCG(grid, EigenConfig(), precondition="similarity")
        f = solver._cusp_factor(hamiltonian)
        assert f is not None
        generator = torch.Generator().manual_seed(11)
        a = torch.randn((1, grid.n_points), dtype=torch.float64, generator=generator)
        b = torch.randn((1, grid.n_points), dtype=torch.float64, generator=generator)
        weight = factor.weight
        apply = lambda v: solver._apply_preconditioner(v * f) / f  # noqa: E731
        left = float((weight * a * apply(b)).sum())
        right = float((weight * b * apply(a)).sum())
        assert abs(left - right) < 1e-10 * max(1.0, abs(left))
        plain_left = float((weight * a * solver._apply_preconditioner(b)).sum())
        plain_right = float((weight * b * solver._apply_preconditioner(a)).sum())
        assert abs(plain_left - plain_right) > 1e-6 * max(1.0, abs(plain_left))

    def test_frames_are_named_and_the_default_is_the_similarity_frame(self) -> None:
        from cdft.eigen.lobpcg import PRECONDITIONER_FRAMES, LOBPCG
        from contract import EigenConfig

        grid = _grid(box=4.0, h=1.0)
        assert LOBPCG(grid, EigenConfig()).frame == "similarity"
        assert LOBPCG(grid, EigenConfig(), precondition=False).frame == "none"
        for frame in PRECONDITIONER_FRAMES:
            assert LOBPCG(grid, EigenConfig(), precondition=frame).frame == frame
        with pytest.raises(ValueError):
            LOBPCG(grid, EigenConfig(), precondition="psi")

    def test_plain_hamiltonian_has_no_factor_and_uses_the_filter_directly(self) -> None:
        from cdft.eigen.lobpcg import LOBPCG
        from contract import EigenConfig

        grid = _grid(box=4.0, h=1.0)
        plain = LocalHamiltonian(grid, torch.zeros(grid.n_points, dtype=torch.float64))
        assert LOBPCG(grid, EigenConfig())._cusp_factor(plain) is None


class TestLOBPCGHelperMeasure:
    """O-27: the LOBPCG helpers judge in the operator's measure, as the solve does (D-56)."""

    @staticmethod
    def _hydrogen_problem():
        """The registered ``h_atom`` scenario on its cusp-factorised operator, a 21^3 grid."""
        import dataclasses

        from cdft.config import numerics_for_spec
        from cdft.physics_config import REGISTRY
        from cdft.scf.noninteracting import build_hamiltonian

        scenario = REGISTRY["h_atom"]
        numerics, _ = numerics_for_spec(scenario, device="cpu")
        numerics = dataclasses.replace(
            numerics,
            grid=dataclasses.replace(numerics.grid, spacing=0.5, box_lengths=(10.0, 10.0, 10.0)),
            eigen=dataclasses.replace(
                numerics.eigen, cross_check=False, max_iterations=3, chebyshev_degree=8
            ),
        )
        grid, hamiltonian, _ = build_hamiltonian(
            scenario, numerics, torch.device("cpu"), derive_grid=False
        )
        return grid, hamiltonian, numerics.eigen

    def test_helpers_agree_with_chefsi_in_the_operator_measure(self) -> None:
        from cdft.eigen.chefsi import ChebyshevFilteredSubspace
        from cdft.eigen.lobpcg import LOBPCG
        from cdft.eigen.rayleigh_ritz import orthonormality_error, residual_norms

        grid, hamiltonian, config = self._hydrogen_problem()
        assert isinstance(hamiltonian, CuspFactoredHamiltonian)
        result = ChebyshevFilteredSubspace(grid, config).solve(
            hamiltonian, 2, generator=torch.Generator().manual_seed(0)
        )
        helper = LOBPCG(grid, config)
        residual = helper.residuals(hamiltonian, result.eigenvalues, result.eigenvectors)
        # CheFSI's own residuals are residual_norms(..., hamiltonian.measure): the same function,
        # the same measure, so the two agree to round-off (1e-12 is far above float64 noise here).
        assert float((residual - result.residuals).abs().max()) < 1e-12
        expected = residual_norms(
            hamiltonian, result.eigenvalues, result.eigenvectors, hamiltonian.measure
        )
        assert torch.equal(residual, expected)
        orth = helper.orthonormality_error(hamiltonian, result.eigenvectors)
        assert orth == orthonormality_error(result.eigenvectors, hamiltonian.measure)
        assert orth < 1e-12
        # The uniform-grid measure is the wrong inner product on this path: it says the block is
        # far from orthonormal and its residuals are off by orders, which is what O-27 reported.
        assert orthonormality_error(result.eigenvectors, grid) > 1.0
        wrong = residual_norms(hamiltonian, result.eigenvalues, result.eigenvectors, grid)
        assert float((wrong - expected).abs().max()) > 1.0

    def test_plain_operator_falls_back_to_the_uniform_measure(self) -> None:
        from cdft.eigen.lobpcg import LOBPCG
        from cdft.eigen.rayleigh_ritz import orthonormality_error
        from contract import EigenConfig

        grid = _grid(box=4.0, h=1.0)
        plain = LocalHamiltonian(grid, torch.zeros(grid.n_points, dtype=torch.float64))
        generator = torch.Generator().manual_seed(3)
        block = torch.randn((2, grid.n_points), dtype=torch.float64, generator=generator)
        helper = LOBPCG(grid, EigenConfig())
        assert helper.orthonormality_error(plain, block) == orthonormality_error(block, grid)
