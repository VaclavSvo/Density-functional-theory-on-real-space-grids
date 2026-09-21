"""Operator invariants that hold for any system, grid, spacing or functional.

Grouped by what makes them true: exact algebra, variational structure, continuum symmetries
(marked ``physics``, grid-limited tolerances) and conservation. All ``fast``; seconds.
"""

from __future__ import annotations


import numpy as np
import pytest
import torch

from cdft.eigen.rayleigh_ritz import orthonormalise, self_adjointness_error
from cdft.grid import UniformGrid
from cdft.operators.cusp import CuspFactor
from cdft.operators.divergence import (
    _pad_axis,
    staggered_coefficients,
    staggered_derivative,
    staggered_derivative_transpose,
    weighted_divergence,
)
from cdft.operators.hamiltonian import CuspFactoredHamiltonian, LocalHamiltonian
from contract import BoundaryMode, DomainMode, GridConfig

pytestmark = pytest.mark.fast


def _small_grid(spacing: float = 0.5, length: float = 6.0) -> UniformGrid:
    """A grid small enough to diagonalise densely, large enough to have an interior."""
    return UniformGrid.from_config(
        GridConfig(
            spacing=spacing,
            fd_order=8,
            domain=DomainMode.BOX,
            boundary=BoundaryMode.ZERO,
            box_lengths=(length, length, length),
            mask_radius=length,
        ),
        device=torch.device("cpu"),
        dtype=torch.float64,
    )


def _cusp_operator(
    charges: tuple[float, ...] = (1.0,), offset: float = 0.0, spacing: float = 0.5
) -> tuple[UniformGrid, CuspFactoredHamiltonian]:
    """A cusp-factorised operator, with the nuclei optionally displaced off the grid points."""
    grid = _small_grid(spacing=spacing)
    positions = torch.tensor(
        [[offset, 0.0, 0.0] for _ in charges], dtype=torch.float64
    )
    if len(charges) == 2:
        positions = torch.tensor(
            [[0.0, 0.0, -1.0 + offset], [0.0, 0.0, 1.0 + offset]], dtype=torch.float64
        )
    factor = CuspFactor(grid, charges, positions)
    return grid, CuspFactoredHamiltonian(grid, factor)


# --- exact algebra: a failure here is a bug, never a tolerance --------------------------------


class TestExactAlgebra:
    """Properties that hold to round-off whatever the physics."""

    @pytest.mark.parametrize("accuracy", [2, 4, 6, 8])
    def test_staggered_transpose_is_the_actual_transpose(self, accuracy: int) -> None:
        """``D^T`` is the matrix transpose of ``D``, compared on assembled dense matrices."""
        # Assembled, not checked by algebra: the algebra is what a one-index slip would satisfy.
        size, spacing = 14, 0.37
        forward = np.zeros((size + accuracy, size))
        for column in range(size):
            basis = torch.zeros((size, 1, 1), dtype=torch.float64)
            basis[column, 0, 0] = 1.0
            faces = staggered_derivative(
                _pad_axis(basis, 0, accuracy), 0, spacing, accuracy, accuracy, size
            )
            forward[:, column] = faces[:, 0, 0].numpy()

        backward = np.zeros((size, size + accuracy))
        for column in range(size + accuracy):
            basis = torch.zeros((size + accuracy, 1, 1), dtype=torch.float64)
            basis[column, 0, 0] = 1.0
            nodes = staggered_derivative_transpose(basis, 0, spacing, accuracy, size)
            backward[:, column] = nodes[:, 0, 0].numpy()

        assert np.abs(backward - forward.T).max() == 0.0

    @pytest.mark.parametrize("accuracy", [2, 4, 6, 8])
    def test_composite_operator_is_symmetric_positive_semidefinite(self, accuracy: int) -> None:
        """``D^T w D`` is symmetric with no negative eigenvalue for a positive weight ``w``."""
        size, spacing = 14, 0.37
        generator = torch.Generator().manual_seed(11)
        weight = torch.rand((size + accuracy, 1, 1), generator=generator, dtype=torch.float64) + 0.5
        matrix = np.zeros((size, size))
        for column in range(size):
            basis = torch.zeros((size, 1, 1), dtype=torch.float64)
            basis[column, 0, 0] = 1.0
            faces = staggered_derivative(
                _pad_axis(basis, 0, accuracy), 0, spacing, accuracy, accuracy, size
            )
            nodes = staggered_derivative_transpose(
                faces * weight, 0, spacing, accuracy, size
            )
            matrix[:, column] = nodes[:, 0, 0].numpy()
        assert np.abs(matrix - matrix.T).max() < 1e-12 * np.abs(matrix).max()
        assert torch.linalg.eigvalsh(torch.as_tensor(matrix)).min().item() > -1e-10 * np.abs(matrix).max()

    def test_staggered_weights_sum_to_zero_and_reproduce_a_linear_slope(self) -> None:
        """The staggered stencil annihilates a constant and returns the slope of a ramp."""
        for accuracy in (2, 4, 6, 8):
            coefficients = staggered_coefficients(accuracy)
            half = accuracy // 2
            assert abs(sum(coefficients)) < 1e-13
            slope = sum(c * (k + 0.5) for c, k in zip(coefficients, range(-half, half)))
            assert abs(slope - 1.0) < 1e-13

    @pytest.mark.parametrize(
        ("charges", "offset"), [((1.0,), 0.0), ((1.0,), 0.17), ((2.0,), 0.0), ((1.0, 1.0), 0.13)]
    )
    def test_cusp_operator_is_self_adjoint_in_its_measure(
        self, charges: tuple[float, ...], offset: float
    ) -> None:
        """``<a|A|b>_w = <b|A|a>_w`` to round-off, nuclei on or off the grid points."""
        # The invariant the staggered divergence form exists for: D-38. A Chebyshev filter does
        # not converge on an operator that is only almost self-adjoint.
        _, hamiltonian = _cusp_operator(charges, offset)
        generator = torch.Generator().manual_seed(5)
        error = self_adjointness_error(
            hamiltonian, hamiltonian.measure, n_vectors=12, generator=generator
        )
        assert error < 1e-12, f"{charges} at offset {offset}: asymmetry {error:.3e}"

    def test_no_nuclei_reduces_to_the_untransformed_operator(self) -> None:
        """With no nuclei the transformed operator is the plain one bit for bit (one code path)."""
        grid = _small_grid()
        potential = torch.linspace(-1.0, 1.0, grid.n_points, dtype=torch.float64)
        plain = LocalHamiltonian(grid, potential)
        transformed = CuspFactoredHamiltonian(grid, CuspFactor(grid, (), torch.zeros((0, 3))), potential)
        generator = torch.Generator().manual_seed(3)
        field = torch.randn((2, grid.n_points), dtype=torch.float64, generator=generator)
        assert torch.equal(plain.apply(field), transformed.apply(field))
        assert transformed.measure.is_uniform


# --- variational structure: a failure means the wrong discretisation --------------------------


class TestVariationalStructure:
    """The discrete problem must be the variational problem it claims to be."""

    @pytest.mark.parametrize("charges", [(1.0,), (2.0,), (1.0, 1.0)])
    def test_the_checkerboard_has_a_large_positive_energy(
        self, charges: tuple[float, ...]
    ) -> None:
        """The checkerboard field is high in the spectrum, so the kinetic form has no null space."""
        # A central first-derivative stencil annihilates ``(-1)^(i+j+k)``, putting a zero-kinetic
        # mode at ``-Z^2/2`` -- the true ground-state energy, with a contaminated eigenvector.
        # The staggered stencil (D-38) cannot. The threshold tests the sign and scale only.
        grid, hamiltonian = _cusp_operator(charges)
        alternating = ((-1.0) ** grid._box_index.sum(dim=-1)).to(torch.float64).unsqueeze(0)
        alternating = alternating / hamiltonian.measure.norm(alternating)[0]
        quotient = float(
            hamiltonian.measure.cross(alternating, hamiltonian.apply(alternating))[0, 0]
        )
        assert quotient > 1.0, (
            f"the checkerboard has Rayleigh quotient {quotient:.4e} Ha; a value near the ground "
            f"state means the kinetic form has a null space and the eigenvectors are contaminated"
        )

    @pytest.mark.parametrize("charges", [(1.0,), (2.0,), (1.0, 1.0)])
    def test_kinetic_form_is_never_negative(self, charges: tuple[float, ...]) -> None:
        """``<phi| -div(w grad)/2 |phi>_w >= 0`` for every field, weight and geometry."""
        grid, hamiltonian = _cusp_operator(charges)
        generator = torch.Generator().manual_seed(19)
        fields = torch.randn((6, grid.n_points), dtype=torch.float64, generator=generator)
        box = grid.scatter_to_box(fields)
        divergence = weighted_divergence(
            box,
            hamiltonian.factor.face_weights(grid.fd_order),
            grid.spacing,
            grid.fd_order,
            grid.shape,
        )
        kinetic = grid.gather_from_box(divergence)
        quadratic = 0.5 * grid.integrate(fields * kinetic)
        assert float(quadratic.min()) >= -1e-16 * float(quadratic.abs().max())

    def test_transformed_potential_is_constant_for_a_single_nucleus(self) -> None:
        """``W = -|grad u|^2/2 = -Z^2/2`` identically for one centre, the nuclear cell included."""
        # Also the guard on cell averaging: the cell holding the nucleus must average to ``Z^2``.
        for charge in (1.0, 2.0, 8.0):
            grid = _small_grid()
            factor = CuspFactor(grid, (charge,), torch.zeros((1, 3), dtype=torch.float64))
            potential = factor.transformed_potential
            assert float((potential + 0.5 * charge**2).abs().max()) < 1e-10 * charge**2

    def test_cell_averaging_is_insensitive_to_where_the_nucleus_sits(self) -> None:
        """The weighted integral of ``W`` barely moves as the nuclei slide across one cell."""
        # Without cell averaging the point value of ``W`` near a nucleus is direction-dependent,
        # which made a finer H2+ grid give a worse answer.
        integrals = []
        for offset in (0.0, 0.1, 0.2, 0.3, 0.4):
            grid, hamiltonian = _cusp_operator((1.0, 1.0), offset=offset * 0.5)
            weight = hamiltonian.factor.weight
            integrals.append(
                float(grid.integrate(weight * hamiltonian.factor.transformed_potential))
                / float(grid.integrate(weight))
            )
        spread = max(integrals) - min(integrals)
        assert spread < 5e-3, f"weighted mean of W varies by {spread:.3e} across one cell: {integrals}"


# --- continuum symmetries: true only as well as the grid represents them -----------------------


@pytest.mark.physics
class TestContinuumSymmetries:
    """Invariances the physics has and the discretisation only approximates."""

    def test_kinetic_energy_obeys_coordinate_scaling(self) -> None:
        """G1.12: ``T_s`` scales as ``lambda^2`` under ``psi_lambda(r) = lambda^(3/2) psi(lambda r)``."""
        # On a Gaussian, whose scaled form is another Gaussian: no interpolation enters.
        for scale in (1.3, 2.0):
            energies = []
            for width in (1.0, 1.0 / scale):
                grid = _small_grid(spacing=0.2, length=8.0)
                radius = grid.points().norm(dim=-1)
                field = torch.exp(-(radius**2) / (2.0 * width**2)).unsqueeze(0)
                field = field / grid.norm(field)[..., None]
                gradient = grid.gradient(field)
                energies.append(0.5 * float(grid.integrate((gradient * gradient).sum(dim=-2))[0]))
            ratio = energies[1] / energies[0]
            assert abs(ratio - scale**2) < 2e-3 * scale**2, f"scaling ratio {ratio} vs {scale**2}"

    def test_translating_the_whole_system_leaves_the_operator_alone(self) -> None:
        """A lattice translation leaves the spectrum unchanged to round-off."""
        # Sub-cell shifts genuinely do break the symmetry and are gated separately as G2.7.
        grid = _small_grid(spacing=0.5, length=12.0)
        spectra = []
        for shift in (0.0, 0.5, 1.0):
            position = torch.tensor([[0.0, 0.0, shift]], dtype=torch.float64)
            factor = CuspFactor(grid, (1.0,), position)
            hamiltonian = CuspFactoredHamiltonian(grid, factor)
            # The trial block is built around the nucleus, so it translates with it. A fixed random
            # block in a shifted potential spans a different subspace and measures nothing.
            offset = grid.points() - position[0]
            radial = offset.norm(dim=-1)
            envelope = torch.exp(-0.35 * radial**2)
            block = torch.stack(
                [envelope, envelope * offset[:, 0], envelope * offset[:, 1], envelope * offset[:, 2]]
            )
            block = orthonormalise(block, hamiltonian.measure)
            matrix = hamiltonian.measure.cross(block, hamiltonian.apply(block))
            spectra.append(torch.linalg.eigvalsh(0.5 * (matrix + matrix.transpose(-1, -2))))
        for index in (1, 2):
            worst = float((spectra[0] - spectra[index]).abs().max())
            assert worst < 1e-8, f"lattice shift of {index} half-cells moved the spectrum by {worst:.2e}"

    def test_grid_is_isotropic_for_a_spherical_potential(self) -> None:
        """The three Cartesian directions are interchangeable for a spherical potential."""
        # Catches an axis swapped in the stencil machinery, which else shows as p-level splitting.
        grid, hamiltonian = _cusp_operator((1.0,))
        points = grid.points()
        values = []
        for axis in range(3):
            field = torch.exp(-points.norm(dim=-1)) * points[:, axis]
            field = field.unsqueeze(0)
            field = field / hamiltonian.measure.norm(field)[..., None]
            values.append(
                float(hamiltonian.measure.cross(field, hamiltonian.apply(field))[0, 0])
            )
        assert max(values) - min(values) < 1e-10 * max(abs(v) for v in values)


# --- conservation ------------------------------------------------------------------------------


class TestConservation:
    """Charge, normalisation and finiteness, whatever else changed."""

    def test_density_integrates_to_the_electron_count(self) -> None:
        """``integral n = N`` for normalised orbitals at fractional occupations."""
        grid, hamiltonian = _cusp_operator((2.0,))
        generator = torch.Generator().manual_seed(23)
        block = orthonormalise(
            torch.randn((3, grid.n_points), dtype=torch.float64, generator=generator),
            hamiltonian.measure,
        )
        occupations = torch.tensor([2.0, 1.0, 0.5], dtype=torch.float64)
        density = hamiltonian.factor.density(block, occupations)
        assert float(grid.integrate(density)) == pytest.approx(3.5, abs=1e-10)

    def test_orthonormalisation_is_idempotent_in_a_weighted_measure(self) -> None:
        """Orthonormalising an already-orthonormal block in the weighted measure is a no-op."""
        grid, hamiltonian = _cusp_operator((1.0,))
        generator = torch.Generator().manual_seed(29)
        block = orthonormalise(
            torch.randn((5, grid.n_points), dtype=torch.float64, generator=generator),
            hamiltonian.measure,
        )
        again = orthonormalise(block, hamiltonian.measure)
        gram = hamiltonian.measure.gram(again)
        assert float((gram - torch.eye(5, dtype=torch.float64)).abs().max()) < 1e-12

    def test_every_operator_field_is_finite(self) -> None:
        """No NaN and no Inf in the weight, the potential or an application, for any geometry."""
        for charges, offset in (((1.0,), 0.0), ((2.0,), 0.21), ((1.0, 1.0), 0.09)):
            grid, hamiltonian = _cusp_operator(charges, offset)
            generator = torch.Generator().manual_seed(31)
            field = torch.randn((2, grid.n_points), dtype=torch.float64, generator=generator)
            for name, tensor in (
                ("weight", hamiltonian.factor.weight),
                ("potential", hamiltonian.factor.transformed_potential),
                ("apply", hamiltonian.apply(field)),
            ):
                assert bool(torch.isfinite(tensor).all()), f"{name} is not finite for {charges}"

    def test_the_weight_refuses_to_underflow_silently(self) -> None:
        """A domain too large for the charge raises rather than returning zeros and then NaN."""
        # ``exp(-2u)`` underflows float64 past u = 354: the weight hits zero at the far corners
        # and its reciprocal infinity. No physics is lost by refusing and using a smaller box.
        grid = _small_grid(spacing=2.0, length=400.0)
        factor = CuspFactor(grid, (8.0,), torch.zeros((1, 3), dtype=torch.float64))
        with pytest.raises(ValueError, match="underflow"):
            factor.face_weights(8)
