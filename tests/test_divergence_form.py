"""The staggered divergence form (D-38), checked against explicit dense matrices.

``D^T`` being the exact transpose of ``D`` is an index claim, so every operator is materialised by
applying it to unit vectors and the identities are asserted on the matrices, not inferred from a
solve. All marked ``fast``; under a second. Closes defect D-6.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cdft.operators.divergence import (
    staggered_coefficients,
    staggered_derivative,
    staggered_derivative_transpose,
    weighted_divergence,
)

pytestmark = [pytest.mark.fast]

#: Orders exercised. 2 is the case with a hand-checkable answer, 8 is what the solver runs at.
_ORDERS = [2, 4, 6, 8]

#: Interior length of the one-dimensional test grid. Comfortably longer than the widest stencil.
_EXTENT = 12

#: Spacing. Deliberately not 1.0, so a missing or doubled ``1/h`` cannot pass.
_SPACING = 0.37


def _forward_matrix(accuracy: int, extent: int = _EXTENT, spacing: float = _SPACING) -> np.ndarray:
    """Materialise :func:`staggered_derivative` as ``(extent + accuracy, extent)``."""
    # Padding matches :func:`weighted_divergence`; anything else tests a configuration never used.
    columns = []
    for j in range(extent):
        box = torch.zeros((extent, 1, 1), dtype=torch.float64)
        box[j, 0, 0] = 1.0
        padded = torch.zeros((extent + 2 * accuracy, 1, 1), dtype=torch.float64)
        padded[accuracy : accuracy + extent] = box
        faces = staggered_derivative(padded, 0, spacing, accuracy, accuracy, extent)
        columns.append(faces[:, 0, 0].numpy().copy())
    return np.stack(columns, axis=1)


def _transpose_matrix(accuracy: int, extent: int = _EXTENT, spacing: float = _SPACING) -> np.ndarray:
    """Materialise :func:`staggered_derivative_transpose` as ``(extent, extent + accuracy)``."""
    n_faces = extent + accuracy
    columns = []
    for j in range(n_faces):
        faces = torch.zeros((n_faces, 1, 1), dtype=torch.float64)
        faces[j, 0, 0] = 1.0
        out = staggered_derivative_transpose(faces, 0, spacing, accuracy, extent)
        columns.append(out[:, 0, 0].numpy().copy())
    return np.stack(columns, axis=1)


class TestTransposeIsExact:
    """``D^T`` is the exact transpose of ``D``, the claim the module's symmetry rests on."""

    @pytest.mark.parametrize("accuracy", _ORDERS)
    def test_transpose_matrix_is_the_transpose(self, accuracy: int) -> None:
        """``Dt == D.T`` to the last bit, not to a tolerance."""
        # Both come from the same coefficients scaled by the same 1/h, so equality is exact iff
        # the index arithmetic agrees; a one-index slip still looks nearly symmetric.
        forward = _forward_matrix(accuracy)
        transpose = _transpose_matrix(accuracy)
        assert transpose.shape == forward.T.shape
        assert np.array_equal(transpose, forward.T), (
            f"order {accuracy}: the transpose operator is not the transpose of the forward one. "
            f"Largest entry difference {np.abs(transpose - forward.T).max():.3e}."
        )

    @pytest.mark.parametrize("accuracy", _ORDERS)
    def test_inner_product_identity(self, accuracy: int) -> None:
        """``<Da, b>_faces == <a, D^T b>_points`` for random vectors."""
        rng = np.random.default_rng(11)
        a = rng.standard_normal(_EXTENT)
        b = rng.standard_normal(_EXTENT + accuracy)
        forward = _forward_matrix(accuracy)
        transpose = _transpose_matrix(accuracy)
        assert float(b @ (forward @ a)) == pytest.approx(float((transpose @ b) @ a), rel=1e-14)

    def test_order_two_weights_are_the_difference_stencil(self) -> None:
        """At order 2 the face derivative is ``(f_{i+1} - f_i)/h``, which is checkable by eye."""
        assert staggered_coefficients(2) == (-1.0, 1.0)


class TestCompositeOperator:
    """``D^T w D``, the operator the Hamiltonian applies."""

    @pytest.mark.parametrize("accuracy", _ORDERS)
    def test_unweighted_composite_is_symmetric_and_positive_semidefinite(self, accuracy: int) -> None:
        """With ``w = 1`` the composite ``D^T D`` is symmetric and never negative."""
        forward = _forward_matrix(accuracy)
        composite = forward.T @ forward
        assert np.allclose(composite, composite.T, rtol=0.0, atol=1e-300) or np.array_equal(
            composite, composite.T
        )
        eigenvalues = torch.linalg.eigvalsh(torch.as_tensor(composite)).numpy()
        assert eigenvalues.min() > -1e-12 * max(abs(eigenvalues).max(), 1.0)

    @pytest.mark.parametrize("accuracy", _ORDERS)
    def test_weighted_composite_stays_symmetric_for_any_positive_weight(self, accuracy: int) -> None:
        """Symmetry survives a weight spanning nine orders, the regime ``exp(-2u)`` creates."""
        rng = np.random.default_rng(3)
        weight = np.exp(rng.uniform(-10.0, 10.0, size=_EXTENT + accuracy))
        forward = _forward_matrix(accuracy)
        composite = forward.T @ (weight[:, None] * forward)
        asymmetry = np.abs(composite - composite.T).max()
        scale = np.abs(composite).max()
        assert asymmetry <= 1e-14 * scale, f"order {accuracy}: relative asymmetry {asymmetry / scale:.3e}"

    def test_order_two_composite_is_the_three_point_laplacian(self) -> None:
        """``D^T D`` at order 2 is ``(2 f_m - f_{m+1} - f_{m-1}) / h^2``, a closed-form anchor."""
        composite = _forward_matrix(2).T @ _forward_matrix(2)
        expected = np.zeros((_EXTENT, _EXTENT))
        for i in range(_EXTENT):
            expected[i, i] = 2.0 / _SPACING**2
            if i + 1 < _EXTENT:
                expected[i, i + 1] = -1.0 / _SPACING**2
                expected[i + 1, i] = -1.0 / _SPACING**2
        assert np.allclose(composite, expected, rtol=1e-13, atol=0.0)


class TestWeightedDivergenceAgreesWithTheMatrices:
    """The public entry point is the operator the matrices above describe."""

    @pytest.mark.parametrize("accuracy", [2, 8])
    def test_matches_the_dense_composite_summed_over_all_three_axes(self, accuracy: int) -> None:
        """``weighted_divergence`` on a 3-D box equals ``sum_d D_d^T w_d D_d`` assembled densely."""
        # The one-point axes are not inert: each contributes ``D^T D`` on a length-one grid, so
        # this is the assertion that all three axes accumulate into one output.
        rng = np.random.default_rng(5)
        shape = (_EXTENT, 1, 1)
        w_axis0 = np.exp(rng.uniform(-3.0, 3.0, size=_EXTENT + accuracy))
        face_weights = (
            torch.tensor(w_axis0.reshape(_EXTENT + accuracy, 1, 1), dtype=torch.float64),
            torch.ones((_EXTENT, 1 + accuracy, 1), dtype=torch.float64),
            torch.ones((_EXTENT, 1, 1 + accuracy), dtype=torch.float64),
        )

        forward = _forward_matrix(accuracy)
        dense = forward.T @ (w_axis0[:, None] * forward)
        # Each degenerate axis adds the length-one composite, a 1x1 matrix, times the identity.
        thin = _forward_matrix(accuracy, extent=1)
        dense = dense + 2.0 * float((thin.T @ thin)[0, 0]) * np.eye(_EXTENT)

        field = rng.standard_normal(_EXTENT)
        box = torch.tensor(field.reshape(shape), dtype=torch.float64)
        result = weighted_divergence(box, face_weights, _SPACING, accuracy, shape)
        assert np.allclose(result.numpy().reshape(-1), dense @ field, rtol=1e-11, atol=1e-12)
