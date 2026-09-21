"""Staggered divergence form: the discretisation that makes the weighted operator exactly symmetric.

    A_w phi = -(1/2 f^2) sum_d D_d^T [ f^2_face,d  D_d phi ] + W phi

with ``D_d`` the staggered first derivative onto cell faces and ``D_d^T`` its exact transpose. The
form is self-adjoint in the ``f^2`` measure by construction, at any stencil order and for any
positive weight, and has no checkerboard null space. Face weights are evaluated analytically at the
face coordinate, never interpolated. Why the two discretisations this replaced failed: D-38,
``docs/03_METHOD.md (Part E)``.
"""

from __future__ import annotations

import math
from fractions import Fraction
from functools import lru_cache

import torch

__all__ = [
    "staggered_coefficients",
    "staggered_derivative",
    "staggered_derivative_transpose",
    "weighted_divergence",
]


@lru_cache(maxsize=32)
def staggered_coefficients(accuracy: int) -> tuple[float, ...]:
    """Weights for the first derivative at a face, from the ``accuracy`` nearest grid points.

    The face lies at ``x_i + h/2``; the stencil uses nodes ``x_{i+1+k}``, ``k = -p .. p-1``,
    ``p = accuracy // 2``, at signed distances ``(k + 1/2) h``, solving

        sum_k c_k (k + 1/2)^m = 1! delta_{m,1},   m = 0 .. accuracy - 1.

    Returned in order ``k = -p .. p-1``, normalised so
    ``f'(x_i + h/2) ~= h^-1 sum_k c_k f(x_{i+1+k})``. Solved in exact rational arithmetic.

    Examples
    --------
    >>> staggered_coefficients(2)
    (-1.0, 1.0)
    >>> tuple(round(c, 10) for c in staggered_coefficients(4))
    (0.0416666667, -1.125, 1.125, -0.0416666667)
    """
    if accuracy % 2 or accuracy < 2:
        raise ValueError(f"accuracy must be a positive even integer, got {accuracy}")
    half = accuracy // 2
    positions = [Fraction(2 * k + 1, 2) for k in range(-half, half)]
    size = len(positions)

    matrix = [[pos**m for pos in positions] for m in range(size)]
    rhs = [Fraction(0) for _ in range(size)]
    rhs[1] = Fraction(math.factorial(1))

    for col in range(size):
        pivot = next((r for r in range(col, size) if matrix[r][col] != 0), None)
        if pivot is None:  # pragma: no cover - a Vandermonde matrix is never singular
            raise ArithmeticError("singular staggered finite-difference system")
        matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
        rhs[col], rhs[pivot] = rhs[pivot], rhs[col]
        inverse = matrix[col][col]
        matrix[col] = [v / inverse for v in matrix[col]]
        rhs[col] = rhs[col] / inverse
        for row in range(size):
            if row == col or matrix[row][col] == 0:
                continue
            factor = matrix[row][col]
            matrix[row] = [a - factor * b for a, b in zip(matrix[row], matrix[col])]
            rhs[row] = rhs[row] - factor * rhs[col]
    return tuple(float(v) for v in rhs)


def _slice_along(tensor: torch.Tensor, axis: int, start: int, length: int) -> torch.Tensor:
    """Return a view of ``length`` entries starting at ``start`` along a trailing spatial ``axis``."""
    index = [slice(None)] * 3
    index[axis] = slice(start, start + length)
    return tensor[(Ellipsis, *index)]


def _pad_axis(box: torch.Tensor, axis: int, width: int) -> torch.Tensor:
    """Return ``box`` zero-padded by ``width`` on both sides of one trailing spatial ``axis``.

    By hand, not ``torch.nn.functional.pad``: G5.7 forbids the import, and one axis at a time keeps
    the transient at ~1.3x the box instead of 2.3x.
    """
    shape = list(box.shape)
    shape[box.dim() - 3 + axis] += 2 * width
    out = torch.zeros(shape, device=box.device, dtype=box.dtype)
    index: list[slice] = [slice(None)] * 3
    index[axis] = slice(width, width + box.shape[box.dim() - 3 + axis])
    out[(Ellipsis, *index)] = box
    return out


def staggered_derivative(
    padded: torch.Tensor, axis: int, spacing: float, accuracy: int, pad: int, extent: int
) -> torch.Tensor:
    """Differentiate onto faces along one axis.

    ``padded`` is the field zero-padded by ``pad >= accuracy`` along ``axis`` (trailing dimensions
    spatial); ``extent`` is the interior length. The result has ``extent + accuracy`` entries,
    faces ``-p .. extent + p - 1``, which is exactly what the transpose needs; face ``n`` sits at
    ``x_n + h/2``.
    """
    half = accuracy // 2
    coefficients = staggered_coefficients(accuracy)
    length = extent + accuracy
    scale = 1.0 / spacing

    out: torch.Tensor | None = None
    for index, coefficient in enumerate(coefficients):
        offset = index - half  # k
        # Face n (running from -half) reads node n + 1 + k; in padded coordinates that is
        # pad + n + 1 + k, so the slice starts at pad - half + 1 + k.
        start = pad - half + 1 + offset
        term = _slice_along(padded, axis, start, length)
        out = term * (coefficient * scale) if out is None else out.add_(term, alpha=coefficient * scale)
    assert out is not None
    return out


def staggered_derivative_transpose(
    faces: torch.Tensor, axis: int, spacing: float, accuracy: int, extent: int
) -> torch.Tensor:
    """Apply the exact transpose of :func:`staggered_derivative`, faces back to interior points.

    Row ``n`` of the forward matrix holds ``c_k / h`` in column ``n + 1 + k``, so row ``m`` of the
    transpose holds ``c_k / h`` in column ``m - 1 - k``. Off by one index the operator is only
    almost symmetric, which is the failure this module exists to remove; ``test_divergence_form.py``
    materialises both as dense matrices and asserts ``Dt == D.T`` exactly. ``faces`` must carry the
    ``extent + accuracy`` entries the forward pass produces, from face ``-p``.
    """
    half = accuracy // 2
    coefficients = staggered_coefficients(accuracy)
    scale = 1.0 / spacing

    out: torch.Tensor | None = None
    for index, coefficient in enumerate(coefficients):
        offset = index - half  # k
        # Interior point m reads face m - 1 - k; faces are stored from -half, so the array index is
        # m - 1 - k + half and the slice starts at half - 1 - k.
        start = half - 1 - offset
        term = _slice_along(faces, axis, start, extent)
        out = term * (coefficient * scale) if out is None else out.add_(term, alpha=coefficient * scale)
    assert out is not None
    return out


def weighted_divergence(
    box: torch.Tensor,
    face_weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    spacing: float,
    accuracy: int,
    shape: tuple[int, int, int],
) -> torch.Tensor:
    """Return ``sum_d D_d^T [ w_d  D_d phi ]`` on the box, which equals ``-div(w grad phi)``.

    The CPU path is :func:`_weighted_divergence_eager`; CUDA dispatches to
    :func:`cdft.operators.fused.device_weighted_divergence`, which may re-associate the sums (D-65).
    """
    if box.device.type == "cuda":
        from .fused import device_weighted_divergence

        return device_weighted_divergence(box, face_weights, spacing, accuracy, shape)
    return _weighted_divergence_eager(box, face_weights, spacing, accuracy, shape)


def _weighted_divergence_eager(
    box: torch.Tensor,
    face_weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    spacing: float,
    accuracy: int,
    shape: tuple[int, int, int],
) -> torch.Tensor:
    """The eager staggered stencil behind :func:`weighted_divergence` (the CPU audit path, D-38).

    ``D^T D`` is *already* the positive Laplacian, so no sign is applied here; the form is positive
    semi-definite for any positive ``w`` at any stencil order.

    ``box`` is the field on the enclosing box, ``(..., *shape)``, zero outside the domain mask and
    with a zero halo -- the Dirichlet condition for a decayed bound state. ``face_weights[d]`` is
    ``w`` at the faces of axis ``d``, ``shape`` extended by ``accuracy`` there and indexed from face
    ``-p``, built analytically by the caller. The result is box-shaped with ``box``'s batch dims.
    """
    out: torch.Tensor | None = None
    for axis in range(3):
        padded = _pad_axis(box, axis, accuracy)
        faces = staggered_derivative(padded, axis, spacing, accuracy, accuracy, shape[axis])
        faces = faces * face_weights[axis]
        term = staggered_derivative_transpose(faces, axis, spacing, accuracy, shape[axis])
        out = term if out is None else out.add_(term)
    assert out is not None
    return out
