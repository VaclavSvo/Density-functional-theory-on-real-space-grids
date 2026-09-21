"""High-order central finite differences: weights, and their application to a padded box.

The discretisation of D-03. Weights are solved in exact rational arithmetic rather than transcribed,
and nothing here imports ``torch.nn`` (gate G5.7 forbids it). Rationale:
``docs/03_METHOD.md (Part E)``.
"""

from __future__ import annotations

import math
from fractions import Fraction
from functools import lru_cache

import torch

__all__ = [
    "fd_coefficients",
    "stencil_half_width",
    "shifted_view",
    "laplacian_from_padded",
    "gradient_from_padded",
]


def stencil_half_width(derivative: int, accuracy: int) -> int:
    """Return the half-width of the central stencil, ``accuracy // 2 + (derivative - 1) // 2``."""
    if accuracy % 2 or accuracy < 2:
        raise ValueError(f"accuracy must be a positive even integer, got {accuracy}")
    if derivative < 1:
        raise ValueError(f"derivative must be at least 1, got {derivative}")
    return accuracy // 2 + (derivative - 1) // 2


@lru_cache(maxsize=64)
def fd_coefficients(derivative: int, accuracy: int) -> tuple[float, ...]:
    """Return central finite-difference weights for ``d^n f / dx^n`` at the given accuracy order.

    Ordered from offset ``-hw`` to ``+hw`` and normalised so
    ``f^(n)(x) ~= h**-n * sum_j c_j f(x + j h)``. They solve the Taylor condition
    ``sum_j c_j j**m = n! delta_{m,n}`` by exact Gaussian elimination over
    :class:`fractions.Fraction`; the only floating-point error is the final conversion.

    Examples
    --------
    >>> fd_coefficients(2, 2)
    (1.0, -2.0, 1.0)
    >>> fd_coefficients(1, 2)
    (-0.5, 0.0, 0.5)
    """
    hw = stencil_half_width(derivative, accuracy)
    offsets = list(range(-hw, hw + 1))
    size = len(offsets)

    # Vandermonde system A[m][j] = offsets[j]**m, right-hand side derivative! at row `derivative`.
    matrix = [[Fraction(off) ** m for off in offsets] for m in range(size)]
    rhs = [Fraction(0) for _ in range(size)]
    rhs[derivative] = Fraction(math.factorial(derivative))

    for col in range(size):
        pivot = next((r for r in range(col, size) if matrix[r][col] != 0), None)
        if pivot is None:  # pragma: no cover - a Vandermonde matrix is never singular
            raise ArithmeticError("singular finite-difference system")
        matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
        rhs[col], rhs[pivot] = rhs[pivot], rhs[col]
        inv = matrix[col][col]
        matrix[col] = [v / inv for v in matrix[col]]
        rhs[col] = rhs[col] / inv
        for row in range(size):
            if row == col or matrix[row][col] == 0:
                continue
            factor = matrix[row][col]
            matrix[row] = [a - factor * b for a, b in zip(matrix[row], matrix[col])]
            rhs[row] = rhs[row] - factor * rhs[col]
    return tuple(float(v) for v in rhs)


def shifted_view(padded: torch.Tensor, axis: int, offset: int, pad: int, shape: tuple[int, int, int]) -> torch.Tensor:
    """Return the interior of a padded box shifted by ``offset`` along ``axis``; a view, not a copy."""
    slices = [slice(pad, pad + shape[d]) for d in range(3)]
    slices[axis] = slice(pad + offset, pad + offset + shape[axis])
    return padded[(Ellipsis, *slices)]


def laplacian_from_padded(
    padded: torch.Tensor, spacing: float, accuracy: int, shape: tuple[int, int, int], pad: int
) -> torch.Tensor:
    """Apply the order-``accuracy`` Cartesian Laplacian to a padded box.

    ``padded`` has trailing spatial dimensions of size ``shape[d] + 2 * pad`` with the halo already
    filled per the boundary mode; this function does not know which condition produced it, which is
    what lets one stencil serve a molecule in vacuum and a hard-walled box. ``spacing`` is in bohr;
    ``accuracy`` is the nominal order ``2p`` that G0.1 checks the measured convergence against. The
    result is the interior, with ``padded``'s batch dimensions.
    """
    coeffs = fd_coefficients(2, accuracy)
    hw = stencil_half_width(2, accuracy)
    scale = 1.0 / (spacing * spacing)

    out: torch.Tensor | None = None
    for axis in range(3):
        for index, coeff in enumerate(coeffs):
            if coeff == 0.0:
                continue
            term = shifted_view(padded, axis, index - hw, pad, shape)
            out = term * (coeff * scale) if out is None else out.add_(term, alpha=coeff * scale)
    assert out is not None  # every stencil has at least one non-zero weight
    return out


def component_from_padded(
    padded: torch.Tensor, spacing: float, accuracy: int, shape: tuple[int, int, int], pad: int, axis: int
) -> torch.Tensor:
    """Apply the order-``accuracy`` first derivative along ``axis`` to a padded box.

    One component of :func:`gradient_from_padded`, accumulated over the same taps in the same order,
    so the two agree bit for bit; asking for one component costs one sweep instead of three.
    """
    coeffs = fd_coefficients(1, accuracy)
    hw = stencil_half_width(1, accuracy)
    scale = 1.0 / spacing

    acc: torch.Tensor | None = None
    for index, coeff in enumerate(coeffs):
        if coeff == 0.0:
            continue
        term = shifted_view(padded, axis, index - hw, pad, shape)
        acc = term * (coeff * scale) if acc is None else acc.add_(term, alpha=coeff * scale)
    assert acc is not None
    return acc


def gradient_from_padded(
    padded: torch.Tensor, spacing: float, accuracy: int, shape: tuple[int, int, int], pad: int
) -> torch.Tensor:
    """Apply the order-``accuracy`` Cartesian gradient to a padded box.

    Returns ``(*batch, 3, *interior_shape)``: the component axis precedes the spatial axes, matching
    the ``(n_spin, 3, n_pts)`` layout the contract declares for density gradients.
    """
    components = [
        component_from_padded(padded, spacing, accuracy, shape, pad, axis) for axis in range(3)
    ]
    return torch.stack(components, dim=-4)
