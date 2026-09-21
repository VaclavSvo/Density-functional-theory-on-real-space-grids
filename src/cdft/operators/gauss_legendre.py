"""Gauss--Legendre rules without numpy's LAPACK.

``numpy.polynomial.legendre.leggauss`` solves the Golub--Welsch eigenproblem in numpy's bundled
OpenBLAS, which can abort the interpreter once ``torch`` has been imported. These rules use
``torch.linalg.eigvalsh`` and then the same Newton polish, symmetrisation and normalisation as
numpy, so nodes and weights are bit-identical to ``leggauss`` for every order 1..240; no CPU number
moves. Only ``numpy.polynomial``'s polynomial evaluations are used below -- plain elementwise
arithmetic that never enters BLAS or LAPACK.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from numpy.polynomial import legendre as _legendre

__all__ = ["gauss_legendre"]


@lru_cache(maxsize=None)
def _rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute (and memoise) the ``order``-point rule on ``[-1, 1]``.

    Mirrors ``leggauss`` step for step -- companion-matrix eigenvalues, one Newton step on ``P_n``,
    weights ``1 / (P_{n-1} P_n')`` scaled to sum to two, both halves symmetrised -- so the order
    must not be rearranged. The Newton step removes any dependence on which LAPACK gave the
    eigenvalues.
    """
    if order < 1:
        raise ValueError("Gauss-Legendre order must be at least one")
    coefficients = np.zeros(order + 1, dtype=np.float64)
    coefficients[-1] = 1.0
    companion = _legendre.legcompanion(coefficients)
    nodes = torch.linalg.eigvalsh(torch.from_numpy(np.ascontiguousarray(companion))).numpy()
    nodes = np.sort(nodes)
    value = _legendre.legval(nodes, coefficients)
    derivative = _legendre.legval(nodes, _legendre.legder(coefficients))
    nodes = nodes - value / derivative
    lower = _legendre.legval(nodes, coefficients[1:])
    lower /= np.abs(lower).max()
    derivative /= np.abs(derivative).max()
    weights = 1.0 / (lower * derivative)
    weights = (weights + weights[::-1]) / 2.0
    nodes = (nodes - nodes[::-1]) / 2.0
    weights *= 2.0 / weights.sum()
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights


def gauss_legendre(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights of the ``order``-point Gauss--Legendre rule on ``[-1, 1]``.

    Drop-in replacement for ``leggauss(order)`` with identical output and no numpy LAPACK call.
    The arrays are cached per order and read-only; copy before mutating.
    """
    return _rule(int(order))
