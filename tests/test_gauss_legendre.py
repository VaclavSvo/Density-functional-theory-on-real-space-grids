"""The LAPACK-free Gauss--Legendre rule: exactness, the closed form, caching, numpy agreement.

All marked ``fast``; instant. The numpy comparison is skipped where its LAPACK aborts.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest
import torch

from cdft.operators.gauss_legendre import gauss_legendre

pytestmark = pytest.mark.fast

_NUMPY_LAPACK_ABORTS = sys.platform == "win32" and os.environ.get("CDFT_NUMPY_LAPACK_OK") != "1"


@pytest.mark.parametrize("order", [1, 2, 3, 4, 8, 16, 61, 160, 240])
def test_is_exact_on_polynomials_and_symmetric(order: int) -> None:
    """Integrates ``x^k`` exactly for ``k <= 2n - 1``, nodes antisymmetric, weights sum to two."""
    nodes, weights = gauss_legendre(order)
    assert nodes.shape == (order,) and weights.shape == (order,)
    assert np.array_equal(nodes, -nodes[::-1]) and np.array_equal(weights, weights[::-1])
    assert abs(weights.sum() - 2.0) <= 4.0 * np.finfo(np.float64).eps
    for k in range(0, 2 * order, max(1, (2 * order) // 12)):
        exact = 0.0 if k % 2 else 2.0 / (k + 1)
        assert abs(float(weights @ nodes**k) - exact) <= 1e-14 * max(1.0, order)


def test_order_four_matches_the_closed_form() -> None:
    """Order 4, the cell-average rule of :class:`CuspFactor`, against its textbook values."""
    nodes, weights = gauss_legendre(4)
    inner = math.sqrt(3.0 / 7.0 - 2.0 / 7.0 * math.sqrt(6.0 / 5.0))
    outer = math.sqrt(3.0 / 7.0 + 2.0 / 7.0 * math.sqrt(6.0 / 5.0))
    assert np.allclose(nodes, [-outer, -inner, inner, outer], rtol=0.0, atol=4e-16)
    w_inner = (18.0 + math.sqrt(30.0)) / 36.0
    w_outer = (18.0 - math.sqrt(30.0)) / 36.0
    assert np.allclose(weights, [w_outer, w_inner, w_inner, w_outer], rtol=0.0, atol=4e-16)


def test_rule_is_cached_and_read_only() -> None:
    """Same arrays on repeat, and they cannot be mutated by a caller."""
    a = gauss_legendre(8)
    b = gauss_legendre(8)
    assert a[0] is b[0] and a[1] is b[1]
    with pytest.raises(ValueError):
        a[0][0] = 0.0


@pytest.mark.skipif(
    _NUMPY_LAPACK_ABORTS,
    reason="numpy's eigvalsh aborts the interpreter after torch is imported on this platform "
    "(03_METHOD.md Part A section 9); set CDFT_NUMPY_LAPACK_OK=1 to run the comparison anyway",
)
@pytest.mark.parametrize("order", list(range(1, 33)) + [61, 100, 160, 240])
def test_bit_identical_to_numpy_leggauss(order: int) -> None:
    """The reference implementation and numpy agree to the last bit, nodes and weights."""
    nodes, weights = gauss_legendre(order)
    ref_nodes, ref_weights = np.polynomial.legendre.leggauss(order)
    assert np.array_equal(nodes, ref_nodes)
    assert np.array_equal(weights, ref_weights)


def test_torch_path_does_not_touch_numpy_linalg(monkeypatch: pytest.MonkeyPatch) -> None:
    """The rule never calls numpy's eigvalsh, whose LAPACK aborts after torch is imported."""
    from cdft.operators import gauss_legendre as module

    module._rule.cache_clear()

    def boom(*args, **kwargs):  # pragma: no cover - reached only on regression
        raise AssertionError("numpy.linalg.eigvalsh must not be called")

    monkeypatch.setattr(np.linalg, "eigvalsh", boom)
    monkeypatch.setattr(np.linalg, "eigh", boom)
    nodes, weights = gauss_legendre(6)
    assert nodes.shape == (6,)
    assert torch.is_tensor(torch.from_numpy(np.array(weights)))
