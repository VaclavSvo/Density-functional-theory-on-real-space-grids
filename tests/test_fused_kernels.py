"""The fused device kernels (D-65), checked on the CPU.

The GEMM tier is the eager operator to round-off (BLAS re-orders the inner products), the registry
picks the tier the rules say, and the CPU path keeps the pre-fusion bits. Marked ``fast``; seconds.
"""

from __future__ import annotations

import pytest
import torch

from cdft.operators import fused
from cdft.operators.divergence import (
    _pad_axis,
    staggered_derivative,
    staggered_derivative_transpose,
    weighted_divergence,
)

pytestmark = [pytest.mark.fast]

_SPACING = 0.37
_SHAPE = (9, 11, 13)


def _pre_wp3_weighted_divergence(box, face_weights, spacing, accuracy, shape):
    """The body of ``weighted_divergence`` as it was before the fused tiers, copied verbatim."""
    out = None
    for axis in range(3):
        padded = _pad_axis(box, axis, accuracy)
        faces = staggered_derivative(padded, axis, spacing, accuracy, accuracy, shape[axis])
        faces = faces * face_weights[axis]
        term = staggered_derivative_transpose(faces, axis, spacing, accuracy, shape[axis])
        out = term if out is None else out.add_(term)
    return out


def _weights(accuracy: int, generator: torch.Generator) -> tuple[torch.Tensor, ...]:
    """Positive random face weights of the shapes the operator expects."""
    out = []
    for axis in range(3):
        sizes = list(_SHAPE)
        sizes[axis] += accuracy
        out.append(0.1 + torch.rand(sizes, generator=generator, dtype=torch.float64))
    return tuple(out)


def _box(batch: int, generator: torch.Generator) -> torch.Tensor:
    """A random batch of boxes."""
    return torch.randn((batch, *_SHAPE), generator=generator, dtype=torch.float64)


class TestGemmTier:
    """The GEMM tier is the D-38 operator, up to summation order."""

    @pytest.mark.parametrize("accuracy", [2, 4, 8])
    def test_matrix_is_the_eager_stencil(self, accuracy: int) -> None:
        """``D`` applied to unit vectors reproduces :func:`staggered_derivative` exactly."""
        extent = 10
        forward, transpose = fused.staggered_matrices(extent, _SPACING, accuracy)
        assert forward.shape == (extent + accuracy, extent)
        assert torch.equal(transpose, forward.T)
        for j in range(extent):
            unit = torch.zeros((extent, 1, 1), dtype=torch.float64)
            unit[j] = 1.0
            faces = staggered_derivative(_pad_axis(unit, 0, accuracy), 0, _SPACING, accuracy, accuracy, extent)
            assert torch.equal(faces[:, 0, 0], forward[:, j])

    @pytest.mark.parametrize("accuracy", [2, 4, 8])
    def test_equals_eager_to_round_off(self, accuracy: int) -> None:
        """Agreement with the eager stencil to 1e-13 relative on random boxes and weights."""
        generator = torch.Generator().manual_seed(accuracy)
        weights = _weights(accuracy, generator)
        box = _box(3, generator)
        eager = weighted_divergence(box, weights, _SPACING, accuracy, _SHAPE)
        gemm = fused.gemm_weighted_divergence(box, weights, _SPACING, accuracy, _SHAPE)
        assert gemm.shape == eager.shape
        scale = float(eager.abs().max())
        assert float((gemm - eager).abs().max()) <= 1e-13 * scale

    @pytest.mark.parametrize("accuracy", [2, 4, 8])
    def test_unbatched_box(self, accuracy: int) -> None:
        """A single box without a batch dimension."""
        generator = torch.Generator().manual_seed(10 + accuracy)
        weights = _weights(accuracy, generator)
        box = _box(1, generator)[0]
        eager = weighted_divergence(box, weights, _SPACING, accuracy, _SHAPE)
        gemm = fused.gemm_weighted_divergence(box, weights, _SPACING, accuracy, _SHAPE)
        assert float((gemm - eager).abs().max()) <= 1e-13 * float(eager.abs().max())

    @pytest.mark.parametrize("accuracy", [2, 4, 8])
    def test_transpose_symmetric(self, accuracy: int) -> None:
        """``<a|K|b> == <b|K|a>`` to round-off for the GEMM operator."""
        generator = torch.Generator().manual_seed(20 + accuracy)
        weights = _weights(accuracy, generator)
        a, b = _box(1, generator)[0], _box(1, generator)[0]
        kab = float((a * fused.gemm_weighted_divergence(b, weights, _SPACING, accuracy, _SHAPE)).sum())
        kba = float((b * fused.gemm_weighted_divergence(a, weights, _SPACING, accuracy, _SHAPE)).sum())
        norm = float((a.abs() * fused.gemm_weighted_divergence(b.abs(), weights, _SPACING, accuracy, _SHAPE).abs()).sum())
        assert abs(kab - kba) <= 1e-13 * max(norm, abs(kab))


class TestRegistry:
    """Tier selection, with ``CDFT_FUSED`` forced and the probes faked."""

    @pytest.fixture(autouse=True)
    def _clean(self, monkeypatch):
        """Reset the registry around each test."""
        monkeypatch.delenv(fused.ENV_FUSED, raising=False)
        fused.reset_registry()
        yield
        fused.reset_registry()

    def test_cpu_is_always_eager(self, monkeypatch) -> None:
        """Whatever is requested, the CPU resolves to eager without probing."""
        def boom():
            raise AssertionError("probe must not run on the CPU")

        for requested in ("auto", "compile", "gemm", "eager"):
            tier, reason = fused.resolve_tier(requested, False, boom, boom)
            assert tier == "eager" and "CPU" in reason
        monkeypatch.setenv(fused.ENV_FUSED, "gemm")
        assert fused.kernel_record("cpu")["tier"] == "eager"

    def test_auto_order(self) -> None:
        """compile, then gemm, then eager, with every refusal in the reason."""
        ok = lambda: None  # noqa: E731
        assert fused.resolve_tier("auto", True, ok, ok)[0] == "compile"
        tier, reason = fused.resolve_tier("auto", True, lambda: "no triton", ok)
        assert tier == "gemm" and "no triton" in reason
        tier, reason = fused.resolve_tier("auto", True, lambda: "no triton", lambda: "slower")
        assert tier == "eager" and "no triton" in reason and "slower" in reason

    def test_forced(self) -> None:
        """A forced tier is taken if its probe passes and raises (G5.5) if it does not."""
        ok = lambda: None  # noqa: E731
        assert fused.resolve_tier("eager", True, ok, ok) == ("eager", f"forced by {fused.ENV_FUSED}=eager")
        assert fused.resolve_tier("gemm", True, lambda: "x", ok)[0] == "gemm"
        assert fused.resolve_tier("compile", True, ok, lambda: "x")[0] == "compile"
        with pytest.raises(RuntimeError, match="no triton"):
            fused.resolve_tier("compile", True, lambda: "no triton", ok)

    def test_environment_is_read(self, monkeypatch) -> None:
        """``CDFT_FUSED`` values are validated."""
        for value in ("compile", "gemm", "eager", "auto"):
            monkeypatch.setenv(fused.ENV_FUSED, value)
            assert fused._requested() == value
        monkeypatch.setenv(fused.ENV_FUSED, "triton")
        with pytest.raises(ValueError):
            fused._requested()

    def test_unselected_cuda_record(self) -> None:
        """Before first use the CUDA record says so rather than guessing."""
        assert fused.kernel_record("cuda:0")["tier"] == "unselected"


class TestCpuPathUntouched:
    """The CPU production path is the pre-fusion code, bit for bit."""

    @pytest.mark.parametrize("accuracy", [2, 4, 8])
    def test_weighted_divergence_bitwise(self, accuracy: int, monkeypatch) -> None:
        """Even with a fused tier requested, the CPU returns the captured eager bits."""
        monkeypatch.setenv(fused.ENV_FUSED, "gemm")
        generator = torch.Generator().manual_seed(30 + accuracy)
        weights = _weights(accuracy, generator)
        box = _box(2, generator)
        got = weighted_divergence(box, weights, _SPACING, accuracy, _SHAPE)
        want = _pre_wp3_weighted_divergence(box, weights, _SPACING, accuracy, _SHAPE)
        assert torch.equal(got, want)

    def test_chebyshev_update_bitwise(self, monkeypatch) -> None:
        """The CPU branch is the plain recurrence expression, bit for bit."""
        monkeypatch.setenv(fused.ENV_FUSED, "compile")
        generator = torch.Generator().manual_seed(7)
        applied, y, previous = (torch.randn((4, 50), generator=generator, dtype=torch.float64) for _ in range(3))
        c, e, sigma, sigma2 = 1700.3, 812.9, -0.731, -0.402
        want = (applied - c * y) * (2.0 * sigma2 / e) - (sigma * sigma2) * previous
        got = fused.chebyshev_update(applied, y, previous, c, 2.0 * sigma2 / e, sigma * sigma2)
        assert torch.equal(got, want)

    @pytest.mark.parametrize("accuracy", [2, 4, 8])
    def test_traceable_stencil_agrees(self, accuracy: int) -> None:
        """The function the compiled tier traces is the eager stencil (checked uncompiled)."""
        generator = torch.Generator().manual_seed(40 + accuracy)
        weights = _weights(accuracy, generator)
        box = _box(2, generator)
        forward, transpose = fused._stencil_taps(_SPACING, accuracy)
        got = fused._divergence_traceable(box, *weights, forward, transpose, accuracy, _SHAPE)
        want = weighted_divergence(box, weights, _SPACING, accuracy, _SHAPE)
        assert float((got - want).abs().max()) <= 1e-14 * float(want.abs().max())

    def test_pure_functions_agree(self) -> None:
        """The functions the compiled tier traces compute the eager values (checked uncompiled)."""
        generator = torch.Generator().manual_seed(8)
        applied, y, previous = (torch.randn((3, 20), generator=generator, dtype=torch.float64) for _ in range(3))
        c, a, b = 3.5, -0.25, 0.125
        want = (applied - c * y) * a - b * previous
        got = fused._chebyshev_pointwise(applied, y, previous, torch.tensor([c, a, b], dtype=torch.float64))
        assert torch.equal(got, want)
