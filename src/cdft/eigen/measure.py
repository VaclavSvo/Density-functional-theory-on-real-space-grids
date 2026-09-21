"""The inner product an eigensolver works in.

Every eigensolver is written against an abstract symmetric positive-definite product rather than a
hard-coded ``h**3`` sum, because the cusp-transformed operator is self-adjoint only in
``<a,b>_w = integral(f^2 a b)``. The uniform case is dispatched separately, so an unweighted run
pays nothing. Rationale: ``docs/03_METHOD.md (Part E)``.
"""

from __future__ import annotations

import torch

from ..grid import UniformGrid

__all__ = ["Measure"]


class Measure:
    """A quadrature measure on a grid: ``<a,b> = sum_r q(r) a(r) b(r)``."""

    def __init__(self, grid: UniformGrid, weight: torch.Tensor | None = None) -> None:
        """Build a measure. ``weight = None`` means the plain uniform quadrature ``h**3``."""
        if weight is not None:
            if weight.shape[-1] != grid.n_points:
                raise ValueError(
                    f"measure weight has {weight.shape[-1]} points, grid has {grid.n_points}"
                )
            if bool((weight <= 0.0).any()):
                raise ValueError(
                    "a quadrature weight must be strictly positive everywhere; a zero or negative "
                    "weight makes the inner product degenerate and the Gram matrix singular"
                )
        self.grid = grid
        self.weight = weight
        self._quadrature = (
            None if weight is None else weight.to(torch.float64) * grid.volume_element
        )

    @classmethod
    def uniform(cls, grid: UniformGrid) -> "Measure":
        """Return the plain ``h**3`` measure, used by every non-transformed run."""
        return cls(grid)

    @property
    def is_uniform(self) -> bool:
        """Whether this is the unweighted measure, which has a faster code path."""
        return self.weight is None

    @property
    def quadrature(self) -> torch.Tensor:
        """The per-point quadrature weight, materialised. Uniform measures build it on demand."""
        if self._quadrature is None:
            return torch.full(
                (self.grid.n_points,),
                self.grid.volume_element,
                device=self.grid.device,
                dtype=torch.float64,
            )
        return self._quadrature

    @property
    def dynamic_range(self) -> float:
        """Ratio of the largest to the smallest quadrature weight, recorded on transformed runs.

        About ``exp(2 Z R)`` for the cusp factor. Harmless in itself -- physical quantities see
        ``w * phi**2`` -- but it is what would explain a badly conditioned Gram matrix.
        """
        if self._quadrature is None:
            return 1.0
        return float(self._quadrature.max() / self._quadrature.min())

    def inner(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Return ``<a,b>``, accumulated in float64 whatever the operand dtype."""
        product = a.to(torch.float64) * b.to(torch.float64)
        if self._quadrature is None:
            return product.sum(dim=-1) * self.grid.volume_element
        return (product * self._quadrature).sum(dim=-1)

    def norm(self, a: torch.Tensor) -> torch.Tensor:
        """Return ``sqrt(<a,a>)``."""
        return self.inner(a, a).clamp_min(0.0).sqrt()

    def gram(self, block: torch.Tensor) -> torch.Tensor:
        """Return the Gram matrix ``S_ij = <b_i, b_j>`` for a block of shape ``(m, n_points)``."""
        flat = block.to(torch.float64).reshape(-1, self.grid.n_points)
        if self._quadrature is None:
            return (flat @ flat.transpose(-1, -2)) * self.grid.volume_element
        return (flat * self._quadrature) @ flat.transpose(-1, -2)

    def cross(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Return ``M_ij = <l_i, r_j>`` for two blocks, the subspace matrix of a Rayleigh--Ritz."""
        a = left.to(torch.float64).reshape(-1, self.grid.n_points)
        b = right.to(torch.float64).reshape(-1, self.grid.n_points)
        if self._quadrature is None:
            return (a @ b.transpose(-1, -2)) * self.grid.volume_element
        return (a * self._quadrature) @ b.transpose(-1, -2)
