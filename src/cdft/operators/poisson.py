"""Open-boundary Hartree solver: the Coulomb-cutoff kernel of Rozzi et al. [F15].

An unmodified FFT solves the *periodic* problem, which on an isolated molecule adds the interaction
with a lattice of copies. The fix is a kernel truncated at radius ``R`` whose transform is known
analytically -- ``K(G) = (4 pi / G^2)[1 - cos(G R)]``, ``K(0) = 2 pi R^2`` -- applied to a
zero-padded box. Validity is measured, not assumed: see
:meth:`CoulombCutoffPoisson.charge_outside_cutoff`. Rationale: ``docs/03_METHOD.md (Part E)``.
"""

from __future__ import annotations

import math

import torch

from ..grid import UniformGrid

__all__ = ["CoulombCutoffPoisson", "analytic_gaussian_potential", "analytic_gaussian_hartree_energy"]


class CoulombCutoffPoisson:
    """Free-space Hartree solver by zero-padded FFT with a spherically truncated kernel.

    The kernel depends only on the grid and the padding, so it is built once per instance.
    """

    def __init__(self, grid: UniformGrid, pad_factor: float = 2.0) -> None:
        """Build the solver and its kernel for one grid.

        ``pad_factor`` must be at least 2, the condition for the FFT's circular convolution to equal
        the linear convolution electrostatics requires.
        """
        if pad_factor < 2.0:
            raise ValueError("pad_factor must be at least 2.0 for the convolution to be linear")
        self.grid = grid
        self.pad_factor = pad_factor
        self.padded_shape = tuple(
            _fft_friendly(int(math.ceil(pad_factor * n))) for n in grid.shape
        )
        self.cutoff_radius = 0.5 * min(n * grid.spacing for n in self.padded_shape)
        self._kernel = self._build_kernel()

    def _build_kernel(self) -> torch.Tensor:
        """Return the truncated Coulomb kernel on the half-spectrum reciprocal grid.

        From the analytic transform, never the DFT of real-space samples: ``1/r`` is not
        band-limited, so sampling first aliases at the singularity and refinement does not fix it.
        """
        h = self.grid.spacing
        n0, n1, n2 = self.padded_shape
        two_pi = 2.0 * math.pi
        kx = two_pi * torch.fft.fftfreq(n0, d=h, device=self.grid.device, dtype=torch.float64)
        ky = two_pi * torch.fft.fftfreq(n1, d=h, device=self.grid.device, dtype=torch.float64)
        kz = two_pi * torch.fft.rfftfreq(n2, d=h, device=self.grid.device, dtype=torch.float64)
        g_sq = kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2
        g = g_sq.sqrt()
        radius = self.cutoff_radius
        with torch.no_grad():
            kernel = torch.where(
                g_sq > 0.0,
                4.0 * math.pi * (1.0 - torch.cos(g * radius)) / g_sq.clamp_min(1e-300),
                torch.full_like(g_sq, 2.0 * math.pi * radius**2),
            )
        return kernel

    def solve(self, density: torch.Tensor, grid: UniformGrid | None = None) -> torch.Tensor:
        """Return ``v_H`` in Hartree on the domain, ``(n_points,)``, with free boundary conditions.

        ``density`` is the *total* density, ``(n_points,)``: a spin-resolved one must be summed over
        spin first, since a per-channel Hartree potential double-counts.
        """
        target = grid or self.grid
        if density.dim() != 1:
            raise ValueError(
                f"expected a total density of shape (n_points,), got {tuple(density.shape)}; "
                "sum over the spin axis before calling the Hartree solver"
            )
        box = target.scatter_to_box(density.to(torch.float64))
        padded = torch.zeros(self.padded_shape, device=box.device, dtype=torch.float64)
        padded[: box.shape[0], : box.shape[1], : box.shape[2]] = box
        transformed = torch.fft.rfftn(padded, s=self.padded_shape)
        potential_box = torch.fft.irfftn(transformed * self._kernel, s=self.padded_shape)
        interior = potential_box[: box.shape[0], : box.shape[1], : box.shape[2]]
        return target.gather_from_box(interior)

    def energy(
        self, density: torch.Tensor, v_hartree: torch.Tensor, grid: UniformGrid | None = None
    ) -> float:
        """Return the Hartree energy ``0.5 * integral(n v_H)`` in Hartree."""
        target = grid or self.grid
        return float(0.5 * target.integrate(density * v_hartree))

    def charge_outside_cutoff(self, density: torch.Tensor) -> float:
        """Return the fraction of the charge farther than half the cutoff from its centroid.

        The truncated kernel is exact only while every source--evaluation pair is closer than the
        cutoff radius; this number, recorded per run, says whether that held.
        """
        points = self.grid.points().to(torch.float64)
        weights = density.to(torch.float64) * self.grid.volume_element
        total = float(weights.sum())
        if abs(total) < 1e-30:
            return 0.0
        centroid = (points * weights[:, None]).sum(dim=0) / total
        radius = (points - centroid).norm(dim=-1)
        outside = float(weights[radius > 0.5 * self.cutoff_radius].sum())
        return abs(outside / total)


def _fft_friendly(n: int) -> int:
    """Round up to the next integer whose only prime factors are 2, 3 and 5 (FFT-friendly)."""
    candidate = n
    while True:
        residue = candidate
        for prime in (2, 3, 5):
            while residue % prime == 0:
                residue //= prime
        if residue == 1:
            return candidate
        candidate += 1


def analytic_gaussian_potential(radius: torch.Tensor, sigma: float, charge: float = 1.0) -> torch.Tensor:
    """Return the exact potential of a Gaussian charge, ``Q erf(r/(sqrt(2) sigma))/r`` (G0.2 [F15])."""
    scaled = radius / (math.sqrt(2.0) * sigma)
    limit = charge * math.sqrt(2.0 / math.pi) / sigma
    safe_r = radius.clamp_min(1e-30)
    return torch.where(
        radius > 1e-12, charge * torch.erf(scaled) / safe_r, torch.full_like(radius, limit)
    )


def analytic_gaussian_hartree_energy(sigma: float, charge: float = 1.0) -> float:
    """Return the exact Hartree energy of a Gaussian charge: ``Q^2 / (2 sqrt(pi) sigma)``."""
    return charge**2 / (2.0 * math.sqrt(math.pi) * sigma)
