"""Density mixers for the self-consistent iteration: linear, Anderson/Pulay, periodic Pulay, Kerker.

All of them change only the path to the fixed point, never the point itself (D-16; G3.4 proves it
by converging from three starts). Each implements :class:`~contract.MixerProtocol`:
``mix(n_in, n_out, iteration) -> n_next``, with a ``reset`` for the fallback ladder.

* :class:`LinearMixer` -- ``n_in + alpha (n_out - n_in)``; the rung nothing falls below.
* :class:`PulayMixer` -- DIIS on the residual ``F = n_out - n_in`` [F8]. With ``period > 1`` it is
  the periodic Pulay of [F9], extrapolating every ``period``-th step and mixing linearly in
  between, which is more robust for the same cost; ``period = 1`` is ordinary Pulay, which in this
  formulation is Anderson mixing [F10] up to bookkeeping.
* :class:`KerkerPreconditioner` -- ``F -> F * q^2 / (q^2 + q0^2)`` by FFT on the box [F11]. Off by
  default: it damps long-wavelength charge sloshing, a metallic pathology, and for an isolated atom
  only slows the first iterations.

The residual norm reported to the loop is the L1 norm in the caller's measure, so it counts
electrons and :class:`~contract.SCFConfig`'s density tolerance is meaningful.
"""

from __future__ import annotations

import math

import torch

from contract import MixingConfig, MixingScheme

from ..precision import host_floats

__all__ = ["LinearMixer", "PulayMixer", "KerkerPreconditioner", "make_mixer"]


class KerkerPreconditioner:
    """Kerker preconditioning of a residual on the enclosing box, ``F(q) q^2 / (q^2 + q0^2)`` [F11]."""

    def __init__(self, grid, q0: float) -> None:
        """Build the reciprocal-space filter for one grid."""
        self.grid = grid
        h = grid.spacing
        two_pi = 2.0 * math.pi
        kx = two_pi * torch.fft.fftfreq(grid.shape[0], d=h, device=grid.device, dtype=torch.float64)
        ky = two_pi * torch.fft.fftfreq(grid.shape[1], d=h, device=grid.device, dtype=torch.float64)
        kz = two_pi * torch.fft.rfftfreq(grid.shape[2], d=h, device=grid.device, dtype=torch.float64)
        q2 = kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2
        self._filter = q2 / (q2 + q0 * q0)

    def __call__(self, residual: torch.Tensor) -> torch.Tensor:
        """Filter a residual of shape ``(n_spin, n_points)``."""
        out = torch.empty_like(residual)
        for s in range(residual.shape[0]):
            box = self.grid.scatter_to_box(residual[s].to(torch.float64))
            filtered = torch.fft.irfftn(torch.fft.rfftn(box) * self._filter, s=self.grid.shape)
            out[s] = self.grid.gather_from_box(filtered)
        return out


class LinearMixer:
    """``n_next = n_in + alpha (n_out - n_in)``."""

    def __init__(self, alpha: float, preconditioner=None) -> None:
        """``alpha`` in ``(0, 1]``; an optional residual preconditioner (Kerker)."""
        if not 0.0 < alpha <= 1.0:
            raise ValueError("mixing alpha must lie in (0, 1]")
        self.alpha = alpha
        self.preconditioner = preconditioner
        self.name = f"linear(alpha={alpha:g})"

    @property
    def history_size(self) -> int:
        """Linear mixing keeps no history."""
        return 0

    def reset(self) -> None:
        """Nothing to clear."""

    def mix(self, n_in: torch.Tensor, n_out: torch.Tensor, iteration: int) -> torch.Tensor:
        """Damped step along the residual."""
        residual = n_out - n_in
        if self.preconditioner is not None:
            residual = self.preconditioner(residual)
        return n_in + self.alpha * residual


class PulayMixer:
    """Pulay/DIIS mixing on the density residual [F8], periodic in the sense of [F9].

    On an extrapolation step the coefficients ``c`` over the stored pairs ``(n_in^k, F^k)``
    minimise ``|sum_k c_k F^k|`` subject to ``sum_k c_k = 1``, and the next input is
    ``sum_k c_k (n_in^k + alpha F^k)``; on other steps it is the linear mixer.
    """

    def __init__(self, alpha: float, history: int, period: int, inner, preconditioner=None) -> None:
        """``inner(a, b)`` is the quadrature inner product the residual norm is measured in."""
        if history < 2:
            raise ValueError("Pulay mixing needs a history of at least 2")
        if period < 1:
            raise ValueError("pulay_period must be at least 1")
        self.alpha = alpha
        self.history = history
        self.period = period
        self.inner = inner
        self.preconditioner = preconditioner
        self._inputs: list[torch.Tensor] = []
        self._residuals: list[torch.Tensor] = []
        self.name = (
            f"periodic_pulay(alpha={alpha:g}, history={history}, period={period})"
            if period > 1
            else f"pulay(alpha={alpha:g}, history={history})"
        )
        self.last_event = ""

    @property
    def history_size(self) -> int:
        """Number of stored input/residual pairs."""
        return len(self._inputs)

    def reset(self) -> None:
        """Clear the history, for example after a fallback event."""
        self._inputs.clear()
        self._residuals.clear()

    def mix(self, n_in: torch.Tensor, n_out: torch.Tensor, iteration: int) -> torch.Tensor:
        """Store the pair, then extrapolate on a period step or damp otherwise."""
        residual = n_out - n_in
        if self.preconditioner is not None:
            residual = self.preconditioner(residual)
        self._inputs.append(n_in.clone())
        self._residuals.append(residual.clone())
        if len(self._inputs) > self.history:
            self._inputs.pop(0)
            self._residuals.pop(0)
        m = len(self._inputs)
        if m < 2 or iteration % self.period != 0:
            self.last_event = "linear step"
            return n_in + self.alpha * residual
        # Residual Gram matrix with a Lagrange multiplier for sum c = 1. The Tikhonov term guards
        # against a linearly dependent history, which otherwise gives coefficients of 1e6 and a
        # density with negative regions. The matrix is history-sized (m <= 8), so it is built on
        # the CPU; the pair products are reduced on the residuals' device and read in one transfer.
        pairs = [(i, j) for i in range(m) for j in range(i, m)]
        values = host_floats(
            *(self.inner(self._residuals[i].reshape(-1), self._residuals[j].reshape(-1)).sum() for i, j in pairs)
        )
        gram = torch.zeros((m, m), dtype=torch.float64, device="cpu")
        for (i, j), value in zip(pairs, values):
            gram[i, j] = gram[j, i] = value
        scale = float(gram.diagonal().max()) or 1.0
        gram = gram + 1.0e-12 * scale * torch.eye(m, dtype=torch.float64, device="cpu")
        system = torch.zeros((m + 1, m + 1), dtype=torch.float64, device="cpu")
        system[:m, :m] = gram
        system[:m, m] = -1.0
        system[m, :m] = 1.0
        rhs = torch.zeros(m + 1, dtype=torch.float64, device="cpu")
        rhs[m] = 1.0
        coefficients = torch.linalg.solve(system, rhs)[:m]
        self.last_event = f"Pulay extrapolation over {m} vectors"
        mixed = torch.zeros_like(n_in)
        for c, stored_in, stored_res in zip(coefficients, self._inputs, self._residuals):
            mixed = mixed + float(c) * (stored_in + self.alpha * stored_res)
        return mixed


def make_mixer(config: MixingConfig, grid, inner):
    """Build the mixer a :class:`~contract.MixingConfig` asks for."""
    preconditioner = KerkerPreconditioner(grid, config.kerker_q0) if config.kerker else None
    if config.scheme is MixingScheme.LINEAR:
        return LinearMixer(config.alpha, preconditioner)
    if config.scheme is MixingScheme.KERKER:
        # Kerker is a preconditioner, not a scheme: the enum pairs it with periodic Pulay.
        return PulayMixer(
            config.alpha, config.history, config.pulay_period, inner,
            KerkerPreconditioner(grid, config.kerker_q0),
        )
    period = config.pulay_period if config.scheme is MixingScheme.PERIODIC_PULAY else 1
    return PulayMixer(config.alpha, config.history, period, inner, preconditioner)
