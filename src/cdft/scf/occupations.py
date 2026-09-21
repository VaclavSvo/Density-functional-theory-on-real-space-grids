"""Occupation numbers -- floating point from the first increment (D-11).

Nothing here is integer-typed, including the closed-shell aufbau case. Smearing arrives with spin
polarisation and is deliberately not stubbed in. Rationale: ``docs/03_METHOD.md (Part E)``.
"""

from __future__ import annotations

import torch

__all__ = ["aufbau_occupations", "electrons_from_occupations"]


def aufbau_occupations(
    eigenvalues: torch.Tensor,
    n_electrons: float,
    n_spin: int = 1,
    magnetisation: float = 0.0,
) -> torch.Tensor:
    """Fill states from the bottom, allowing a fractional electron in the frontier orbital.

    ``eigenvalues`` ``(n_spin, n_states)`` ascending; ``n_electrons`` may be fractional (D-11);
    ``n_spin`` 1 means capacity 2 per spatial orbital, 2 means capacity 1; ``magnetisation``
    ``M = N_up - N_down`` requires ``n_spin == 2``. Returns float64 ``(n_spin, n_states)``.
    """
    if eigenvalues.dim() != 2:
        raise ValueError(f"expected eigenvalues of shape (n_spin, n_states), got {tuple(eigenvalues.shape)}")
    if n_spin not in (1, 2):
        raise ValueError(f"n_spin must be 1 or 2, got {n_spin}")
    if magnetisation != 0.0 and n_spin != 2:
        raise ValueError("a non-zero magnetisation requires n_spin == 2")
    if n_electrons < -1e-12:
        raise ValueError(f"n_electrons must be non-negative, got {n_electrons}")

    occupations = torch.zeros_like(eigenvalues, dtype=torch.float64)
    if n_spin == 1:
        _fill_channel(occupations[0], n_electrons, capacity=2.0)
        return occupations

    n_up = 0.5 * (n_electrons + magnetisation)
    n_down = 0.5 * (n_electrons - magnetisation)
    if n_up < -1e-12 or n_down < -1e-12:
        raise ValueError(f"|magnetisation|={magnetisation} exceeds n_electrons={n_electrons}")
    _fill_channel(occupations[0], n_up, capacity=1.0)
    _fill_channel(occupations[1], n_down, capacity=1.0)
    return occupations


def _fill_channel(target: torch.Tensor, electrons: float, capacity: float) -> None:
    """Fill one spin channel in place from the lowest state upward, raising on leftover charge."""
    remaining = float(electrons)
    n_states = target.shape[0]
    for index in range(n_states):
        if remaining <= 1e-14:
            return
        take = min(capacity, remaining)
        target[index] = take
        remaining -= take
    if remaining > 1e-10:
        raise ValueError(
            f"{electrons} electrons do not fit in {n_states} states of capacity {capacity}; "
            f"{remaining} left over. Increase the state count rather than discarding charge."
        )


def electrons_from_occupations(occupations: torch.Tensor) -> float:
    """Return the total electron number implied by an occupation array."""
    return float(occupations.to(torch.float64).sum())
