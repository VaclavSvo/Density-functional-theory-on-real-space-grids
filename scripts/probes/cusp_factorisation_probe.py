"""Probe: can the nuclear cusp be removed by factorising it out of the wavefunction?

    python scripts/probes/cusp_factorisation_probe.py     # prints the ladder, writes nothing

The bare -Z/r all-electron path converges as O(h^1) (D-32, the ladder of the retired gate G1.10,
which lives here now). Writing psi = f phi with the Kato factor f = exp(-Z r) gives

    A phi = f^-1 H (f phi) = -1/2 lap(phi) + Z (rhat . grad phi) - (Z^2 / 2) phi,

since the singular +Z/r from lap(f)/f cancels the -Z/r of the external potential exactly. A is
self-adjoint in the weighted inner product <a,b>_w = integral(f^2 a b) and has no singular
potential and no cusp, so for a one-electron atom, where the exact solution is phi = const, a
correct implementation must return -Z^2/2 to round-off on any grid. This measures that and the same
quantity on the untransformed operator over a ladder of spacings. A probe, not a gate: nothing here
is imported by ``cdft``.
"""

from __future__ import annotations

import sys
import time

import torch

import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from cdft.grid import GridGeometry, UniformGrid  # noqa: E402
from cdft.operators.external import external_potential  # noqa: E402
from cdft.operators.hamiltonian import LocalHamiltonian  # noqa: E402
from contract import (  # noqa: E402
    AtomicStructure,
    BoundaryMode,
    ExternalPotentialKind,
    ExternalPotentialSpec,
)


def build_grid(box: float, h: float, order: int = 8) -> UniformGrid:
    """Build a cubic grid with the coordinate origin exactly on a grid point."""
    n = int(round(box / h))
    n = n if n % 2 else n + 1  # odd count puts a point at the centre
    origin = -0.5 * (n - 1) * h
    geometry = GridGeometry(shape=(n, n, n), spacing=h, origin=(origin, origin, origin))
    mask = torch.ones((n, n, n), dtype=torch.bool)
    return UniformGrid(geometry, mask, fd_order=order, gradient_order=order, boundary=BoundaryMode.ZERO)


def transformed_ground_state(grid: UniformGrid, charge: float) -> tuple[float, int]:
    """Return the ground-state eigenvalue of the cusp-factored operator, and the subspace size.

    Rayleigh--Ritz in the weighted inner product on a deliberately crude basis: the probe measures
    the discretisation error, not the eigensolver's.
    """
    points = grid.points()
    radius = points.norm(dim=-1)
    weight = torch.exp(-2.0 * charge * radius) * grid.volume_element

    def w_inner(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Weighted inner product, the one the transformed operator is symmetric in."""
        return (a.to(torch.float64) * b.to(torch.float64) * weight).sum(dim=-1)

    unit = torch.where(
        radius > 1e-14, 1.0 / radius.clamp_min(1e-300), torch.zeros_like(radius)
    )

    def apply(phi: torch.Tensor) -> torch.Tensor:
        """A phi = -1/2 lap(phi) + Z (rhat . grad phi) - (Z^2/2) phi."""
        lap = grid.laplacian(phi)
        grad = grid.gradient(phi)
        radial = (grad * points.transpose(0, 1)).sum(dim=-2) * unit
        return -0.5 * lap + charge * radial - 0.5 * charge**2 * phi

    # phi = 1 is the exact one-electron answer, so the trial space contains it on purpose: what is
    # measured is whether the discretised operator returns -Z^2/2 when handed it.
    trials = [torch.ones_like(radius)]
    for scale in (0.5, 1.0, 2.0):
        trials.append(torch.exp(-scale * charge * radius))
        trials.append(radius * torch.exp(-scale * charge * radius))
    block = torch.stack(trials).to(torch.float64)

    gram = torch.stack([torch.stack([w_inner(a, b) for b in block]) for a in block])
    evals_g, evecs_g = torch.linalg.eigh(0.5 * (gram + gram.T))
    keep = evals_g > evals_g[-1] * 1e-13
    basis = (evecs_g[:, keep] / evals_g[keep].sqrt()).T @ block

    applied = torch.stack([apply(v) for v in basis])
    sub = torch.stack([torch.stack([w_inner(a, b) for b in applied]) for a in basis])
    evals = torch.linalg.eigvalsh(0.5 * (sub + sub.T))
    return float(evals[0]), int(basis.shape[0])


def untransformed_ground_state(grid: UniformGrid, charge: float) -> float:
    """Return the ground-state eigenvalue of the bare −Z/r operator, for comparison."""
    from cdft.eigen.chefsi import ChebyshevFilteredSubspace
    from contract import EigenConfig

    structure = AtomicStructure(numbers=(int(charge),), positions=((0.0, 0.0, 0.0),))
    spec = ExternalPotentialSpec(kind=ExternalPotentialKind.NUCLEAR_COULOMB, charges=(charge,))
    v_ext = external_potential(spec, structure, grid)
    hamiltonian = LocalHamiltonian(grid, v_ext)
    solver = ChebyshevFilteredSubspace(
        grid, EigenConfig(n_extra_states=4, chebyshev_degree=20, max_iterations=400, residual_tol=1e-9)
    )
    result = solver.solve(hamiltonian, 1, generator=torch.Generator(device="cpu").manual_seed(0))
    return float(result.eigenvalues[0])


def main() -> None:
    """Run both operators over a ladder of spacings for hydrogen and He+."""
    print(f"{'Z':>3} {'box':>5} {'h':>6} {'n^3':>10} "
          f"{'bare -Z/r err':>15} {'cusp-factored err':>19} {'seconds':>9}")
    print("-" * 74)
    for charge, box in ((1.0, 24.0), (2.0, 16.0)):
        exact = -0.5 * charge**2
        for h in (0.40, 0.30, 0.20, 0.15):
            start = time.perf_counter()
            grid = build_grid(box, h)
            factored, _ = transformed_ground_state(grid, charge)
            bare = untransformed_ground_state(grid, charge)
            print(
                f"{charge:>3.0f} {box:>5.0f} {h:>6.3f} {grid.n_points:>10d} "
                f"{bare - exact:>15.3e} {factored - exact:>19.3e} "
                f"{time.perf_counter() - start:>9.1f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
