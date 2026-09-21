"""Orthonormalisation and the Rayleigh--Ritz step -- float64, unconditionally.

Cheap (``n_states**2`` and ``**3``) and the documented way mixed precision breaks an eigensolver,
so G0.3 holds at 1e-12 whatever the hot-path dtype. Everything works in a
:class:`~cdft.eigen.measure.Measure`: ``h**3`` on the ordinary path, ``h**3 exp(-2u)`` on the
cusp-transformed one. Rationale: ``docs/03_METHOD.md (Part E)``.
"""

from __future__ import annotations

import torch

from ..grid import UniformGrid
from ..precision import seeded_randn, solve_triangular_wide
from .measure import Measure

__all__ = [
    "gram_matrix",
    "orthonormalise",
    "orthonormality_error",
    "orthonormality_error_tensor",
    "rayleigh_ritz",
    "residual_norms",
    "self_adjointness_error",
    "as_measure",
]


def as_measure(target: Measure | UniformGrid) -> Measure:
    """Accept either a measure or a bare grid, so callers that have no weight stay simple."""
    return target if isinstance(target, Measure) else Measure.uniform(target)


def gram_matrix(block: torch.Tensor, measure: Measure | UniformGrid) -> torch.Tensor:
    """Return the Gram matrix ``S_ij = <psi_i|psi_j>``, shape ``(m, m)``, in float64."""
    return as_measure(measure).gram(block)


def orthonormality_error(block: torch.Tensor, measure: Measure | UniformGrid) -> float:
    """Return ``max |S - I|``, the quantity gate G0.3 thresholds at 1e-12."""
    return float(orthonormality_error_tensor(block, measure))


def orthonormality_error_tensor(block: torch.Tensor, measure: Measure | UniformGrid) -> torch.Tensor:
    """:func:`orthonormality_error` as a 0-d tensor on the block's device (no host read)."""
    gram = gram_matrix(block, measure)
    identity = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    return (gram - identity).abs().max()


def orthonormalise(block: torch.Tensor, measure: Measure | UniformGrid) -> torch.Tensor:
    """Return an orthonormal basis for the span of ``block``, shape ``(m, n_points)``.

    Cholesky where it succeeds; on a rank-deficient block (a filter having driven several vectors
    onto the same eigenvector) it falls back to a symmetric eigendecomposition with the small
    eigenvalues dropped, reported by returning fewer vectors, never a basis that is not
    orthonormal.
    """
    m = as_measure(measure)
    flat = block.to(torch.float64).reshape(-1, m.grid.n_points)
    gram = m.gram(flat)
    try:
        chol = torch.linalg.cholesky(gram)
    except RuntimeError:
        evals, evecs = torch.linalg.eigh(0.5 * (gram + gram.transpose(-1, -2)))
        keep = evals > evals[-1] * 1e-14
        if not bool(keep.any()):
            raise ValueError("the orbital block collapsed to rank zero") from None
        scaled = evecs[:, keep] / evals[keep].sqrt()
        return scaled.transpose(-1, -2) @ flat
    # Chunked on CUDA above 2^18 columns (cuBLAS trsm cliff); plain elsewhere.
    return solve_triangular_wide(chol, flat, upper=False)


def rayleigh_ritz(
    hamiltonian: "object",
    block: torch.Tensor,
    measure: Measure | UniformGrid,
    *,
    host: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, float | torch.Tensor]:
    """Diagonalise a Hamiltonian within the span of ``block``.

    ``block`` is ``(m, n_points)`` trial orbitals, not necessarily orthonormal; ``measure`` is the
    inner product the Hamiltonian is self-adjoint in, a bare grid meaning the uniform one;
    ``host=False`` returns the orthonormality error as a 0-d device tensor, so a solver can read it
    with its other per-iteration scalars in one transfer.

    Returns ``(eigenvalues, eigenvectors, orthonormality_error)`` with eigenvalues ascending,
    eigenvectors orthonormal in the measure, and the error measured after the rotation, on the
    block the caller keeps (G0.3).
    """
    m = as_measure(measure)
    basis = orthonormalise(block, m)
    # The staggered divergence form is exactly self-adjoint, so <a|A|b>_w built from `apply` is
    # the weak form; the symmetrisation only removes the ~1e-16 round-off asymmetry.
    subspace = m.cross(basis, hamiltonian.apply(basis))  # type: ignore[attr-defined]
    subspace = 0.5 * (subspace + subspace.transpose(-1, -2))
    evals, rotation = torch.linalg.eigh(subspace)
    vectors = rotation.transpose(-1, -2) @ basis
    error = orthonormality_error_tensor(vectors, m)
    return evals, vectors, (float(error) if host else error)


def residual_norms(
    hamiltonian: "object",
    eigenvalues: torch.Tensor,
    vectors: torch.Tensor,
    measure: Measure | UniformGrid,
) -> torch.Tensor:
    """Return ``||H psi_i - eps_i psi_i||`` per state (G2.4), in the operator's own inner product."""
    m = as_measure(measure)
    applied = hamiltonian.apply(vectors)  # type: ignore[attr-defined]
    residual = applied - eigenvalues[:, None] * vectors
    return m.norm(residual)


def self_adjointness_error(
    hamiltonian: "object",
    measure: Measure | UniformGrid,
    n_vectors: int = 32,
    generator: torch.Generator | None = None,
) -> float:
    """Return ``max |<a|A|b> - <b|A|a>|`` over random pairs, in the given measure (G0.12).

    Separate from the plain Hermiticity probe of G0.4: the transformed operator is not symmetric in
    the plain inner product.
    """
    m = as_measure(measure)
    probes = seeded_randn(
        (n_vectors, m.grid.n_points), generator, device=m.grid.device, dtype=torch.float64
    )
    applied = hamiltonian.apply(probes)  # type: ignore[attr-defined]
    matrix = m.cross(probes, applied)
    scale = float(matrix.abs().max())
    error = float((matrix - matrix.transpose(-1, -2)).abs().max())
    return error / scale if scale > 0.0 else error
