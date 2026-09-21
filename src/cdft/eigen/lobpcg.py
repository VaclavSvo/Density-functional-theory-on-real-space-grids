"""LOBPCG [F4] -- kept permanently as an oracle, not as a fallback.

G2.5 requires the two eigensolvers to agree to 1e-8 Ha on every occupied state: a disagreement
localises a fault to the solver, an agreement to the Hamiltonian. ``torch.lobpcg`` takes a matrix
and the architecture never forms one (D-06), so this is matrix-free with the usual conditioning
safeguards. Rationale: ``docs/03_METHOD.md (Part E)``.

On the cusp-factorised path the preconditioner acts in the similarity frame (D-56): the filter
models the plain Laplacian, which is the operator on ``psi = f phi``, not on ``phi``. Applied to
``phi`` directly it is worse than no preconditioner and stagnates on the molecules.
"""

from __future__ import annotations

import torch

from contract import EigenConfig, EigenResult

from ..grid import UniformGrid
from ..precision import host_floats, seeded_randn, solve_triangular_wide
from .measure import Measure
from .rayleigh_ritz import orthonormality_error, residual_norms

__all__ = ["LOBPCG"]


PRECONDITIONER_FRAMES = ("similarity", "transformed", "none")
"""Where the kinetic preconditioner acts (D-56).

``"similarity"`` (default)
    ``f^-1 K f``: the residual is mapped to the wavefunction frame (``r_psi = f r_phi``, exact for
    ``H_phi = f^-1 H_psi f``), the Teter--Payne--Allan filter [F7] is applied there, where the
    operator it models is the plain Laplacian, and the direction is mapped back. Symmetric in the
    ``f^2`` measure, so a legitimate LOBPCG preconditioner.
``"transformed"``
    ``K`` applied directly to ``r_phi``, kept for the comparison that justified the change.
``"none"``
    Steepest-descent LOBPCG, the baseline every preconditioner has to beat.
"""


class LOBPCG:
    """Locally optimal block preconditioned conjugate gradient, implementing the eigensolver protocol."""

    def __init__(self, grid: UniformGrid, config: EigenConfig, precondition: bool | str = True) -> None:
        """Build the solver for one grid and one eigensolver configuration.

        ``precondition`` is a frame name from :data:`PRECONDITIONER_FRAMES`; ``True`` means the
        default frame and ``False`` means ``"none"``.
        """
        self.grid = grid
        self.config = config
        if precondition is True:
            precondition = "similarity"
        elif precondition is False:
            precondition = "none"
        if precondition not in PRECONDITIONER_FRAMES:
            raise ValueError(f"unknown preconditioner frame {precondition!r}; one of {PRECONDITIONER_FRAMES}")
        self.frame: str = precondition
        self.history: list[dict[str, float]] = []
        self._kinetic_g2: torch.Tensor | None = None
        self._kinetic_reference = 1.0e-6
        #: Ritz-value stability required before convergence: an order tighter than G2.5's
        #: ``cross_check_tol`` and nothing further.
        self.eigenvalue_tol = 0.1 * config.cross_check_tol
        #: Largest Ritz-value movement in the final iteration, for the gate's detail.
        self.eigenvalue_drift = float("inf")

    def solve(
        self,
        hamiltonian: object,
        n_states: int,
        initial: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> EigenResult:
        """Return the lowest ``n_states`` eigenpairs by locally optimal block preconditioned CG."""
        grid = self.grid
        cfg = self.config
        measure = self._measure_of(hamiltonian)
        cusp_factor = self._cusp_factor(hamiltonian)

        if initial is None:
            x = seeded_randn((n_states, grid.n_points), generator, device=grid.device, dtype=torch.float64)
        else:
            x = initial.to(torch.float64).reshape(-1, grid.n_points)[:n_states].clone()

        x = _orthonormalise(x, measure)
        hx = hamiltonian.apply(x)  # type: ignore[attr-defined]
        subspace = _symmetrise(measure.cross(x, hx))
        evals, rotation = torch.linalg.eigh(subspace)
        x = rotation.transpose(-1, -2) @ x
        hx = rotation.transpose(-1, -2) @ hx

        p: torch.Tensor | None = None
        converged = False
        iteration = 0
        residual = measure.norm(hx - evals[:, None] * x)
        previous_evals = evals.clone()
        stable_iterations = 0
        self.eigenvalue_drift = float("inf")

        for iteration in range(1, cfg.max_iterations + 1):
            r = hx - evals[:, None] * x
            residual = measure.norm(r)
            keep = residual > 0.1 * cfg.residual_tol
            # One host read per iteration: residual, drift and the deflation test.
            if iteration > 1:
                worst, drift, any_kept = host_floats(
                    residual.max(), (evals - previous_evals).abs().max(), keep.any()
                )
            else:
                worst, any_kept = host_floats(residual.max(), keep.any())
                drift = float("inf")
            previous_evals = evals.clone()
            self.eigenvalue_drift = drift
            self.history.append(
                {"iteration": float(iteration), "residual": worst, "eigenvalue_drift": drift}
            )
            if worst < cfg.residual_tol:
                converged = True
                break
            # Second, weaker criterion: the Ritz values have stopped moving -- the quantity G2.5
            # compares, and quadratically faster than the residual.
            stable_iterations = stable_iterations + 1 if drift < self.eigenvalue_tol else 0
            if stable_iterations >= 2:
                converged = True
                break

            # Deflate converged directions: searching along one at the noise floor injects noise
            # into the others.
            w = r[keep] if any_kept else r
            if self.frame == "similarity" and cusp_factor is not None:
                w = self._apply_preconditioner(w * cusp_factor) / cusp_factor
            elif self.frame != "none":
                w = self._apply_preconditioner(w)
            w = _orthogonalise_against(w, x, measure)
            if w.shape[0] == 0:
                converged = True
                break

            blocks = [x, w] if p is None else [x, w, p]
            basis, dropped = _stable_basis(torch.cat(blocks, dim=0), measure)
            h_basis = hamiltonian.apply(basis)  # type: ignore[attr-defined]
            subspace = _symmetrise(measure.cross(basis, h_basis))
            sub_evals, sub_vectors = torch.linalg.eigh(subspace)
            coeff = sub_vectors[:, :n_states]

            x_new = coeff.transpose(-1, -2) @ basis
            hx_new = coeff.transpose(-1, -2) @ h_basis
            # The conjugate direction is the part of the update that is not already in X.
            p = x_new - (measure.cross(x_new, x) @ x)
            x, hx = x_new, hx_new
            evals = sub_evals[:n_states]
            if dropped:
                self.history[-1]["dropped_directions"] = float(dropped)

        return EigenResult(
            eigenvalues=evals[:n_states],
            eigenvectors=x[:n_states],
            residuals=residual[:n_states],
            n_iterations=iteration,
            converged=converged,
        )

    @staticmethod
    def _cusp_factor(hamiltonian: object) -> torch.Tensor | None:
        """Return ``f = exp(-u)`` of a cusp-factorised Hamiltonian, or ``None`` for a plain one.

        Clamped below at ``1e-150`` so the division in the similarity frame stays finite in the far
        corners of a large box, where ``f`` underflows toward zero and the residual with it.
        """
        factor = getattr(hamiltonian, "factor", None)
        if factor is None or getattr(factor, "is_identity", True):
            return None
        return factor.factor().clamp_min(1.0e-150)

    def _apply_preconditioner(self, block: torch.Tensor) -> torch.Tensor:
        """Apply the Teter--Payne--Allan kinetic preconditioner [F7] to a block of residuals.

        ``K(x) = (27 + 18x + 12x^2 + 8x^3) / (27 + 18x + 12x^2 + 8x^3 + 16x^4)`` with
        ``x = (G^2/2) / E_ref``, on the enclosing box by FFT. ``E_ref`` is the grid's mean kinetic
        scale, not a per-band eigenvalue, which can be negative and would invert the filter. On a
        masked domain the transform wraps around; a preconditioner moves only the path, not the
        fixed point.
        """
        grid = self.grid
        if self._kinetic_g2 is None:
            import math

            h = grid.spacing
            two_pi = 2.0 * math.pi
            kx = two_pi * torch.fft.fftfreq(grid.shape[0], d=h, device=grid.device, dtype=torch.float64)
            ky = two_pi * torch.fft.fftfreq(grid.shape[1], d=h, device=grid.device, dtype=torch.float64)
            kz = two_pi * torch.fft.rfftfreq(grid.shape[2], d=h, device=grid.device, dtype=torch.float64)
            self._kinetic_g2 = 0.5 * (
                kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2
            )
            # A grid constant: read once, not once per iteration.
            self._kinetic_reference = max(float(self._kinetic_g2.mean()), 1.0e-6)
        kinetic = self._kinetic_g2

        box = grid.scatter_to_box(block)
        transformed = torch.fft.rfftn(box, dim=(-3, -2, -1))
        reference = self._kinetic_reference
        x = kinetic / reference
        poly = 27.0 + 18.0 * x + 12.0 * x**2 + 8.0 * x**3
        filt = poly / (poly + 16.0 * x**4)
        preconditioned = torch.fft.irfftn(transformed * filt, s=grid.shape, dim=(-3, -2, -1))
        return grid.gather_from_box(preconditioned)

    def _measure_of(self, hamiltonian: object) -> Measure:
        """Return the operator's own inner product, the uniform one when it carries none (D-56)."""
        return getattr(hamiltonian, "measure", None) or Measure.uniform(self.grid)

    def orthonormality_error(self, hamiltonian: object, vectors: torch.Tensor) -> float:
        """Return ``max |S - I|`` of a block in the operator's measure, for gate G0.3 (O-27)."""
        return orthonormality_error(vectors, self._measure_of(hamiltonian))

    def residuals(self, hamiltonian: object, evals: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
        """Return the per-state residual norms in the operator's measure, for gate G2.4 (O-27)."""
        return residual_norms(hamiltonian, evals, vectors, self._measure_of(hamiltonian))


def _symmetrise(matrix: torch.Tensor) -> torch.Tensor:
    """Return the symmetric part of a matrix that is symmetric up to round-off."""
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def _orthonormalise(block: torch.Tensor, measure: Measure) -> torch.Tensor:
    """Orthonormalise a block in the given measure by Cholesky, falling back to eigendecomposition."""
    gram = measure.gram(block)
    try:
        chol = torch.linalg.cholesky(gram)
    except RuntimeError:
        basis, _ = _stable_basis(block, measure)
        return basis
    return solve_triangular_wide(chol, block, upper=False)


def _orthogonalise_against(block: torch.Tensor, basis: torch.Tensor, measure: Measure) -> torch.Tensor:
    """Project ``basis`` out of ``block`` twice, then orthonormalise what is left.

    Twice, because one pass of classical Gram--Schmidt loses orthogonality on nearly dependent
    vectors, this solver's regime.
    """
    for _ in range(2):
        block = block - (measure.cross(block, basis) @ basis)
    out, _ = _stable_basis(block, measure)
    return out


def _stable_basis(block: torch.Tensor, measure: Measure, threshold: float = 1e-12) -> tuple[torch.Tensor, int]:
    """Return an orthonormal basis with near-dependent directions dropped, and how many were dropped.

    The count goes into the solver history: dropping directions every iteration means the block
    size is wrong.
    """
    gram = measure.gram(block)
    evals, evecs = torch.linalg.eigh(_symmetrise(gram))
    if not evals.numel():
        return block[:0], block.shape[0]
    # Compared on the device so ``largest`` and the dropped count come back in one read.
    keep = evals > evals[-1] * threshold
    largest, dropped_count = host_floats(evals[-1], (~keep).sum())
    if largest <= 0.0:
        return block[:0], block.shape[0]
    dropped = int(dropped_count)
    scaled = evecs[:, keep] / evals[keep].sqrt()
    return scaled.transpose(-1, -2) @ block, dropped
