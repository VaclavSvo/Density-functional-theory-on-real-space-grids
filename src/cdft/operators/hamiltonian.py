"""The Kohn--Sham Hamiltonian, exposed only through its action on a block of orbitals.

The seam of the whole architecture (D-06): ``H`` is never materialised, so memory scales as
``n_states * n_points``, the Chebyshev filter needs nothing else, and a TDDFT propagator can be
added later without touching anything below this line. :class:`LocalHamiltonian` is kinetic plus a
fixed local potential; :class:`CuspFactoredHamiltonian` is ``A = f^-1 H f`` in staggered divergence
form. Rationale: ``docs/03_METHOD.md (Part E)``.
"""

from __future__ import annotations

from typing import Callable

import torch

from ..eigen.measure import Measure
from ..grid import UniformGrid
from ..precision import host_floats, seeded_randn
from .cusp import CuspFactor
from .divergence import weighted_divergence
from .quadrature import CuspQuadrature

__all__ = ["LocalHamiltonian", "CuspFactoredHamiltonian"]


def _lanczos_bounds(
    apply: Callable[[torch.Tensor], torch.Tensor],
    start: torch.Tensor,
    inner: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    norm: Callable[[torch.Tensor], torch.Tensor],
    steps: int,
    safety: float,
) -> tuple[float, float]:
    """Return ``(lambda_min, lambda_max_upper)`` from a short Lanczos run in a given inner product.

    The upper bound must genuinely bound: an underestimate makes the Chebyshev filter amplify the
    subspace it was meant to damp, and that looks like slow convergence, not an error [F2]. Returned
    is the extreme Ritz value plus (minus) the final residual norm -- a rigorous enclosure -- widened
    by ``safety``. ``inner`` and ``norm`` must be the products the operator is symmetric in; in the
    wrong one the recurrence loses symmetry and the bound can fall below the true maximum.

    ``alpha`` and ``beta`` stay 0-d device tensors inside the loop and are read in one transfer
    after it, so the breakdown test ``beta < 1e-12`` is applied after the fact: the tridiagonal
    matrix is cut at the first step that meets it, exactly the matrix a step-by-step early exit
    would have built.
    """
    v = start / norm(start)
    alpha_t: list[torch.Tensor] = []
    beta_t: list[torch.Tensor] = []
    v_prev = torch.zeros_like(v)
    beta: torch.Tensor | float = 0.0
    for _ in range(steps):
        w = apply(v)
        alpha = inner(v, w)
        w = w - alpha * v - beta * v_prev
        # One reorthogonalisation pass against the previous two vectors: plain Lanczos loses
        # orthogonality fast, and a spurious Ritz value here is a wrong bound, not a visible error.
        w = w - inner(w, v) * v - inner(w, v_prev) * v_prev
        beta = norm(w)
        alpha_t.append(alpha)
        beta_t.append(beta)
        v_prev, v = v, w / beta

    values = host_floats(*alpha_t, *beta_t)
    alphas = values[: len(alpha_t)]
    betas = values[len(alpha_t) :]
    for index, value in enumerate(betas):
        if value < 1e-12:
            del alphas[index + 1 :], betas[index + 1 :]
            break

    size = len(alphas)
    # A handful of host floats: built and diagonalised on the CPU whatever the operator's device.
    tri = torch.zeros((size, size), dtype=torch.float64, device="cpu")
    for i in range(size):
        tri[i, i] = alphas[i]
        if i + 1 < size:
            tri[i, i + 1] = betas[i]
            tri[i + 1, i] = betas[i]
    ritz = torch.linalg.eigvalsh(tri)
    residual = betas[-1] if betas else 0.0
    lo = float(ritz[0]) - residual
    hi = float(ritz[-1]) + residual
    centre = 0.5 * (hi + lo)
    half = 0.5 * (hi - lo) * safety
    return centre - half, centre + half


class LocalHamiltonian:
    """``H = -1/2 laplacian + v_local(r)`` with a fixed local potential."""

    def __init__(self, grid: UniformGrid, v_local: torch.Tensor) -> None:
        """Build the Hamiltonian from a grid and a local potential of shape ``(n_points,)``."""
        if v_local.shape[-1] != grid.n_points:
            raise ValueError(
                f"potential has {v_local.shape[-1]} points, grid has {grid.n_points}"
            )
        self.grid = grid
        self._v_local = v_local
        self._apply_count = 0
        #: The inner product this operator is self-adjoint in. Uniform here; the eigensolvers read
        #: it rather than assume it, which is what lets the transformed operator share them.
        self.measure = Measure.uniform(grid)

    @property
    def apply_count(self) -> int:
        """Number of Hamiltonian applications so far -- the cost metric for a matrix-free solver."""
        return self._apply_count

    def apply(self, psi: torch.Tensor) -> torch.Tensor:
        """Return ``H psi`` for one or many orbitals, shape ``(..., n_points)``; batch dims preserved."""
        self._apply_count += 1
        return self.apply_with_potential(psi, self._v_local)

    def apply_with_potential(self, psi: torch.Tensor, potential: torch.Tensor) -> torch.Tensor:
        """``-1/2 lap psi + potential * psi``: :meth:`apply` with the potential passed in.

        Same operations in the same order as :meth:`apply`; a CUDA graph calls it with a static
        buffer refreshed before each replay. Does not count an application.
        """
        kinetic = self.grid.laplacian(psi).mul_(-0.5)
        return kinetic.add_(psi * potential)

    def graph_potential(self) -> torch.Tensor:
        """The tensor :meth:`apply` multiplies by: what a graph copies into its potential buffer."""
        return self._v_local

    def graph_geometry_key(self) -> tuple:
        """Identity of every tensor the kinetic part reads, for CUDA-graph reuse.

        Equal keys mean the same kinetic arithmetic on the same device memory, so a graph captured
        on one operator replays for the other once the potential is copied. The grid owns every
        stencil tensor and the capturing operator keeps it alive, so its ``id`` is a safe key.
        """
        return ("local", id(self.grid))

    def note_applications(self, count: int) -> None:
        """Add ``count`` applications made outside :meth:`apply` (a graph replay of the filter)."""
        self._apply_count += int(count)

    def local_potential(self) -> torch.Tensor:
        """Return the current total local potential, for diagnostics and for the record."""
        return self._v_local

    def update_potentials(self, density: torch.Tensor) -> None:
        """Refuse: this Hamiltonian has no density-dependent terms.

        Always raises: a caller here has confused a fixed-potential eigenproblem with an SCF and
        would otherwise get a converged answer to the wrong question (G5.5).
        """
        raise NotImplementedError(
            "LocalHamiltonian has no density-dependent terms; the self-consistent Hamiltonian "
            "arrives with Increment 3"
        )

    def kinetic_energy(self, psi: torch.Tensor, occupations: torch.Tensor) -> float:
        """Return ``sum_i f_i <psi_i| -1/2 lap |psi_i>``.

        From the Laplacian, not ``|grad psi|^2``: the two differ on a grid by a boundary term, and
        this form is consistent with the operator the eigensolver diagonalised (G2.9 closes to 1e-12).
        """
        lap = self.grid.laplacian(psi)
        per_state = -0.5 * self.grid.inner(psi, lap)
        return float((occupations.to(torch.float64) * per_state).sum())

    def spectral_bounds(
        self, steps: int = 12, safety: float = 1.02, generator: torch.Generator | None = None
    ) -> tuple[float, float]:
        """Estimate ``(lambda_min, lambda_max_upper)`` by a short Lanczos run in the plain product."""
        grid = self.grid
        start = seeded_randn(grid.n_points, generator, device=grid.device, dtype=torch.float64)
        return _lanczos_bounds(self.apply, start, grid.inner, grid.norm, steps, safety)


class CuspFactoredHamiltonian:
    """``A = f^-1 H f`` with the Kato factor ``f = exp(-sum_a Z_a |r - R_a|)``.

    Same seam as :class:`LocalHamiltonian`, so every eigensolver, gate and diagnostic works on it
    unchanged; but the orbitals are ``phi``, not ``psi``, and it is self-adjoint in :attr:`measure`
    rather than in the plain integral::

        A phi = -1/2 lap(phi) + (grad u) . grad(phi) - 1/2 |grad u|^2 phi

    With no nuclei the factor is identically 1 and this is bit-for-bit :class:`LocalHamiltonian`, so
    the transform can be the default for every all-electron scenario.

    ``extra_potential`` carries everything that is not a bare nuclear Coulomb term (Hartree, XC, an
    analytic well) and is added unchanged; on the interacting path it is already lumped through
    :meth:`~cdft.operators.quadrature.CuspQuadrature.effective_potential`.

    With a :class:`~cdft.operators.quadrature.CuspQuadrature` the inner product is its mass weights
    ``omega_i``, not the plain ``f^2(r_i) h^3`` (D-54), so the weak form is
    ``(1/2) sum_faces f^2_face Da Db h^3 + sum_i omega_i (W~_i + v~_i) a_i b_i``: the kinetic term
    keeps the staggered divergence form (exact for constant ``phi``, positive semi-definite, no null
    space -- D-38) and every potential term is a lumped Galerkin integral. ``apply`` therefore
    divides the kinetic term by ``omega_i / h^3`` instead of ``f^2_i`` and ``W`` enters as ``W~``.
    For one nucleus ``W~ = -Z^2/2`` exactly, so single-centre eigenvalues do not move; molecular
    ones do, and the density is normalised in a measure that integrates the cusp weight exactly
    (G1.13).
    """

    def __init__(
        self,
        grid: UniformGrid,
        factor: CuspFactor,
        extra_potential: torch.Tensor | None = None,
        quadrature: CuspQuadrature | None = None,
    ) -> None:
        """Build the transformed Hamiltonian from a grid, a cusp factor and (optionally) the quadrature."""
        self.grid = grid
        self.factor = factor
        self.quadrature = quadrature
        self._extra = extra_potential
        if quadrature is not None and not factor.is_identity and quadrature.factor is factor:
            # W lumped through the same rule as every other potential term; geometry-only, so
            # built once per quadrature (the SCF builds one operator per iteration).
            self._potential, self._inverse_weight, self.measure = quadrature.bare_operator_terms()
        elif quadrature is not None and not factor.is_identity:
            # W lumped through the same rule as every other potential term.
            self._potential = quadrature.effective_potential(
                quadrature.transformed_potential, factor.transformed_potential
            )
            self._inverse_weight = grid.volume_element / quadrature.mass_weights
            self.measure = Measure(grid, quadrature.mass_weights / grid.volume_element)
        else:
            self._potential = factor.transformed_potential
            self._inverse_weight = None if factor.is_identity else 1.0 / factor.weight
            self.measure = Measure.uniform(grid) if factor.is_identity else Measure(grid, factor.weight)
        self._bare_potential = self._potential
        if extra_potential is not None:
            if extra_potential.shape[-1] != grid.n_points:
                raise ValueError(
                    f"extra potential has {extra_potential.shape[-1]} points, "
                    f"grid has {grid.n_points}"
                )
            self._potential = self._potential + extra_potential
        self._grad_u = factor.grad_u
        self._apply_count = 0
        #: True when ``apply`` uses the staggered divergence form, exactly self-adjoint in
        #: :attr:`measure`. Recorded per run and read by the eigensolver's second criterion (D-44).
        self.uses_divergence_form = not factor.is_identity

    @property
    def apply_count(self) -> int:
        """Number of applications so far, the cost metric for a matrix-free solver."""
        return self._apply_count

    def apply(self, phi: torch.Tensor) -> torch.Tensor:
        """Return ``A phi`` for one or many transformed orbitals, shape ``(..., n_points)``.

        In staggered divergence form::

            A phi = (1 / 2 f^2) sum_d D_d^T [ f^2_face,d  D_d phi ]  +  W phi

        Self-adjointness in the measure, a non-negative kinetic term and the absence of a
        checkerboard null space hold by construction, not to the order of the stencil. Do not
        substitute the collocation form. One staggered derivative and one transpose per axis.
        """
        self._apply_count += 1
        return self.apply_with_potential(phi, self._potential)

    def apply_with_potential(self, phi: torch.Tensor, potential: torch.Tensor) -> torch.Tensor:
        """:meth:`apply` with the diagonal potential passed in, without counting.

        Same operations in the same order as :meth:`apply`. A CUDA graph of the Chebyshev filter
        calls it with a static buffer ``copy_``-refreshed from :meth:`graph_potential` before each
        replay, so one capture serves every SCF iteration (the kinetic part is geometry-only).
        """
        if self.factor.is_identity:
            out = self.grid.laplacian(phi).mul_(-0.5)
            return out.add_(phi * potential)

        return self.kinetic_apply(phi).add_(phi * potential)

    def graph_potential(self) -> torch.Tensor:
        """The full diagonal potential :meth:`apply` multiplies by (bare ``W~`` plus the extra term)."""
        return self._potential

    def graph_geometry_key(self) -> tuple:
        """Identity of every tensor :meth:`kinetic_apply` reads, for CUDA-graph reuse.

        The grid, the factor's cached face weights and the inverse measure weight; on the SCF path
        the last comes from the quadrature's geometry cache, so every iteration's operator has the
        same key. Data pointers identify live tensors the capturing operator keeps alive.
        """
        if self.factor.is_identity:
            return ("cusp-identity", id(self.grid))
        faces = self.factor.face_weights(self.grid.fd_order)
        return (
            "cusp",
            id(self.grid),
            tuple(w.data_ptr() for w in faces),
            self._inverse_weight.data_ptr(),
        )

    def note_applications(self, count: int) -> None:
        """Add ``count`` applications made outside :meth:`apply` (a graph replay of the filter)."""
        self._apply_count += int(count)

    def kinetic_apply(self, phi: torch.Tensor) -> torch.Tensor:
        """Return the kinetic part of ``A phi`` alone: ``(1 / 2 w) sum_d D_d^T [f^2_face D_d phi]``.

        ``w`` is the measure's per-point weight (``f^2`` or ``omega / h^3``), so that
        ``<a|kinetic|b>_measure = (1/2) sum_faces f^2_face (D a)(D b) h^3`` in either case.
        """
        accuracy = self.grid.fd_order
        box = self.grid.scatter_to_box(phi)
        divergence = weighted_divergence(
            box,
            self.factor.face_weights(accuracy),
            self.grid.spacing,
            accuracy,
            self.grid.shape,
        )
        return self.grid.gather_from_box(divergence).mul_(0.5) * self._inverse_weight

    def local_potential(self) -> torch.Tensor:
        """Return the transformed local potential, for diagnostics and for the record."""
        return self._potential

    def update_potentials(self, density: torch.Tensor) -> None:
        """Refuse: the density-dependent terms arrive with Increment 3."""
        raise NotImplementedError(
            "CuspFactoredHamiltonian carries a fixed extra_potential; the self-consistent "
            "Hamiltonian arrives with Increment 3"
        )

    def kinetic_energy(self, phi: torch.Tensor, occupations: torch.Tensor) -> float:
        """Return ``T_s = (1/2) sum_i f_i integral |grad psi_i|^2`` from the transformed orbitals.

        With ``psi = f phi``: ``T = (1/2) int f^2 |grad phi|^2 - (1/2) int f^2 grad rho . grad u
        + (1/2) int f^2 rho |grad u|^2``, ``rho = sum_i f_i phi_i^2``. On the quadrature path the
        first term is the operator's own kinetic form (so the band-energy identity closes to
        round-off) and the other two are cusp-aware integrals with ``grad u`` analytic at the nodes.
        Without the quadrature, the D-35 gradient form with plain sampling.
        """
        if self.quadrature is None or self.factor.is_identity:
            return self.factor.kinetic_energy(phi, occupations)
        return float(self.kinetic_energy_tensor(phi, occupations))

    def kinetic_energy_tensor(self, phi: torch.Tensor, occupations: torch.Tensor) -> torch.Tensor:
        """:meth:`kinetic_energy` as a 0-d device tensor: the same operations, no host read.

        For the SCF energy assembly, which reads all its scalars in one transfer.
        """
        if self.quadrature is None or self.factor.is_identity:
            return torch.tensor(
                self.factor.kinetic_energy(phi, occupations), dtype=torch.float64, device=phi.device
            )
        occ = occupations.to(torch.float64)
        kinetic_form = self.measure.inner(phi, self.kinetic_apply(phi))  # per state
        first = (occ * kinetic_form).sum()
        rho = (occ[..., None] * phi.to(torch.float64).pow(2)).sum(dim=-2)
        cross, potential = self.external_energy_parts_tensor(rho)
        return first - cross - potential

    def external_energy_parts_tensor(self, rho: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``((1/2) Q[f^2 grad rho . grad u], Q[f^2 rho W])`` as two 0-d device tensors.

        These two bounded integrals carry the external energy of a transformed density: by parts
        with ``lap u = -2 v_ext``, ``E_ext = (1/2) Q[f^2 grad rho . grad u] + 2 Q[f^2 rho W]``, so
        the singular ``int n v_ext`` never appears (D-52 item 2, D-54).
        """
        q = self.quadrature
        assert q is not None
        rho_node = q.interpolate(rho)
        grad_rho_node = q.interpolate_gradient(rho)
        cross_node = q.f2 * (grad_rho_node * q.grad_u).sum(dim=0)
        cross_grid = self.factor.weight * (self.grid.gradient(rho) * self.factor.grad_u).sum(dim=0)
        cross = 0.5 * q.integrate(cross_node, cross_grid)
        potential = q.integrate(
            q.f2 * rho_node * q.transformed_potential, self.factor.weight * rho * self.factor.transformed_potential
        )
        return cross, potential

    def external_energy_direct(self, rho: torch.Tensor) -> float:
        """``E_ext`` by the bounded by-parts formula, for the record's ``external_direct`` field."""
        return float(self.external_energy_direct_tensor(rho))

    def external_energy_direct_tensor(self, rho: torch.Tensor) -> torch.Tensor:
        """:meth:`external_energy_direct` as a 0-d device tensor (same operations, no host read)."""
        cross, potential = self.external_energy_parts_tensor(rho)
        return cross + 2.0 * potential

    def spectral_bounds(
        self, steps: int = 12, safety: float = 1.02, generator: torch.Generator | None = None
    ) -> tuple[float, float]:
        """Estimate ``(lambda_min, lambda_max_upper)`` by a short Lanczos run in the weighted measure."""
        measure = self.measure
        start = seeded_randn(
            self.grid.n_points, generator, device=self.grid.device, dtype=torch.float64
        )
        return _lanczos_bounds(self.apply, start, measure.inner, measure.norm, steps, safety)
