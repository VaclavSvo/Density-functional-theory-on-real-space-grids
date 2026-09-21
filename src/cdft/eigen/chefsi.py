"""Chebyshev-filtered subspace iteration -- the production eigensolver [F1], [F2], [F3].

Matrix-free; the scaled three-term recurrence of [F2] never forms the polynomial. The one bad
failure mode is a spectral upper bound that does not bound, hence the rigorous Lanczos enclosure
of ``spectral_bounds``. Rationale: ``docs/03_METHOD.md (Part E)``.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Callable

import torch

from contract import EigenConfig, EigenResult

from ..grid import UniformGrid
from ..operators.fused import chebyshev_update
from ..precision import ENV_CUDA_GRAPHS, cuda_graphs_enabled, host_floats, seeded_randn
from .measure import Measure
from .rayleigh_ritz import rayleigh_ritz, residual_norms

__all__ = [
    "ChebyshevFilteredSubspace",
    "chebyshev_filter",
    "chebyshev_coefficients",
    "chebyshev_recurrence",
    "GraphedChebyshevFilter",
    "graphed_filter_for",
    "SpectralBoundsHint",
    "BOUND_REUSE_RITZ_FRACTION",
    "set_graph_memory_limits",
]

#: Reject reused bounds once the top Ritz value reaches this fraction of the enclosing interval,
#: ``(theta_top - lambda_min) / (lambda_max - lambda_min)`` (D-68): near 1 the "upper bound" may
#: not bound (the D-44 failure), so Lanczos is rerun at once. Random warm-start padding sits ~0.5.
BOUND_REUSE_RITZ_FRACTION = 0.9

#: Warm-up filter runs on a side stream before a CUDA-graph capture, letting the caching allocator
#: and lazy kernel initialisation settle outside the graph.
GRAPH_WARMUP_RUNS = 2

#: Relative gap (against the eager output's largest entry) above which a first replay is discarded.
#: Both paths are expected bitwise equal; this only tolerates a last-digit kernel difference and is
#: two decades inside the 1e-10 GPU-vs-CPU gate.
GRAPH_VALIDATION_RTOL = 1.0e-12

#: Captures allowed per engine before it stays eager (a block whose shape keeps changing -- a
#: rank-deficient Rayleigh--Ritz dropping vectors -- would otherwise recapture forever).
GRAPH_MAX_CAPTURES = 8

#: Largest graph pool attempted, as a fraction of the card's memory; it lives as long as the engine
#: and is estimated as :data:`GRAPH_POOL_BLOCKS` copies of the block. Over the limit, capture is
#: skipped with the reason recorded.
GRAPH_MEMORY_FRACTION = 0.25

#: Block-sized buffers a captured filter keeps alive (input, output, three recurrence vectors, the
#: stencil's box copy, padded-axis and face transients), for the estimate above.
GRAPH_POOL_BLOCKS = 10

#: The shipped value, kept so a derived policy can be compared against it.
GRAPH_POOL_BLOCKS_DEFAULT = GRAPH_POOL_BLOCKS


def set_graph_memory_limits(fraction: float, pool_blocks: int) -> None:
    """Set :data:`GRAPH_MEMORY_FRACTION` and :data:`GRAPH_POOL_BLOCKS` process-wide (CUDA only)."""
    global GRAPH_MEMORY_FRACTION, GRAPH_POOL_BLOCKS
    if not 0.0 <= float(fraction) <= 1.0:
        raise ValueError(f"graph memory fraction {fraction!r} is not in [0, 1]")
    if int(pool_blocks) < 1:
        raise ValueError(f"graph pool blocks {pool_blocks!r} must be at least 1")
    GRAPH_MEMORY_FRACTION = float(fraction)
    GRAPH_POOL_BLOCKS = int(pool_blocks)


def chebyshev_filter(
    hamiltonian: object,
    block: torch.Tensor,
    degree: int,
    lower: float,
    upper: float,
    spectrum_min: float,
) -> torch.Tensor:
    """Apply the scaled Chebyshev filter of [F2] to a block of orbitals.

    Damps ``[lower, upper]``: the top of the wanted subspace up to the rigorous spectral upper
    bound; ``spectrum_min`` normalises the amplification only. The returned block has the same
    shape and is not orthonormal -- follow with Rayleigh--Ritz, which repairs the collapse onto
    the dominant direction.
    """
    if upper <= lower:
        raise ValueError(f"filter interval is empty: lower={lower}, upper={upper}")
    e = 0.5 * (upper - lower)
    c = 0.5 * (upper + lower)
    sigma = e / (spectrum_min - c)
    sigma1 = sigma

    applied = hamiltonian.apply(block)  # type: ignore[attr-defined]
    y = (applied - c * block) * (sigma1 / e)
    previous = block
    for _ in range(2, degree + 1):
        sigma2 = 1.0 / (2.0 / sigma1 - sigma)
        applied = hamiltonian.apply(y)  # type: ignore[attr-defined]
        # Fused update (D-65); bitwise the unfused expression on the CPU.
        y_new = chebyshev_update(applied, y, previous, c, 2.0 * sigma2 / e, sigma * sigma2)
        previous, y = y, y_new
        sigma = sigma2
    return y


def chebyshev_coefficients(
    degree: int, lower: float, upper: float, spectrum_min: float
) -> list[tuple[float, float, float]]:
    """The scalars of :func:`chebyshev_filter`'s recurrence, one row ``(c, a_k, b_k)`` per step.

    Step 1 is ``y = (H x - c x) a_1`` (``b_1 = 0``); step ``k >= 2`` is
    ``y_k = (H y - c y) a_k - b_k y_{k-1}`` [F2]. Same float64 operations in the same order as the
    eager filter, so the scalars are bitwise the ones it uses.
    """
    if upper <= lower:
        raise ValueError(f"filter interval is empty: lower={lower}, upper={upper}")
    e = 0.5 * (upper - lower)
    c = 0.5 * (upper + lower)
    sigma = e / (spectrum_min - c)
    sigma1 = sigma
    rows = [(c, sigma1 / e, 0.0)]
    for _ in range(2, degree + 1):
        sigma2 = 1.0 / (2.0 / sigma1 - sigma)
        rows.append((c, 2.0 * sigma2 / e, sigma * sigma2))
        sigma = sigma2
    return rows


def chebyshev_recurrence(
    apply_with_potential: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    block: torch.Tensor,
    coefficients: torch.Tensor,
    potential: torch.Tensor,
) -> torch.Tensor:
    """:func:`chebyshev_filter` with its scalars read from a ``(degree, 3)`` tensor (CUDA only).

    The body a CUDA graph captures: every per-call scalar is a row of
    :func:`chebyshev_coefficients` and the potential is a tensor argument, so a replay reads new
    values from static buffers instead of constants baked in at capture. Multiplying by a 0-d
    tensor is the same IEEE operation as by a float, so the result is bitwise the eager filter's.
    """
    degree = coefficients.shape[0]
    applied = apply_with_potential(block, potential)
    y = (applied - coefficients[0, 0] * block) * coefficients[0, 1]
    previous = block
    for k in range(1, degree):
        applied = apply_with_potential(y, potential)
        y_new = (applied - coefficients[k, 0] * y) * coefficients[k, 1] - coefficients[k, 2] * previous
        previous, y = y, y_new
    return y


def _interval_fraction(value: float, low: float, high: float) -> float:
    """``(value - low) / (high - low)``; ``inf`` for an empty interval (the filter then refuses it)."""
    width = high - low
    return (value - low) / width if width > 0.0 else float("inf")


@dataclass(frozen=True)
class SpectralBoundsHint:
    """Spectral bounds carried over from the previous SCF iteration (D-68).

    ``shift`` bounds the operator's change, ``max_i |v_new(i) - v_old(i)|`` for a diagonal-potential
    change alone (0-d device tensor, read with the solver's first host read, or a float). By Weyl,
    every eigenvalue moves by at most the perturbation's norm -- for a diagonal operator in any
    positive diagonal measure, its largest entry -- so ``[lower - shift, upper + shift]`` encloses
    the new spectrum whenever ``[lower, upper]`` enclosed the old one.
    """

    lower: float
    upper: float
    shift: torch.Tensor | float = 0.0


class GraphedChebyshevFilter:
    """The Chebyshev filter replayed as a ``torch.cuda.CUDAGraph`` per block shape (D-67).

    Captures :func:`chebyshev_recurrence` for a fixed block shape and operator geometry; three
    static buffers are refreshed per call (block, potential from ``graph_potential`` so a new SCF
    Hamiltonian reuses the graph, and the coefficient table, a non-blocking copy from pinned host
    memory -- no host read). A change of shape, degree or geometry key recaptures.

    No silent fallback (G5.5): non-CUDA block, an operator without the graph seam (the
    level-shifted rung), a capture or validation failure, the memory limit or too many captures all
    run :func:`chebyshev_filter` eagerly, counted by reason; a failure disables the engine with its
    reason in :meth:`record`, which the solvers write to ``measurements["cuda_graphs"]``. Each
    capture's first replay is checked against the eager filter (:data:`GRAPH_VALIDATION_RTOL`) and
    that call returns the eager result. The graph pool goes through the caching allocator, so it
    counts in ``gpu_peak_bytes``.
    """

    def __init__(self, device: torch.device | str, reason_disabled: str | None = None) -> None:
        """Prepare an engine for ``device``; ``reason_disabled`` makes it permanently eager."""
        self.device = torch.device(device)
        self.disabled_reason: str | None = reason_disabled
        self.captures = 0
        self.replays = 0
        self.eager_calls: collections.Counter = collections.Counter()
        self.warmup_applications = 0
        self.validation_max_rel_diff: float | None = None
        self._release()

    # -- bookkeeping --

    def record(self) -> dict[str, object]:
        """The outcome for the run record: ``used`` plus captures, replays and every eager reason."""
        used = self.disabled_reason is None and self.replays > 0
        out: dict[str, object] = {
            "used": used,
            "captures": self.captures,
            "replays": self.replays,
            "eager_calls": dict(self.eager_calls),
            "warmup_applications": self.warmup_applications,
            "validation_max_rel_diff": self.validation_max_rel_diff,
        }
        if not used:
            out["reason"] = self.disabled_reason or (
                "no filter call was graph-capable" if self.eager_calls else "the filter was never called"
            )
        return out

    def _disable(self, reason: str) -> None:
        """Stay eager from now on, with ``reason`` in the record, and free the graph."""
        self.disabled_reason = reason
        self._release()

    def _release(self) -> None:
        """Drop the graph, its pool and the static buffers."""
        self._graph = None
        self._key: tuple | None = None
        self._template: object | None = None
        self._static_input: torch.Tensor | None = None
        self._static_potential: torch.Tensor | None = None
        self._static_coefficients: torch.Tensor | None = None
        self._static_output: torch.Tensor | None = None
        self._validated = False

    # -- the call --

    def _eager(self, reason: str, hamiltonian, block, degree, lower, upper, spectrum_min) -> torch.Tensor:
        """Run the eager filter and count why."""
        self.eager_calls[reason] += 1
        return chebyshev_filter(hamiltonian, block, degree, lower=lower, upper=upper, spectrum_min=spectrum_min)

    @staticmethod
    def graph_capable(hamiltonian: object, block: torch.Tensor) -> str | None:
        """Return why a call with this operator and block cannot replay, or ``None`` if it can."""
        if block.device.type != "cuda":
            return f"block on {block.device.type}; graphs are CUDA-only"
        for name in ("apply_with_potential", "graph_potential", "graph_geometry_key", "note_applications"):
            if not hasattr(hamiltonian, name):
                return f"operator {type(hamiltonian).__name__} has no graph seam"
        return None

    def __call__(
        self,
        hamiltonian: object,
        block: torch.Tensor,
        degree: int,
        lower: float,
        upper: float,
        spectrum_min: float,
    ) -> torch.Tensor:
        """Filter ``block`` as :func:`chebyshev_filter` does, by graph replay where possible."""
        if self.disabled_reason is not None:
            return self._eager("disabled", hamiltonian, block, degree, lower, upper, spectrum_min)
        why = self.graph_capable(hamiltonian, block)
        if why is not None:
            return self._eager(why, hamiltonian, block, degree, lower, upper, spectrum_min)
        rows = chebyshev_coefficients(degree, lower, upper, spectrum_min)
        potential = hamiltonian.graph_potential()  # type: ignore[attr-defined]
        key = (
            hamiltonian.graph_geometry_key(),  # type: ignore[attr-defined]
            tuple(block.shape),
            block.dtype,
            tuple(potential.shape),
            potential.dtype,
            degree,
        )
        if key != self._key:
            if self.captures >= GRAPH_MAX_CAPTURES:
                self._disable(f"more than {GRAPH_MAX_CAPTURES} captures (block shape or geometry keeps changing)")
                return self._eager("disabled", hamiltonian, block, degree, lower, upper, spectrum_min)
            try:
                self._capture(hamiltonian, block, potential, rows, key)
            except Exception as exc:  # noqa: BLE001 - any capture failure is recorded, then eager
                self._disable(f"capture failed: {type(exc).__name__}: {exc}")
            if self.disabled_reason is not None:
                return self._eager("disabled", hamiltonian, block, degree, lower, upper, spectrum_min)
        try:
            self._load(block, potential, rows)
            self._replay()
        except Exception as exc:  # noqa: BLE001 - a replay failure is recorded, then eager
            self._disable(f"replay failed: {type(exc).__name__}: {exc}")
            return self._eager("disabled", hamiltonian, block, degree, lower, upper, spectrum_min)
        if not self._validated:
            return self._validate(hamiltonian, block, degree, lower, upper, spectrum_min)
        self.replays += 1
        hamiltonian.note_applications(degree)  # type: ignore[attr-defined]
        return self._static_output.clone()  # type: ignore[union-attr]

    def _replay(self) -> None:
        """Run the captured graph once (its outputs land in the static output buffer)."""
        self._graph.replay()  # type: ignore[union-attr]

    def _load(self, block: torch.Tensor, potential: torch.Tensor, rows: list[tuple[float, float, float]]) -> None:
        """Refresh the three static inputs of the graph (no host read).

        The coefficient table goes through pinned memory with a non-blocking copy, because a
        pageable copy would synchronise the stream; the caching host allocator will not reuse the
        pinned block before its copy has run.
        """
        table = torch.tensor(rows, dtype=torch.float64)
        if self._static_coefficients.device.type == "cuda":  # type: ignore[union-attr]
            try:
                table = table.pin_memory()
            except RuntimeError:
                pass
        self._static_coefficients.copy_(table, non_blocking=table.is_pinned())  # type: ignore[union-attr]
        if potential.data_ptr() != self._static_potential.data_ptr():  # type: ignore[union-attr]
            self._static_potential.copy_(potential)  # type: ignore[union-attr]
        self._static_input.copy_(block)  # type: ignore[union-attr]

    def _validate(self, hamiltonian, block, degree, lower, upper, spectrum_min) -> torch.Tensor:
        """Keep a capture only if its first replay matches the eager filter; return the eager output."""
        eager = chebyshev_filter(hamiltonian, block, degree, lower=lower, upper=upper, spectrum_min=spectrum_min)
        diff, scale = host_floats((eager - self._static_output).abs().max(), eager.abs().max())
        relative = diff / scale if scale > 0.0 else diff
        self.validation_max_rel_diff = max(relative, self.validation_max_rel_diff or 0.0)
        self.eager_calls["validation of a new capture"] += 1
        if not relative <= GRAPH_VALIDATION_RTOL:
            self._disable(
                f"replay differs from the eager filter by {relative:.3e} (relative) > {GRAPH_VALIDATION_RTOL:g}"
            )
        else:
            self._validated = True
        return eager

    def _capture(
        self,
        hamiltonian: object,
        block: torch.Tensor,
        potential: torch.Tensor,
        rows: list[tuple[float, float, float]],
        key: tuple,
    ) -> None:
        """Warm up on a side stream, then capture :func:`chebyshev_recurrence`."""
        self._release()
        device = block.device
        estimate = GRAPH_POOL_BLOCKS * block.numel() * block.element_size()
        total = torch.cuda.get_device_properties(device).total_memory
        if estimate > GRAPH_MEMORY_FRACTION * total:
            self._disable(
                f"estimated graph pool {estimate / 2**30:.2f} GiB exceeds {GRAPH_MEMORY_FRACTION:g} "
                f"of the card's {total / 2**30:.2f} GiB"
            )
            return
        static_input = block.detach().clone()
        static_potential = potential.detach().clone()
        static_coefficients = torch.tensor(rows, dtype=torch.float64, device=device)
        apply = hamiltonian.apply_with_potential  # type: ignore[attr-defined]
        degree = len(rows)
        side = torch.cuda.Stream(device=device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            for _ in range(GRAPH_WARMUP_RUNS):
                chebyshev_recurrence(apply, static_input, static_coefficients, static_potential)
                self.warmup_applications += degree
        torch.cuda.current_stream(device).wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = chebyshev_recurrence(apply, static_input, static_coefficients, static_potential)
        self.warmup_applications += degree
        self._graph = graph
        self._template = hamiltonian  # keeps every geometry tensor the graph reads alive
        self._static_input = static_input
        self._static_potential = static_potential
        self._static_coefficients = static_coefficients
        self._static_output = static_output
        self._key = key
        self._validated = False
        self.captures += 1


def graphed_filter_for(device: torch.device) -> GraphedChebyshevFilter | None:
    """The CUDA-graph engine a solve on ``device`` uses; ``None`` off CUDA (eager filter).

    On CUDA an engine is always returned, disabled with its reason under ``CDFT_CUDA_GRAPHS=0``,
    so that the record says what happened (G5.5).
    """
    if device.type != "cuda":
        return None
    if not cuda_graphs_enabled(device):
        return GraphedChebyshevFilter(device, reason_disabled=f"disabled by {ENV_CUDA_GRAPHS}=0")
    return GraphedChebyshevFilter(device)


class ChebyshevFilteredSubspace:
    """Chebyshev-filtered subspace iteration, implementing :class:`~contract.EigensolverProtocol`."""

    def __init__(
        self,
        grid: UniformGrid,
        config: EigenConfig,
        ritz_stop: bool = True,
        graphed_filter: GraphedChebyshevFilter | None = None,
    ) -> None:
        """Build the solver for one grid and one eigensolver configuration.

        ``graphed_filter`` (CUDA only) replays the filter as a graph where it can; ``None``
        -- always on the CPU -- calls :func:`chebyshev_filter`.

        ``ritz_stop=False`` disables the weak-form Ritz-stability criterion (D-44), so the filter
        runs until the residual is below ``residual_tol`` or ``max_iterations`` is reached. The
        self-consistent loop uses this (D-55 item 8): the Ritz criterion stops with eigenvectors,
        and so a density, still carrying a residual of order 1e-6, the SCF's own density tolerance,
        which the loop would then read as an SCF residual.
        """
        self.grid = grid
        self.config = config
        self.ritz_stop = ritz_stop
        self.graphed_filter = graphed_filter
        self.history: list[dict[str, float]] = []
        #: What the last solve filtered with: the ``(lambda_min, lambda_max)`` interval, its origin
        #: (``"lanczos"``, ``"reused"``, ``"lanczos (reuse rejected: ...)"``), the shift a reuse
        #: widened it by, and the largest top-Ritz fraction of the interval seen.
        self.spectral_bounds: tuple[float, float] | None = None
        self.bounds_source = "unset"
        self.bounds_shift = 0.0
        self.top_ritz_fraction = float("nan")

    def solve(
        self,
        hamiltonian: object,
        n_states: int,
        initial: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        bounds: SpectralBoundsHint | None = None,
    ) -> EigenResult:
        """Return the lowest ``n_states`` eigenpairs.

        ``bounds`` carries a previous operator's spectral interval, widened by its shift, in place
        of the Lanczos estimate; ``None`` -- every non-SCF solve and the first and final SCF
        diagonalisations -- runs Lanczos. A carried interval is replaced by a fresh estimate once
        the top Ritz value reaches :data:`BOUND_REUSE_RITZ_FRACTION` of it.

        The subspace carries ``n_extra_states`` beyond those requested and does not return them:
        the filter damps imperfectly near the boundary of the wanted set, so a subspace with no
        slack converges slowly for the highest requested state and not at all if it is nearly
        degenerate with the first unwanted one.
        """
        grid = self.grid
        cfg = self.config
        # The operator declares the inner product it is self-adjoint in; the wrong one gives a
        # non-symmetric subspace matrix whose eigenvalues are the Ritz values of nothing.
        measure: Measure = getattr(hamiltonian, "measure", None) or Measure.uniform(grid)
        extra = max(cfg.n_extra_states, int(cfg.extra_states_fraction * n_states))
        subspace_size = min(n_states + extra, grid.n_points)

        if initial is None:
            block = seeded_randn(
                (subspace_size, grid.n_points), generator, device=grid.device, dtype=torch.float64
            )
        else:
            block = initial.to(torch.float64).reshape(-1, grid.n_points)
            if block.shape[0] < subspace_size:
                pad = seeded_randn(
                    (subspace_size - block.shape[0], grid.n_points),
                    generator,
                    device=grid.device,
                    dtype=torch.float64,
                )
                block = torch.cat([block, pad], dim=0)

        def lanczos() -> tuple[float, float]:
            """The Lanczos enclosure of the operator's spectrum (``_lanczos_bounds``)."""
            return hamiltonian.spectral_bounds(  # type: ignore[attr-defined]
                steps=cfg.lanczos_steps, safety=cfg.bound_safety, generator=generator
            )

        if bounds is None:
            spectrum_min, spectrum_max = lanczos()
            self.bounds_source = "lanczos"
            self.bounds_shift = 0.0
        evals, vectors, orth = rayleigh_ritz(hamiltonian, block, measure, host=False)

        converged = False
        iteration = 0
        residual = residual_norms(hamiltonian, evals, vectors, measure)
        previous_evals = evals.clone()
        stable = 0
        self.eigenvalue_drift = float("inf")
        self.stop_reason = "iteration limit"
        # Second, weaker criterion -- Ritz values stable -- on the weak-form path ONLY (D-44),
        # where the eigenvalue comes from an exactly symmetric Galerkin matrix and the residual
        # from `apply`, so the two may legitimately disagree by orders. On a collocation operator
        # it is a premature stop reporting "converged". `stop_reason` records which fired (G5.5).
        weak_form = self.ritz_stop and bool(getattr(hamiltonian, "uses_divergence_form", False))
        eigenvalue_tol = 0.1 * cfg.cross_check_tol
        # One host read per iteration: the filter's lower bound is the top Ritz value just
        # computed, so it travels with that iteration's residual, drift and orthonormality error.
        # Only this first read stands alone.
        if bounds is None:
            (next_lower,) = host_floats(evals[min(subspace_size, evals.shape[0]) - 1])
        else:
            # The reuse's shift travels with the first read (no extra synchronisation).
            next_lower, shift = host_floats(evals[min(subspace_size, evals.shape[0]) - 1], bounds.shift)
            spectrum_min, spectrum_max = bounds.lower - shift, bounds.upper + shift
            self.bounds_source = "reused"
            self.bounds_shift = shift
        top_fraction = float("-inf")
        filter_step = self.graphed_filter if self.graphed_filter is not None else chebyshev_filter
        for iteration in range(1, cfg.max_iterations + 1):
            lower = next_lower
            fraction = _interval_fraction(lower, spectrum_min, spectrum_max)
            if self.bounds_source == "reused" and not fraction < BOUND_REUSE_RITZ_FRACTION:
                # A carried "upper bound" this close to the subspace may not bound (D-44):
                # re-estimate rather than filter with it.
                spectrum_min, spectrum_max = lanczos()
                self.bounds_source = (
                    f"lanczos (reuse rejected: top Ritz fraction {fraction:.3f} >= "
                    f"{BOUND_REUSE_RITZ_FRACTION:g} at iteration {iteration})"
                )
                fraction = _interval_fraction(lower, spectrum_min, spectrum_max)
            top_fraction = max(top_fraction, fraction)
            # The damping interval must start above the wanted states and stay inside the spectrum.
            lower = min(max(lower, spectrum_min + 1e-8), spectrum_max - 1e-8)
            filtered = filter_step(
                hamiltonian,
                vectors,
                cfg.chebyshev_degree,
                lower=lower,
                upper=spectrum_max,
                spectrum_min=spectrum_min,
            )
            evals, vectors, orth_error = rayleigh_ritz(hamiltonian, filtered, measure, host=False)
            residual = residual_norms(hamiltonian, evals, vectors, measure)
            worst, drift, orth, next_lower = host_floats(
                residual[:n_states].max(),
                (evals[:n_states] - previous_evals[:n_states]).abs().max(),
                orth_error,
                evals[min(subspace_size, evals.shape[0]) - 1],
            )
            previous_evals = evals.clone()
            self.eigenvalue_drift = drift
            self.history.append(
                {
                    "iteration": float(iteration),
                    "residual": worst,
                    "orthonormality": orth,
                    "eigenvalue_drift": drift,
                }
            )
            if worst < cfg.residual_tol:
                converged = True
                self.stop_reason = "residual below residual_tol"
                break
            stable = stable + 1 if (weak_form and drift < eigenvalue_tol) else 0
            if stable >= 2:
                converged = True
                self.stop_reason = (
                    "Ritz values stable on the weak-form path; the collocation residual is "
                    f"{worst:.3e}, above residual_tol {cfg.residual_tol:.1e}"
                )
                break

        self.spectral_bounds = (spectrum_min, spectrum_max)
        self.top_ritz_fraction = top_fraction
        return EigenResult(
            eigenvalues=evals[:n_states],
            eigenvectors=vectors[:n_states],
            residuals=residual[:n_states],
            n_iterations=iteration,
            converged=converged,
        )
