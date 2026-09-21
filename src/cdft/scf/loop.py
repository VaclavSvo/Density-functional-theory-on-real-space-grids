"""The self-consistent field driver: iterate the pure map of :mod:`cdft.scf.step`.

The loop owns what changes the path to the fixed point, not the point itself: the initial guess
[F13], the mixer [F8]--[F11], the fallback ladder of [F12] (periodic Pulay -> damped Pulay ->
linear -> level shift) and the convergence test. Every rung, mixer event and stop reason is
recorded (G5.5: fallbacks are allowed, hidden fallbacks are not).

Convergence requires both ``|dE| < energy_tol`` and ``||dn||_1 < density_tol`` (G3.3); either alone
is satisfied by a stalled iteration. ``dn`` is the input-density change in the cusp-aware L1 norm,
in electrons. The energy per iteration is the variational Kohn--Sham energy of that iteration's
output orbitals (G2.3), with the Harris--Foulkes energy at the input density beside it (G2.1).
"""

from __future__ import annotations

import dataclasses
import time
import uuid

import torch

from contract import (
    EigenResult,
    NumericsConfig,
    RunArtifact,
    RunStatus,
    ScenarioSpec,
    SCFResult,
    SCFTrajectory,
)

from ..eigen.chefsi import BOUND_REUSE_RITZ_FRACTION, SpectralBoundsHint, graphed_filter_for
from ..eigen.lobpcg import LOBPCG
from ..eigen.rayleigh_ritz import orthonormality_error, self_adjointness_error
from ..grid import UniformGrid, grid_config_for_scenario, lattice_anchor
from ..io.provenance import capture
from ..operators import geometry_cache
from ..operators.cusp import CuspFactor
from ..operators.fused import kernel_record
from ..operators.hamiltonian import CuspFactoredHamiltonian
from ..operators.poisson import CoulombCutoffPoisson
from ..operators.quadrature import CuspQuadrature
from ..precision import (
    ENV_LANCZOS_REUSE,
    configure_device,
    env_switch,
    host_floats,
    reset_peak_memory,
    resolve_device,
)
from ..structure import electron_count, nuclear_charges, positions_tensor
from ..xc.dispatch import functional_from_spec
from .mixing import LinearMixer, make_mixer
from .noninteracting import (
    _capture_after_error,
    _hermiticity_error,
    _nucleus_grid_offsets,
    _record_peak,
    _use_cusp_factorisation,
)
from .step import KohnShamStep, StepGeometry

__all__ = ["solve_self_consistent", "initial_density", "build_interacting_step", "INITIAL_GUESSES"]

#: Initial-guess strategies: three genuinely different starting points for G3.4.
INITIAL_GUESSES = ("hydrogenic", "diffuse", "compact")

#: Chebyshev filter steps per SCF iteration after the first [F1]. A path choice: same fixed point
#: to 4e-12 with 2 or 4, but with 2 the output density carries eigenvector error at the density
#: tolerance and the residual wanders in the tail (D-55 item 8). The final diagonalisation always
#: runs to the eigensolver's own tolerance.
FILTER_STEPS_PER_ITERATION = 4

#: Reuse the Lanczos spectral bounds across SCF iterations (D-68): a path choice (D-16) worth ~12
#: of ~92 operator applications per iteration. ``CDFT_LANCZOS_REUSE=0`` switches it off.
LANCZOS_BOUND_REUSE = True

#: Force a fresh Lanczos estimate this often: the backstop against a carried interval that is
#: rigorous but has drifted wide.
LANCZOS_REUSE_BACKSTOP = 8

#: Re-estimate once the accumulated Weyl shifts exceed this fraction of the interval width: still
#: an enclosure, but a wider one damps less per filter step. On the tail, ~1e-4 Ha against ~1e2 Ha.
LANCZOS_REUSE_MAX_WIDENING = 0.05


def lanczos_reuse_enabled() -> bool:
    """:data:`LANCZOS_BOUND_REUSE`, unless ``CDFT_LANCZOS_REUSE`` switches it off."""
    return LANCZOS_BOUND_REUSE and env_switch(ENV_LANCZOS_REUSE, True)


class BoundReusePolicy:
    """When an SCF iteration may filter with the previous iteration's spectral bounds (D-68).

    The Weyl bound ``max |v_eff,new - v_eff,old|`` widens the carried interval, keeping it an
    enclosure (:class:`~cdft.eigen.chefsi.SpectralBoundsHint`); :meth:`decide` lists the cases that
    force a fresh estimate instead. The level-shift rung is one of them: its projector is not a
    diagonal perturbation, so the Weyl shift does not cover it. Reads nothing from the device.
    """

    def __init__(self, enabled: bool) -> None:
        """Start with no carried interval."""
        self.enabled = enabled
        self.bounds: tuple[float, float] | None = None
        self.since_estimate = 0
        self.widening = 0.0
        self.top_fraction = float("nan")
        self.history: list[str] = []
        self.reestimates = 0
        self.reuses = 0

    def decide(self, *, residual_rose: bool, rung_changed: bool, level_shift: bool) -> str | None:
        """Return the reason for a fresh estimate this iteration, or ``None`` to reuse."""
        if not self.enabled:
            return f"reuse off ({ENV_LANCZOS_REUSE}=0 or LANCZOS_BOUND_REUSE=False)"
        if self.bounds is None:
            return "no carried interval"
        if level_shift:
            return "level-shift rung active"
        if rung_changed:
            return "fallback rung taken"
        if residual_rose:
            return "SCF residual rose"
        if self.since_estimate >= LANCZOS_REUSE_BACKSTOP:
            return f"backstop ({LANCZOS_REUSE_BACKSTOP} iterations)"
        low, high = self.bounds
        if self.widening > LANCZOS_REUSE_MAX_WIDENING * (high - low):
            return f"accumulated shift {self.widening:.3e} > {LANCZOS_REUSE_MAX_WIDENING:g} of the width"
        if not self.top_fraction < BOUND_REUSE_RITZ_FRACTION:
            return f"top Ritz fraction {self.top_fraction:.3f} near the upper bound"
        return None

    def hint(self, shift: torch.Tensor | float) -> SpectralBoundsHint:
        """The carried interval with this iteration's Weyl shift; valid only after ``decide`` said reuse."""
        assert self.bounds is not None
        return SpectralBoundsHint(self.bounds[0], self.bounds[1], shift)

    def observe(self, reason: str | None, info: dict[str, object]) -> None:
        """Take the bounds the solver used and log the choice."""
        bounds = info.get("eigen_spectral_bounds") or None
        self.bounds = (float(bounds[0]), float(bounds[1])) if bounds else None
        self.top_fraction = float(info.get("eigen_top_ritz_fraction", float("nan")))
        source = str(info.get("eigen_bounds_source", "lanczos"))
        if source == "reused":
            shift = float(info.get("eigen_bounds_shift", 0.0))
            self.reuses += 1
            self.since_estimate += 1
            self.widening += shift
            self.history.append(f"reused (shift {shift:.3e})")
        else:
            self.reestimates += 1
            self.since_estimate = 0
            self.widening = 0.0
            self.history.append(f"lanczos: {reason}" if reason is not None else source)

    def record(self) -> dict[str, object]:
        """The measurements of this policy (counts, per-iteration history, constants)."""
        return {
            "lanczos_bound_reuse": self.enabled,
            "lanczos_reestimates": self.reestimates,
            "lanczos_reuses": self.reuses,
            "lanczos_bound_history": list(self.history),
            "lanczos_reuse_constants": {
                "backstop_iterations": LANCZOS_REUSE_BACKSTOP,
                "max_widening": LANCZOS_REUSE_MAX_WIDENING,
                "ritz_fraction": BOUND_REUSE_RITZ_FRACTION,
            },
        }


def initial_density(step: KohnShamStep, kind: str = "hydrogenic") -> torch.Tensor:
    """Return a starting density ``(n_spin, n_points)`` normalised to the electron count.

    ``hydrogenic`` is the superposition of ``(Z_a^3/pi) e^{-2 Z_a s_a}`` (the all-electron analogue
    of the SAP guess of [F13], exact for the bare one-electron atom); ``diffuse`` and ``compact``
    halve and double every ``Z_a``. Normalisation is in the cusp-aware measure, so the first
    iteration already sees the right charge.
    """
    scale = {"hydrogenic": 1.0, "diffuse": 0.5, "compact": 2.0}
    if kind not in scale:
        raise ValueError(f"unknown initial guess {kind!r}; choose from {INITIAL_GUESSES}")
    points = step.grid.points().to(torch.float64)
    total = torch.zeros(step.grid.n_points, dtype=torch.float64, device=step.grid.device)
    for atom, charge in enumerate(step.charges):
        zeta = charge * scale[kind]
        distance = (points - step.positions[atom]).norm(dim=-1)
        total = total + (zeta**3 / torch.pi) * torch.exp(-2.0 * zeta * distance)
    rho = total / step._f2_grid
    charge_now = float((step.quadrature.mass_weights * rho).sum())
    total = total * (step.n_electrons / charge_now)
    return (total / step.n_spin).unsqueeze(0).repeat(step.n_spin, 1)


def build_interacting_step(
    scenario: ScenarioSpec,
    numerics: NumericsConfig,
    device: torch.device | None = None,
    *,
    derive_grid: bool = True,
    n_states: int | None = None,
) -> tuple[UniformGrid, KohnShamStep]:
    """Build the grid, factor, quadrature and the pure map for an interacting scenario.

    Geometry-only objects come from :mod:`cdft.operators.geometry_cache`; whether they were cached
    is in ``grid.geometry_cache_record`` (written to ``measurements["geometry_cache"]``).
    """
    if not _use_cusp_factorisation(scenario):
        raise NotImplementedError(
            "the self-consistent path exists on the cusp-factorised all-electron route only "
            "(NUCLEAR_COULOMB with at least one nucleus); model potentials with Hartree and XC are "
            "not a Phase 1 system"
        )
    dev = device or resolve_device(numerics.device)
    grid_config, overrides = grid_config_for_scenario(
        scenario, numerics.grid, derive_grid=derive_grid, interacting=True
    )
    grid = UniformGrid.from_config(
        grid_config,
        structure=scenario.structure,
        device=dev,
        dtype=torch.float64,
        anchor=lattice_anchor(scenario, derive_grid=derive_grid),
    )
    grid.resolution_overrides = overrides
    charges = nuclear_charges(scenario.structure, scenario.external)
    positions = positions_tensor(scenario.structure, device=grid.device)
    # Geometry-only objects through the cache: the non-interacting path's key for the factor and
    # quadrature, plus the step's Hartree-split tensors and the Poisson kernel (keyed by its
    # padding factor, a numerics setting).
    cache = geometry_cache.lease(grid, charges, positions)
    factor = cache.part("factor", lambda: CuspFactor(grid, charges, positions))
    quadrature = cache.part("quadrature", lambda: CuspQuadrature(grid, factor))
    geometry = cache.part(
        "step_geometry",
        lambda: StepGeometry.build(
            grid, quadrature, charges, positions_tensor(scenario.structure, device=grid.device, dtype=torch.float64)
        ),
    )
    pad_factor = numerics.grid.poisson_pad_factor
    poisson = cache.part(("poisson", pad_factor), lambda: CoulombCutoffPoisson(grid, pad_factor=pad_factor))
    n_electrons = electron_count(scenario.structure, scenario.external, scenario.electrons.n_electrons)
    capacity = 2.0 if scenario.electrons.n_spin == 1 else 1.0
    needed = max(1, int(-(-n_electrons // capacity)))
    requested = max(needed, n_states or needed)
    functional = functional_from_spec(scenario.xc)
    step = KohnShamStep(
        scenario, numerics, grid, factor, quadrature, requested, functional, geometry=geometry, poisson=poisson
    )
    grid.geometry_cache_record = cache.record()
    return grid, step


def solve_self_consistent(
    scenario: ScenarioSpec,
    numerics: NumericsConfig,
    n_states: int | None = None,
    cross_check: bool | None = None,
    *,
    derive_grid: bool = True,
    initial_guess: str = "hydrogenic",
    fixed_occupations: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> RunArtifact:
    """Solve one interacting scenario self-consistently and return a fully gated artifact.

    ``fixed_occupations`` ``(n_spin, n_states)`` overrides the aufbau occupations for Janak's
    theorem (G2.2); ``device`` defaults to ``numerics.device``. An artifact is always returned: a
    failed convergence is ``UNCONVERGED``, an exception ``ERROR`` with a message.
    """
    started = time.perf_counter()
    run_id = f"run_{uuid.uuid4().hex[:16]}"
    device = device if device is not None else resolve_device(numerics.device)
    measurements: dict[str, object] = {}
    scf = numerics.scf
    graphs = None
    try:
        configure_device(device, numerics.deterministic)
        reset_peak_memory(device)
        if fixed_occupations is not None:
            fixed_occupations = fixed_occupations.to(device)
        grid, step = build_interacting_step(scenario, numerics, device=device, derive_grid=derive_grid, n_states=n_states)
        quadrature = step.quadrature
        n_spin = step.n_spin
        functional = step.functional
        measurements["functional"] = functional.name
        measurements["grid"] = grid.describe()
        measurements["geometry_cache"] = grid.geometry_cache_record
        measurements["grid_overrides"] = getattr(grid, "resolution_overrides", {})
        measurements["n_electrons"] = step.n_electrons
        measurements["nucleus_offgrid_bohr"] = _nucleus_grid_offsets(scenario, grid)
        measurements["cusp_factorisation"] = True
        measurements["divergence_form"] = True
        measurements["cusp_quadrature"] = quadrature.describe()
        measurements["initial_guess"] = initial_guess

        def norm_l1(delta: torch.Tensor) -> torch.Tensor:
            """Cusp-aware L1 norm of a density change, in electrons (0-d device tensor)."""
            rho = delta.sum(dim=0).abs() / step._f2_grid
            return (quadrature.mass_weights * rho).sum()

        # Geometry-only: built once instead of in every `inner` call.
        inner_weights = (quadrature.mass_weights / step._f2_grid**2).repeat(n_spin)

        def inner(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            """The mixer's quadrature inner product of two flattened residuals (0-d device tensor)."""
            return (inner_weights * a * b).sum()

        n_in = initial_density(step, initial_guess)
        mixer = make_mixer(numerics.mixing, grid, inner)
        ladder = [
            ("pulay", lambda: make_mixer(numerics.mixing, grid, inner)),
            ("damped pulay", lambda: make_mixer(dataclasses.replace(numerics.mixing, alpha=0.5 * numerics.mixing.alpha), grid, inner)),
            ("linear", lambda: LinearMixer(0.5 * numerics.mixing.alpha)),
            ("linear, level shift 0.5 Ha", lambda: LinearMixer(0.5 * numerics.mixing.alpha)),
        ]
        rung = 0
        level_shift = 0.0
        fallback_events: list[str] = []
        mixing_events: list[str] = []
        energies_out: list[float] = []
        harris: list[float] = []
        residual_norms: list[float] = []
        density_changes: list[float] = []
        eigenvalue_history: list[list[float]] = []
        converged = False
        stop_reason = "iteration limit"
        worsened = 0
        previous_residual = float("inf")
        initial_block: torch.Tensor | None = None
        eigen: EigenResult | None = None
        potentials_in = potentials_out = None
        occupations = None
        n_out = rho_out = None
        iteration = 0
        # Path state, owned here so the step stays pure: the CUDA-graph engine (None on the CPU)
        # and the spectral-bound carry-over with the previous input potential.
        graphs = graphed_filter_for(device)
        bound_policy = BoundReusePolicy(lanczos_reuse_enabled())
        previous_effective: torch.Tensor | None = None
        rung_changed = False
        applications: list[int] = []

        for iteration in range(1, scf.max_iterations + 1):
            potentials_in = step.potentials(n_in, functional)
            hamiltonian = step.hamiltonian(potentials_in)
            if level_shift:
                hamiltonian = _LevelShifted(hamiltonian, eigen, occupations, level_shift)
            # Chebyshev-filtered SCF [F1]: from a warm start a few filter steps suffice, the
            # subspace only tracking a slowly changing Hamiltonian; the first and final
            # diagonalisations converge fully.
            steps = None if initial_block is None else FILTER_STEPS_PER_ITERATION
            bound_reason = bound_policy.decide(
                residual_rose=worsened > 0, rung_changed=rung_changed, level_shift=bool(level_shift)
            )
            hint = None
            if bound_reason is None and previous_effective is not None:
                # Weyl: only the diagonal potential changed, so no eigenvalue moved further than this.
                hint = bound_policy.hint((potentials_in.effective - previous_effective).abs().max())
            elif bound_reason is None:
                bound_reason = "no previous potential"
            eigen, eigen_info = step.diagonalise(
                hamiltonian, initial=initial_block, max_iterations=steps, bounds=hint, graphed_filter=graphs
            )
            bound_policy.observe(bound_reason, eigen_info)
            previous_effective = potentials_in.effective
            rung_changed = False
            applications.append(int(hamiltonian.apply_count))
            initial_block = eigen.eigenvectors  # warm start: path only, not the fixed point
            occupations = fixed_occupations if fixed_occupations is not None else step.occupy(eigen.eigenvalues)
            n_out, rho_out = step.output_density(eigen, occupations)
            potentials_out = step.potentials(n_out, functional)
            terms = step.energy_terms(eigen, occupations, rho_out, potentials_in, potentials_out)
            residual = n_out - n_in
            res_norm_device = norm_l1(residual)
            n_next = mixer.mix(n_in, n_out, iteration)
            change_device = norm_l1(n_next - n_in)
            # The iteration's one host read: energy terms, both L1 norms and the eigenvalues.
            host = host_floats(*terms.values(), res_norm_device, change_device, eigen.eigenvalues)
            n_terms = len(terms)
            breakdown, extra = step.energies_from_host(dict(zip(terms, host[:n_terms])))
            res_norm, change = host[n_terms], host[n_terms + 1]
            energies_out.append(breakdown.total)
            harris.append(breakdown.harris_foulkes)
            residual_norms.append(res_norm)
            eigenvalue_history.append(host[n_terms + 2 :])
            density_changes.append(change)
            mixing_events.append(getattr(mixer, "last_event", mixer.name))
            energy_change = abs(energies_out[-1] - energies_out[-2]) if len(energies_out) > 1 else float("inf")
            if (
                iteration >= scf.min_iterations
                and energy_change < scf.energy_tol
                and res_norm < scf.density_tol
            ):
                converged = True
                stop_reason = f"|dE| = {energy_change:.2e} < {scf.energy_tol:g} and ||dn||_1 = {res_norm:.2e} < {scf.density_tol:g}"
                break
            # Fallback ladder [F12]: three consecutive residual increases step one rung down.
            worsened = worsened + 1 if res_norm > previous_residual else 0
            previous_residual = res_norm
            if scf.fallback_ladder and worsened >= 3 and rung + 1 < len(ladder):
                rung += 1
                name, factory = ladder[rung]
                mixer = factory()
                if "level shift" in name:
                    level_shift = 0.5
                fallback_events.append(f"iteration {iteration}: residual grew three times, stepping down to {name}")
                worsened = 0
                rung_changed = True
                n_next = n_in + 0.5 * numerics.mixing.alpha * residual
            n_in = n_next

        assert eigen is not None and potentials_in is not None and potentials_out is not None
        # Final, fully converged diagonalisation at the last input potential, with fresh Lanczos
        # bounds (no hint) so the recorded eigenpairs do not depend on the carry-over.
        final_input_hamiltonian = step.hamiltonian(potentials_in)
        eigen, eigen_info = step.diagonalise(
            final_input_hamiltonian, initial=eigen.eigenvectors, graphed_filter=graphs
        )
        occupations = fixed_occupations if fixed_occupations is not None else step.occupy(eigen.eigenvalues)
        n_out, rho_out = step.output_density(eigen, occupations)
        potentials_out = step.potentials(n_out, functional)
        breakdown, extra = step.energies(eigen, occupations, rho_out, potentials_in, potentials_out)
        measurements["scf_filter_steps_per_iteration"] = FILTER_STEPS_PER_ITERATION
        # Operator applications per SCF iteration, and in total with the final diagonalisation
        # (the gates' probes below are not counted).
        measurements["scf_hamiltonian_applications"] = list(applications)
        measurements["hamiltonian_applications"] = float(sum(applications) + final_input_hamiltonian.apply_count)
        measurements.update(bound_policy.record())
        measurements.update(extra)
        measurements["scf_iterations"] = float(iteration)
        measurements["scf_stop_reason"] = stop_reason
        measurements["scf_converged"] = converged
        measurements["scf_fallbacks"] = list(fallback_events)
        measurements["mixer"] = mixer.name
        measurements["kerker"] = bool(numerics.mixing.kerker)
        record = potentials_out.record()
        measurements["hartree_split_coefficients"] = list(record["hartree_coefficients"])
        measurements["smooth_charge_outside_cutoff"] = record["smooth_charge_outside_cutoff"]
        measurements["lieb_oxford_integral"] = record["lieb_oxford_integral"]
        measurements["xc_double_counting"] = record["xc_double_counting"]
        if record["virial_scaling_term"] is not None:
            measurements["virial_xc_scaling_term"] = record["virial_scaling_term"]
        measurements["harris_foulkes_gap"] = abs(breakdown.total - breakdown.harris_foulkes)
        measurements["external_method"] = "band_energy_complement"
        measurements["external_method_note"] = (
            "E_ext = E - T - E_H - E_xc - E_ii, the exact complement (D-52 item 2); external_direct "
            "is the bounded by-parts integral (1/2) Q[f^2 grad rho . grad u] + 2 Q[f^2 rho W] "
            "evaluated on the output density and external_consistency their difference, which "
            "vanishes with the SCF residual (D-54)."
        )
        measurements["orthonormality_error"] = orthonormality_error(eigen.eigenvectors, step._bare.measure)
        measurements["eigen_residual_max"] = float(eigen.residuals.max())
        measurements["eigen_iterations"] = float(eigen.n_iterations)
        measurements.update(eigen_info)
        final_hamiltonian = step.hamiltonian(potentials_out)
        generator = torch.Generator(device="cpu").manual_seed(numerics.seed)
        measurements["hermiticity_error"] = _hermiticity_error(final_hamiltonian, grid, generator)
        measurements["self_adjointness_error"] = self_adjointness_error(
            final_hamiltonian, final_hamiltonian.measure, n_vectors=16, generator=generator
        )
        measurements["weight_dynamic_range"] = final_hamiltonian.measure.dynamic_range
        # G1.13 and the charge, exactly as on the non-interacting path.
        exact_weight = step.factor.weight_integral_exact()
        grid_weight = float(quadrature.mass_weights.sum())
        measurements["cusp_weight_integral_grid"] = grid_weight
        measurements["cusp_weight_integral_plain"] = step.factor.weight_integral_grid()
        if exact_weight is None:
            measurements["cusp_weight_quadrature_note"] = (
                "no closed form for integral f^2 with this geometry; the quadrature error is unmeasured here"
            )
        else:
            # Closed form over all space minus the tail beyond the box, so the gate reads the
            # quadrature error and not the box truncation.
            outside = step.factor.weight_integral_outside_box()
            in_box = exact_weight - outside
            measurements["cusp_weight_integral_exact"] = exact_weight
            measurements["cusp_weight_integral_outside_box"] = outside
            measurements["cusp_weight_quadrature_error"] = abs(grid_weight - in_box) / exact_weight
            measurements["density_amplitude_ratio"] = in_box / grid_weight
        measurements["charge_error"] = abs(float((quadrature.mass_weights * rho_out).sum()) - step.n_electrons)
        measurements["charge_error_plain"] = abs(float(grid.integrate(n_out.sum(dim=0))) - step.n_electrons)

        do_cross = numerics.eigen.cross_check if cross_check is None else cross_check
        if do_cross:
            reference = LOBPCG(grid, numerics.eigen)
            second = reference.solve(
                final_hamiltonian, step.n_states, generator=torch.Generator(device="cpu").manual_seed(numerics.seed + 1)
            )
            first, _ = step.diagonalise(final_hamiltonian, initial=eigen.eigenvectors, graphed_filter=graphs)
            delta = (first.eigenvalues - second.eigenvalues).abs()
            measurements["two_solver_max_delta"] = float(delta.max())
            measurements["two_solver_converged"] = bool(second.converged)
            measurements["lobpcg_iterations"] = float(second.n_iterations)

        finite = bool(torch.isfinite(n_out).all() and torch.isfinite(eigen.eigenvalues).all())
        measurements["all_finite"] = finite
        eigenvalues = eigen.eigenvalues.reshape(1, -1).repeat(n_spin, 1)
        orbitals = eigen.eigenvectors.reshape(1, step.n_states, grid.n_points).repeat(n_spin, 1, 1)
        result = SCFResult(
            converged=converged and eigen.converged,
            n_iterations=iteration,
            density=n_out,
            eigenvalues=eigenvalues,
            occupations=occupations,
            energies=breakdown,
            trajectory=SCFTrajectory(
                energies=energies_out,
                residual_norms=residual_norms,
                density_changes=density_changes,
                eigenvalue_history=eigenvalue_history,
                mixing_events=mixing_events,
                fallback_events=fallback_events,
            ),
            v_hartree=potentials_out.v_hartree,
            v_xc=potentials_out.v_xc,
            orbitals=orbitals if numerics.output.store_orbitals else None,
        )
        measurements["harris_foulkes_history"] = harris
        # The fused-kernel tier is chosen at the first stencil apply, so it is read after the solve.
        measurements["fused_kernels"] = kernel_record(grid.device)
        if graphs is not None:
            measurements["cuda_graphs"] = graphs.record()
        status = RunStatus.VALID if result.converged else RunStatus.UNCONVERGED
        peak = _record_peak(measurements, device)
        return RunArtifact(
            run_id=run_id,
            scenario=scenario,
            numerics=numerics,
            provenance=capture(
                numerics, time.perf_counter() - started, device=device, gpu_peak_bytes=peak
            ),
            result=result,
            measurements=measurements,
            status=status,
        )
    except Exception as exc:  # noqa: BLE001 - a physics run must report, not crash the batch
        if graphs is not None:
            measurements["cuda_graphs"] = graphs.record()
        return RunArtifact(
            run_id=run_id,
            scenario=scenario,
            numerics=numerics,
            provenance=_capture_after_error(numerics, started, device, measurements),
            result=None,
            measurements=measurements,
            status=RunStatus.ERROR,
            error_message=f"{type(exc).__name__}: {exc}",
        )


class _LevelShifted:
    """``H + Delta (1 - P_occ)``: the last rung of the fallback ladder [F12].

    ``P_occ`` projects onto the previous iteration's occupied orbitals in the operator's measure,
    so the virtuals move up by ``Delta`` and the occupied manifold is untouched; the fixed point is
    unchanged and the shift is recorded.
    """

    def __init__(self, hamiltonian: CuspFactoredHamiltonian, eigen: EigenResult | None, occupations, shift: float) -> None:
        self.inner = hamiltonian
        self.measure = hamiltonian.measure
        self.grid = hamiltonian.grid
        self.uses_divergence_form = hamiltonian.uses_divergence_form
        self.shift = shift
        occupied = None
        if eigen is not None and occupations is not None:
            mask = occupations.sum(dim=0) > 1.0e-12
            occupied = eigen.eigenvectors[mask].to(torch.float64) if bool(mask.any()) else None
        self.occupied = occupied
        self.apply_count = 0

    def apply(self, phi: torch.Tensor) -> torch.Tensor:
        """``H phi + Delta (phi - P phi)``."""
        self.apply_count += 1
        out = self.inner.apply(phi)
        if self.occupied is None:
            return out
        coefficients = self.measure.cross(phi, self.occupied)  # (..., n_occ)
        projected = coefficients @ self.occupied
        return out + self.shift * (phi - projected)

    def spectral_bounds(self, **kwargs):
        lo, hi = self.inner.spectral_bounds(**kwargs)
        return lo, hi + self.shift

    def local_potential(self):
        return self.inner.local_potential()
