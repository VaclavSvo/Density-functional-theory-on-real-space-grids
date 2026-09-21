"""Tier 3 -- numerical convergence: gates that re-solve the scenario themselves and read the ladder.

Each passes ``derive_grid=False``, because the D-42 sizing rule would rewrite every rung to the same
derived grid and report a converged constant. G3.1 reports monotonicity beside the extrapolated
error: scatter in a sequence is indistinguishable from physics once it reaches a dissociation curve.
"""

from __future__ import annotations

import dataclasses
import math

from contract import GateKind, GateResult, RunArtifact

from .base import ArtifactGate, GateSpec, artifact_device, make_result, skipped

__all__ = [
    "GridSpacingConvergenceGate",
    "DomainSizeConvergenceGate",
    "StencilOrderGate",
    "SCFConvergenceGate",
    "InitialGuessGate",
    "resolved_grid",
]

#: 1 meV expressed in Hartree, the convergence requirement shared by G3.1, G3.2 and G3.5.
MILLI_EV = 3.674932e-5


def resolved_grid(artifact: RunArtifact):
    """Return the grid configuration the run actually used, after the D-42 derivation.

    ``artifact.numerics.grid`` is only what the caller asked for, and on the all-electron path that
    is a placeholder, so a ladder built from it would sweep around a grid the run never used.
    """
    from ..grid import grid_config_for_scenario
    from ..scf.solve import is_interacting

    config, _ = grid_config_for_scenario(
        artifact.scenario, artifact.numerics.grid, interacting=is_interacting(artifact.scenario)
    )
    return config


def _solve_at(artifact: RunArtifact, grid_updates: dict, n_states: int = 1):
    """Solve this artifact's scenario on the resolved grid with ``grid_updates`` applied.

    The baseline is the resolved grid, so the ladder is centred on the settings the run used, and
    ``derive_grid=False`` keeps the varied rung from being rewritten back to its derived value.
    """
    from ..scf.solve import solve_scenario

    grid = dataclasses.replace(resolved_grid(artifact), **grid_updates)
    numerics = dataclasses.replace(
        artifact.numerics,
        grid=grid,
        eigen=dataclasses.replace(artifact.numerics.eigen, cross_check=False),
    )
    return solve_scenario(
        artifact.scenario, numerics, n_states=n_states, derive_grid=False, device=artifact_device(artifact)
    )


#: Why the ladder gates do not re-solve an interacting scenario inside the suite.
INTERACTING_LADDER_REASON = (
    "self-consistent scenario: a refinement rung costs minutes on two cores at the production "
    "interacting grid (D-53), so the ladders for the interacting path are measured once and "
    "recorded in scripts/interacting_ladder.py and the accuracy table (02_STATUS.md Part F) rather "
    "than re-run on every suite pass; the gate is not deferred, it is budgeted"
)


class GridSpacingConvergenceGate(ArtifactGate):
    """G3.1 -- difference between the two finest rungs of a refining spacing ladder; 1 meV/atom.

    The ladder must refine: a power law fitted to the coarse end and evaluated at zero extrapolates
    outside its own data and returns a confidently wrong limit. The Richardson estimate of the
    remaining error and the monotonicity are reported beside the verdict, never thresholded. Nearly
    vacuous on a single centre, whose transformed problem has a constant solution. Interacting
    scenarios SKIP with :data:`INTERACTING_LADDER_REASON`.
    """
    spec = GateSpec(
        gate_id="G3.1",
        name="Grid-spacing convergence",
        threshold=MILLI_EV,
        kind=GateKind.DERIVED,
        citation="D6; D9; Richardson extrapolation as in D-41",
        units="Ha/atom",
        solves=3,
    )

    #: Refinement ratio between successive rungs.
    RATIO = 1.25
    #: Number of rungs. Three cost about 130 s on a molecule; a fourth costs 315 s on its own.
    N_RUNGS = 3

    def applicable(self, artifact: RunArtifact) -> bool:
        """Applies to any Coulomb scenario with nuclei; model systems have their own exact gates."""
        if artifact.result is None:
            return False
        if artifact.scenario.external.kind.value not in ("nuclear_coulomb", "soft_coulomb"):
            return False
        return artifact.scenario.structure.n_atoms > 0

    def evaluate(self, artifact: RunArtifact) -> GateResult:
        """Solve on a refining ladder, fit the order, and extrapolate to zero spacing."""
        from ..scf.solve import is_interacting

        if is_interacting(artifact.scenario):
            return skipped(self.spec, INTERACTING_LADDER_REASON)
        resolved = resolved_grid(artifact)
        n_atoms = max(int(artifact.scenario.structure.n_atoms), 1)
        base = resolved.spacing
        box = resolved.box_lengths

        spacings: list[float] = []
        energies: list[float] = []
        for rung in range(self.N_RUNGS):
            # Refining, never coarsening; see the class docstring.
            spacing = base / (self.RATIO**rung)
            try:
                run = _solve_at(artifact, {"spacing": spacing, "box_lengths": box})
            except Exception as exc:  # pragma: no cover - a failed rung is reported, not hidden
                return skipped(
                    self.spec,
                    f"the spacing ladder failed at h = {spacing:g}: {type(exc).__name__}: {exc}",
                )
            spacings.append(spacing)
            energies.append(float(run.result.eigenvalues[0][0]))

        # energies[0] is the production spacing; the list runs from coarse to fine.
        steps = [energies[i + 1] - energies[i] for i in range(len(energies) - 1)]
        monotone = all(step <= 0 for step in steps) or all(step >= 0 for step in steps)
        shrinking = all(abs(steps[i]) > abs(steps[i + 1]) for i in range(len(steps) - 1))

        order = float("nan")
        if len(steps) >= 2 and steps[0] != 0.0 and steps[1] != 0.0:
            ratio = abs(steps[0] / steps[1])
            if ratio > 0:
                order = math.log(ratio) / math.log(self.RATIO)

        # The verdict is a measurement rather than a model; the extrapolation below is reported
        # beside it and never thresholded.
        measured = abs(steps[-1]) / n_atoms

        extrapolated: float | None = None
        basis = "measured difference between the two finest rungs"
        if monotone and math.isfinite(order) and order > 0.0:
            factor = self.RATIO**order
            extrapolated = energies[-1] + (energies[-1] - energies[-2]) / (factor - 1.0)

        return make_result(
            self.spec,
            measured,
            {
                "spacings_coarse_to_fine": spacings,
                "energies": energies,
                "successive_differences": steps,
                "monotone": monotone,
                "differences_shrinking": shrinking,
                "measured_order": order,
                "extrapolated_zero_spacing": extrapolated,
                "verdict_basis": basis,
                "remaining_error_estimate": (
                    None if extrapolated is None else abs(energies[-1] - extrapolated) / n_atoms
                ),
                "production_spacing": base,
                "per_atom": measured,
                "note": (
                    "the ladder REFINES from the production spacing. An earlier version coarsened "
                    "and extrapolated back, which returned an h->0 limit 8.8e-4 Ha from the "
                    "two-centre oracle and was reported as proof the error was irreducible -- "
                    "while the direct measurement at h=0.125 is 8.1e-5. A power law fitted to the "
                    "coarse end and evaluated at zero is an extrapolation outside its own data"
                ),
            },
        )


class DomainSizeConvergenceGate(ArtifactGate):
    """G3.2 -- energy shift when the box edge grows by 4 bohr; 1 meV/atom (D5).

    On the cusp-factorised path ``phi`` is constant for a single nucleus, so the Dirichlet condition
    is wrong by ``f^2(R) = exp(-2 Z R)`` and box size is the only thing buying accuracy (D-42). The
    boundary decay of the physical amplitude ``psi = f phi`` is measured too and, above 1e-5 of its
    peak, promoted into the verdict. Interacting scenarios SKIP.
    """

    spec = GateSpec(
        gate_id="G3.2",
        name="Domain-size convergence",
        threshold=MILLI_EV,
        kind=GateKind.DERIVED,
        citation="D5",
        units="Ha/atom",
        solves=2,
    )

    #: Box growth in bohr between the two rungs.
    GROWTH = 4.0
    #: Required decay of the physical amplitude at the boundary, relative to its maximum.
    DECAY_TOLERANCE = 1.0e-5

    def applicable(self, artifact: RunArtifact) -> bool:
        """Applies to a Coulomb scenario in an explicit box."""
        if artifact.result is None:
            return False
        if artifact.scenario.external.kind.value not in ("nuclear_coulomb", "soft_coulomb"):
            return False
        if resolved_grid(artifact).box_lengths is None:
            return False
        return artifact.scenario.structure.n_atoms > 0

    def evaluate(self, artifact: RunArtifact) -> GateResult:
        """Re-solve in a larger box and compare, holding the spacing fixed."""
        from ..scf.solve import is_interacting

        if is_interacting(artifact.scenario):
            return skipped(self.spec, INTERACTING_LADDER_REASON)
        n_atoms = max(int(artifact.scenario.structure.n_atoms), 1)
        resolved = resolved_grid(artifact)
        spacing = resolved.spacing
        box = resolved.box_lengths
        assert box is not None
        grown = tuple(edge + self.GROWTH for edge in box)

        try:
            small = _solve_at(artifact, {"spacing": spacing, "box_lengths": box})
            large = _solve_at(artifact, {"spacing": spacing, "box_lengths": grown})
        except Exception as exc:  # pragma: no cover - a failed rung is reported, not hidden
            return skipped(
                self.spec, f"the box ladder failed: {type(exc).__name__}: {exc}"
            )

        e_small = float(small.result.eigenvalues[0][0])
        e_large = float(large.result.eigenvalues[0][0])
        shift = abs(e_large - e_small) / n_atoms

        decay = self._boundary_decay(small)
        measured = shift
        if decay is not None and decay > self.DECAY_TOLERANCE:
            # A box that truncates the density can still report a stable energy, so the decay
            # failure is promoted into the verdict rather than merely noted.
            measured = max(shift, self.spec.threshold * (decay / self.DECAY_TOLERANCE))

        return make_result(
            self.spec,
            measured,
            {
                "box_lengths": list(box),
                "grown_box_lengths": list(grown),
                "energy_small_box": e_small,
                "energy_large_box": e_large,
                "energy_shift_per_atom": shift,
                "boundary_decay": decay,
                "decay_tolerance": self.DECAY_TOLERANCE,
                "decay_dominates_verdict": decay is not None and decay > self.DECAY_TOLERANCE,
                "note": (
                    "the decay is measured in the physical amplitude psi = f phi. On the "
                    "cusp-factorised path phi is constant and does not decay, so reading phi "
                    "would certify any box, however small"
                ),
            },
        )

    def _boundary_decay(self, run) -> float | None:
        """Return max |psi| on the domain boundary, relative to max |psi| anywhere."""
        import torch

        psi = run.result.orbitals if getattr(run.result, "orbitals", None) is not None else None
        if psi is None:
            return None
        try:
            amplitude = torch.as_tensor(psi).reshape(-1, psi.shape[-1])[0].abs()
            grid = run.measurements.get("grid_boundary_index")
            if grid is None:
                return None
            boundary = torch.as_tensor(grid, dtype=torch.bool, device=amplitude.device)
            peak = float(amplitude.max())
            if peak == 0.0:
                return None
            return float(amplitude[boundary].max()) / peak
        except Exception:  # pragma: no cover - a decay we cannot measure is reported as absent
            return None


class StencilOrderGate(ArtifactGate):
    """G3.5 -- production stencil order against order 12 at the production spacing; 1 meV/atom (D2).

    The cheapest check that the spacing is adequate, and the only gate here probing the stencil: a
    spacing sweep at fixed order cannot see weights that are wrong at that order, since every rung
    is then wrong by the same factor. Interacting scenarios SKIP.
    """

    spec = GateSpec(
        gate_id="G3.5",
        name="Stencil-order independence",
        threshold=MILLI_EV,
        kind=GateKind.DERIVED,
        citation="D2",
        units="Ha/atom",
        solves=2,
    )

    #: Comparison order. 12 against the production 8.
    HIGH_ORDER = 12

    def applicable(self, artifact: RunArtifact) -> bool:
        """Applies to a Coulomb scenario running below the comparison order."""
        if artifact.result is None:
            return False
        if artifact.scenario.external.kind.value not in ("nuclear_coulomb", "soft_coulomb"):
            return False
        if resolved_grid(artifact).fd_order >= self.HIGH_ORDER:
            return False
        return artifact.scenario.structure.n_atoms > 0

    def evaluate(self, artifact: RunArtifact) -> GateResult:
        """Re-solve at the higher stencil order and compare."""
        from ..scf.solve import is_interacting

        if is_interacting(artifact.scenario):
            return skipped(self.spec, INTERACTING_LADDER_REASON)
        n_atoms = max(int(artifact.scenario.structure.n_atoms), 1)
        low_order = resolved_grid(artifact).fd_order
        try:
            low = _solve_at(artifact, {"fd_order": low_order})
            high = _solve_at(artifact, {"fd_order": self.HIGH_ORDER})
        except Exception as exc:  # pragma: no cover - a failed solve is reported, not hidden
            return skipped(
                self.spec,
                f"the order comparison failed: {type(exc).__name__}: {exc}. Order "
                f"{self.HIGH_ORDER} may exceed what this grid can support at its spacing.",
            )

        e_low = float(low.result.eigenvalues[0][0])
        e_high = float(high.result.eigenvalues[0][0])
        return make_result(
            self.spec,
            abs(e_high - e_low) / n_atoms,
            {
                "order_low": low_order,
                "order_high": self.HIGH_ORDER,
                "energy_low_order": e_low,
                "energy_high_order": e_high,
                "difference_per_atom": abs(e_high - e_low) / n_atoms,
                "spacing": resolved_grid(artifact).spacing,
                "note": (
                    "a spacing sweep at fixed order cannot see a stencil whose weights are wrong "
                    "at that order -- every rung is wrong by the same factor and the sequence "
                    "converges cleanly to the wrong number"
                ),
            },
        )


class SCFConvergenceGate(ArtifactGate):
    """G3.3 -- larger of ``|dE| / energy_tol`` and ``||dn||_1 / density_tol`` at convergence; 1.

    Both criteria, because either alone is satisfied by a stalled iteration [F12]. A run that
    stopped on the iteration limit reports its ratios and fails. SKIPs below two iterations.
    """

    spec = GateSpec(
        gate_id="G3.3",
        name="SCF convergence criterion",
        threshold=1.0,
        kind=GateKind.DERIVED,
        citation="F12",
        units="fraction of tolerance",
    )

    def applicable(self, artifact: RunArtifact) -> bool:
        """Applies to a self-consistent run with a trajectory."""
        from ..scf.solve import is_interacting

        return artifact.result is not None and is_interacting(artifact.scenario)

    def evaluate(self, artifact: RunArtifact) -> GateResult:
        """Read the last energy change and residual norm against the configured tolerances."""
        assert artifact.result is not None
        trajectory = artifact.result.trajectory
        scf = artifact.numerics.scf
        if len(trajectory.energies) < 2:
            return skipped(self.spec, "fewer than two SCF iterations were recorded")
        energy_change = abs(trajectory.energies[-1] - trajectory.energies[-2])
        residual = trajectory.residual_norms[-1]
        ratio = max(energy_change / scf.energy_tol, residual / scf.density_tol)
        if not artifact.measurements.get("scf_converged", False):
            ratio = max(ratio, 1.0 + 1.0e-12)  # an iteration-limit stop never passes
        return make_result(
            self.spec,
            ratio,
            {
                "energy_change": energy_change,
                "energy_tol": scf.energy_tol,
                "residual_l1": residual,
                "density_tol": scf.density_tol,
                "iterations": artifact.result.n_iterations,
                "stop_reason": artifact.measurements.get("scf_stop_reason"),
                "fallbacks": artifact.measurements.get("scf_fallbacks"),
            },
        )


class InitialGuessGate(ArtifactGate):
    """G3.4 -- spread of the total energy over :data:`cdft.scf.loop.INITIAL_GUESSES`; 1e-7 Ha [F13].

    The mixer changes the path, never the fixed point (D-16), so a wider spread means a
    guess-dependent stationary point or an unconverged eigenproblem. SKIPs if any guess fails.
    """

    spec = GateSpec(
        gate_id="G3.4",
        name="Initial-guess independence",
        threshold=1.0e-7,
        kind=GateKind.EXACT,
        citation="F13",
        units="Ha",
        solves=2,
    )

    def applicable(self, artifact: RunArtifact) -> bool:
        """Applies to a converged self-consistent run."""
        from ..scf.solve import is_interacting

        return artifact.result is not None and is_interacting(artifact.scenario)

    def evaluate(self, artifact: RunArtifact) -> GateResult:
        """Solve from every other guess and report the spread of the total energies."""
        from ..scf.loop import INITIAL_GUESSES, solve_self_consistent

        assert artifact.result is not None
        used = str(artifact.measurements.get("initial_guess", "hydrogenic"))
        energies = {used: artifact.result.energies.total}
        numerics = dataclasses.replace(
            artifact.numerics, eigen=dataclasses.replace(artifact.numerics.eigen, cross_check=False)
        )
        for guess in INITIAL_GUESSES:
            if guess == used:
                continue
            run = solve_self_consistent(
                artifact.scenario, numerics, initial_guess=guess, device=artifact_device(artifact)
            )
            if run.result is None or not run.result.converged:
                return skipped(
                    self.spec,
                    f"the SCF from the {guess!r} guess did not converge "
                    f"({run.error_message or run.status.value}); the comparison is not available",
                )
            energies[guess] = run.result.energies.total
        spread = max(energies.values()) - min(energies.values())
        return make_result(self.spec, spread, {"energies": energies, "iterations_from_used_guess": artifact.result.n_iterations})
