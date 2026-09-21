"""Tier 2 -- internal consistency of the converged solution, checked against the run itself.

Nothing here needs an external number, so it applies to scenarios with no reference value at all.
"""

from __future__ import annotations

from contract import GateKind, GateResult, RunArtifact

from .base import ArtifactGate, GateSpec, artifact_device, make_result, skipped

__all__ = [
    "EggBoxGate",
    "EigenvalueResidualGate",
    "TwoSolverAgreementGate",
    "EnergyClosureGate",
    "HarrisFoulkesGate",
]


class EigenvalueResidualGate(ArtifactGate):
    """G2.4 -- ``max_i ||H psi_i - eps_i psi_i||``; 1e-6 Ha [F1].

    A small residual bounds the eigenvalue error directly, where a small change between iterations
    is also what a stall produces. SKIPs when the run recorded neither measure.
    """

    spec = GateSpec(
        gate_id="G2.4",
        name="Eigenvalue residual",
        threshold=1.0e-6,
        kind=GateKind.DERIVED,
        citation="F1",
        units="Ha",
    )

    def evaluate(self, artifact: RunArtifact) -> GateResult:
        """Read the convergence measure appropriate to how the eigenvalues were obtained.

        On a Galerkin subspace matrix -- the cusp-transformed path -- the eigenvalue does not come
        from ``apply``, so the Ritz-value drift bounds its error and the collocation residual does
        not. Both are recorded, and the detail says which was thresholded.
        """
        # On the weak-form path the residual is bounded only by the filter's stopping rule (D-44),
        # not driven to ``residual_tol``, so it overstates the eigenvalue error by orders.
        galerkin = bool(
            artifact.measurements.get("galerkin_subspace", artifact.measurements.get("divergence_form", False))
        )
        key = "eigen_eigenvalue_drift" if galerkin else "eigen_residual_max"
        value = artifact.measurements.get(key)
        if value is None:
            return skipped(self.spec, f"the run recorded no {key}")
        return make_result(
            self.spec,
            float(value),
            {
                "thresholded_quantity": "Ritz-value drift" if galerkin else "eigenvalue residual",
                "eigen_residual_max": artifact.measurements.get("eigen_residual_max"),
                "eigen_eigenvalue_drift": artifact.measurements.get("eigen_eigenvalue_drift"),
                "iterations": artifact.measurements.get("eigen_iterations"),
            },
        )


class TwoSolverAgreementGate(ArtifactGate):
    """G2.5 -- CheFSI against LOBPCG on the same Hamiltonian; 1e-8 Ha [F1], [F4].

    Diagnostic: a disagreement localises the fault to the eigensolver, an agreement to the
    Hamiltonian. SKIPs without ``cross_check``, or when the LOBPCG oracle did not converge.
    """

    spec = GateSpec(
        gate_id="G2.5",
        name="Two-solver eigenvalue agreement",
        threshold=1.0e-8,
        kind=GateKind.EXACT,
        citation="F1; F4",
        units="Ha",
    )

    def evaluate(self, artifact: RunArtifact) -> GateResult:
        """Read the largest eigenvalue disagreement between the two solvers."""
        value = artifact.measurements.get("two_solver_max_delta")
        if value is None:
            return skipped(
                self.spec,
                "cross_check was not enabled for this run; set EigenConfig.cross_check=True",
            )
        if not artifact.measurements.get("two_solver_converged", True):
            return skipped(
                self.spec,
                "the LOBPCG oracle did not converge, so the comparison would measure the oracle "
                "rather than the production solver",
            )
        return make_result(
            self.spec,
            float(value),
            {"lobpcg_iterations": artifact.measurements.get("lobpcg_iterations")},
        )


class EnergyClosureGate(ArtifactGate):
    """G2.9 -- recorded energy terms against the recorded total; 1e-12 Ha [A2].

    Only floating-point addition stands between the two, hence the threshold. Catches a term double
    counted or dropped when a contribution is added to the Hamiltonian. SKIPs with no result.
    """

    spec = GateSpec(
        gate_id="G2.9",
        name="Energy term closure",
        threshold=1.0e-12,
        kind=GateKind.EXACT,
        citation="A2",
        units="Ha",
    )

    def evaluate(self, artifact: RunArtifact) -> GateResult:
        """Compare the total with the sum of its recorded parts."""
        if artifact.result is None:
            return skipped(self.spec, "the run produced no result")
        energies = artifact.result.energies
        return make_result(
            self.spec,
            energies.closure_error(),
            {
                "total": energies.total,
                "kinetic": energies.kinetic,
                "external": energies.external,
                "hartree": energies.hartree,
                "xc": energies.xc,
                "ion_ion": energies.ion_ion,
            },
        )


class HarrisFoulkesGate(ArtifactGate):
    """G2.1 -- band-structure energy against the self-consistent total; 1e-6 Ha [F22].

    Their difference is second order in the density error, so it is a sensitive convergence probe.
    SKIPs on a non-interacting run, where the two are equal by construction.
    """

    spec = GateSpec(
        gate_id="G2.1",
        name="Harris-Foulkes agreement",
        threshold=1.0e-6,
        kind=GateKind.DERIVED,
        citation="F22",
        units="Ha",
    )

    def evaluate(self, artifact: RunArtifact) -> GateResult:
        """Compare the band energy with the total, unless the comparison is trivially satisfied."""
        if artifact.result is None:
            return skipped(self.spec, "the run produced no result")
        energies = artifact.result.energies
        if energies.hartree == 0.0 and energies.xc == 0.0:
            return skipped(
                self.spec,
                "non-interacting run: with no Hartree or XC term the band energy equals the total "
                "by construction, so this comparison tests nothing (it becomes meaningful at I3)",
            )
        if energies.harris_foulkes is None:
            return skipped(self.spec, "the run recorded no Harris-Foulkes energy")
        return make_result(self.spec, abs(energies.total - energies.harris_foulkes))


class JanakGate(ArtifactGate):
    """G2.2 -- Janak's theorem ``dE/df_i = eps_i`` by central difference in the HOMO occupation.

    1e-4 Ha [F21]: the finite difference itself carries ``O(delta^2 d^3E/df^3)``, of order 1e-5 Ha
    at ``delta = 0.01``. An error in the double-counting terms shows up here at first order. Costs
    two extra self-consistent solves; SKIPs if either fails to converge.
    """

    spec = GateSpec(
        gate_id="G2.2",
        name="Janak's theorem",
        threshold=1.0e-4,
        kind=GateKind.EXACT,
        citation="F21",
        units="Ha",
        solves=2,
    )

    DELTA = 0.01

    def applicable(self, artifact: RunArtifact) -> bool:
        """Applies to a converged self-consistent run."""
        from ..scf.solve import is_interacting

        return artifact.result is not None and is_interacting(artifact.scenario)

    def evaluate(self, artifact: RunArtifact) -> GateResult:
        """Re-converge at ``f_HOMO +- delta`` and compare the energy slope with ``eps_HOMO``."""
        import dataclasses

        import torch

        from ..scf.loop import solve_self_consistent

        result = artifact.result
        assert result is not None
        occupations = result.occupations.to(torch.float64)
        occupied = torch.nonzero(occupations[0] > 1.0e-12).flatten()
        if occupied.numel() == 0:
            return skipped(self.spec, "no occupied orbital to perturb")
        homo = int(occupied[-1])
        eigenvalue = float(result.eigenvalues[0, homo])
        numerics = dataclasses.replace(
            artifact.numerics, eigen=dataclasses.replace(artifact.numerics.eigen, cross_check=False)
        )
        energies: dict[str, float] = {}
        for sign in (+1.0, -1.0):
            perturbed = occupations.clone()
            perturbed[0, homo] += sign * self.DELTA
            run = solve_self_consistent(
                artifact.scenario, numerics, fixed_occupations=perturbed, device=artifact_device(artifact)
            )
            if run.result is None or not run.result.converged:
                return skipped(
                    self.spec,
                    f"the SCF at f_HOMO {'+' if sign > 0 else '-'} {self.DELTA} did not converge "
                    f"({run.error_message or run.status.value}); Janak's theorem cannot be evaluated",
                )
            energies["plus" if sign > 0 else "minus"] = run.result.energies.total
        slope = (energies["plus"] - energies["minus"]) / (2.0 * self.DELTA)
        return make_result(
            self.spec,
            abs(slope - eigenvalue),
            {
                "orbital": homo,
                "occupation": float(occupations[0, homo]),
                "delta": self.DELTA,
                "dE_df": slope,
                "eigenvalue": eigenvalue,
                "energy_plus": energies["plus"],
                "energy_minus": energies["minus"],
            },
        )


class VariationalTailGate(ArtifactGate):
    """G2.3 -- energy rises and residual rises over the last five SCF iterations; 0 violations.

    The trajectory energy is variational, so a rise in the tail means the mixer moved away from the
    minimum; rises below the SCF energy tolerance are not counted, since at the fixed point the last
    digits belong to the eigensolver. The residual must fall strictly; DIIS may wander before the
    window [F8], [F12].
    """

    spec = GateSpec(
        gate_id="G2.3",
        name="Variational tail of the SCF",
        threshold=0.0,
        kind=GateKind.EMPIRICAL,
        citation="F8; F12",
        units="violations",
    )

    WINDOW = 5

    def applicable(self, artifact: RunArtifact) -> bool:
        """Applies to a self-consistent run with a recorded trajectory."""
        from ..scf.solve import is_interacting

        return (
            artifact.result is not None
            and is_interacting(artifact.scenario)
            and len(artifact.result.trajectory.energies) >= 2
        )

    def evaluate(self, artifact: RunArtifact) -> GateResult:
        """Count energy rises and residual rises in the last five iterations."""
        assert artifact.result is not None
        trajectory = artifact.result.trajectory
        energies = trajectory.energies[-self.WINDOW:]
        residuals = trajectory.residual_norms[-self.WINDOW:]
        tolerance = artifact.numerics.scf.energy_tol
        energy_rises = [b - a for a, b in zip(energies[:-1], energies[1:]) if b - a > tolerance]
        # Strict on the residual: a rising tail has meant too few filter steps per SCF iteration
        # (D-55 item 8), which a rule loosened to the mixer's wander would not have found.
        residual_rises = [b - a for a, b in zip(residuals[:-1], residuals[1:]) if b > a]
        return make_result(
            self.spec,
            float(len(energy_rises) + len(residual_rises)),
            {
                "window": len(energies),
                "energies": energies,
                "residual_norms": residuals,
                "energy_rises_above_tolerance": energy_rises,
                "residual_rises": residual_rises,
                "energy_tolerance": tolerance,
                "n_iterations": artifact.result.n_iterations,
            },
        )


class EggBoxGate(ArtifactGate):
    """G2.7 -- peak-to-peak energy spread as the nuclei move through one cell; 1 meV/atom (D16-D18).

    A rigid translation changes no physics, so the whole spread is discretisation artefact -- one
    that is periodic in the nuclear position and therefore biases a corpus instead of averaging out
    (D-39). Eight offsets along the body diagonal, since an axis-aligned sweep can sit in a symmetry
    plane of the stencil. Interacting scenarios SKIP.
    """

    spec = GateSpec(
        gate_id="G2.7",
        name="Egg-box error",
        threshold=3.674932e-5,
        kind=GateKind.EMPIRICAL,
        citation="D16; D17; D18",
        units="Ha/atom",
        solves=8,
    )

    #: Offsets sampled along the cell body diagonal, as fractions of one spacing.
    N_OFFSETS = 8

    def applicable(self, artifact: RunArtifact) -> bool:
        """Applies to any scenario with nuclei to translate and a computed spectrum."""
        if artifact.result is None:
            return False
        if artifact.scenario.external.kind.value not in ("nuclear_coulomb", "soft_coulomb"):
            return False
        return artifact.scenario.structure.n_atoms > 0

    def evaluate(self, artifact: RunArtifact) -> GateResult:
        """Translate the structure through one cell and measure the spread in the ground energy."""
        import dataclasses

        from ..scf.solve import is_interacting, solve_scenario
        from .tier3 import INTERACTING_LADDER_REASON, resolved_grid

        if is_interacting(artifact.scenario):
            return skipped(self.spec, INTERACTING_LADDER_REASON + "; the egg-box of the self-consistent path is an Increment 7 (forces) measurement")
        scenario = artifact.scenario
        structure = scenario.structure
        n_atoms = int(structure.n_atoms)
        # The resolved spacing, not the requested placeholder: the threshold is written against a
        # one-cell amplitude, and a placeholder cell spans several real ones on a molecule.
        spacing = resolved_grid(artifact).spacing

        # The cross-check is off for the sweep only: the gate measures a difference between runs
        # that share every setting, so the absolute solver does not enter.
        # The probes use ``derive_grid=False`` (D-64): a derived lattice is anchored on the first
        # nucleus, so it would travel with the structure and the sweep would read zero.
        numerics = dataclasses.replace(
            artifact.numerics,
            grid=resolved_grid(artifact),
            eigen=dataclasses.replace(artifact.numerics.eigen, cross_check=False),
        )

        energies: list[float] = []
        offsets: list[float] = []
        for step in range(self.N_OFFSETS):
            fraction = step / self.N_OFFSETS
            shift = fraction * spacing
            positions = tuple(
                tuple(float(x) + shift for x in position) for position in structure.positions
            )
            moved = dataclasses.replace(structure, positions=positions)
            probe = dataclasses.replace(scenario, structure=moved)
            try:
                run = solve_scenario(
                    probe, numerics, n_states=1, derive_grid=False, device=artifact_device(artifact)
                )
            except Exception as exc:  # pragma: no cover - a failed probe is reported, not hidden
                return skipped(
                    self.spec,
                    f"the egg-box sweep failed at offset {fraction:.3f} cells: "
                    f"{type(exc).__name__}: {exc}",
                )
            energies.append(float(run.result.eigenvalues[0][0]))
            offsets.append(fraction)

        spread = (max(energies) - min(energies)) / max(n_atoms, 1)
        return make_result(
            self.spec,
            spread,
            {
                "offsets_in_cells": offsets,
                "energies": energies,
                "peak_to_peak": max(energies) - min(energies),
                "per_atom": spread,
                "spacing": spacing,
                "n_atoms": n_atoms,
                "direction": "body diagonal (1,1,1)",
                "note": (
                    "a rigid translation changes no physics; the whole spread is discretisation "
                    "artefact, and it is periodic in the nuclear position rather than random, so "
                    "it biases a corpus systematically rather than averaging out"
                ),
            },
        )
