"""Tier 4 -- agreement with published and independently computed values, on the interacting path.

The radial and two-centre reference gates G4.8 and G4.9 live in :mod:`cdft.gates.tier1`, beside the
closed forms they extend.
"""

from __future__ import annotations

import dataclasses

from contract import GateKind, GateResult, ReferenceKind, RunArtifact

from .base import ArtifactGate, GateSpec, artifact_device, make_result, skipped

__all__ = ["PublishedValueGate", "SelfInteractionGate", "reference_key_for"]


def reference_key_for(artifact: RunArtifact) -> str | None:
    """Return the ``cdft.reference.literature`` key for an all-electron atom at a matched functional.

    NIST SRD 141 covers H, He and He+ at spin-restricted VWN LDA; ``None`` when no row applies.
    """
    scenario = artifact.scenario
    if scenario.structure.n_atoms != 1 or scenario.xc.name.lower() != "lda_vwn":
        return None
    if scenario.electrons.spin_polarised:
        return None
    charge = int(scenario.structure.numbers[0])
    n_electrons = float(artifact.measurements.get("n_electrons", float("nan")))
    system = {(1, 1.0): "H", (2, 2.0): "He", (2, 1.0): "He+"}.get((charge, n_electrons))
    return None if system is None else f"{system}/lda/total_energy"


class PublishedValueGate(ArtifactGate):
    """G4.7 -- self-consistent total energy against a transcribed or computed reference; 1e-3 Ha.

    Two sources, both with provenance (D-29): NIST SRD 141 (:mod:`cdft.reference.literature`,
    accuracy 1e-6) and PySCF all-electron RKS totals for what NIST does not cover
    (:mod:`cdft.reference.computed`, O-12). The threshold is not tightened past the reference's own
    accuracy. SKIPs when no reference row exists for the scenario.
    """

    spec = GateSpec(
        gate_id="G4.7",
        name="All-electron agreement with published values",
        threshold=1.0e-3,
        kind=GateKind.EMPIRICAL,
        citation="NIST SRD 141, doi:10.18434/T4ZP4F; PySCF (I2) as a computed oracle",
        units="Ha",
    )

    def applicable(self, artifact: RunArtifact) -> bool:
        """Applies to a converged, spin-restricted, all-electron self-consistent run."""
        from ..scf.solve import is_interacting

        return artifact.result is not None and is_interacting(artifact.scenario) and artifact.scenario.all_electron

    def evaluate(self, artifact: RunArtifact) -> GateResult:
        """Look the reference up, compare, and record which source it came from."""
        from ..reference.literature import lookup

        assert artifact.result is not None
        total = float(artifact.result.energies.total)
        key = reference_key_for(artifact)
        if key is not None:
            value = lookup(key)
            return make_result(
                self.spec,
                abs(total - value.value),
                {
                    "reference_key": key,
                    "reference": value.value,
                    "reference_accuracy": value.accuracy,
                    "reference_method": value.method,
                    "computed": total,
                    "difference": total - value.value,
                    "source": "NIST SRD 141 (literature)",
                },
            )
        try:
            from ..reference.computed import lookup_computed
        except ImportError:
            return skipped(self.spec, "no computed reference table is present (run scripts/pyscf_reference.py)")
        entry = lookup_computed(artifact.scenario.scenario_id)
        if entry is None:
            return skipped(
                self.spec,
                f"no transcribed or computed reference for scenario {artifact.scenario.scenario_id!r} "
                f"at {artifact.scenario.xc.name}; NIST covers H, He, He+ at LDA only and the PySCF "
                f"table has no row for it (scripts/pyscf_reference.py)",
            )
        return make_result(
            self.spec,
            abs(total - entry["value"]),
            {
                "reference": entry["value"],
                "reference_uncertainty": entry.get("uncertainty"),
                "reference_method": entry.get("method"),
                "computed": total,
                "difference": total - entry["value"],
                "source": "PySCF all-electron RKS, computed oracle (O-12)",
                "kind": ReferenceKind.CROSS_CODE.value,
            },
        )


class SelfInteractionGate(ArtifactGate):
    """G4.5 -- ``E(H2+, R) - E(H)`` at the same functional must be negative; 0 Ha (D-12).

    A semi-local functional does not cancel the Hartree self-repulsion of this one-electron system,
    so at large separation the molecule sits below the atom -- the delocalisation error [A9], [C1],
    which the gate requires to appear. Its magnitude is diagnostic D1.8, recorded and never
    thresholded (D-20). Costs one extra solve; SKIPs if the atom does not converge.
    """

    spec = GateSpec(
        gate_id="G4.5",
        name="Self-interaction error, reproduced deliberately",
        threshold=0.0,
        kind=GateKind.EMPIRICAL,
        citation="A9; C1; A17",
        units="Ha (positive part of E(H2+) - E(H))",
        solves=1,
    )

    #: Separation beyond which the delocalisation error is the dominant feature of the curve.
    MIN_BOND_LENGTH = 6.0

    def applicable(self, artifact: RunArtifact) -> bool:
        """Applies to a converged, interacting, two-proton, one-electron run at large separation."""
        from ..scf.solve import is_interacting

        if artifact.result is None or not is_interacting(artifact.scenario):
            return False
        structure = artifact.scenario.structure
        if structure.n_atoms != 2 or any(int(z) != 1 for z in structure.numbers):
            return False
        if abs(float(artifact.measurements.get("n_electrons", 0.0)) - 1.0) > 1.0e-9:
            return False
        import torch

        positions = torch.tensor(structure.positions, dtype=torch.float64, device="cpu")
        return float((positions[0] - positions[1]).norm()) >= self.MIN_BOND_LENGTH

    def evaluate(self, artifact: RunArtifact) -> GateResult:
        """Solve the atom at the same functional and compare."""
        from ..physics_config import hydrogenic, interacting
        from ..scf.loop import solve_self_consistent

        assert artifact.result is not None
        atom = interacting(hydrogenic(1.0, "H"), artifact.scenario.xc, "g45_hydrogen_atom", gates=())
        numerics = dataclasses.replace(
            artifact.numerics, eigen=dataclasses.replace(artifact.numerics.eigen, cross_check=False)
        )
        run = solve_self_consistent(atom, numerics, device=artifact_device(artifact))
        if run.result is None or not run.result.converged:
            return skipped(self.spec, f"the hydrogen atom at {artifact.scenario.xc.name} did not converge: {run.error_message or run.status.value}")
        molecule = float(artifact.result.energies.total)
        atom_energy = float(run.result.energies.total)
        gap = molecule - atom_energy
        tolerance = artifact.numerics.scf.energy_tol
        return make_result(
            self.spec,
            max(0.0, gap + tolerance),
            {
                "E_H2plus": molecule,
                "E_H_atom_same_functional": atom_energy,
                "delocalisation_error_D1_8": gap,
                "exact_asymptote": -0.5,
                "functional": artifact.scenario.xc.name,
                "note": (
                    "a negative gap is the textbook semi-local failure and is what passes; its size "
                    "is diagnostic D1.8, recorded and never thresholded (D-20)"
                ),
            },
        )
