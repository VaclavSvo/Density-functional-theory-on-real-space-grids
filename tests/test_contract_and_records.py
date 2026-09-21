"""Contract invariants, configuration loading, the gate framework, provenance and the corpus.

Pure algebra and bookkeeping, all marked ``fast``: a wrong configuration must fail before any
compute, a wrong record must fail visibly. Under a second.
"""

from __future__ import annotations

import dataclasses
import math
import pathlib

import pytest
import torch

import cdft
from cdft.config import numerics_from_dict, numerics_to_dict
from cdft.eigen.rayleigh_ritz import orthonormality_error, rayleigh_ritz
from cdft.gates.base import GateSpec, make_result, skipped
from cdft.gates.runner import GateSuite, states_required
from cdft.gates.tier5 import ScopeBoundaryGate
from cdft.grid import GridGeometry, UniformGrid
from cdft.io.hdf5 import CorpusWriter
from cdft.io.provenance import capture
from cdft.operators.hamiltonian import LocalHamiltonian
from cdft.physics_config import REGISTRY, ScenarioRegistry, hydrogenic
from cdft.scf.occupations import aufbau_occupations
from contract import (
    CONTRACT_VERSION,
    AtomicStructure,
    Device,
    ElectronSpec,
    ExternalPotentialKind,
    ExternalPotentialSpec,
    GateKind,
    GateReport,
    GateVerdict,
    GridConfig,
    NumericsConfig,
    Precision,
    PrecisionConfig,
    PseudoSpec,
    ReferenceKind,
    ReferenceValue,
    RunArtifact,
    RunStatus,
    ScenarioSpec,
)

#: Markers for every test in this module (see ``tests/conftest.py``).
pytestmark = [pytest.mark.fast]


class TestContract:
    """The frozen API refuses configurations that would defeat its own gates."""

    def test_package_matches_the_contract_it_was_written_against(self) -> None:
        assert CONTRACT_VERSION == cdft.CONTRACT_VERSION_EXPECTED

    def test_inversion_protocol_is_not_a_component_gate(self) -> None:
        # O-29 (contract 1.6.1): an inverter needs ``kind`` and ``invert`` only; the gate pair
        # copied from ComponentGateProtocol is gone and a gate is not an inverter.
        from contract import ComponentGateProtocol, InversionKind, InversionProtocol

        class Inverter:
            kind = InversionKind.ZMP

            def invert(self, density, grid, n_electrons, v_ext, v_hartree):
                return density, density

        class Gate:
            gate_id = "G0.1"

            def evaluate(self, numerics):
                raise NotImplementedError

        assert isinstance(Inverter(), InversionProtocol)
        assert not isinstance(Gate(), InversionProtocol)
        assert isinstance(Gate(), ComponentGateProtocol)
        assert not isinstance(Inverter(), ComponentGateProtocol)
        assert not {"gate_id", "evaluate"} & set(vars(InversionProtocol))
        assert CONTRACT_VERSION == "1.6.1"

    def test_precision_policy_cannot_relax_the_float64_stages(self) -> None:
        with pytest.raises(ValueError, match="float64"):
            PrecisionConfig(subspace=Precision.FLOAT32)

    def test_fingerprint_is_stable_and_sensitive(self) -> None:
        a, b = NumericsConfig(), NumericsConfig()
        assert a.fingerprint() == b.fingerprint()
        assert a.fingerprint() != NumericsConfig(seed=1).fingerprint()
        assert len(a.fingerprint()) == 64

    def test_literature_reference_without_a_method_is_refused(self) -> None:
        with pytest.raises(ValueError, match="method"):
            ReferenceValue(value=-2.9, kind=ReferenceKind.LITERATURE)
        ReferenceValue(value=-0.5, kind=ReferenceKind.ANALYTIC)  # needs none

    def test_fractional_spin_requires_a_spin_axis(self) -> None:
        with pytest.raises(ValueError, match="spin_polarised"):
            ElectronSpec(magnetisation=0.5)
        assert ElectronSpec(magnetisation=0.5, spin_polarised=True).n_spin == 2

    def test_magnetisation_cannot_exceed_the_electron_count(self) -> None:
        with pytest.raises(ValueError):
            ElectronSpec(n_electrons=1.0, magnetisation=2.0, spin_polarised=True)

    def test_scenario_and_pseudopotential_must_agree(self) -> None:
        structure = AtomicStructure(numbers=(1,), positions=((0.0, 0.0, 0.0),))
        with pytest.raises(ValueError, match="pseudopotential path"):
            ScenarioSpec(
                "x",
                structure,
                external=ExternalPotentialSpec(kind=ExternalPotentialKind.PSEUDOPOTENTIAL),
            )
        with pytest.raises(ValueError, match="not a pseudopotential"):
            ScenarioSpec("x", structure, pseudo=PseudoSpec())

    def test_external_potential_parameters_must_match_the_kind(self) -> None:
        with pytest.raises(ValueError):
            ExternalPotentialSpec(kind=ExternalPotentialKind.SOFT_COULOMB)
        with pytest.raises(ValueError):
            ExternalPotentialSpec(kind=ExternalPotentialKind.PARTICLE_IN_BOX)


class TestConfigLoading:
    """The loader turns a mistyped key into an error rather than a silent default."""

    def test_round_trip(self) -> None:
        config = NumericsConfig(grid=GridConfig(spacing=0.17, fd_order=6))
        assert numerics_from_dict(numerics_to_dict(config)).fingerprint() == config.fingerprint()

    def test_unknown_key_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown key"):
            numerics_from_dict({"grid": {"spaceing": 0.2}})

    def test_unknown_top_level_key_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown top-level"):
            numerics_from_dict({"gird": {}})

    def test_unknown_enum_value_lists_the_alternatives(self) -> None:
        with pytest.raises(ValueError, match="periodic_pulay"):
            numerics_from_dict({"mixing": {"scheme": "pulya"}})


class TestOccupations:
    """Occupations are floating point (D-11): fractional charge and spin are in scope."""

    def test_closed_shell_aufbau(self) -> None:
        eigenvalues = torch.tensor([[-1.0, -0.5, 0.1]], dtype=torch.float64)
        occupations = aufbau_occupations(eigenvalues, 3.0, n_spin=1)
        assert occupations.tolist() == [[2.0, 1.0, 0.0]]

    def test_fractional_electron_number(self) -> None:
        eigenvalues = torch.tensor([[-1.0, -0.5]], dtype=torch.float64)
        occupations = aufbau_occupations(eigenvalues, 1.4, n_spin=1)
        assert occupations[0, 0] == pytest.approx(1.4)
        assert float(occupations.sum()) == pytest.approx(1.4)

    def test_fractional_spin(self) -> None:
        eigenvalues = torch.tensor([[-1.0, -0.5], [-1.0, -0.5]], dtype=torch.float64)
        occupations = aufbau_occupations(eigenvalues, 2.0, n_spin=2, magnetisation=0.6)
        assert float(occupations[0].sum()) == pytest.approx(1.3)
        assert float(occupations[1].sum()) == pytest.approx(0.7)

    def test_charge_is_never_discarded_silently(self) -> None:
        eigenvalues = torch.tensor([[-1.0]], dtype=torch.float64)
        with pytest.raises(ValueError, match="do not fit"):
            aufbau_occupations(eigenvalues, 5.0, n_spin=1)


class TestRayleighRitz:
    """Float64 unconditionally; orthonormality in the grid measure, not the Euclidean one."""

    def test_orthonormal_in_the_quadrature_measure(self) -> None:
        geometry = GridGeometry(shape=(8, 8, 8), spacing=0.7, origin=(0.0, 0.0, 0.0))
        grid = UniformGrid(geometry, torch.ones((8, 8, 8), dtype=torch.bool), fd_order=4)
        hamiltonian = LocalHamiltonian(grid, torch.zeros(grid.n_points, dtype=torch.float64))
        block = torch.randn((5, grid.n_points), dtype=torch.float64)
        evals, vectors, error = rayleigh_ritz(hamiltonian, block, grid)
        assert error < 1e-12
        assert orthonormality_error(vectors, grid) < 1e-12
        assert torch.all(evals[:-1] <= evals[1:])


class TestGateFramework:
    """Verdicts, margins, and the rule that a skip must say why."""

    spec = GateSpec("G9.9", "test gate", threshold=1.0, kind=GateKind.EXACT)

    def test_verdicts(self) -> None:
        assert make_result(self.spec, 0.5).verdict is GateVerdict.PASS
        assert make_result(self.spec, 0.95).verdict is GateVerdict.MARGINAL
        assert make_result(self.spec, 1.5).verdict is GateVerdict.FAIL

    def test_nan_fails_rather_than_passing_by_comparison(self) -> None:
        """``nan > x`` is False, so a naive threshold test reports NaN as a pass."""
        assert make_result(self.spec, float("nan")).verdict is GateVerdict.FAIL
        assert make_result(self.spec, float("inf")).verdict is GateVerdict.FAIL

    def test_skip_requires_a_reason(self) -> None:
        assert skipped(self.spec, "not applicable here").verdict is GateVerdict.SKIPPED
        with pytest.raises(ValueError):
            skipped(self.spec, "")

    def test_report_status_derivation(self) -> None:
        report = GateReport(results=[make_result(self.spec, 0.5)])
        assert report.status is RunStatus.VALID
        report.results.append(make_result(self.spec, 0.95))
        assert report.status is RunStatus.MARGINAL
        report.results.append(make_result(self.spec, 2.0))
        assert report.status is RunStatus.INVALID

    def test_unimplemented_gate_named_by_a_scenario_fails_the_record(self) -> None:
        scenario = ScenarioSpec(
            "made_up",
            AtomicStructure(numbers=(1,), positions=((0.0, 0.0, 0.0),)),
            gates=("G9.99",),
        )
        numerics = NumericsConfig()
        artifact = RunArtifact(
            run_id="r", scenario=scenario, numerics=numerics, provenance=capture(numerics),
            status=RunStatus.VALID,
        )
        GateSuite().run_on_artifact(artifact)
        offending = [r for r in artifact.gates.results if r.gate_id == "G9.99"]
        assert offending and offending[0].verdict is GateVerdict.FAIL
        assert artifact.status is RunStatus.INVALID

    def test_unimplemented_gate_is_classified_unimplemented_not_new(self) -> None:
        # O-26: the runner's FAIL row must reach the ``unimplemented`` bucket, not count as new.
        from cdft.gates.known_open import partition_failures
        from cdft.gates.runner import UNEVALUABLE_GATE_NAME

        scenario = ScenarioSpec(
            "made_up",
            AtomicStructure(numbers=(1,), positions=((0.0, 0.0, 0.0),)),
            gates=("G9.99",),
        )
        numerics = NumericsConfig()
        artifact = RunArtifact(
            run_id="r", scenario=scenario, numerics=numerics, provenance=capture(numerics),
            status=RunStatus.VALID,
        )
        GateSuite().run_on_artifact(artifact)
        offending = [r for r in artifact.gates.results if r.gate_id == "G9.99"]
        assert offending[0].name == UNEVALUABLE_GATE_NAME
        split = partition_failures("made_up", offending)
        assert (split.unimplemented, split.new, split.known) == (1, 0, 0)
        assert not split.blocking

    def test_states_required_comes_from_the_gates(self) -> None:
        assert states_required(REGISTRY["harmonic_w1"]) == 10
        assert states_required(REGISTRY["box_L10"]) == 4
        assert states_required(REGISTRY["h_atom"]) == 0


class TestScopeBoundary:
    """G5.7: the no-machine-learning scope boundary is scanned for, not merely stated."""

    def test_package_contains_no_machine_learning(self) -> None:
        result = ScopeBoundaryGate().evaluate(NumericsConfig())
        assert result.verdict is GateVerdict.PASS, result.detail["violations"]
        assert result.detail["modules_scanned"] > 20

    def test_the_scan_actually_detects_a_violation(self, tmp_path: pathlib.Path) -> None:
        """The scan flags a planted violation, so a pass means something (G5.7)."""
        (tmp_path / "offender.py").write_text(
            "import torch.optim\n\n\ndef train(x):\n    return x\n", encoding="utf-8"
        )
        result = ScopeBoundaryGate(package_root=tmp_path).evaluate(NumericsConfig())
        assert result.verdict is GateVerdict.FAIL
        assert len(result.detail["violations"]) == 2

    def test_a_forbidden_name_inside_a_string_is_not_a_violation(self, tmp_path: pathlib.Path) -> None:
        (tmp_path / "fine.py").write_text('NOTE = "do not import torch.optim here"\n', encoding="utf-8")
        assert ScopeBoundaryGate(package_root=tmp_path).evaluate(NumericsConfig()).measured == 0.0


class TestProvenance:
    """G5.1: every record carries what is needed to reproduce it."""

    def test_capture_is_complete(self) -> None:
        provenance = capture(NumericsConfig(), wall_time_s=1.5)
        assert provenance.contract_version == CONTRACT_VERSION
        assert len(provenance.config_hash) == 64
        assert "torch" in provenance.package_versions
        assert provenance.precision_policy["reduction"] == "float64"
        assert provenance.timestamp_utc.endswith("+00:00")

    def test_device_block_is_captured_and_says_what_ran(self) -> None:
        """The device block is non-empty and reports the device that actually ran (G5.1)."""
        import torch

        provenance = capture(
            dataclasses.replace(NumericsConfig(), device=Device.CPU), device=torch.device("cpu")
        )
        assert provenance.device == "cpu"
        assert provenance.device_capability == "n/a"
        assert provenance.cuda_driver_version == "n/a"
        assert provenance.torch_cuda_version
        assert provenance.deterministic_algorithms in ("strict", "warn_only", "off")
        assert provenance.cublas_workspace_config
        assert provenance.gpu_peak_bytes is None
        assert provenance.precision_policy["hot_path"] == "float64"
        assert provenance.precision_policy["device_type"] == "cpu"

    def _gated(self, provenance):
        from cdft.gates.tier5 import ProvenanceGate

        artifact = RunArtifact(
            run_id="r", scenario=hydrogenic(charge=1.0, scenario_id="h_prov"),
            numerics=NumericsConfig(), provenance=provenance, status=RunStatus.VALID,
        )
        return ProvenanceGate().evaluate(artifact)

    def test_g51_passes_on_a_captured_record_and_reads_the_device_block(self) -> None:
        result = self._gated(capture(NumericsConfig()))
        assert result.verdict is GateVerdict.PASS, result.detail["missing"]
        assert result.detail["device"] == capture(NumericsConfig()).device
        assert "deterministic_algorithms" in result.detail

    def test_g51_counts_a_missing_device_field(self) -> None:
        provenance = dataclasses.replace(capture(NumericsConfig()), device="", cublas_workspace_config="")
        result = self._gated(provenance)
        assert result.verdict is GateVerdict.FAIL
        assert set(result.detail["missing"]) == {"device", "cublas_workspace_config"}

    def test_g51_demands_gpu_peak_bytes_on_a_completed_cuda_run(self) -> None:
        """A completed CUDA record without ``gpu_peak_bytes`` is incomplete (G5.1)."""
        from cdft.gates.tier5 import ProvenanceGate

        cuda_like = dataclasses.replace(
            capture(NumericsConfig()), device="cuda:0", device_capability="7.5",
            cuda_driver_version="unavailable", deterministic_algorithms="strict",
            cublas_workspace_config=":4096:8",
        )
        artifact = RunArtifact(
            run_id="r", scenario=hydrogenic(charge=1.0, scenario_id="h_prov"),
            numerics=NumericsConfig(), provenance=cuda_like, status=RunStatus.VALID,
            result=object(),  # only its presence is read
        )
        assert ProvenanceGate().evaluate(artifact).detail["missing"] == ["gpu_peak_bytes"]
        artifact.provenance = dataclasses.replace(cuda_like, gpu_peak_bytes=123)
        assert ProvenanceGate().evaluate(artifact).verdict is GateVerdict.PASS


class TestCorpus:
    """G5.4: the corpus is append-only and reading invalid records is an explicit act."""

    def _artifact(self, status: RunStatus, run_id: str) -> RunArtifact:
        numerics = NumericsConfig()
        return RunArtifact(
            run_id=run_id,
            scenario=hydrogenic(charge=1.0, scenario_id="h_test"),
            numerics=numerics,
            provenance=capture(numerics),
            status=status,
            measurements={"charge_error": 1e-12},
        )

    def test_default_iterator_excludes_invalid_records(self, tmp_path: pathlib.Path) -> None:
        writer = CorpusWriter(tmp_path / "corpus.h5")
        writer.write(self._artifact(RunStatus.VALID, "run_good"))
        writer.write(self._artifact(RunStatus.INVALID, "run_bad"))
        assert writer.run_ids() == ("run_good",)
        assert set(writer.run_ids(include_invalid=True)) == {"run_good", "run_bad"}
        assert [a.run_id for a in writer.iterate()] == ["run_good"]

    def test_corpus_is_append_only(self, tmp_path: pathlib.Path) -> None:
        writer = CorpusWriter(tmp_path / "corpus.h5")
        writer.write(self._artifact(RunStatus.VALID, "run_x"))
        with pytest.raises(ValueError, match="append-only"):
            writer.write(self._artifact(RunStatus.VALID, "run_x"))

    def test_round_trip_preserves_provenance_and_configuration(self, tmp_path: pathlib.Path) -> None:
        writer = CorpusWriter(tmp_path / "corpus.h5")
        original = self._artifact(RunStatus.VALID, "run_rt")
        writer.write(original)
        restored = writer.read("run_rt")
        assert restored.provenance.config_hash == original.provenance.config_hash
        for key in ("device", "device_capability", "torch_cuda_version", "cuda_driver_version",
                    "deterministic_algorithms", "cublas_workspace_config", "gpu_peak_bytes"):
            assert getattr(restored.provenance, key) == getattr(original.provenance, key), key
        assert restored.numerics.fingerprint() == original.numerics.fingerprint()
        assert restored.measurements["charge_error"] == pytest.approx(1e-12)


class TestRegistry:
    """Scenario identifiers are written into every record, so they must be unique."""

    def test_duplicate_identifier_is_refused(self) -> None:
        registry = ScenarioRegistry()
        registry.register(hydrogenic(charge=1.0, scenario_id="dup"))
        with pytest.raises(ValueError, match="already registered"):
            registry.register(hydrogenic(charge=2.0, scenario_id="dup"))

    def test_phase_one_systems_are_all_electron(self) -> None:
        assert len(REGISTRY) >= 7
        for scenario in REGISTRY:
            assert scenario.all_electron
            assert scenario.pseudo is None

    def test_hydrogenic_reference_is_analytic_and_exact(self) -> None:
        scenario = REGISTRY["he_plus"]
        assert scenario.reference["eigenvalue_0"].value == pytest.approx(-2.0)
        assert scenario.reference["eigenvalue_0"].kind is ReferenceKind.ANALYTIC

    def test_unknown_scenario_names_the_registered_ones(self) -> None:
        with pytest.raises(KeyError, match="h_atom"):
            REGISTRY["not_a_scenario"]


def test_harmonic_levels_include_degeneracies() -> None:
    """The harmonic level table G1.x compares against carries the 1, 3, 6 degeneracies."""
    from cdft.gates.tier1 import _harmonic_levels

    levels = _harmonic_levels(1.0, 10)
    assert levels.tolist() == [1.5] + [2.5] * 3 + [3.5] * 6
    assert math.isclose(float(levels[-1]), 3.5)


class TestKnownOpenRegistry:
    """``cdft.gates.known_open`` excuses recorded failures only, never a new or foreign one."""

    def test_a_regression_beyond_the_drift_allowance_is_new(self) -> None:
        """A measurement past the drift allowance classifies as new and still names the entry."""
        from cdft.gates.known_open import KNOWN_OPEN, classify

        entry = KNOWN_OPEN[0]
        worse = entry.recorded * entry.drift * 1.01
        kind, found = classify(entry.scenario_id, entry.gate_id, worse)
        assert kind == "new", (
            f"{entry.scenario_id}/{entry.gate_id} at {worse:.3e} is beyond the "
            f"{entry.drift:g}x allowance on {entry.recorded:.3e} and must fail the build"
        )
        assert found is entry, "the entry must still be reported so the message can name it"

    def test_a_measurement_inside_the_band_is_known(self) -> None:
        """The recorded value itself classifies as known."""
        from cdft.gates.known_open import KNOWN_OPEN, classify

        entry = KNOWN_OPEN[0]
        kind, _ = classify(entry.scenario_id, entry.gate_id, entry.recorded)
        assert kind == "known"

    def test_a_fixed_defect_is_reported_as_stale(self) -> None:
        """A measurement far better than recorded classifies as improved, flagging a stale entry."""
        from cdft.gates.known_open import KNOWN_OPEN, classify

        entry = KNOWN_OPEN[0]
        kind, _ = classify(entry.scenario_id, entry.gate_id, entry.recorded * 0.1)
        assert kind == "improved"

    def test_an_entry_never_excuses_another_scenario(self) -> None:
        """An entry matches its own scenario only; another scenario classifies as new."""
        from cdft.gates.known_open import KNOWN_OPEN, classify

        entry = KNOWN_OPEN[0]
        kind, found = classify("a_scenario_with_no_entry", entry.gate_id, entry.recorded)
        assert kind == "new" and found is None

    def test_every_entry_names_an_open_item_and_a_reason(self) -> None:
        """Every entry names an open item, a note and a drift allowance of at least one."""
        from cdft.gates.known_open import KNOWN_OPEN

        for entry in KNOWN_OPEN:
            assert entry.open_item, f"{entry.scenario_id}/{entry.gate_id} names no open item"
            assert entry.note, f"{entry.scenario_id}/{entry.gate_id} carries no explanation"
            assert entry.drift >= 1.0, "a drift allowance below 1 would fail on the recorded value"

    def test_every_entry_refers_to_a_gate_that_exists(self) -> None:
        """Every excused gate id is in the catalogue, so no entry is dead weight."""
        from cdft.gates.catalogue import by_id
        from cdft.gates.known_open import KNOWN_OPEN

        for entry in KNOWN_OPEN:
            assert by_id(entry.gate_id) is not None, (
                f"{entry.gate_id} is excused in known_open but is not in the gate catalogue"
            )
