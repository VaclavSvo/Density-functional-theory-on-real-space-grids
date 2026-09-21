"""The oracle switch (O-24): ``CDFT_ORACLES``, the pytest deselection and the runner's plan.

Off by default the PySCF/libxc oracles are off every diagnostic surface: the tests are deselected,
G0.6 leaves the plan and the record, and the record says ``oracles: off``. All ``fast``.
"""

from __future__ import annotations

import sys

import pytest

from cdft.gates.base import GateSpec
from cdft.gates.runner import (
    GateSuite,
    default_artifact_gates,
    default_component_gates,
    planned,
    states_required,
)
from cdft.io.provenance import capture
from cdft.reference import ENV_ORACLES, enable_oracles, oracles_enabled, oracles_mode
from contract import (
    AtomicStructure,
    GateKind,
    GateVerdict,
    NumericsConfig,
    RunArtifact,
    RunStatus,
    ScenarioSpec,
)

pytestmark = [pytest.mark.fast]


def _artifact(gates: tuple[str, ...] = ()) -> RunArtifact:
    """A completed-looking record with no solve behind it; every artifact gate then skips."""
    scenario = ScenarioSpec(
        "made_up", AtomicStructure(numbers=(1,), positions=((0.0, 0.0, 0.0),)), gates=gates
    )
    numerics = NumericsConfig()
    return RunArtifact(
        run_id="r", scenario=scenario, numerics=numerics, provenance=capture(numerics),
        status=RunStatus.VALID,
    )


class TestSwitch:
    """``CDFT_ORACLES`` through ``env_switch``: default off, on/off spellings, a typo raises."""

    def test_default_is_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ENV_ORACLES, raising=False)
        assert oracles_enabled() is False
        assert oracles_mode() == "off"

    @pytest.mark.parametrize("value", ["1", "true", "on", "YES"])
    def test_on_spellings(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        monkeypatch.setenv(ENV_ORACLES, value)
        assert oracles_enabled() is True
        assert oracles_mode() == "on"

    @pytest.mark.parametrize("value", ["0", "false", "off", ""])
    def test_off_spellings(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        monkeypatch.setenv(ENV_ORACLES, value)
        assert oracles_enabled() is False

    def test_a_typo_raises_rather_than_reading_as_off(self, monkeypatch) -> None:
        monkeypatch.setenv(ENV_ORACLES, "maybe")
        with pytest.raises(ValueError, match=ENV_ORACLES):
            oracles_enabled()

    def test_enable_sets_the_variable_for_child_processes(self, monkeypatch) -> None:
        monkeypatch.setenv(ENV_ORACLES, "0")  # set first, so the session's value is restored after
        enable_oracles()
        assert oracles_enabled() is True


class TestDeselection:
    """The conftest hook removes ``oracle`` items from the session instead of skipping them."""

    class _Item:
        def __init__(self, name: str, *keywords: str) -> None:
            self.name = name
            self.keywords = set(keywords)

    @staticmethod
    def _hook():
        conftest = sys.modules.get("tests.conftest") or sys.modules.get("conftest")
        assert conftest is not None, "the session conftest is imported before any test runs"
        return conftest.deselect_oracle_tests, conftest.oracles_requested

    def test_off_removes_exactly_the_marked_items(self) -> None:
        deselect, _ = self._hook()
        items = [
            self._Item("a", "fast"), self._Item("b", "fast", "oracle"), self._Item("c", "oracle")
        ]
        removed = deselect(items, oracles=False)
        assert [item.name for item in removed] == ["b", "c"]
        assert [item.name for item in items] == ["a"]

    def test_on_keeps_every_item(self) -> None:
        deselect, _ = self._hook()
        items = [self._Item("a", "fast"), self._Item("b", "oracle")]
        assert deselect(items, oracles=True) == []
        assert [item.name for item in items] == ["a", "b"]

    def test_a_mis_set_switch_is_a_usage_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _, requested = self._hook()
        monkeypatch.setenv(ENV_ORACLES, "maybe")
        with pytest.raises(pytest.UsageError, match=ENV_ORACLES):
            requested()

    def test_this_session_deselected_the_libxc_tests_unless_requested(self, request) -> None:
        _, requested = self._hook()
        collected = {item.nodeid for item in request.session.items}
        libxc = [nodeid for nodeid in collected if "TestAgainstLibxc" in nodeid]
        assert bool(libxc) == requested()


class TestRunnerPlan:
    """``requires_oracle`` gates leave the plan and the record when the oracles are off."""

    def test_g06_is_the_oracle_gate_and_g47_is_not(self) -> None:
        by_id = {gate.gate_id: gate.spec for gate in default_component_gates()}
        assert by_id["G0.6"].requires_oracle is True
        assert all(not spec.requires_oracle for gid, spec in by_id.items() if gid != "G0.6")
        artifact_specs = {gate.gate_id: gate.spec for gate in default_artifact_gates()}
        assert artifact_specs["G4.7"].requires_oracle is False  # stored references are data
        assert not any(spec.requires_oracle for spec in artifact_specs.values())
        assert GateSpec("G9.9", "x", 1.0, GateKind.EXACT).requires_oracle is False

    def test_planned_filters_only_when_off(self) -> None:
        gates = default_component_gates()
        assert [g.gate_id for g in planned(gates, True)] == [g.gate_id for g in gates]
        assert "G0.6" not in [g.gate_id for g in planned(gates, False)]
        assert len(planned(gates, False)) == len(gates) - 1

    def test_default_reads_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ENV_ORACLES, raising=False)
        off = GateSuite()
        assert off.oracles is False and off.oracles_mode == "off"
        assert "G0.6" not in [g.gate_id for g in off.component_gates]
        assert list(off.dropped_gates) == ["G0.6"]
        monkeypatch.setenv(ENV_ORACLES, "1")
        on = GateSuite()
        assert on.oracles is True and on.oracles_mode == "on"
        assert "G0.6" in [g.gate_id for g in on.component_gates]
        assert not on.dropped_gates

    def test_the_wiring_check_still_sees_the_gate(self) -> None:
        """G5.8 counts G0.6 as wired whatever the switch, or it would flag a drift each run."""
        assert "G0.6" in [g.gate_id for g in default_component_gates()]

    def test_the_record_says_which_mode_ran(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ENV_ORACLES, raising=False)
        artifact = GateSuite(oracles=False).run_on_artifact(_artifact())
        assert artifact.measurements["oracles"] == "off"
        assert "G0.6" not in [r.gate_id for r in artifact.gates.results]
        artifact = GateSuite(oracles=True).run_on_artifact(_artifact())
        assert artifact.measurements["oracles"] == "on"

    def test_a_scenario_naming_a_dropped_gate_gets_a_skipped_row(self) -> None:
        artifact = GateSuite(oracles=False).run_on_artifact(_artifact(gates=("G0.6",)))
        row = [r for r in artifact.gates.results if r.gate_id == "G0.6"]
        assert row and row[0].verdict is GateVerdict.SKIPPED and ENV_ORACLES in row[0].reason
        assert artifact.status is RunStatus.VALID

    def test_states_required_follows_the_plan(self) -> None:
        from cdft.physics_config import REGISTRY

        suite = GateSuite(oracles=False)
        assert states_required(REGISTRY["harmonic_w1"], suite.artifact_gates) == 10
