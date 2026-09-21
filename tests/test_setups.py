"""The per-system setups (D-73): the level tables, the scoped grid rule, and the CLI they feed.

Nothing here solves anything. What is asserted is that ``standard`` is bit-for-bit the production
configuration, that a non-production level is visible in the record, and that a level or a field a
system cannot use is refused before any tensor is allocated.
"""

from __future__ import annotations

import pytest
import torch

import cdft.grid as grid_module
import run
from cdft.grid import (
    PRODUCTION_RULE,
    RULE_TARGETS,
    UniformGrid,
    active_rule,
    grid_config_for_scenario,
    grid_rule,
    interacting_box_edge,
    lattice_anchor,
    lattice_box_edge,
    lattice_offsets,
    lattice_spacing,
)
from cdft.scf.solve import is_interacting
from cdft.structure import electron_count
from config import (
    ALL_ELECTRON_NUMERICS,
    LEVELS,
    MODEL_SYSTEM_NUMERICS,
    SETUPS,
    Resolution,
    numerics_for,
    numerics_for_spec,
)
from contract import ExternalPotentialKind
from physics_config import REGISTRY

pytestmark = pytest.mark.fast

_BASES = {"model": MODEL_SYSTEM_NUMERICS, "all-electron": ALL_ELECTRON_NUMERICS}


def _rule_constants() -> dict[str, float]:
    """The module constants the rule swaps, read straight off :mod:`cdft.grid`."""
    return {name: getattr(grid_module, name.upper()) for name in RULE_TARGETS}


def _expected_production_grid(scenario) -> tuple[float, float] | None:
    """The spacing and box edge D-42/D-53/D-64 give this scenario, from PRODUCTION_RULE."""
    if (
        scenario.external.kind is not ExternalPotentialKind.NUCLEAR_COULOMB
        or scenario.structure.n_atoms == 0
    ):
        return None
    if is_interacting(scenario):
        electrons = electron_count(
            scenario.structure, scenario.external, scenario.electrons.n_electrons
        )
        spacing, _ = lattice_spacing(scenario.structure, PRODUCTION_RULE["interacting_spacing"])
        edge = lattice_box_edge(interacting_box_edge(scenario.structure, electrons), spacing)
        return spacing, edge
    if scenario.structure.n_atoms == 1:
        z_max = max(int(z) for z in scenario.structure.numbers)
        edge = 2.0 * PRODUCTION_RULE["coulomb_half_box_per_z"] / z_max
        return edge / (PRODUCTION_RULE["points_per_edge"] - 1), edge
    positions = [tuple(float(x) for x in r) for r in scenario.structure.positions]
    span = max(max(abs(a[d] - b[d]) for d in range(3)) for a in positions for b in positions)
    spacing, _ = lattice_spacing(scenario.structure, PRODUCTION_RULE["molecular_spacing"])
    return spacing, lattice_box_edge(span + PRODUCTION_RULE["vacuum_padding"], spacing)


class TestSetupsTable:
    """The table covers the registry exactly, and ``standard`` is the production configuration."""

    def test_every_registered_scenario_has_one_setup_in_registry_order(self) -> None:
        assert tuple(SETUPS) == REGISTRY.ids()
        for scenario_id, setup in SETUPS.items():
            assert setup.scenario == scenario_id
            assert setup.base in _BASES
            assert len(setup.about) <= 90
            assert tuple(setup.levels) == LEVELS

    def test_standard_is_the_base_preset_and_no_rule(self) -> None:
        for scenario_id, setup in SETUPS.items():
            assert setup.levels["standard"] == Resolution()
            numerics, targets = numerics_for(scenario_id)
            assert targets == {}
            assert numerics == _BASES[setup.base]

    def test_the_device_replace_is_the_only_change_standard_allows(self) -> None:
        numerics, targets = numerics_for("h_atom", device=ALL_ELECTRON_NUMERICS.device)
        assert (numerics, targets) == (ALL_ELECTRON_NUMERICS, {})

    def test_unknown_scenario_level_and_field_list_the_alternatives(self) -> None:
        with pytest.raises(ValueError, match="unknown scenario 'no_such_scenario'"):
            numerics_for("no_such_scenario")
        with pytest.raises(ValueError, match="unknown level 'exquisite'.*draft"):
            numerics_for("he_atom_lda", "exquisite")
        with pytest.raises(ValueError, match="unknown resolution field.*spacing_bohr"):
            numerics_for("he_atom_lda", spacing_bohr=0.2)

    def test_a_field_that_means_nothing_for_the_system_is_refused(self) -> None:
        with pytest.raises(ValueError, match="points_per_edge"):
            numerics_for("he_atom_lda", points_per_edge=49)
        with pytest.raises(ValueError, match="spacing"):
            numerics_for("h_atom", spacing=0.2)
        with pytest.raises(ValueError, match="box"):
            numerics_for("box_L10", box=12.0)
        with pytest.raises(ValueError, match="half_box"):
            numerics_for("harmonic_w1", half_box=8.0)

    def test_a_value_the_rule_cannot_use_is_refused_before_any_solve(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            numerics_for("he_atom_lda", spacing=-0.2)
        with pytest.raises(ValueError, match="odd integer"):
            numerics_for("h_atom", points_per_edge=32)
        with pytest.raises(ValueError, match="odd integer"):
            numerics_for("h_atom", points_per_edge=7)

    def test_overrides_split_between_the_numerics_and_the_rule(self) -> None:
        numerics, targets = numerics_for(
            "he_atom_lda", "fine", eigen_tol=1.0e-10, max_scf_iterations=5
        )
        assert targets == {"interacting_spacing": 0.20}
        assert numerics.eigen.residual_tol == 1.0e-10
        assert numerics.scf.max_iterations == 5
        # A Coulomb grid is derived, never configured: the spacing went into the rule instead.
        assert numerics.grid == ALL_ELECTRON_NUMERICS.grid

    def test_a_model_system_carries_its_resolution_in_the_numerics(self) -> None:
        numerics, targets = numerics_for("harmonic_w1", "fine")
        assert targets == {}
        assert numerics.grid.spacing == 0.16

    def test_every_model_level_divides_its_box_edge_exactly(self) -> None:
        # from_config re-derives h from the edge; a spacing that does not divide it comes back
        # snapped, and G5.5 reads the snap as an unexplained change (measured on harmonic_w1).
        for scenario_id in ("harmonic_w1", "box_L10"):
            scenario = REGISTRY[scenario_id]
            for level in LEVELS:
                numerics, _ = numerics_for(scenario_id, level)
                config, _ = grid_config_for_scenario(scenario, numerics.grid)
                cells = config.box_lengths[0] / config.spacing
                assert abs(cells - round(cells)) < 1.0e-9

    def test_a_scenario_without_a_setup_falls_back_to_the_base_rule(self) -> None:
        assert numerics_for_spec(REGISTRY["harmonic_w1"]) == (MODEL_SYSTEM_NUMERICS, {})
        numerics, targets = numerics_for_spec(REGISTRY["h2plus_R2"], spacing=0.2)
        assert numerics == ALL_ELECTRON_NUMERICS
        assert targets == {"molecular_spacing": 0.2}


class TestGridRule:
    """The rule is scoped: it swaps the derivation constants and always puts them back."""

    def test_nesting_exceptions_and_exit_all_restore_the_constants(self) -> None:
        before = _rule_constants()
        assert before == dict(PRODUCTION_RULE)
        with grid_rule(interacting_spacing=0.2):
            assert grid_module.INTERACTING_SPACING == 0.2
            with grid_rule(interacting_spacing=0.16, points_per_edge=49):
                assert grid_module.POINTS_PER_EDGE == 49
                assert active_rule() == {"interacting_spacing": 0.16, "points_per_edge": 49}
            assert active_rule() == {"interacting_spacing": 0.2}
        assert _rule_constants() == before
        assert active_rule() == {}

        with pytest.raises(RuntimeError), grid_rule(molecular_spacing=0.2):
            raise RuntimeError("the solve failed")
        assert _rule_constants() == before

    def test_no_arguments_is_a_no_op(self) -> None:
        with grid_rule():
            assert active_rule() == {}

    def test_an_unknown_target_or_a_bad_value_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown grid rule target"), grid_rule(spacing=0.2):
            pass
        with pytest.raises(ValueError, match="positive"), grid_rule(interacting_spacing=0.0):
            pass
        with pytest.raises(ValueError, match="odd integer"), grid_rule(points_per_edge=8):
            pass
        assert _rule_constants() == dict(PRODUCTION_RULE)


class TestDerivationUnderARule:
    """What the rule does to :func:`cdft.grid.grid_config_for_scenario`, and what it records."""

    def test_the_production_derivation_is_unchanged_and_unannotated(self) -> None:
        for scenario in REGISTRY:
            base = _BASES["model" if scenario.structure.n_atoms == 0 else "all-electron"]
            config, overrides = grid_config_for_scenario(
                scenario, base.grid, interacting=is_interacting(scenario)
            )
            assert "grid_rule" not in overrides
            expected = _expected_production_grid(scenario)
            if expected is None:
                continue
            spacing, edge = expected
            assert config.spacing == pytest.approx(spacing, rel=1.0e-12)
            assert config.box_lengths == (edge, edge, edge)

    def test_a_rule_changes_the_derivation_and_says_so_in_the_record(self) -> None:
        scenario = REGISTRY["he_atom_lda"]
        with grid_rule(interacting_spacing=0.2):
            config, overrides = grid_config_for_scenario(
                scenario, ALL_ELECTRON_NUMERICS.grid, interacting=True
            )
        assert config.spacing == pytest.approx(0.2)
        assert "interacting_spacing=0.2" in overrides["grid_rule"]
        assert "production 0.25" in overrides["grid_rule"]

    def test_a_molecule_at_fine_still_puts_every_nucleus_on_the_lattice(self) -> None:
        for scenario_id in ("h2plus_R2", "h2_R1.4"):
            scenario = REGISTRY[scenario_id]
            numerics, targets = numerics_for(scenario_id, "fine")
            with grid_rule(**targets):
                config, _ = grid_config_for_scenario(
                    scenario, numerics.grid, interacting=is_interacting(scenario)
                )
                grid = UniformGrid.from_config(
                    config,
                    scenario.structure,
                    anchor=lattice_anchor(scenario, derive_grid=True),
                )
            offsets = lattice_offsets(scenario.structure, grid.origin, grid.spacing)
            assert max(offsets) < 1.0e-9


class TestCommandLine:
    """``--setups``, and the refusal to mix an explicit numerics file with a level."""

    def test_setups_prints_every_scenario_and_level_and_exits_zero(self, capsys) -> None:
        assert run.main(["--setups"]) == 0
        printed = capsys.readouterr().out
        for scenario_id in SETUPS:
            assert scenario_id in printed
        for level in LEVELS:
            assert level in printed

    def test_an_explicit_preset_refuses_a_level_and_an_override(self, capsys) -> None:
        assert run.main(["he_atom_lda", "--numerics", "model", "--res", "fine"]) == 2
        assert "--numerics" in capsys.readouterr().err
        assert run.main(["he_atom_lda", "--numerics", "model", "--spacing", "0.2"]) == 2
        assert "--numerics" in capsys.readouterr().err

    def test_an_unknown_scenario_is_a_command_mistake_not_a_failed_run(self, capsys) -> None:
        assert run.main(["no_such_scenario"]) == 2
        assert "unknown scenario" in capsys.readouterr().err


class TestPlaceholderGridAndComponentGates:
    """B3: ``--numerics all-electron`` fails G0.2 at the placeholder spacing by construction."""

    def test_g02_at_the_placeholder_spacing_is_a_fixture_limit_not_a_solver_error(self) -> None:
        from cdft.gates.tier0 import PoissonAnalyticGate

        assert ALL_ELECTRON_NUMERICS.grid.spacing == 1.0
        failed = PoissonAnalyticGate().evaluate(ALL_ELECTRON_NUMERICS)
        assert failed.verdict.value == "fail" and failed.detail["binding_quantity"] == "energy"
        assert abs(failed.measured - 5.77e-3) < 5e-5  # the B3 number: sigma = 4 in a 16-bohr box
        # The same solver at the same spacing, in a box that holds the 4-bohr Gaussian: round-off.
        held = PoissonAnalyticGate(box=64.0).evaluate(ALL_ELECTRON_NUMERICS)
        assert held.verdict.value == "pass" and held.measured < 1e-12
        # And the model preset, which auto runs the component gates at, passes on the 16-bohr box.
        assert PoissonAnalyticGate().evaluate(MODEL_SYSTEM_NUMERICS).verdict.value == "pass"

    def test_auto_runs_the_component_gates_at_the_model_preset(self, monkeypatch) -> None:
        seen: dict[str, object] = {}

        def fake_run_scenarios(scenario_ids, numerics, **kwargs):
            seen["numerics"] = numerics
            seen["plan"] = kwargs["numerics_for_scenario"]
            return 0, ""

        monkeypatch.setattr(run, "run_scenarios", fake_run_scenarios)
        assert run.main(["h_atom"]) == 0
        assert seen["numerics"] == MODEL_SYSTEM_NUMERICS
        assert seen["plan"]("h_atom")[0] == ALL_ELECTRON_NUMERICS
        assert run.main(["h_atom", "--numerics", "all-electron"]) == 0
        assert seen["numerics"] == ALL_ELECTRON_NUMERICS and seen["plan"] is None


@pytest.mark.fast
class TestHostMemoryExclusion:
    """A scenario whose measured CPU peak exceeds the host is skipped visibly, never silently (O-22)."""

    def test_host_memory_is_read_and_the_table_names_registered_scenarios(self) -> None:
        import test_suite

        limit = test_suite.host_memory_bytes()
        assert limit is None or limit > 2**28
        for scenario_id, needed in test_suite.HOST_MEMORY_NEEDED_BYTES.items():
            assert scenario_id in REGISTRY and needed > 2**30
        assert "h2plus_R8_lda" in test_suite.HOST_MEMORY_NEEDED_BYTES

    def test_card_memory_table_names_registered_scenarios_and_is_off_the_cpu(self) -> None:
        """The 6 GiB card's `h2plus_R8_lda` error row of 2026-09-18/19 becomes a visible skip (D-83)."""
        import test_suite

        for scenario_id, needed in test_suite.DEVICE_MEMORY_NEEDED_BYTES.items():
            assert scenario_id in REGISTRY and needed > 2**30
        assert test_suite.DEVICE_MEMORY_NEEDED_BYTES["h2plus_R8_lda"] > 6 * 2**30
        assert test_suite.device_memory_bytes(torch.device("cpu")) is None
        assert test_suite.HOST_MEMORY_REASON != test_suite.DEVICE_MEMORY_REASON
