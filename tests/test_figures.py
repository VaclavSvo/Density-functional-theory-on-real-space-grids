"""Tests for the publication-figure layer, ``figures.py`` at the repository root (D-72).

Covers scenario selection and ids, the drawing policy, the field evaluator and its cusp factor,
the quadratures and de-combed distributions, rendering (journal widths, provenance, byte-for-byte
repeats, partial redraws) and the suite runner ``scripts/figures_suite.py``.

The synthetic records are exact closed forms on small grids, so these tests need no solve; the
single end-to-end test (``slow``) solves the hydrogen atom through the command line.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib.util
import json
import math
import pathlib
import sys
import types

import numpy as np
import pytest

import cdft  # noqa: F401  (the package puts the repository root on sys.path)
import figures
import physics_config
from config import ALL_ELECTRON_NUMERICS, MODEL_SYSTEM_NUMERICS
from contract import DomainMode, GateKind, GateResult, GateVerdict, NumericsConfig, RunStatus

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

#: ``2 (3 pi^2)^(1/3)``, as in the module.
S_PREFACTOR = 2.0 * (3.0 * math.pi**2) ** (1.0 / 3.0)


# Synthetic records: exact densities on small grids


def _gate(gate_id: str, verdict: GateVerdict, measured: float = 0.0, reason: str = "") -> GateResult:
    return GateResult(gate_id=gate_id, name=f"test {gate_id}", verdict=verdict, measured=measured,
                      threshold=1.0e-8, kind=GateKind.EXACT, reason=reason)


def synthetic_record(
    scenario_id: str,
    density,
    *,
    spacing: float,
    half_width: float,
    eigenvalues=(-0.5,),
    occupations=(1.0,),
    status: str = "valid",
    gates=(),
    numerics: NumericsConfig = ALL_ELECTRON_NUMERICS,
) -> figures.FigureRecord:
    """A record of ``scenario_id`` whose density is the closed form ``density(x, y, z)``.

    Box centred on the origin with a lattice point there, as derived grids place nuclei (D-64);
    ``density=None`` leaves the record field-less.
    """
    scenario = physics_config.REGISTRY[scenario_id]
    count = int(round(2.0 * half_width / spacing)) + 1
    axis = -half_width + spacing * np.arange(count)
    field = None
    if density is not None:
        x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
        field = density(x, y, z)
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    occupations = np.asarray(occupations, dtype=np.float64)
    band = float(eigenvalues @ occupations)
    energies = {name: 0.0 for name in ("hartree", "xc", "nonlocal_ps", "ion_ion", "dispersion", "entropy")}
    energies.update(total=band, kinetic=-band, external=2.0 * band, harris_foulkes=float("nan"))
    return figures.FigureRecord(
        run_id=f"synthetic_{scenario_id}",
        scenario=scenario,
        numerics=numerics,
        status=status,
        error_message="",
        gates=list(gates),
        measurements={"n_electrons": float(occupations.sum())},
        provenance={"git_sha": None, "git_dirty": False, "config_hash": "synthetic",
                    "contract_version": figures.CONTRACT_VERSION, "device": "cpu"},
        shape=(count, count, count),
        spacing=spacing,
        origin=(-half_width, -half_width, -half_width),
        boundary="zero",
        density=field,
        v_hartree=None,
        v_xc=None,
        eigenvalues=eigenvalues,
        occupations=occupations,
        energies=energies,
        trajectory={},
        converged=True,
        n_iterations=1,
        source="synthetic",
    )


def hydrogen_1s(x, y, z):
    """``n = exp(-2 r) / pi``."""
    return np.exp(-2.0 * np.sqrt(x * x + y * y + z * z)) / math.pi


def harmonic_ground_state(x, y, z):
    """``n = pi^(-3/2) exp(-r^2)`` (omega = 1)."""
    return math.pi ** -1.5 * np.exp(-(x * x + y * y + z * z))


@pytest.fixture(scope="module")
def hydrogen_record() -> figures.FigureRecord:
    # 12 bohr keeps the zero halo (the solver's convention for rho = n / f^2) out of every stencil
    # the scaled-error metrics look at, as D-53 box sizing guarantees for real records.
    return synthetic_record("h_atom", hydrogen_1s, spacing=0.4, half_width=12.0)


@pytest.fixture(scope="module")
def harmonic_record() -> figures.FigureRecord:
    # h = 0.25: the degree-7 interpolant of the Gaussian's gradient is then good to ~1e-5, below the
    # quantile tolerance; at h = 0.4 the interpolation error alone reaches 2e-3 in s.
    return synthetic_record("harmonic_w1", harmonic_ground_state, spacing=0.25, half_width=6.0,
                            eigenvalues=(1.5,), numerics=MODEL_SYSTEM_NUMERICS)


def _verdict(record: figures.FigureRecord, allow_invalid: bool = False) -> figures.Verdict:
    return figures.classify_record(record.scenario.scenario_id, record.status, record.gates,
                                   has_fields=record.density is not None, allow_invalid=allow_invalid)


# Selection


@pytest.mark.fast
class TestSelection:
    """Scenario ids, overrides, and the preference for registered entries."""

    def test_every_registered_id_round_trips(self) -> None:
        for scenario in physics_config.REGISTRY:
            parsed = figures.parse_scenario_id(scenario.scenario_id)
            assert figures.compose_scenario_id(*parsed) == scenario.scenario_id

    @pytest.mark.parametrize(
        ("scenario_id", "expected"),
        [
            ("h2plus_R2_lda", ("h2plus", 2.0, None, "lda_vwn")),
            ("h_atom_N0.5_lda", ("h_atom", None, 0.5, "lda_vwn")),
            ("harmonic_w1", ("harmonic_w1", None, None, "none")),
            ("h2_R1.4_pbe", ("h2", 1.4, None, "pbe")),
            ("h2_R1.4_lda_pw92", ("h2", 1.4, None, "lda_pw92")),
        ],
    )
    def test_parse(self, scenario_id: str, expected: tuple) -> None:
        assert figures.parse_scenario_id(scenario_id) == expected

    def test_functional_aliases(self) -> None:
        assert figures.canonical_xc("LDA") == "lda_vwn"
        assert figures.canonical_xc("pz81") == "lda_pz81"
        assert figures.canonical_xc(None) is None
        with pytest.raises(ValueError, match="unknown functional"):
            figures.canonical_xc("b3lyp")

    def test_no_override_is_the_registry_entry(self) -> None:
        spec, notes = figures.select_scenario("he_atom_lda")
        assert spec is physics_config.REGISTRY["he_atom_lda"] and notes == []

    @pytest.mark.parametrize(
        ("scenario_id", "overrides", "registered"),
        [
            ("h2plus_R2_lda", {"bond_length": 8.0}, "h2plus_R8_lda"),
            ("h_atom_lda", {"n_electrons": 0.5}, "h_atom_N0.5_lda"),
            ("h_atom", {"xc": "lda"}, "h_atom_lda"),
            ("h2_R1.4_lda", {"xc": "pbe"}, "h2_R1.4_pbe"),
        ],
    )
    def test_registered_ids_win(self, scenario_id: str, overrides: dict, registered: str) -> None:
        spec, _ = figures.select_scenario(scenario_id, **overrides)
        assert spec is physics_config.REGISTRY[registered]

    def test_built_geometry_moves_the_nuclei_and_inherits_the_nearest_gate_list(self) -> None:
        spec, notes = figures.select_scenario("h2plus_R2_lda", bond_length=3.0)
        positions = np.asarray(spec.structure.positions)
        assert spec.scenario_id == "h2plus_R3_lda" and spec.xc.name == "lda_vwn"
        assert np.linalg.norm(positions[1] - positions[0]) == pytest.approx(3.0, abs=1e-12)
        assert spec.gates == physics_config.REGISTRY["h2plus_R2_lda"].gates
        stretched, _ = figures.select_scenario("h2plus_R2_lda", bond_length=7.0)
        assert stretched.gates == physics_config.REGISTRY["h2plus_R8_lda"].gates  # G4.5 included
        assert any("nearest registered relative" in note for note in notes)

    def test_other_functional_and_fractional_counts(self) -> None:
        spec, _ = figures.select_scenario("h2_R1.4", xc="pw92")
        assert (spec.scenario_id, spec.xc.name) == ("h2_R1.4_lda_pw92", "lda_pw92")
        fractional, _ = figures.select_scenario("he_atom_lda", n_electrons=1.5)
        assert fractional.electrons.n_electrons == 1.5
        assert fractional.gates == figures._FRACTIONAL_N_GATES

    def test_bare_potential_with_another_count_drops_its_references(self) -> None:
        spec, notes = figures.select_scenario("h2plus_R2", n_electrons=2.0)
        assert spec.scenario_id == "h2plus_R2_N2" and spec.xc.name == "none"
        assert spec.reference == {} and any("references dropped" in note for note in notes)

    @pytest.mark.parametrize(
        ("scenario_id", "overrides", "error"),
        [
            ("he_atom_lda", {"bond_length": 2.0}, ValueError),
            ("h2_R1.4_lda", {"bond_length": -1.0}, ValueError),
            ("h_atom_lda", {"n_electrons": 0.0}, ValueError),
            ("h_atom_lda", {"xc": "scan"}, ValueError),
            ("no_such_system", {}, KeyError),
        ],
    )
    def test_nonsense_is_refused(self, scenario_id: str, overrides: dict, error: type) -> None:
        with pytest.raises(error):
            figures.select_scenario(scenario_id, **overrides)

    def test_a_yaml_diatomic_moves_its_nuclei_and_drops_its_references(self, tmp_path: pathlib.Path) -> None:
        source = tmp_path / "mine.yaml"
        source.write_text(
            "scenarios:\n"
            "  - scenario_id: my_pair\n"
            "    symbols: [H, H]\n"
            "    positions: [[0.0, 0.0, -1.25], [0.0, 0.0, 1.25]]\n"
            "    charge: 1.0\n"
            "    xc: {name: none, rung: LDA, libxc_reference: []}\n"
            "    electrons: {n_electrons: 1.0}\n"
            "    gates: [G2.4, G0.3]\n"
            "    reference:\n"
            "      total_energy: {value: -0.5, kind: analytic, citation: test, tolerance: 1.0e-3}\n",
            encoding="utf-8",
        )
        registry = physics_config.load_scenarios(source)
        spec, notes = figures.select_scenario("my_pair", registry, bond_length=3.0)
        positions = np.asarray(spec.structure.positions)
        assert spec.scenario_id == "my_pair_R3" and spec.reference == {}
        assert np.allclose(positions, [[0.0, 0.0, -1.5], [0.0, 0.0, 1.5]])
        assert spec.gates == ("G2.4", "G0.3")
        assert any("references of the original geometry are dropped" in note for note in notes)

    def test_overrides_that_restate_the_scenario_are_not_overrides(self) -> None:
        for scenario_id, overrides in (
            ("h2plus_R2_lda", {"bond_length": 2.0}),
            ("he_atom_lda", {"n_electrons": 2.0}),
            ("h_atom_lda", {"xc": "lda"}),
            ("h2_R1.4_pbe", {"bond_length": 1.4, "xc": "PBE", "n_electrons": 2.0}),
        ):
            spec, notes = figures.select_scenario(scenario_id, **overrides)
            assert spec is physics_config.REGISTRY[scenario_id], scenario_id
            assert not any(note.startswith("built") for note in notes)

    def test_a_functional_needs_coulomb_nuclei(self) -> None:
        with pytest.raises(ValueError, match="needs Coulomb nuclei"):
            figures.select_scenario("harmonic_w1", xc="lda")
        with pytest.raises(ValueError, match="needs Coulomb nuclei"):
            figures.select_scenario("box_L10", xc="pbe")
        spec, _ = figures.select_scenario("harmonic_w1", n_electrons=2.0)
        assert spec.xc.name == "none" and spec.electrons.n_electrons == 2.0

    def test_ids_keep_twelve_digits(self) -> None:
        assert figures.compose_scenario_id("h2", 1.4000049, None, "lda_vwn") == "h2_R1.4000049_lda"
        spec, _ = figures.select_scenario("h2_R1.4_lda", bond_length=1.4000049)
        assert spec.scenario_id == "h2_R1.4000049_lda"  # built, not snapped to the registered R = 1.4
        assert spec is not physics_config.REGISTRY["h2_R1.4_lda"]

    def test_the_functional_is_the_scenarios_not_the_ids(self, tmp_path: pathlib.Path) -> None:
        source = tmp_path / "mine.yaml"
        source.write_text(
            "scenarios:\n"
            "  - scenario_id: my_he\n"
            "    symbols: [He]\n"
            "    positions: [[0.0, 0.0, 0.0]]\n"
            "    xc: {name: pbe, rung: GGA, libxc_reference: [gga_x_pbe, gga_c_pbe]}\n"
            "    electrons: {n_electrons: 2.0}\n"
            "    gates: [G2.4]\n",
            encoding="utf-8",
        )
        registry = physics_config.load_scenarios(source)
        spec, notes = figures.select_scenario("my_he", registry, n_electrons=1.5)
        assert (spec.scenario_id, spec.xc.name, spec.electrons.n_electrons) == ("my_he_N1.5_pbe", "pbe", 1.5)
        assert spec.gates == figures._FRACTIONAL_N_GATES
        assert not any("non-interacting" in note for note in notes)

    def test_scope_is_checked_before_the_solve(self) -> None:
        atom = physics_config.REGISTRY["h_atom_lda"]
        assert figures.figure_scope_problem(atom, ALL_ELECTRON_NUMERICS) is None
        polarised = dataclasses.replace(atom, electrons=dataclasses.replace(atom.electrons, spin_polarised=True))
        assert "spin-polarised" in figures.figure_scope_problem(polarised, ALL_ELECTRON_NUMERICS)
        masked = dataclasses.replace(
            MODEL_SYSTEM_NUMERICS,
            grid=dataclasses.replace(MODEL_SYSTEM_NUMERICS.grid, domain=DomainMode.MASKED_SPHERES),
        )
        well = physics_config.REGISTRY["harmonic_w1"]
        assert "masked domain" in figures.figure_scope_problem(well, masked)
        # A Coulomb scenario is put on its derived box whatever the preset says (D-42), as the solver does.
        all_electron_masked = dataclasses.replace(
            ALL_ELECTRON_NUMERICS,
            grid=dataclasses.replace(ALL_ELECTRON_NUMERICS.grid, domain=DomainMode.MASKED_SPHERES,
                                     box_lengths=None),
        )
        assert figures.figure_scope_problem(atom, all_electron_masked) is None

    def test_model_centres(self) -> None:
        well = synthetic_record("harmonic_w1", None, spacing=0.5, half_width=2.0, eigenvalues=(1.5,),
                                numerics=MODEL_SYSTEM_NUMERICS)
        shifted = dataclasses.replace(well, origin=(0.5, 1.0, 1.5))
        assert figures.model_centre(shifted).tolist() == [0.0, 0.0, 0.0]  # the well sits at the origin
        box = dataclasses.replace(synthetic_record("box_L10", None, spacing=0.5, half_width=2.0,
                                                   eigenvalues=(0.3,), numerics=MODEL_SYSTEM_NUMERICS),
                                  origin=(0.5, 1.0, 1.5))
        assert figures.model_centre(box).tolist() == [2.5, 3.0, 3.5]  # the box is the domain

    def test_axis_labels_name_the_plotted_coordinates(self) -> None:
        def labels(record):
            ctx = figures.FigureContext(record, _verdict(record), figures.FigureOptions())
            return ctx.axis_label("u"), ctx.axis_label("v")

        pair = synthetic_record("h2plus_R2", None, spacing=1.0, half_width=1.0)
        assert labels(pair) == ("position along the bond axis, $b$ (bohr)",
                                "perpendicular, $a$ (bohr)")  # b_bohr, a_bohr in the tables
        atom = synthetic_record("h_atom", None, spacing=1.0, half_width=1.0)
        assert labels(atom) == ("$z$ (bohr)", "$x$ (bohr)")
        structure = dataclasses.replace(atom.scenario.structure, positions=((-1.5, 0.0, 3.0),))
        moved = dataclasses.replace(atom, scenario=dataclasses.replace(atom.scenario, structure=structure))
        assert labels(moved) == ("$z - 3$ (bohr)", "$x + 1.5$ (bohr)")  # ticks are offsets from the nucleus
        assert figures._glossary(["b_bohr", "density"]).startswith("Columns: b = signed position")
        assert figures._glossary(["density"]) == ""

    def test_numerics_presets_and_profiles(self) -> None:
        model = figures.select_numerics(physics_config.REGISTRY["harmonic_w1"], "auto", figures.GATE_PROFILES["full"])
        atom = figures.select_numerics(physics_config.REGISTRY["h_atom"], "auto", figures.GATE_PROFILES["quick"])
        assert model.grid == MODEL_SYSTEM_NUMERICS.grid and model.eigen.cross_check
        assert atom.grid == ALL_ELECTRON_NUMERICS.grid and not atom.eigen.cross_check


# The drawing policy (D-72)


@pytest.mark.fast
class TestDrawingPolicy:
    """Which records are drawn, how they are marked, and what sets the exit code."""

    @pytest.mark.parametrize("status", [RunStatus.VALID.value, RunStatus.MARGINAL.value])
    def test_trusted_records_are_drawn_plain(self, status: str) -> None:
        verdict = figures.classify_record("h_atom", status, [_gate("G2.4", GateVerdict.PASS)], has_fields=True)
        assert verdict.drawable and verdict.mark == "" and verdict.badge() == "" and not verdict.blocking

    def test_deferred_is_drawn_only_on_request_and_marked(self) -> None:
        refused = figures.classify_record("h_atom", RunStatus.DEFERRED.value, [], has_fields=True)
        drawn = figures.classify_record("h_atom", RunStatus.DEFERRED.value, [], has_fields=True,
                                        allow_invalid=True)
        assert not refused.drawable and refused.mark == "deferred" and not refused.blocking
        assert drawn.drawable and "not for publication" in drawn.badge()

    # The registry's one row since D-75: G2.3 = 1 on h2_R1.4_lda (D-75 follow-up).
    def test_known_open_failure_is_drawn_only_on_request_and_marked(self) -> None:
        gates = [_gate("G2.3", GateVerdict.FAIL, measured=1.0)]
        refused = figures.classify_record("h2_R1.4_lda", RunStatus.INVALID.value, gates, has_fields=True)
        drawn = figures.classify_record("h2_R1.4_lda", RunStatus.INVALID.value, gates, has_fields=True,
                                        allow_invalid=True)
        assert not refused.drawable and refused.mark == "known-open" and refused.known_open == 1
        assert drawn.drawable and not drawn.blocking
        assert "known-open G2.3" in drawn.badge() and "not for publication" in drawn.badge()

    def test_a_regressed_known_open_row_is_a_new_failure(self) -> None:
        gates = [_gate("G2.3", GateVerdict.FAIL, measured=4.0)]  # far past recorded x drift
        verdict = figures.classify_record("h2_R1.4_lda", RunStatus.INVALID.value, gates, has_fields=True)
        assert not verdict.drawable and verdict.blocking

    def test_new_failure_needs_allow_invalid(self) -> None:
        gates = [_gate("G1.13", GateVerdict.FAIL, measured=2.29e-8)]  # not registered for this id
        refused = figures.classify_record("h2plus_R2_lda", RunStatus.INVALID.value, gates, has_fields=True)
        forced = figures.classify_record("h2plus_R2_lda", RunStatus.INVALID.value, gates, has_fields=True,
                                         allow_invalid=True)
        assert not refused.drawable and refused.mark == "invalid" and refused.blocking
        assert forced.drawable and forced.blocking
        assert forced.badge().startswith("INVALID record: G1.13 failed")

    def test_new_and_known_failures_are_told_apart(self) -> None:
        # G2.3 inside its registered known-open row; G1.13 no longer registered at all (D-75).
        gates = [_gate("G2.3", GateVerdict.FAIL, measured=1.0), _gate("G1.13", GateVerdict.FAIL, 2.03e-8)]
        verdict = figures.classify_record("h2_R1.4_lda", RunStatus.INVALID.value, gates, has_fields=True,
                                          allow_invalid=True)
        assert verdict.mark == "invalid" and verdict.blocking
        assert verdict.new_failures == 1 and verdict.known_gates == ("G2.3",)
        assert verdict.badge() == "INVALID record: G1.13 failed (known-open: G2.3) -- not for publication"
        assert "registry for 'h2_R1.4_lda': G1.13; known-open: G2.3" in verdict.reason

    @pytest.mark.parametrize("status", [RunStatus.UNCONVERGED.value, RunStatus.ERROR.value])
    def test_failed_runs_are_refused_unless_forced(self, status: str) -> None:
        verdict = figures.classify_record("he_atom_lda", status, [], has_fields=True)
        assert not verdict.drawable and verdict.blocking
        assert f"status {status}" in figures.classify_record(
            "he_atom_lda", status, [], has_fields=True, allow_invalid=True
        ).badge()

    def test_a_record_without_fields_is_never_drawn(self) -> None:
        verdict = figures.classify_record("he_atom_lda", RunStatus.ERROR.value, [], has_fields=False,
                                          allow_invalid=True, error_message="boom")
        assert not verdict.drawable and verdict.reason == "boom"

    def test_gate_summary_names_budget_exclusions(self) -> None:
        gates = [
            _gate("G2.4", GateVerdict.PASS),
            _gate("G3.1", GateVerdict.SKIPPED, reason="costs 3 extra solves and this profile allows 0; ..."),
            _gate("G1.2", GateVerdict.SKIPPED, reason="not applicable to scenario h_atom"),
        ]
        summary = figures.gate_summary(gates)
        assert summary["skipped_for_budget"] == ["G3.1"]
        assert summary["counts"]["pass"] == 1 and summary["counts"]["skipped"] == 2
        assert "--gates full" in figures.gate_summary_line(gates)


# Fields at points


@pytest.mark.fast
class TestEvaluator:
    """The degree-7 interpolant, the analytic cusp factor, and the face treatment."""

    def test_polynomials_are_reproduced_to_round_off(self) -> None:
        def cubic(x, y, z):
            return 1.0 + 0.3 * x - 0.2 * y * y + 0.05 * x * y * z**3 + 0.01 * x**7

        record = synthetic_record("harmonic_w1", cubic, spacing=0.5, half_width=4.0,
                                  eigenvalues=(1.5,), numerics=MODEL_SYSTEM_NUMERICS)
        evaluator = figures.FieldEvaluator(record)
        rng = np.random.default_rng(7)
        points = rng.uniform(-1.4, 1.4, size=(500, 3))  # stencils stay inside the box
        fields = evaluator.density(points, order=2)
        x, y, z = points.T
        grad = np.stack([0.3 + 0.05 * y * z**3 + 0.07 * x**6, -0.4 * y + 0.05 * x * z**3,
                         0.15 * x * y * z * z], axis=1)
        hess = np.empty((x.size, 3, 3))
        hess[:, 0, 0] = 0.42 * x**5
        hess[:, 1, 1] = -0.2 * 2.0
        hess[:, 2, 2] = 0.3 * x * y * z
        hess[:, 0, 1] = hess[:, 1, 0] = 0.05 * z**3
        hess[:, 0, 2] = hess[:, 2, 0] = 0.15 * y * z * z
        hess[:, 1, 2] = hess[:, 2, 1] = 0.15 * x * z * z
        lap = -0.4 + 0.3 * x * y * z + 0.42 * x**5
        assert np.max(np.abs(fields["n"] - cubic(x, y, z))) < 1e-11
        assert np.max(np.abs(fields["grad"] - grad)) < 1e-10
        assert np.max(np.abs(fields["hess"] - hess)) < 1e-8
        assert np.max(np.abs(fields["lap"] - lap)) < 1e-8

    def test_grid_values_are_the_record(self, hydrogen_record: figures.FigureRecord) -> None:
        evaluator = figures.FieldEvaluator(hydrogen_record)
        axis = evaluator.axes()[0]
        points = np.array([[axis[3], axis[20], axis[31]], [axis[40], axis[40], axis[40]]])
        values = evaluator.density(evaluator.nudge(points))["n"]
        expected = hydrogen_record.density[[3, 40], [20, 40], [31, 40]]
        assert np.allclose(values, expected, rtol=1e-8, atol=0.0)

    def test_one_sided_stencils_reach_the_faces(self) -> None:
        record = synthetic_record("harmonic_w1", None, spacing=0.3, half_width=3.0,
                                  eigenvalues=(1.5,), numerics=MODEL_SYSTEM_NUMERICS)
        record = dataclasses.replace(record, density=np.ones(record.shape))
        evaluator = figures.FieldEvaluator(record)
        axis = evaluator.axes()[0]
        x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")

        def potential(a, b, c):
            return 1.0 / np.sqrt(1.0 + (a - 4.0) ** 2 + b * b + 0.5 * c * c)

        evaluator.add_smooth_field("v", potential(x, y, z))
        rng = np.random.default_rng(3)
        near = rng.uniform(-1.8, 1.8, size=(400, 3))
        near[:, 0] = rng.uniform(2.7, 3.0, size=400)
        values, _ = evaluator.smooth("v", near)
        assert np.max(np.abs(values - potential(*near.T))) < 1e-4
        centred, _, _ = evaluator._interpolate(evaluator._smooth["v"], near, 0, one_sided=False)
        assert np.max(np.abs(centred - potential(*near.T))) > 1e-2  # what a zero halo would draw

    def test_hydrogen_is_exact_through_the_cusp(self, hydrogen_record: figures.FigureRecord) -> None:
        evaluator = figures.FieldEvaluator(hydrogen_record)
        reference = figures.density_reference(hydrogen_record)
        rng = np.random.default_rng(11)
        points = rng.normal(scale=1.5, size=(400, 3))
        points = points[np.linalg.norm(points, axis=1) < 7.0]
        points[0] = (1.0e-7, 0.0, 0.0)  # next to the nucleus
        fields = evaluator.density(points, order=2)
        n_ref, grad_ref, lap_ref = reference.evaluate(points)
        r = np.linalg.norm(points, axis=1)
        assert np.max(np.abs(fields["n"] / n_ref - 1.0)) < 1e-10
        assert np.max(np.abs(fields["grad"] - grad_ref) / (2.0 * n_ref[:, None])) < 1e-10
        # lap n = n (4 - 4/r) changes sign at r = 1: scale by the size of its two terms.
        assert np.max(np.abs(fields["lap"] - lap_ref) / (n_ref * (4.0 + 4.0 / r))) < 1e-9

    def test_factor_times_a_polynomial_with_its_hessian(self) -> None:
        """``n = f^2 rho`` with a non-constant ``rho``: every term of the product rule is exercised."""
        torch = pytest.importorskip("torch")

        def rho(x, y, z):
            return 1.0 + 0.2 * x - 0.1 * y * y + 0.03 * x * y * z + 0.01 * z**3

        def density(x, y, z):
            return np.exp(-2.0 * np.sqrt(x * x + y * y + z * z)) * rho(x, y, z)

        record = synthetic_record("h_atom", density, spacing=0.5, half_width=4.0)
        assert record.cusp_factorised
        evaluator = figures.FieldEvaluator(record)
        rng = np.random.default_rng(5)
        points = rng.uniform(-1.4, 1.4, size=(60, 3))
        points = points[np.linalg.norm(points, axis=1) > 0.05]
        fields = evaluator.density(points, order=2)

        def closed_form(p):
            x, y, z = p[0], p[1], p[2]
            return torch.exp(-2.0 * torch.sqrt(x * x + y * y + z * z)) * (
                1.0 + 0.2 * x - 0.1 * y * y + 0.03 * x * y * z + 0.01 * z**3
            )

        for k, point in enumerate(points):
            p = torch.tensor(point, dtype=torch.float64)
            value = float(closed_form(p))
            gradient = torch.autograd.functional.jacobian(closed_form, p).numpy()
            hessian = torch.autograd.functional.hessian(closed_form, p).numpy()
            scale = abs(value) + 1e-300
            assert abs(fields["n"][k] - value) / scale < 1e-11
            assert np.max(np.abs(fields["grad"][k] - gradient)) / scale < 1e-10
            assert np.max(np.abs(fields["hess"][k] - hessian)) / scale < 1e-8 * (1.0 + 1.0 / np.linalg.norm(point))
            assert fields["lap"][k] == pytest.approx(np.trace(hessian), rel=1e-8, abs=1e-8 * scale)

    def test_gga_potential_is_the_radial_formula(self) -> None:
        """``xc_at`` expands the GGA divergence with the Hessian; the radial formula needs neither."""
        torch = pytest.importorskip("torch")
        record = synthetic_record("he_atom_pbe", None, spacing=1.0, half_width=1.0, occupations=(2.0,))
        functional = record.functional()
        z = 2.0
        radii = np.array([0.05, 0.2, 0.5, 1.0, 2.0, 4.0, 6.0])
        r = torch.tensor(radii, dtype=torch.float64, requires_grad=True)
        with torch.enable_grad():
            n = z**3 / math.pi * torch.exp(-2.0 * z * r)
            slope = -2.0 * z * (z**3 / math.pi) * torch.exp(-2.0 * z * r)  # its own node: sigma must not feed n
            sigma = slope * slope
            energy = functional.energy(n[None, :], sigma[None, :])
            e_n, e_sigma = torch.autograd.grad(energy.sum(), (n, sigma), create_graph=True)
            flux = r * r * 2.0 * e_sigma * slope
            (d_flux,) = torch.autograd.grad(flux.sum(), r)
        radial = (e_n - d_flux / r**2).detach().numpy()
        direction = np.array([1.0, 2.0, 2.0]) / 3.0
        n_np, d1 = n.detach().numpy(), slope.detach().numpy()
        projector = np.outer(direction, direction)
        hessian = ((2.0 * z) ** 2 * n_np)[:, None, None] * projector + (d1 / radii)[:, None, None] * (
            np.eye(3) - projector
        )
        e_xc, v_xc = figures.xc_at(record, n_np, d1[:, None] * direction, hessian)
        assert np.allclose(e_xc, energy.detach().numpy().ravel(), rtol=1e-14, atol=0.0)
        assert np.max(np.abs(v_xc / radial - 1.0)) < 1e-12

    def test_the_cusp_flag_is_the_records(self, hydrogen_record: figures.FigureRecord) -> None:
        # The corpus does not store the scenario's flag; the solver's own measurement decides.
        assert hydrogen_record.cusp_factorised
        bare = dataclasses.replace(hydrogen_record, measurements={"n_electrons": 1.0, "cusp_factorisation": False})
        assert not bare.cusp_factorised
        assert figures.FieldEvaluator(bare).charges.size == 0  # no factor is invented for it

    def test_external_potentials_are_the_solvers(self) -> None:
        from cdft.grid import UniformGrid
        from cdft.operators.external import external_potential
        from contract import ExternalPotentialKind, ExternalPotentialSpec, GridConfig

        pair = physics_config.REGISTRY["h2plus_R2"]
        well = physics_config.REGISTRY["harmonic_w1"]
        soft = dataclasses.replace(pair, external=dataclasses.replace(
            pair.external, kind=ExternalPotentialKind.SOFT_COULOMB, softening=0.7, cusp_factorisation=False))
        gaussian = dataclasses.replace(well, external=ExternalPotentialSpec(
            kind=ExternalPotentialKind.GAUSSIAN_WELL, depth=2.0, width=1.5))
        config = GridConfig(spacing=0.5, domain=DomainMode.BOX, box_lengths=(4.0, 4.0, 4.0))
        template = synthetic_record("h2plus_R2", None, spacing=0.5, half_width=2.0)
        for scenario in (pair, soft, well, gaussian):
            structure = scenario.structure if scenario.structure.n_atoms else None
            grid = UniformGrid.from_config(config, structure)
            expected = external_potential(scenario.external, scenario.structure, grid).numpy()
            points = grid.points().numpy()
            record = dataclasses.replace(template, scenario=scenario)
            got = figures.external_at(record, points)
            far = np.ones(points.shape[0], dtype=bool)
            for centre in record.positions:  # the solver replaces the pole by its cell average there
                far &= np.linalg.norm(points - centre, axis=1) >= 0.5 * grid.spacing
            assert np.allclose(got[far], expected[far], rtol=1e-13, atol=1e-13), scenario.external.kind

    def test_two_spin_channels_are_refused(self) -> None:
        with pytest.raises(ValueError, match="spin channels"):
            figures._restricted_levels(np.zeros((2, 3)), np.ones((2, 3)))
        values, weights = figures._restricted_levels(np.array([[-0.5, 0.1]]), np.array([[2.0, 0.0]]))
        assert values.tolist() == [-0.5, 0.1] and weights.tolist() == [2.0, 0.0]


# Quadrature and distributions


@pytest.mark.fast
class TestQuadrature:
    """Closed-form integrals, the Becke fusion, and the de-combing model."""

    def test_angular_and_radial_rules(self) -> None:
        directions, weights = figures.angular_rule(12, 24)
        assert weights.sum() == pytest.approx(4.0 * math.pi, rel=1e-14)
        assert (weights * directions[:, 0] ** 2).sum() == pytest.approx(4.0 * math.pi / 3.0, rel=1e-13)
        radii, radial_weights = figures.radial_rule(80, 1.0)
        assert (radial_weights * np.exp(-2.0 * radii)).sum() == pytest.approx(0.25, rel=1e-10)

    def test_becke_cells_integrate_a_two_centre_density(self) -> None:
        scenario = physics_config.REGISTRY["h2plus_R2"]
        record = types.SimpleNamespace(
            positions=np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]]), charges=np.array([1.0, 1.0]),
            scenario=scenario, origin=(0.0, 0.0, 0.0), shape=(1, 1, 1), spacing=1.0,
        )
        grid = figures.IntegrationGrid.for_record(record, 80, 20, 40)
        a = np.linalg.norm(grid.points - record.positions[0], axis=1)
        b = np.linalg.norm(grid.points - record.positions[1], axis=1)
        pair = 0.5 * (np.exp(-2.0 * a) + np.exp(-2.0 * b)) / math.pi
        assert (grid.weights * pair).sum() == pytest.approx(1.0, abs=1e-8)
        assert grid.kind == "becke" and len(grid.blocks) == 2

    def test_half_cells_and_the_linear_model(self) -> None:
        grid = figures.IntegrationGrid(np.zeros((4, 3)), np.full(4, 0.25), "becke", ((0, (4, 1, 1)),), ((0, 8),))
        values = np.array([0.0, 1.0, 2.0, 3.0])
        valid = np.ones(4, dtype=bool)
        lower, upper = grid.half_cell_values(values, valid, 0)
        assert lower.tolist() == [0.0, 0.5, 1.5, 2.5] and upper.tolist() == [0.5, 1.5, 2.5, 3.0]
        # The model: the outer half-cells of the end nodes do not spread (no neighbour), so 1/8 of the
        # weight sits at 0 and at 3 each and 3/4 is uniform on [0, 3]: F(q) = 1/8 + q/4 inside.
        weights = np.full(4, 0.25)
        quantiles = figures.radial_model_quantiles(grid, values, weights, valid, (0.25, 0.5, 0.9))
        assert quantiles == pytest.approx([0.5, 1.5, 3.0], abs=1e-9)
        assert figures.radial_model_fraction_above(grid, values, weights, valid, 2.0) == pytest.approx(0.375)

    def test_tex_escape_is_single_pass(self) -> None:
        text = "h2_R1.4 & 50% $x$ #1 {y} ~ ^ \\ end"
        assert figures._tex_escape(text) == (
            r"h2\_R1.4 \& 50\% \$x\$ \#1 \{y\} \textasciitilde{} \^{} \textbackslash{} end"
        )

    def test_one_two_five(self) -> None:
        assert figures._one_two_five(0.15, 12.0) == [0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
        assert figures._one_two_five(1e-3, 1e3) == [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
        assert figures._one_two_five(-1.0, 1.0) == []

    def test_tables_refuse_ragged_or_empty_columns(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="unequal length"):
            figures._write_table(tmp_path / "t.csv", {"a": np.zeros(3), "b": np.zeros(4)}, "header")
        with pytest.raises(ValueError, match="no columns"):
            figures._write_table(tmp_path / "t.csv", {}, "header")


def _closed_form_quantiles(radius_at_quantile, s_of_r, rs_of_r, quantiles=(0.1, 0.5, 0.9)):
    rows = []
    for q in quantiles:
        r = radius_at_quantile(q)
        rows.append((s_of_r(r), rs_of_r(r)))
    return rows


def _distribution_quantiles(record: figures.FigureRecord, n_radial: int = 400):
    evaluator = figures.FieldEvaluator(record)
    grid = figures.IntegrationGrid.for_record(record, n_radial, 8, 16)
    fields = evaluator.density(evaluator.nudge(grid.points), order=1)
    n = fields["n"]
    keep = fields["inside"] & (n > float(np.max(record.density)) * 1e-12)
    weights = grid.weights * n
    weights = weights / weights[keep].sum()
    s = np.full(n.size, np.nan)
    rs = np.full(n.size, np.nan)
    s[keep] = np.linalg.norm(fields["grad"][keep], axis=1) / (S_PREFACTOR * n[keep] ** (4.0 / 3.0))
    rs[keep] = (3.0 / (4.0 * math.pi * n[keep])) ** (1.0 / 3.0)
    qs = figures.radial_model_quantiles(grid, s, weights, keep, (0.1, 0.5, 0.9))
    qr = figures.radial_model_quantiles(grid, rs, weights, keep, (0.1, 0.5, 0.9))
    return list(zip(qs, qr, strict=True)), grid, (s, rs, weights, keep)


@pytest.mark.physics
class TestDistributions:
    """g(s) and g(r_s) against closed forms [A32]: the comb is gone and the quantiles are right."""

    def test_hydrogen(self, hydrogen_record: figures.FigureRecord) -> None:
        from scipy.optimize import brentq

        def radius(q):  # charge inside r of a 1s density: 1 - exp(-2r)(1 + 2r + 2r^2)
            return brentq(lambda r: 1.0 - math.exp(-2 * r) * (1 + 2 * r + 2 * r * r) - q, 1e-9, 60.0)

        exact = _closed_form_quantiles(radius, lambda r: math.exp(2 * r / 3) / (3 * math.pi) ** (1 / 3),
                                       lambda r: (0.75 * math.exp(2 * r)) ** (1 / 3))
        got, grid, (s, rs, weights, keep) = _distribution_quantiles(hydrogen_record)
        assert np.allclose(np.array(got), np.array(exact), rtol=1e-3)
        shown = figures.smeared_histograms(grid, weights, keep, {"s": (s, np.linspace(0.0, 3.0, 61))})
        assert int(shown["substeps"][1]) == 1  # a spherical density needs no polar spreading
        mass = shown["s"] * 0.05
        assert mass.sum() == pytest.approx(1.0 - figures.radial_model_fraction_above(grid, s, weights, keep, 3.0),
                                           abs=2e-3)
        # No comb: neighbouring bins of the smooth part differ by far less than a node's weight would.
        smooth = shown["s"][14:50]
        assert np.max(np.abs(np.diff(smooth))) < 0.1

    def test_harmonic_well(self, harmonic_record: figures.FigureRecord) -> None:
        from scipy.stats import chi

        def radius(q):  # r sqrt(2) is chi-distributed with three degrees of freedom
            return float(chi(3).ppf(q)) / math.sqrt(2.0)

        k = (3.0 * math.pi**2) ** (1.0 / 3.0)
        exact = _closed_form_quantiles(radius, lambda r: r * math.exp(r * r / 3) * math.sqrt(math.pi) / k,
                                       lambda r: (0.75 * math.sqrt(math.pi) * math.exp(r * r)) ** (1 / 3))
        got, grid, _ = _distribution_quantiles(harmonic_record)
        assert grid.kind == "centred"
        assert np.allclose(np.array(got), np.array(exact), rtol=5e-4)


# References


def _bare(scenario_id: str, electrons: float) -> figures.FigureRecord:
    return synthetic_record(scenario_id, None, spacing=1.0, half_width=1.0, eigenvalues=(-0.5,),
                            occupations=(electrons,))


@pytest.mark.fast
class TestReferences:
    """The exact -I and total energies the figures print against."""

    @pytest.mark.parametrize(
        ("scenario_id", "electrons", "expected"),
        [
            ("h_atom_lda", 1.0, -0.5),
            ("he_plus_lda", 1.0, -2.0),
            ("h_atom_N0.5_lda", 0.5, -0.5),
            ("he_atom_lda", 2.0, -2.90372437703411959831 + 2.0),
            ("he_atom", 2.0, None),
            ("h2_R1.4_lda", 2.0, None),
        ],
    )
    def test_exact_ionisation_level(self, scenario_id: str, electrons: float, expected) -> None:
        level = figures.exact_ionisation_level(_bare(scenario_id, electrons))
        if expected is None:
            assert level is None
        else:
            assert level.energy == pytest.approx(expected, abs=1e-12)
            assert level.energy < 0.0  # -I, never +I

    @pytest.mark.parametrize(
        ("scenario_id", "electrons", "expected"),
        [
            ("h_atom", 1.0, -0.5),
            ("he_atom", 2.0, -4.0),
            ("h_atom_lda", 1.0, -0.5),
            ("h_atom_N0.5_lda", 0.5, -0.25),
            ("he_atom_lda", 2.0, -2.90372437703411959831),
        ],
    )
    def test_exact_total_energy(self, scenario_id: str, electrons: float, expected: float) -> None:
        value, source = figures.exact_total_energy(_bare(scenario_id, electrons))
        assert value == pytest.approx(expected, abs=1e-12) and source

    def test_fractional_helium_follows_the_straight_line(self) -> None:
        record = _bare("he_atom_lda", 1.5)
        value, source = figures.exact_total_energy(record)
        assert value == pytest.approx(-2.0 + 0.5 * (-2.90372437703411959831 + 2.0), abs=1e-12)
        assert "piecewise linearity" in source



@pytest.mark.physics
def test_radial_oracle_reference_is_regular_at_the_nucleus() -> None:
    """The oracle's density near r = 0 is its fitted Kato form, not its Dirichlet boundary layer.

    Marked ``physics``: it runs a radial SCF of LDA hydrogen (tens of seconds on two cores).
    """
    reference = figures._radial_ks_reference(1.0, 1.0, "lda_vwn", np.zeros(3))
    points = np.array([[0.0, 0.0, r] for r in (1e-9, 1e-6, 1e-4, 2.9e-3, 3.1e-3, 0.1)])
    n, grad, _ = reference.evaluate(points)
    ratio = grad[:, 2] / n
    assert np.all(np.abs(ratio + 2.0) < 1e-2)  # Kato for LDA, and no 1/r blow-up of the slope
    assert np.all(np.diff(n) < 0.0)
    assert reference.z_eff == pytest.approx(1.0, abs=1e-4)


# Rendering

SMALL = dict(map_points=81, profile_points=241, radial_points=120, grid_radial=40, grid_theta=10,
             grid_phi=20, dist_radial=120, dist_theta=6, dist_phi=12)

def _coulomb_1s_1s(distance: float) -> float:
    """``J(R)``, the Coulomb energy between two normalised 1s densities (Z = 1) at distance ``R``."""
    return 1.0 / distance - math.exp(-2.0 * distance) * (
        1.0 / distance + 11.0 / 8.0 + 0.75 * distance + distance * distance / 6.0
    )


def _pair_hartree(record: figures.FigureRecord, points: np.ndarray) -> np.ndarray:
    """The exact v_H of the synthetic density: the split coefficients times the 1s potentials."""
    total = np.zeros(points.shape[0])
    for c, centre in zip(record.measurements["hartree_split_coefficients"], record.positions, strict=True):
        total += c * figures._hydrogenic_potential(1.0, np.linalg.norm(points - centre, axis=1))
    return total


def _self_consistent_record(pair: bool):
    """A converged-looking LDA record with an exact density, ``v_H`` and SCF history.

    Returns ``(record, E_H, density)``. One proton: the 1s density; two protons 2 bohr apart: the
    mean of their 1s densities, whose cusp is weaker than Kato's by ``1 / (1 + e^{-2R})``. The D-54
    split coefficients are the exact cusp weights, so the remainder the figure interpolates is zero.
    """
    scenario_id = "h2plus_R2_lda" if pair else "h_atom_lda"
    centres = np.asarray(physics_config.REGISTRY[scenario_id].structure.positions, dtype=np.float64)
    weight = 1.0 / centres.shape[0]

    def density(x, y, z):
        return sum(weight * hydrogen_1s(x - c[0], y - c[1], z - c[2]) for c in centres)

    spacing, half_width = (0.5, 10.0) if pair else (0.4, 12.0)
    record = synthetic_record(scenario_id, density, spacing=spacing, half_width=half_width,
                              eigenvalues=(-0.3, 0.12, 0.12), occupations=(1.0, 0.0, 0.0))
    record.measurements.update({
        "hartree_split_coefficients": [weight / math.pi] * centres.shape[0],
        "harris_foulkes_history": [-0.42, -0.4449, -0.44491, -0.444912],
        "scf_stop_reason": "energy and density tolerances met",
        "mixer": "Pulay",
    })
    axes = [record.origin[d] + spacing * np.arange(record.shape[d]) for d in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    v_hartree = _pair_hartree(record, grid).reshape(record.shape)
    if pair:
        separation = float(np.linalg.norm(centres[1] - centres[0]))
        hartree = 0.125 * (2.0 * 0.625 + 2.0 * _coulomb_1s_1s(separation))
    else:
        hartree = 5.0 / 16.0
    energies = dict(record.energies, hartree=hartree, xc=-hartree + 0.03, total=-0.444912)
    trajectory = {"energies": [-0.40, -0.444, -0.44491, -0.444912], "residual_norms": [1e-1, 1e-3, 1e-6, 1e-9],
                  "density_changes": [5e-2, 5e-4, 5e-7, 5e-10], "fallback_events": ["iteration 3: damped restart"]}
    record = dataclasses.replace(record, v_hartree=v_hartree, energies=energies, trajectory=trajectory,
                                 n_iterations=4)
    return record, hartree, density


def _sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.physics
class TestRendering:
    """Every applicable figure is drawn, sized, labelled with its provenance and reproducible."""

    def test_house_style_is_accepted_by_matplotlib(self) -> None:
        pytest.importorskip("matplotlib")
        plt = figures.pyplot()
        with plt.rc_context(figures.HouseStyle().rc()):
            assert plt.rcParams["figure.constrained_layout.use"]
            assert plt.rcParams["pdf.fonttype"] == 42

    def test_every_mathtext_label_parses(self) -> None:
        matplotlib = pytest.importorskip("matplotlib")
        from matplotlib.font_manager import FontProperties
        from matplotlib.mathtext import MathTextParser

        parser = MathTextParser("agg")
        tree = ast.parse((REPO_ROOT / "figures.py").read_text(encoding="utf-8"))
        # f-strings are parsed with every replacement field set to "1"; their parts are not labels.
        joined_parts = {id(part) for node in ast.walk(tree) if isinstance(node, ast.JoinedStr)
                        for part in node.values}
        labels = [node.value for node in ast.walk(tree)
                  if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in joined_parts
                  and node.value.count("$") >= 2 and "\n" not in node.value]
        for node in ast.walk(tree):
            if isinstance(node, ast.JoinedStr):
                text = "".join(part.value if isinstance(part, ast.Constant) else "1" for part in node.values)
                if text.count("$") >= 2 and text.count("$") % 2 == 0 and "\n" not in text:
                    labels.append(text)
        assert labels and any(label.startswith("$-10^{") for label in labels)
        for label in labels:
            parser.parse(label, dpi=72, prop=FontProperties())
        assert matplotlib is not None

    def test_hydrogen_figures(self, hydrogen_record: figures.FigureRecord, tmp_path: pathlib.Path) -> None:
        pytest.importorskip("matplotlib")
        from PIL import Image

        options = figures.FigureOptions(**SMALL)
        manifest = figures.render_figures(hydrogen_record, _verdict(hydrogen_record), tmp_path, options,
                                          log=lambda _: None, invocation={"mode": "test"})
        drawn = [entry["name"] for entry in manifest["figures"]]
        assert manifest["errors"] == []
        assert drawn == ["density_map", "density_profile", "radial_density", "cusp_and_decay",
                         "density_error", "reduced_gradient", "spectrum"]
        assert {e["name"] for e in manifest["not_drawn"]} == {"potentials", "self_interaction", "scf_convergence"}
        assert manifest["invocation"] == {"mode": "test"}
        style = options.style
        for entry in manifest["figures"]:
            with Image.open(tmp_path / entry["files"]["png"]) as image:
                width = image.size[0] / style.dpi
                assert image.info["cdft-run-id"] == hydrogen_record.run_id
            assert width == pytest.approx(style.single_width, abs=0.01) or \
                width == pytest.approx(style.double_width, abs=0.01)
            caption = (tmp_path / entry["files"]["caption"]).read_text(encoding="utf-8")
            assert "Sources:" in caption and "Provenance: run synthetic_h_atom" in caption
            assert "Gates:" in caption
            assert (tmp_path / entry["files"]["pdf"]).read_bytes()[:5] == b"%PDF-"
        metrics = manifest["diagnostics"]
        assert metrics["radial_density"]["radial_peak_bohr"] == pytest.approx(1.0, abs=1e-6)
        cusp = metrics["cusp_and_decay"]
        assert cusp["kato_ratio_interpolant_limit_atom0"] == pytest.approx(1.0, abs=1e-8)
        assert cusp["kato_ratio_at_one_spacing_atom0"] == pytest.approx(1.0, abs=1e-8)
        profile = metrics["density_profile"]
        assert profile["kato_jump_factor_nucleus0"] == 4.0
        # rho = n / f^2 is the constant 1/pi here, so the interpolant adds no kink at the node.
        assert abs(profile["interpolant_slope_kink_nucleus0"]) < 1e-9
        assert metrics["density_error"]["max_abs_error_density_scaled"] < 1e-9
        assert metrics["density_error"]["l1_density_error_electrons"] < 1e-6
        assert metrics["spectrum"]["eps_homo"] == -0.5
        table = (tmp_path / "cusp_and_decay_kato_atom0.csv").read_text(encoding="utf-8").splitlines()
        assert table[0].startswith("# cusp_and_decay (kato_atom0)") and "r_bohr" in table[1]
        assert "Columns: r = distance" in table[0]
        profile_header = (tmp_path / "density_profile.csv").read_text(encoding="utf-8").splitlines()[0]
        assert "Columns: b = signed position along the plotted axis" in profile_header
        with np.load(tmp_path / "density_map.npz") as arrays:
            assert "a = in-plane perpendicular" in str(arrays["glossary"])
        assert (tmp_path / "energies.tex").read_text(encoding="utf-8").count(r"\toprule") == 1

    def test_model_systems(self, harmonic_record: figures.FigureRecord, tmp_path: pathlib.Path) -> None:
        """The harmonic well (centred grid) and the cubic well (the mesh, odd reflection at the walls)."""
        pytest.importorskip("matplotlib")
        options = figures.FigureOptions(**SMALL)
        well = figures.render_figures(harmonic_record, _verdict(harmonic_record), tmp_path / "well", options,
                                      log=lambda _: None)
        assert well["errors"] == []
        assert [entry["name"] for entry in well["figures"]] == [
            "density_map", "density_profile", "radial_density", "density_error", "reduced_gradient", "spectrum"]
        metrics = well["diagnostics"]
        assert metrics["radial_density"]["radial_peak_bohr"] == pytest.approx(1.0, abs=1e-5)
        assert metrics["density_error"]["max_abs_error_density_scaled"] < 5e-5  # degree 7 at h = 0.25: 1.8e-5
        assert "harmonic well" in (tmp_path / "well" / "radial_density.caption.txt").read_text(encoding="utf-8")

        length, spacing = 10.0, 0.5
        k = math.pi / length

        def ground_state(x, y, z):
            return (2.0 / length) ** 3 * (np.cos(k * x) * np.cos(k * y) * np.cos(k * z)) ** 2

        box = synthetic_record("box_L10", ground_state, spacing=spacing, half_width=0.5 * length - spacing,
                               eigenvalues=(3.0 * k * k / 2.0,), numerics=MODEL_SYSTEM_NUMERICS)
        box = dataclasses.replace(box, boundary="odd_reflection")
        drawn = figures.render_figures(box, _verdict(box), tmp_path / "box", options, log=lambda _: None)
        assert drawn["errors"] == []
        assert [entry["name"] for entry in drawn["figures"]] == [
            "density_map", "density_profile", "density_error", "reduced_gradient", "spectrum"]
        errors = drawn["diagnostics"]["density_error"]
        # The even halo continues n = psi^2 through the walls, so the interpolant stays exact there.
        assert errors["max_abs_error_density_scaled"] < 1e-6
        assert errors["figure_quadrature_charge"] == pytest.approx(1.0, abs=1e-10)
        assert "the recorded mesh" in (tmp_path / "box" / "density_error.caption.txt").read_text(encoding="utf-8")

    def test_same_record_same_bytes_and_partial_redraw(self, hydrogen_record: figures.FigureRecord,
                                                       tmp_path: pathlib.Path) -> None:
        pytest.importorskip("matplotlib")
        options = figures.FigureOptions(**SMALL)
        verdict = _verdict(hydrogen_record)
        first, second = tmp_path / "a", tmp_path / "b"
        for out in (first, second):
            figures.render_figures(hydrogen_record, verdict, out, options, only=["density_map", "spectrum"],
                                   log=lambda _: None)
        for name in ("density_map.pdf", "density_map.png", "spectrum.pdf", "spectrum.csv"):
            assert _sha(first / name) == _sha(second / name), name
        manifest = figures.render_figures(hydrogen_record, verdict, first, options, only=["spectrum"],
                                          log=lambda _: None)
        assert [e["name"] for e in manifest["figures"]] == ["density_map", "spectrum"]
        assert manifest["drawn_this_call"] == ["spectrum"]
        stored = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
        assert [e["name"] for e in stored["figures"]] == ["density_map", "spectrum"]

    def test_the_profile_legend_lists_this_work_first(self, hydrogen_record: figures.FigureRecord) -> None:
        pytest.importorskip("matplotlib")
        plt = figures.pyplot()
        options = figures.FigureOptions(**SMALL)
        spec = next(spec for spec in figures.FIGURES if spec.name == "density_profile")
        with plt.rc_context(options.style.rc()):
            rendered = spec.render(figures.FigureContext(hydrogen_record, _verdict(hydrogen_record), options))
        try:
            texts = [text.get_text() for text in rendered.figure.legends[0].get_texts()]
        finally:
            plt.close(rendered.figure)
        assert texts[0] == "this work" and "exact" in texts  # the reference is drawn first, listed second

    def test_marked_records_carry_the_badge(self, hydrogen_record: figures.FigureRecord) -> None:
        pytest.importorskip("matplotlib")
        plt = figures.pyplot()
        record = dataclasses.replace(hydrogen_record, status=RunStatus.DEFERRED.value)
        verdict = _verdict(record, allow_invalid=True)
        options = figures.FigureOptions(**SMALL)
        spec = next(spec for spec in figures.FIGURES if spec.name == "spectrum")
        with plt.rc_context(options.style.rc()):
            rendered = spec.render(figures.FigureContext(record, verdict, options))
        try:
            badge = rendered.figure._suptitle.get_text()
        finally:
            plt.close(rendered.figure)
        assert verdict.mark == "deferred" and verdict.drawable
        assert "DEFERRED" in badge and "not for publication" in badge
        assert not _verdict(record).drawable  # and without the request it is not drawn at all

    def test_a_refused_record_gets_a_manifest_and_no_figures(self, hydrogen_record: figures.FigureRecord,
                                                             tmp_path: pathlib.Path) -> None:
        record = dataclasses.replace(hydrogen_record, status=RunStatus.UNCONVERGED.value)
        manifest = figures.render_figures(record, _verdict(record), tmp_path, figures.FigureOptions(**SMALL),
                                          log=lambda _: None)
        assert manifest["figures"] == [] and len(manifest["not_drawn"]) == len(figures.FIGURES)
        assert sorted(p.name for p in tmp_path.iterdir()) == ["manifest.json"]

    def test_a_refusal_keeps_the_description_of_earlier_figures(self, hydrogen_record: figures.FigureRecord,
                                                                 tmp_path: pathlib.Path) -> None:
        pytest.importorskip("matplotlib")
        record = dataclasses.replace(hydrogen_record, status=RunStatus.DEFERRED.value)
        options = figures.FigureOptions(**SMALL)
        figures.render_figures(record, _verdict(record, allow_invalid=True), tmp_path, options,
                               only=["spectrum"], log=lambda _: None)
        refused = figures.render_figures(record, _verdict(record), tmp_path, options, only=["spectrum"],
                                         log=lambda _: None)
        assert refused["drawn_this_call"] == []
        assert [entry["name"] for entry in refused["figures"]] == ["spectrum"]  # still on disk, still described
        assert [entry["name"] for entry in refused["not_drawn"]] == ["spectrum"]
        assert refused["not_drawn"][0]["reason"].startswith("record not drawable")
        assert "previously_drawn_note" in refused
        assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))["figures"] == refused["figures"]
        assert (tmp_path / "spectrum.pdf").is_file()

    @pytest.mark.parametrize("pair", [False, True], ids=["h_atom_lda", "h2plus_R2_lda"])
    def test_self_consistent_records(self, pair: bool, tmp_path: pathlib.Path) -> None:
        pytest.importorskip("matplotlib")
        record, exact_hartree, density = _self_consistent_record(pair)
        options = figures.FigureOptions(use_oracle=False, **SMALL)
        manifest = figures.render_figures(record, _verdict(record), tmp_path, options, log=lambda _: None)
        assert manifest["errors"] == []
        drawn = [entry["name"] for entry in manifest["figures"]]
        for name in ("density_map", "density_profile", "cusp_and_decay", "reduced_gradient", "potentials",
                     "self_interaction", "spectrum", "scf_convergence"):
            assert name in drawn, name
        assert ("radial_density" in drawn) is not pair
        metrics = manifest["diagnostics"]
        figure_hartree = metrics["self_interaction"]["EH_figure_quadrature"]
        if pair:
            # The quadrature itself, on the closed forms: J(R) is reproduced.
            grid = figures.IntegrationGrid.for_record(record, SMALL["grid_radial"], SMALL["grid_theta"],
                                                      SMALL["grid_phi"])
            closed = 0.5 * np.sum(grid.weights * density(*grid.points.T) * _pair_hartree(record, grid.points))
            assert closed == pytest.approx(exact_hartree, abs=1e-7)
            # ~3e-4 Ha through the evaluator: the synthetic density's Kato defect kinks rho = n / f^2
            # at each nucleus, and no interpolant reproduces that (a solved density has no kink).
            assert figure_hartree == pytest.approx(exact_hartree, rel=2e-3)
        else:
            assert figure_hartree == pytest.approx(exact_hartree, abs=2e-6)
        assert metrics["scf_convergence"]["scf_iterations"] == 4
        profile = metrics["density_profile"]
        nuclei = record.positions.shape[0]
        assert [profile[f"kato_jump_factor_nucleus{a}"] for a in range(nuclei)] == [4.0] * nuclei
        # One Kato curve per distinct charge: the second proton of the pair is not drawn again.
        assert "kato_ratio_interpolant_limit_atom0" in metrics["cusp_and_decay"]
        assert "kato_ratio_interpolant_limit_atom1" not in metrics["cusp_and_decay"]
        # v_H along the axis is the exact potential: the D-54 split leaves nothing to interpolate.
        table = tmp_path / "potentials.csv"
        names = table.read_text(encoding="utf-8").splitlines()[1].lstrip("# ").split(",")
        data = np.loadtxt(table, delimiter=",", comments="#")
        column = {name: data[:, k] for k, name in enumerate(names)}
        points = column["b_bohr"][:, None] * np.array([0.0, 0.0, 1.0])
        inside = np.isfinite(column["v_hartree"])
        exact = _pair_hartree(record, points)
        assert np.max(np.abs(column["v_hartree"][inside] - exact[inside])) < 1e-9
        caption = (tmp_path / "potentials.caption.txt").read_text(encoding="utf-8")
        assert "D-54 split" in caption
        assert "Fallbacks: iteration 3" in (tmp_path / "scf_convergence.caption.txt").read_text(encoding="utf-8")

    def test_unknown_figure_names_are_a_usage_error(self, hydrogen_record: figures.FigureRecord,
                                                    tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="unknown figure"):
            figures.render_figures(hydrogen_record, _verdict(hydrogen_record), tmp_path, only=["nope"],
                                   log=lambda _: None)


# Scope and the command line


@pytest.mark.fast
class TestScope:
    """The figure layer sits outside the solver and outside machine learning."""

    def test_g57_still_clean_with_figures_at_the_root(self) -> None:
        from cdft.gates.tier5 import ScopeBoundaryGate

        gate = ScopeBoundaryGate()
        assert (REPO_ROOT / "figures.py") in gate.extra_files
        result = gate.evaluate(NumericsConfig())
        assert result.measured == 0.0, result.detail["violations"]

    def test_nothing_in_the_package_imports_figures(self) -> None:
        offenders = []
        for path in (REPO_ROOT / "src" / "cdft").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                if any(name == "figures" or name.startswith("figures.") for name in names):
                    offenders.append(str(path))
        assert offenders == []

    def test_listing_and_usage_errors(self, capsys: pytest.CaptureFixture[str], tmp_path: pathlib.Path) -> None:
        assert figures.main(["--list"]) == 0
        listing = capsys.readouterr().out
        assert "he_atom_lda" in listing and "reduced_gradient" in listing
        assert figures.main([]) == 2
        assert figures.main(["no_such_system"]) == 2
        assert figures.main(["he_atom_lda", "--bond-length", "2"]) == 2
        assert figures.main(["--from-corpus", "x.h5", "he_atom_lda"]) == 2

    def test_usage_errors_are_found_before_the_solve(self, capsys: pytest.CaptureFixture[str],
                                                     tmp_path: pathlib.Path) -> None:
        # Each of these names a valid scenario, so a late check would solve first.
        out = tmp_path / "out"
        assert figures.main(["h_atom", "--only", "nope"]) == 2
        assert figures.main(["h_atom", "--only", "spectrum", "--skip", "spectrum"]) == 2
        assert figures.main(["h_atom", "--angstrom"]) == 2
        assert figures.main(["h_atom", "--run-id", "x"]) == 2
        assert figures.main(["h_atom", "--out", str(out), "--corpus", str(out / "sub" / ".." / "record.h5")]) == 2
        not_hdf5 = tmp_path / "notes.h5"
        not_hdf5.write_text("not a corpus", encoding="utf-8")
        assert figures.main(["h_atom", "--corpus", str(not_hdf5)]) == 2
        polarised = tmp_path / "polarised.yaml"
        polarised.write_text(
            "scenarios:\n"
            "  - scenario_id: h_polarised\n"
            "    symbols: [H]\n"
            "    positions: [[0.0, 0.0, 0.0]]\n"
            "    xc: {name: lda_vwn, rung: LDA, libxc_reference: []}\n"
            "    electrons: {n_electrons: 1.0, magnetisation: 1.0, spin_polarised: true}\n",
            encoding="utf-8",
        )
        assert figures.main(["--scenario-file", str(polarised), "h_polarised"]) == 2
        errors = capsys.readouterr().err
        assert "spin-polarised" in errors and "appended twice" in errors and "not an HDF5 file" in errors
        assert not out.exists()
        # The stored-record path: a missing file leaves no directory behind, solve options are refused.
        missing = tmp_path / "absent" / "corpus.h5"
        assert figures.main(["--from-corpus", str(missing)]) == 2
        assert not missing.parent.exists()
        assert figures.main(["--from-corpus", str(missing), "--gates", "full"]) == 2
        assert figures.main(["--from-corpus", str(not_hdf5)]) == 2


# The suite runner (scripts/figures_suite.py)


@pytest.fixture(scope="module")
def suite():
    spec = importlib.util.spec_from_file_location("figures_suite", REPO_ROOT / "scripts" / "figures_suite.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module while the file executes
    spec.loader.exec_module(module)
    return module


def _nothing_may_run(*args, **kwargs):
    raise AssertionError("the suite launched a run it should have refused")


@pytest.mark.fast
class TestSuiteRunner:
    """The plan, the checks before the first run, and the summary."""

    def test_default_plan_is_the_registry_without_the_card_breaker(self, suite, tmp_path: pathlib.Path) -> None:
        args = suite.build_parser().parse_args(["--gates", "full", "--allow-invalid"])
        ids, problems = suite.select_scenarios(physics_config.REGISTRY, args.scenarios, args.exclude)
        assert problems == []
        assert ids == [s.scenario_id for s in physics_config.REGISTRY if s.scenario_id != "h2plus_R8_lda"]
        plan = suite.plan_runs(args, ids, tmp_path)
        assert plan[0].argv == (ids[0], "--device", "cuda", "--gates", "full", "--allow-invalid",
                                "--out", str(tmp_path / ids[0]))
        assert plan[0].command("python")[1].endswith("figures.py")
        assert suite.validate_plan(plan, physics_config.REGISTRY) == []

    def test_named_scenarios_and_the_options_passed_through(self, suite, tmp_path: pathlib.Path) -> None:
        args = suite.build_parser().parse_args([
            "--device", "cpu", "--scenarios", "h2plus_R8_lda", "h_atom", "--exclude", "h_atom",
            "--only", "spectrum", "--formats", "pdf", "svg", "--font", "serif", "--dpi", "300", "--no-oracle",
        ])
        ids, problems = suite.select_scenarios(physics_config.REGISTRY, args.scenarios, args.exclude)
        assert ids == ["h2plus_R8_lda"] and problems == []  # named, so the default exclusion does not apply
        (run,) = suite.plan_runs(args, ids, tmp_path)
        assert run.argv == ("h2plus_R8_lda", "--device", "cpu", "--gates", "quick", "--only", "spectrum",
                            "--formats", "pdf", "svg", "--font", "serif", "--dpi", "300", "--no-oracle",
                            "--out", str(tmp_path / "h2plus_R8_lda"))

    def test_every_problem_is_found_before_the_first_run(self, suite, monkeypatch: pytest.MonkeyPatch,
                                                         capsys: pytest.CaptureFixture[str]) -> None:
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        assert suite.main(["--scenarios", "nope", "h_atom", "--only", "spectra"], runner=_nothing_may_run) == 2
        errors = capsys.readouterr().err
        assert "unknown scenario 'nope'" in errors and "unknown figure(s) spectra" in errors
        assert "no CUDA device" in errors  # a request for the card is never answered by the CPU
        assert suite.main(["--device", "cpu", "--timeout-min", "0"], runner=_nothing_may_run) == 2
        assert suite.main(["--device", "cpu", "--scenarios", "h_atom", "--exclude", "h_atom"],
                          runner=_nothing_may_run) == 2

    def test_a_dry_run_only_prints_the_plan(self, suite, monkeypatch: pytest.MonkeyPatch,
                                            capsys: pytest.CaptureFixture[str], tmp_path: pathlib.Path) -> None:
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        root = tmp_path / "suite"
        assert suite.main(["--dry-run", "--out", str(root)], runner=_nothing_may_run) == 0
        out = capsys.readouterr().out
        commands = [line for line in out.splitlines() if "--out" in line]
        assert len(commands) == len(physics_config.REGISTRY) - 1
        assert "left out: h2plus_R8_lda" in out and "no CUDA device" in out  # a note, not an error, when dry
        assert not root.exists()

    def test_runs_are_summarised_from_their_manifests(self, suite, capsys: pytest.CaptureFixture[str],
                                                      tmp_path: pathlib.Path) -> None:
        def fake(run, python, log_path, timeout_s):
            trusted = run.scenario_id == "h_atom"
            assert log_path.parent == run.outdir and run.outdir.is_dir()
            manifest = {
                "run_id": f"run_{run.scenario_id}", "status": "valid" if trusted else "invalid",
                "verdict": {"drawable": True, "mark": "" if trusted else "invalid",
                            "reason": "" if trusted else "G2.3 failed"},
                "figures": [{"name": "spectrum"}], "not_drawn": [{"name": "potentials"}], "errors": [],
            }
            (run.outdir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            return 0 if trusted else 1

        root = tmp_path / "suite"
        code = suite.main(["--device", "cpu", "--scenarios", "h_atom", "h2_R1.4_lda", "--out", str(root)],
                          runner=fake)
        assert code == 1
        summary = json.loads((root / "suite_summary.json").read_text(encoding="utf-8"))
        rows = {row["scenario_id"]: row for row in summary["runs"]}
        assert rows["h_atom"]["exit_code"] == 0 and rows["h_atom"]["status"] == "valid"
        assert rows["h2_R1.4_lda"]["mark"] == "invalid" and rows["h2_R1.4_lda"]["drawn"] == ["spectrum"]
        assert summary["figures_suite"]["device"] == "cpu" and summary["figures_suite"]["interrupted"] is False
        table = (root / "suite_summary.txt").read_text(encoding="utf-8")
        assert "drawn, exit 1" in table and "G2.3 failed" in table
        assert "drawn, exit 1" in capsys.readouterr().out

    def test_an_interrupt_keeps_the_summary(self, suite, tmp_path: pathlib.Path) -> None:
        def interrupt(run, python, log_path, timeout_s):
            raise KeyboardInterrupt

        root = tmp_path / "suite"
        code = suite.main(["--device", "cpu", "--scenarios", "h_atom", "he_plus", "--out", str(root)],
                          runner=interrupt)
        summary = json.loads((root / "suite_summary.json").read_text(encoding="utf-8"))
        assert code == suite.INTERRUPTED and summary["figures_suite"]["interrupted"] is True
        assert [(row["scenario_id"], row["exit_code"]) for row in summary["runs"]] == [("h_atom", 130)]


@pytest.mark.physics
class TestSuiteLauncher:
    """The launcher on a real figures.py process: streamed, logged, stopped on time."""

    def test_output_is_streamed_and_logged(self, suite, capsys: pytest.CaptureFixture[str],
                                           tmp_path: pathlib.Path) -> None:
        run = suite.PlannedRun("listing", tmp_path, ("--list",))
        assert suite.launch(run, sys.executable, tmp_path / "figures.log") == 0
        log = (tmp_path / "figures.log").read_text(encoding="utf-8")
        assert log.startswith("$ ") and "reduced_gradient" in log
        assert "reduced_gradient" in capsys.readouterr().out

    def test_a_run_past_its_time_is_stopped(self, suite, tmp_path: pathlib.Path) -> None:
        run = suite.PlannedRun("h_atom", tmp_path / "h_atom", ("h_atom", "--device", "cpu", "--no-record",
                                                               "--out", str(tmp_path / "h_atom")))
        assert suite.launch(run, sys.executable, tmp_path / "figures.log", timeout_s=0.5) == 1
        assert "stopped by the suite" in (tmp_path / "figures.log").read_text(encoding="utf-8")


@pytest.mark.slow
def test_solve_record_and_redraw_the_hydrogen_atom(tmp_path: pathlib.Path) -> None:
    """The whole path: solve, gate, record, draw; then draw again from the record alone."""
    pytest.importorskip("matplotlib")
    out = tmp_path / "solve"
    assert figures.main(["h_atom", "--out", str(out), "--only", "density_profile", "spectrum",
                         "--map-points", "81"]) == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "valid" and manifest["invocation"]["gate_profile"] == "quick"
    assert manifest["diagnostics"]["spectrum"]["eps_homo"] == pytest.approx(-0.5, abs=1e-5)
    again = tmp_path / "again"
    assert figures.main(["--from-corpus", str(out / "record.h5"), "--out", str(again),
                         "--only", "spectrum"]) == 0
    redrawn = json.loads((again / "manifest.json").read_text(encoding="utf-8"))
    assert redrawn["run_id"] == manifest["run_id"]
    rows = (again / "spectrum.csv").read_bytes().splitlines()[2:]
    assert rows == (out / "spectrum.csv").read_bytes().splitlines()[2:]
