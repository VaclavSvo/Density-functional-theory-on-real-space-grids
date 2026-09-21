"""Throughput mechanisms: CUDA-graph capture and spectral-bound reuse.

All marked ``fast``; the graph engine runs against a CPU stand-in graph, the real capture only
under ``CDFT_DEVICE=cuda``. Tens of seconds (the short SCF run at the end dominates).
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from cdft.eigen import chefsi
from cdft.eigen.chefsi import (
    BOUND_REUSE_RITZ_FRACTION,
    ChebyshevFilteredSubspace,
    GraphedChebyshevFilter,
    SpectralBoundsHint,
    chebyshev_coefficients,
    chebyshev_filter,
    chebyshev_recurrence,
    graphed_filter_for,
)
from cdft.grid import UniformGrid
from cdft.operators.cusp import CuspFactor
from cdft.operators.hamiltonian import CuspFactoredHamiltonian, LocalHamiltonian
from cdft.precision import ENV_CUDA_GRAPHS, ENV_LANCZOS_REUSE, cuda_graphs_enabled, env_switch, seeded_randn
from cdft.scf import loop
from contract import BoundaryMode, DomainMode, EigenConfig, GridConfig

pytestmark = pytest.mark.fast


def _grid(device: torch.device, spacing: float = 0.5, edge: float = 6.0) -> UniformGrid:
    """A small zero-boundary box."""
    config = GridConfig(
        spacing=spacing,
        fd_order=8,
        domain=DomainMode.BOX,
        boundary=BoundaryMode.ZERO,
        box_lengths=(edge, edge, edge),
        use_double_grid=False,
        fourier_filter_projectors=False,
    )
    return UniformGrid.from_config(config, device=device, dtype=torch.float64)


def _harmonic(grid: UniformGrid, omega: float = 1.0, shift: float = 0.0) -> LocalHamiltonian:
    """A harmonic well, optionally shifted by a constant."""
    r2 = (grid.points().to(torch.float64) ** 2).sum(dim=-1)
    return LocalHamiltonian(grid, 0.5 * omega**2 * r2 + shift)


def _cusp(grid: UniformGrid, extra_scale: float = 0.0) -> CuspFactoredHamiltonian:
    """A one-proton transformed operator, no quadrature, with an optional smooth potential."""
    positions = torch.zeros((1, 3), dtype=torch.float64, device=grid.device)
    factor = CuspFactor(grid, (1.0,), positions)
    extra = None
    if extra_scale:
        r2 = (grid.points().to(torch.float64) ** 2).sum(dim=-1)
        extra = extra_scale * torch.exp(-r2)
    return CuspFactoredHamiltonian(grid, factor, extra_potential=extra)


def _block(grid: UniformGrid, rows: int = 5, seed: int = 3) -> torch.Tensor:
    """A seeded random block."""
    return seeded_randn((rows, grid.n_points), torch.Generator().manual_seed(seed), device=grid.device)


# --- the captured body equals the eager filter -------------------------------------------------


class TestRecurrenceIsTheEagerFilter:
    """The tensor-coefficient recurrence is bitwise the eager filter."""

    @pytest.mark.parametrize("kind", ["local", "cusp"])
    def test_bitwise(self, kind: str) -> None:
        grid = _grid(torch.device("cpu"))
        op = _harmonic(grid) if kind == "local" else _cusp(grid, extra_scale=0.3)
        block = _block(grid)
        degree, lower, upper, smin = 7, 2.1, 60.0, -0.7
        eager = chebyshev_filter(op, block, degree, lower=lower, upper=upper, spectrum_min=smin)
        table = torch.tensor(chebyshev_coefficients(degree, lower, upper, smin), dtype=torch.float64)
        graphed = chebyshev_recurrence(op.apply_with_potential, block, table, op.graph_potential())
        assert torch.equal(eager, graphed)

    def test_coefficients_refuse_an_empty_interval(self) -> None:
        with pytest.raises(ValueError):
            chebyshev_coefficients(4, 3.0, 3.0, 0.0)

    @pytest.mark.parametrize("kind", ["local", "cusp"])
    def test_apply_with_potential_is_apply(self, kind: str) -> None:
        grid = _grid(torch.device("cpu"))
        op = _harmonic(grid) if kind == "local" else _cusp(grid, extra_scale=0.3)
        block = _block(grid, rows=2)
        count = op.apply_count
        assert torch.equal(op.apply(block), op.apply_with_potential(block, op.graph_potential()))
        assert op.apply_count == count + 1
        op.note_applications(5)
        assert op.apply_count == count + 6

    def test_geometry_key_is_shared_by_scf_operators_and_not_by_other_geometries(self) -> None:
        from tests.test_device_agnostic import _interacting_step

        grid, step, (density, potentials, n_out, _) = _interacting_step(torch.device("cpu"))
        first = step.hamiltonian(potentials)
        second = step.hamiltonian(step.potentials(n_out, step.functional))
        assert first.graph_geometry_key() == second.graph_geometry_key()
        assert first.graph_potential() is not second.graph_potential()
        assert first.graph_geometry_key() != _cusp(_grid(torch.device("cpu"))).graph_geometry_key()


# --- the engine's bookkeeping, with a stand-in graph on the CPU --------------------------------


class _FakeGraph:
    """Stand-in for ``torch.cuda.CUDAGraph``: ``replay`` recomputes the body into the output."""

    def __init__(self, engine: "_CpuEngine", apply, error: float = 0.0) -> None:
        self.engine, self.apply, self.error = engine, apply, error

    def replay(self) -> None:
        e = self.engine
        out = chebyshev_recurrence(self.apply, e._static_input, e._static_coefficients, e._static_potential)
        e._static_output.copy_(out + self.error)


class _CpuEngine(GraphedChebyshevFilter):
    """The engine with its CUDA-only pieces (capability check, capture) replaced."""

    def __init__(self, error: float = 0.0, fail: bool = False) -> None:
        super().__init__(torch.device("cpu"))
        self.error, self.fail = error, fail

    @staticmethod
    def graph_capable(hamiltonian, block):
        for name in ("apply_with_potential", "graph_potential", "graph_geometry_key", "note_applications"):
            if not hasattr(hamiltonian, name):
                return "no graph seam"
        return None

    def _capture(self, hamiltonian, block, potential, rows, key) -> None:
        if self.fail:
            raise RuntimeError("simulated capture failure")
        self._release()
        self._static_input = block.clone()
        self._static_potential = potential.clone()
        self._static_coefficients = torch.tensor(rows, dtype=torch.float64)
        self._static_output = torch.empty_like(block)
        self._graph = _FakeGraph(self, hamiltonian.apply_with_potential, self.error)
        self._template = hamiltonian
        self._key = key
        self._validated = False
        self.captures += 1


class TestEngineBookkeeping:
    def test_replays_follow_new_scalars_and_new_potentials(self) -> None:
        grid = _grid(torch.device("cpu"))
        engine = _CpuEngine()
        block = _block(grid)
        # A new operator per call, as the SCF builds one per iteration; new scalars and potential.
        for shift, lower, upper, smin in ((0.0, 1.0, 50.0, -0.5), (0.0, 2.0, 55.0, -0.4), (0.7, 2.5, 58.0, 0.1)):
            op = _harmonic(grid, shift=shift)
            got = engine(op, block, 6, lower, upper, smin)
            reference = chebyshev_filter(_harmonic(grid, shift=shift), block, 6, lower=lower, upper=upper, spectrum_min=smin)
            assert torch.equal(got, reference)
            assert op.apply_count == 6
        record = engine.record()
        # One capture serves all three operators (same grid): first call validates, two replay.
        assert record == {
            "used": True,
            "captures": 1,
            "replays": 2,
            "eager_calls": {"validation of a new capture": 1},
            "warmup_applications": 0,
            "validation_max_rel_diff": 0.0,
        }

    def test_returned_block_is_not_the_static_buffer(self) -> None:
        grid = _grid(torch.device("cpu"))
        engine, op, block = _CpuEngine(), _harmonic(grid), _block(grid)
        engine(op, block, 4, 1.0, 50.0, -0.5)
        first = engine(op, block, 4, 1.0, 50.0, -0.5)
        kept = first.clone()
        engine(op, _block(grid, seed=9), 4, 1.0, 50.0, -0.5)
        assert torch.equal(first, kept)

    def test_a_new_shape_or_degree_recaptures(self) -> None:
        grid = _grid(torch.device("cpu"))
        engine, op = _CpuEngine(), _harmonic(grid)
        engine(op, _block(grid, rows=5), 4, 1.0, 50.0, -0.5)
        engine(op, _block(grid, rows=4), 4, 1.0, 50.0, -0.5)
        engine(op, _block(grid, rows=4), 5, 1.0, 50.0, -0.5)
        assert engine.captures == 3
        assert engine.record()["eager_calls"] == {"validation of a new capture": 3}

    def test_too_many_captures_disable_with_a_reason(self) -> None:
        grid = _grid(torch.device("cpu"))
        engine, op = _CpuEngine(), _harmonic(grid)
        for rows in range(1, chefsi.GRAPH_MAX_CAPTURES + 3):
            engine(op, _block(grid, rows=rows), 3, 1.0, 50.0, -0.5)
        record = engine.record()
        assert record["used"] is False and "captures" in record["reason"]

    def test_capture_failure_is_recorded_and_the_result_is_eager(self) -> None:
        grid = _grid(torch.device("cpu"))
        engine, op, block = _CpuEngine(fail=True), _harmonic(grid), _block(grid)
        got = engine(op, block, 5, 1.0, 50.0, -0.5)
        assert torch.equal(got, chebyshev_filter(op, block, 5, lower=1.0, upper=50.0, spectrum_min=-0.5))
        record = engine.record()
        assert record["used"] is False
        assert record["reason"].startswith("capture failed: RuntimeError: simulated capture failure")
        engine(op, block, 5, 1.0, 50.0, -0.5)
        assert engine.record()["eager_calls"] == {"disabled": 2}

    def test_a_replay_that_disagrees_is_discarded(self) -> None:
        grid = _grid(torch.device("cpu"))
        engine, op, block = _CpuEngine(error=1e-3), _harmonic(grid), _block(grid)
        got = engine(op, block, 5, 1.0, 50.0, -0.5)
        assert torch.equal(got, chebyshev_filter(op, block, 5, lower=1.0, upper=50.0, spectrum_min=-0.5))
        record = engine.record()
        assert record["used"] is False and "differs from the eager filter" in record["reason"]
        assert record["validation_max_rel_diff"] > chefsi.GRAPH_VALIDATION_RTOL

    def test_an_operator_without_the_seam_runs_eager_and_is_counted(self) -> None:
        grid = _grid(torch.device("cpu"))
        op = _harmonic(grid)

        class Wrapped:
            def __init__(self, inner):
                self.inner = inner

            def apply(self, x):
                return self.inner.apply(x)

        engine, block = _CpuEngine(), _block(grid)
        got = engine(Wrapped(op), block, 4, 1.0, 50.0, -0.5)
        assert torch.equal(got, chebyshev_filter(op, block, 4, lower=1.0, upper=50.0, spectrum_min=-0.5))
        assert engine.record() == {
            "used": False,
            "captures": 0,
            "replays": 0,
            "eager_calls": {"no graph seam": 1},
            "warmup_applications": 0,
            "validation_max_rel_diff": None,
            "reason": "no filter call was graph-capable",
        }

    def test_the_real_engine_refuses_a_cpu_block(self) -> None:
        grid = _grid(torch.device("cpu"))
        engine, op, block = GraphedChebyshevFilter("cpu"), _harmonic(grid), _block(grid)
        got = engine(op, block, 4, 1.0, 50.0, -0.5)
        assert torch.equal(got, chebyshev_filter(op, block, 4, lower=1.0, upper=50.0, spectrum_min=-0.5))
        assert engine.record()["eager_calls"] == {"block on cpu; graphs are CUDA-only": 1}

    def test_the_solver_with_the_engine_gives_the_eager_result(self) -> None:
        grid = _grid(torch.device("cpu"))
        config = EigenConfig(max_iterations=6, chebyshev_degree=8, cross_check=False)
        results = []
        for engine in (None, _CpuEngine()):
            op = _harmonic(grid)
            solver = ChebyshevFilteredSubspace(grid, config, graphed_filter=engine)
            results.append((solver.solve(op, 2, generator=torch.Generator().manual_seed(1)), op.apply_count))
        (eager, eager_count), (graphed, graphed_count) = results
        assert torch.equal(eager.eigenvalues, graphed.eigenvalues)
        assert torch.equal(eager.eigenvectors, graphed.eigenvectors)
        assert eager_count == graphed_count


class TestSwitches:
    def test_factory(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ENV_CUDA_GRAPHS, raising=False)
        assert graphed_filter_for(torch.device("cpu")) is None
        engine = graphed_filter_for(torch.device("cuda"))  # constructing touches no CUDA API
        assert engine is not None and engine.disabled_reason is None
        monkeypatch.setenv(ENV_CUDA_GRAPHS, "0")
        engine = graphed_filter_for(torch.device("cuda"))
        assert engine.record() == {
            "used": False, "captures": 0, "replays": 0, "eager_calls": {}, "warmup_applications": 0,
            "validation_max_rel_diff": None, "reason": f"disabled by {ENV_CUDA_GRAPHS}=0",
        }
        assert not cuda_graphs_enabled(torch.device("cuda"))
        monkeypatch.setenv(ENV_CUDA_GRAPHS, "1")
        assert not cuda_graphs_enabled(torch.device("cpu"))

    def test_env_switch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        name = "CDFT_TEST_SWITCH"
        monkeypatch.delenv(name, raising=False)
        assert env_switch(name, True) and not env_switch(name, False)
        for raw, value in (("0", False), ("off", False), ("NO", False), ("1", True), (" true ", True)):
            monkeypatch.setenv(name, raw)
            assert env_switch(name, not value) is value
        monkeypatch.setenv(name, "maybe")
        with pytest.raises(ValueError):
            env_switch(name, True)

    def test_lanczos_reuse_switch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ENV_LANCZOS_REUSE, raising=False)
        assert loop.lanczos_reuse_enabled()
        monkeypatch.setenv(ENV_LANCZOS_REUSE, "0")
        assert not loop.lanczos_reuse_enabled()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device (runs on the card)")
class TestRealCapture:
    """On a card, the real capture replays the eager filter (the CPU leg uses a stand-in)."""

    def test_capture_replays_the_eager_filter(self) -> None:
        device = torch.device("cuda", torch.cuda.current_device())
        grid = _grid(device)
        engine = GraphedChebyshevFilter(device)
        block = _block(grid)
        # Same grid, so one capture; the transformed operator has its own geometry key, so a second.
        calls = [(_harmonic(grid, shift=s), lower) for s, lower in ((0.0, 1.0), (0.3, 1.5), (0.6, 2.0))]
        cusp = _cusp(grid, extra_scale=0.3)
        calls += [(cusp, 1.0), (cusp, 1.7)]
        for op, lower in calls:
            got = engine(op, block, 6, lower, 60.0, -0.5)
            ref = chebyshev_filter(op, block, 6, lower=lower, upper=60.0, spectrum_min=-0.5)
            assert (got - ref).abs().max().item() <= 1e-12 * ref.abs().max().item()
        record = engine.record()
        assert record["used"] is True, record
        assert record["captures"] == 2 and record["replays"] == 3, record
        for buffer in (engine._static_input, engine._static_potential, engine._static_coefficients, engine._static_output):
            assert buffer.device == device


# --- spectral-bound reuse ----------------------------------------------------------------------


def _info(source: str = "lanczos", low: float = -1.0, high: float = 99.0, shift: float = 0.0, fraction: float = 0.02):
    return {
        "eigen_spectral_bounds": [low, high],
        "eigen_bounds_source": source,
        "eigen_bounds_shift": shift,
        "eigen_top_ritz_fraction": fraction,
    }


class TestBoundReusePolicy:
    def _decide(self, policy, **kw):
        base = {"residual_rose": False, "rung_changed": False, "level_shift": False}
        base.update(kw)
        return policy.decide(**base)

    def test_triggers_in_order(self) -> None:
        policy = loop.BoundReusePolicy(True)
        assert self._decide(policy) == "no carried interval"
        policy.observe("no carried interval", _info())
        assert self._decide(policy) is None
        assert self._decide(policy, level_shift=True) == "level-shift rung active"
        assert self._decide(policy, rung_changed=True) == "fallback rung taken"
        assert self._decide(policy, residual_rose=True) == "SCF residual rose"

    def test_disabled_always_reestimates(self) -> None:
        policy = loop.BoundReusePolicy(False)
        policy.observe(None, _info())
        assert self._decide(policy).startswith("reuse off")

    def test_backstop(self) -> None:
        policy = loop.BoundReusePolicy(True)
        policy.observe("no carried interval", _info())
        for _ in range(loop.LANCZOS_REUSE_BACKSTOP):
            assert self._decide(policy) is None
            policy.observe(None, _info("reused", shift=1e-6))
        assert self._decide(policy).startswith("backstop")
        policy.observe("backstop", _info())
        assert self._decide(policy) is None
        record = policy.record()
        assert record["lanczos_reestimates"] == 2 and record["lanczos_reuses"] == loop.LANCZOS_REUSE_BACKSTOP
        assert record["lanczos_bound_history"][0] == "lanczos: no carried interval"
        assert record["lanczos_bound_history"][1] == "reused (shift 1.000e-06)"

    def test_widening_and_ritz_fraction(self) -> None:
        policy = loop.BoundReusePolicy(True)
        policy.observe("start", _info(low=0.0, high=100.0))
        policy.observe(None, _info("reused", low=0.0, high=100.0, shift=6.0))
        assert "accumulated shift" in self._decide(policy)
        policy.observe("widening", _info(low=0.0, high=100.0, fraction=BOUND_REUSE_RITZ_FRACTION))
        assert "top Ritz fraction" in self._decide(policy)

    def test_in_solve_rejection_counts_as_a_reestimate(self) -> None:
        policy = loop.BoundReusePolicy(True)
        policy.observe("start", _info())
        policy.observe(None, _info("lanczos (reuse rejected: top Ritz fraction 0.95 >= 0.9 at iteration 1)"))
        assert policy.reestimates == 2 and policy.since_estimate == 0
        assert policy.history[-1].startswith("lanczos (reuse rejected")


class TestSolverWithCarriedBounds:
    def _solve(self, op, grid, bounds=None, iterations=40):
        config = EigenConfig(max_iterations=iterations, chebyshev_degree=10, cross_check=False, residual_tol=1e-7)
        solver = ChebyshevFilteredSubspace(grid, config)
        result = solver.solve(op, 2, generator=torch.Generator().manual_seed(2), bounds=bounds)
        return solver, result

    def test_default_is_a_fresh_estimate(self) -> None:
        grid = _grid(torch.device("cpu"), spacing=0.6)
        solver, result = self._solve(_harmonic(grid), grid)
        assert solver.bounds_source == "lanczos" and solver.bounds_shift == 0.0
        low, high = solver.spectral_bounds
        assert low < float(result.eigenvalues[0]) and high > 10.0
        # The random start's top Ritz value sits mid-spectrum, well below the rejection threshold.
        assert 0.0 < solver.top_ritz_fraction < BOUND_REUSE_RITZ_FRACTION

    def test_weyl_widened_bounds_converge_to_the_same_eigenpairs(self) -> None:
        grid = _grid(torch.device("cpu"), spacing=0.6)
        first = _harmonic(grid)
        solver, _ = self._solve(first, grid)
        low, high = solver.spectral_bounds
        second = _harmonic(grid, omega=1.05)
        shift = (second.graph_potential() - first.graph_potential()).abs().max()
        fresh_solver, fresh = self._solve(second, grid)
        reused_solver, reused = self._solve(second, grid, SpectralBoundsHint(low, high, shift))
        assert reused_solver.bounds_source == "reused"
        assert reused_solver.bounds_shift == pytest.approx(float(shift))
        assert reused_solver.spectral_bounds == (low - float(shift), high + float(shift))
        assert fresh.converged and reused.converged
        assert torch.allclose(fresh.eigenvalues, reused.eigenvalues, atol=1e-9, rtol=0)
        # The widened carried interval contains the fresh estimate's top (Weyl, with slack).
        assert reused_solver.spectral_bounds[1] >= fresh_solver.spectral_bounds[1] - 1e-6 * high

    def test_a_carried_bound_below_the_subspace_is_rejected_in_the_solve(self) -> None:
        grid = _grid(torch.device("cpu"), spacing=0.6)
        op = _harmonic(grid)
        solver, result = self._solve(op, grid, SpectralBoundsHint(-1.0, 1.6, 0.0))
        assert solver.bounds_source.startswith("lanczos (reuse rejected: top Ritz fraction")
        assert solver.spectral_bounds[1] > 10.0
        assert result.converged


def test_short_scf_records_the_bound_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reuse on and off record their bound choices and the operator counts they imply."""
    from cdft.config import ALL_ELECTRON_NUMERICS
    from cdft.physics_config import helium, lda_functional

    numerics = dataclasses.replace(
        ALL_ELECTRON_NUMERICS,
        grid=dataclasses.replace(ALL_ELECTRON_NUMERICS.grid, spacing=0.5, box_lengths=(9.0, 9.0, 9.0)),
        # Bookkeeping, not convergence: short eigensolves keep the two runs to a few seconds.
        eigen=dataclasses.replace(
            ALL_ELECTRON_NUMERICS.eigen, cross_check=False, max_iterations=6, chebyshev_degree=6
        ),
        scf=dataclasses.replace(ALL_ELECTRON_NUMERICS.scf, max_iterations=4),
    )
    scenario = dataclasses.replace(helium(), xc=lda_functional("vwn"))
    runs = {}
    for setting in ("1", "0"):
        monkeypatch.setenv(ENV_LANCZOS_REUSE, setting)
        artifact = loop.solve_self_consistent(scenario, numerics, derive_grid=False, device=torch.device("cpu"))
        assert artifact.result is not None, artifact.error_message
        runs[setting] = artifact
    on, off = runs["1"].measurements, runs["0"].measurements
    assert "cuda_graphs" not in on and "cuda_graphs" not in off
    iterations = len(on["scf_hamiltonian_applications"])
    assert on["lanczos_bound_reuse"] is True and off["lanczos_bound_reuse"] is False
    assert off["lanczos_reuses"] == 0 and off["lanczos_reestimates"] == iterations
    assert on["lanczos_reestimates"] + on["lanczos_reuses"] == iterations
    assert on["lanczos_bound_history"][0] == "lanczos: no carried interval"
    assert on["lanczos_reuses"] >= 1
    # A warm iteration applies the operator 2 + k(degree + 2) times for k filter steps, plus the
    # Lanczos steps when -- and only when -- the bounds are estimated.
    per_step = numerics.eigen.chebyshev_degree + 2
    lanczos_steps = numerics.eigen.lanczos_steps
    for run in (on, off):
        pairs = zip(run["scf_hamiltonian_applications"], run["lanczos_bound_history"], strict=True)
        for count, choice in list(pairs)[1:]:
            filter_part = count - 2 - (0 if choice.startswith("reused") else lanczos_steps)
            steps = filter_part // per_step
            assert filter_part % per_step == 0, (count, choice)
            assert 1 <= steps <= loop.FILTER_STEPS_PER_ITERATION, (count, choice)
    assert on["hamiltonian_applications"] > sum(on["scf_hamiltonian_applications"])
    # The iterates differ: the filter interval is a path choice (D-16). The fixed point is
    # compared on the converged he_atom_lda fingerprint, not here.
    assert len(runs["1"].result.trajectory.energies) == len(runs["0"].result.trajectory.energies)
