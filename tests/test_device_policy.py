"""Device policy derivation, the allocator record, the G5.2 strict pair and the G5.9 budget.

Cards and solves are faked, so everything runs on the CPU. Marked ``fast`` except the end-to-end
G5.9 check, which is ``slow``. Seconds, plus a minute for the slow one.
"""

from __future__ import annotations

import dataclasses
import os
import types

import pytest
import torch

from cdft import device_policy as policy
from cdft import precision
from cdft.device_policy import DeviceProfile, derive_settings

GIB = 2**30


def _card(total: float, free: float, name: str = "fake card") -> DeviceProfile:
    """A fake CUDA profile of ``total``/``free`` GiB."""
    return DeviceProfile(
        name=name, device="cuda:0", capability="7.5", total_bytes=int(total * GIB),
        free_bytes=int(free * GIB), sm_count=1, fp64_gflops=30.0, bandwidth_gbs=100.0,
        torch_version=torch.__version__, driver="610.88",
    )


@pytest.mark.fast
class TestDerivedSettings:
    """The derivation rules of ``cdft.device_policy``, on profiles no machine here has."""

    def test_the_six_gib_laptop_card(self) -> None:
        s = derive_settings(_card(6.0, 5.5))
        assert s.geometry_cache_bytes == int(0.25 * 6 * GIB)
        assert s.graph_memory_fraction == pytest.approx(0.25)
        assert s.graph_pool_blocks == 10
        assert s.trsm_chunk == 2**18
        assert s.use_cuda_graphs is True
        assert s.max_concurrent_solves == 2

    def test_a_nearly_full_card_turns_graphs_off_and_runs_one_solve(self) -> None:
        s = derive_settings(_card(6.0, 1.5))
        assert s.use_cuda_graphs is False
        assert s.graph_memory_fraction == 0.0
        assert s.max_concurrent_solves == 1

    def test_the_graph_pool_never_takes_more_than_half_the_free_memory(self) -> None:
        s = derive_settings(_card(8.0, 2.5))
        assert s.graph_memory_fraction == pytest.approx(0.5 * 2.5 / 8.0)

    def test_a_large_card_is_capped(self) -> None:
        s = derive_settings(_card(48.0, 47.0))
        assert s.geometry_cache_bytes == policy.GEOMETRY_CACHE_CAP_BYTES
        assert s.max_concurrent_solves == policy.MAX_CONCURRENT_CAP
        assert s.graph_memory_fraction == pytest.approx(0.25)

    def test_the_cpu_profile_turns_every_device_feature_off(self) -> None:
        profile = policy.capture_profile("cpu")
        assert profile.fp64_gflops is None and profile.bandwidth_gbs is None
        assert profile.capability == "n/a" and profile.free_bytes is None
        s = derive_settings(profile)
        assert (s.use_cuda_graphs, s.graph_memory_fraction, s.trsm_chunk, s.max_concurrent_solves) == (
            False, 0.0, 0, 1,
        )

    def test_the_cpu_cache_without_psutil_is_the_stated_constant(self) -> None:
        profile = dataclasses.replace(policy.capture_profile("cpu"), total_bytes=0)
        assert derive_settings(profile).geometry_cache_bytes == policy.HOST_CACHE_FALLBACK_BYTES
        assert derive_settings(profile, host_ram_bytes=64 * GIB).geometry_cache_bytes == policy.HOST_CACHE_CAP_BYTES

    def test_the_profile_is_captured_once(self) -> None:
        assert policy.capture_profile("cpu") is policy.capture_profile(torch.device("cpu"))

    @pytest.mark.skipif(torch.cuda.is_available(), reason="needs a machine without CUDA")
    def test_no_card_is_guessed_on_a_cpu_machine(self) -> None:
        with pytest.raises(RuntimeError, match="no CUDA device"):
            policy.capture_profile("cuda")


@pytest.mark.fast
class TestApplyPolicy:
    """Where ``apply_policy`` writes the derived settings, and what the CPU leaves alone."""

    def test_the_cpu_apply_touches_no_solver_setting(self) -> None:
        from cdft.eigen import chefsi

        before = (chefsi.GRAPH_MEMORY_FRACTION, chefsi.GRAPH_POOL_BLOCKS, precision.CUDA_TRSM_CHUNK,
                  precision._CUDA_GRAPHS_ALLOWED)
        settings = policy.apply_policy("cpu")
        assert settings.use_cuda_graphs is False
        after = (chefsi.GRAPH_MEMORY_FRACTION, chefsi.GRAPH_POOL_BLOCKS, precision.CUDA_TRSM_CHUNK,
                 precision._CUDA_GRAPHS_ALLOWED)
        assert before == after
        assert "geometry_cache" in policy.applied_record("cpu")

    def test_an_explicit_cache_budget_is_applied_and_recorded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``CDFT_GEOMETRY_CACHE_BYTES`` overrides the derived budget, visibly; a bad value raises (G5.5)."""
        from cdft.operators import geometry_cache

        monkeypatch.setattr(policy, "_APPLIED", {})
        monkeypatch.setenv(policy.ENV_GEOMETRY_CACHE_BYTES, str(3 * 2**28))
        before = geometry_cache.cache_limit_bytes("cpu")
        try:
            policy.apply_policy("cpu")
            assert geometry_cache.cache_limit_bytes("cpu") == 3 * 2**28
            assert policy.applied_record("cpu")["geometry_cache_override"].startswith(policy.ENV_GEOMETRY_CACHE_BYTES)
        finally:
            geometry_cache.set_cache_limit_bytes("cpu", before)
        monkeypatch.setattr(policy, "_APPLIED", {})
        monkeypatch.setenv(policy.ENV_GEOMETRY_CACHE_BYTES, "lots")
        with pytest.raises(ValueError, match=policy.ENV_GEOMETRY_CACHE_BYTES):
            policy.apply_policy("cpu")

    def test_the_host_ram_seen_by_the_policy_is_capped_by_the_cgroup(self) -> None:
        """The CPU cache budget is derived from the memory an OOM kill enforces (D-81)."""
        limit = policy.cgroup_memory_limit_bytes()
        ram = policy._host_ram_bytes()
        if limit is not None and ram is not None:
            assert ram <= limit

    def test_a_cuda_apply_writes_the_derived_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from cdft.eigen import chefsi

        fake = torch.device("cuda", 0)
        monkeypatch.setattr(policy, "_canonical", lambda device: fake)
        monkeypatch.setattr(policy, "capture_profile", lambda device: _card(6.0, 1.0))
        monkeypatch.setattr(policy, "_APPLIED", {})
        for module, name in ((chefsi, "GRAPH_MEMORY_FRACTION"), (chefsi, "GRAPH_POOL_BLOCKS"),
                             (precision, "CUDA_TRSM_CHUNK"), (precision, "_CUDA_GRAPHS_ALLOWED")):
            monkeypatch.setattr(module, name, getattr(module, name))
        settings = policy.apply_policy(fake)
        assert settings.use_cuda_graphs is False
        assert chefsi.GRAPH_MEMORY_FRACTION == 0.0
        assert precision._CUDA_GRAPHS_ALLOWED is False
        assert precision.cuda_graphs_enabled(fake) is False
        assert precision.cuda_graphs_enabled(torch.device("cpu")) is False
        assert policy.applied_record(fake)["tf32"] == "off (asserted)"

    def test_tf32_set_by_a_caller_is_refused(self) -> None:
        previous = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            with pytest.raises(RuntimeError, match="TF32"):
                policy.enforce_tf32_off()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous
        policy.enforce_tf32_off()
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False

    def test_the_setters_refuse_nonsense(self) -> None:
        from cdft.eigen import chefsi

        with pytest.raises(ValueError):
            chefsi.set_graph_memory_limits(1.5, 10)
        with pytest.raises(ValueError):
            chefsi.set_graph_memory_limits(0.25, 0)
        with pytest.raises(ValueError):
            precision.set_trsm_chunk(0)


@pytest.mark.fast
class TestAllocatorAndRecord:
    """The allocator setting is in the environment and in the record; release is free on the CPU."""

    def test_the_allocator_setting_is_in_force(self) -> None:
        value = os.environ.get(precision.ENV_CUDA_ALLOC_CONF) or os.environ.get(precision.ENV_ALLOC_CONF)
        assert value, "cdft.precision sets PYTORCH_CUDA_ALLOC_CONF at import unless the user set one"
        assert precision.cuda_alloc_conf() == value

    def test_an_explicit_value_is_reported_as_given(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(precision.ENV_CUDA_ALLOC_CONF, "max_split_size_mb:128")
        assert precision.cuda_alloc_conf() == "max_split_size_mb:128"

    def test_the_policy_record_carries_profile_settings_and_allocator(self) -> None:
        import json

        from contract import PrecisionConfig

        record = precision.policy_record(PrecisionConfig(), torch.device("cpu"))
        assert record["cuda_alloc_conf"] == precision.cuda_alloc_conf()
        assert json.loads(record["device_profile"])["device"] == "cpu"
        assert json.loads(record["derived_settings"])["use_cuda_graphs"] is False
        assert all(isinstance(v, str) for v in record.values())

    def test_release_is_a_no_op_on_the_cpu(self) -> None:
        precision.release_device_memory(torch.device("cpu"))


# --- G5.2: the strict pair ---------------------------------------------------------------------


def _fake_artifact(eigenvalues, mode: str = "off", device: str = "cpu"):
    """A completed-run stand-in with the fields G5.2 reads."""
    from cdft.physics_config import REGISTRY
    from config import ALL_ELECTRON_NUMERICS

    return types.SimpleNamespace(
        scenario=REGISTRY["h_atom"],
        numerics=dataclasses.replace(ALL_ELECTRON_NUMERICS, deterministic=False),
        result=types.SimpleNamespace(eigenvalues=torch.tensor([eigenvalues], dtype=torch.float64)),
        provenance=types.SimpleNamespace(device=device, deterministic_algorithms=mode, wall_time_s=2.0),
        error_message="",
    )


@pytest.mark.fast
class TestDeterminismPair:
    """G5.2 runs its own strict pair and thresholds only A against B."""

    def _run(self, monkeypatch, primary, a, b):
        import cdft.scf.solve as solve_module
        from cdft.gates.tier5 import DeterminismGate

        calls: list[tuple] = []
        outputs = iter((a, b))

        def fake_solve(scenario, numerics, n_states=None, cross_check=None, *, derive_grid=True, device=None):
            calls.append((numerics.deterministic, numerics.eigen.cross_check, str(device), n_states))
            return _fake_artifact(next(outputs), mode="strict")

        configured: list[tuple] = []
        monkeypatch.setattr(solve_module, "solve_scenario", fake_solve)
        monkeypatch.setattr(precision, "configure_device", lambda d, det=False: configured.append((str(d), det)) or "off")
        result = DeterminismGate().evaluate(_fake_artifact(primary))
        return result, calls, configured

    def test_two_strict_solves_compared_with_each_other(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result, calls, configured = self._run(monkeypatch, [-0.5, 0.1], [-0.5 + 1e-7, 0.1], [-0.5 + 1e-7, 0.1])
        assert calls == [(True, False, "cpu", 2), (True, False, "cpu", 2)]
        assert result.verdict.value == "pass" and result.measured == 0.0
        detail = result.detail
        assert detail["bitwise_identical"] is True
        assert detail["primary_vs_strict"] == pytest.approx(1e-7)  # recorded, not thresholded
        assert detail["deterministic_algorithms"] == {"primary": "off", "first_run": "strict", "repeat_run": "strict"}
        assert detail["solves"] == 2 and detail["gpu_leg_exercised"] is False
        assert len(detail["strict_pair_wall_s"]) == 2
        assert configured == [("cpu", False)]  # back to the artifact's own mode

    def test_a_non_reproducible_pair_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result, _, _ = self._run(monkeypatch, [-0.5], [-0.5], [-0.5 + 1e-6])
        assert result.verdict.value == "fail"
        assert result.detail["bitwise_identical"] is False

    def test_the_gate_declares_two_solves(self) -> None:
        from cdft.gates.tier5 import DeterminismGate

        assert DeterminismGate.spec.solves == 2
        assert DeterminismGate.spec.threshold == 1.0e-9


# --- G5.9: the sync/launch budget ----------------------------------------------------------------


def _summary(read_sites, implicit_sites=()):
    """A census summary with the given ``(site, count)`` rows."""
    return {
        "read_sites": [{"site": s, "function": "f", "count": c} for s, c in read_sites],
        "implicit_sites": [{"op": "cholesky", "site": s, "function": "g", "count": c} for s, c in implicit_sites],
    }


@pytest.mark.fast
class TestSyncBudget:
    """The loop/setup split and the budget arithmetic of G5.9."""

    def _census(self):
        from cdft.gates.tier5 import _load_census_module

        module, path = _load_census_module()
        if module is None:
            pytest.skip(f"census script absent at {path}")
        return module

    def test_loop_sites_are_the_ones_that_fire_every_iteration(self) -> None:
        census = self._census()
        loop = census.loop_syncs(_summary([("a.py:1", 12), ("a.py:2", 1)], [("b.py:3", 24)]), 12)
        assert loop["loop_syncs_total"] == 36 and loop["setup_syncs_total"] == 1
        assert loop["per_iteration"] == pytest.approx(3.0)
        assert census.loop_syncs(_summary([("a.py:1", 5)]), 0)["per_iteration"] == 0.0

    def test_one_new_read_per_iteration_exceeds_every_recorded_budget(self) -> None:
        from cdft.gates.tier5 import SYNC_BUDGETS

        for budget in SYNC_BUDGETS.values():
            for quantity in ("reads_per_iteration", "implicit_per_iteration"):
                recorded = getattr(budget, quantity)
                assert recorded <= budget.budget(quantity) < recorded + 1.0
            assert budget.budget("launches") == pytest.approx(1.1 * budget.launches)

    def test_a_new_loop_read_is_seen_by_the_split(self) -> None:
        from cdft.gates.tier5 import SYNC_BUDGETS

        census = self._census()
        budget = SYNC_BUDGETS["he_scf_probe"]
        n = budget.iterations
        reads = round(budget.reads_per_iteration * n)
        clean = census.loop_syncs(_summary([("p.py:1", reads)]), n)["per_iteration"]
        dirty = census.loop_syncs(_summary([("p.py:1", reads), ("new.py:9", n)]), n)["per_iteration"]
        assert clean <= budget.budget("reads_per_iteration") < dirty

    def test_the_gate_is_wired_and_catalogued(self) -> None:
        from cdft.gates.catalogue import Lifecycle, by_id
        from cdft.gates.runner import default_component_gates

        assert "G5.9" in {gate.gate_id for gate in default_component_gates()}
        assert by_id("G5.9").lifecycle is Lifecycle.IMPLEMENTED


@pytest.mark.slow
def test_g59_fails_when_a_read_is_added_to_the_eigensolver_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """An extra ``.item()`` per Rayleigh--Ritz read makes G5.9 fail end to end on the CPU."""
    from cdft.eigen import chefsi
    from cdft.gates.tier5 import SyncBudgetGate

    original = chefsi.host_floats

    def leaky(*values):
        tensors = [v for v in values if isinstance(v, torch.Tensor)]
        if tensors:
            tensors[0].reshape(-1)[0].item()
        return original(*values)

    monkeypatch.setattr(chefsi, "host_floats", leaky)
    result = SyncBudgetGate().evaluate(None)
    assert result.verdict.value == "fail", result.detail["probes"]
    assert any("reads_per_iteration" in row for row in result.detail["exceeded"])
