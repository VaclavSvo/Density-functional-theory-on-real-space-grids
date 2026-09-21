"""The geometry cache (``cdft.operators.geometry_cache``) and the CSC view of ``L^T``.

Hits are bitwise, keys separate distinct geometries, the byte budget evicts LRU, and the switch
and bypass leave the store alone. All marked ``fast``; tens of seconds (each case solves twice).
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from cdft.operators import geometry_cache
from cdft.operators import quadrature as quadrature_module
from cdft.operators.cusp import CuspFactor
from cdft.operators.quadrature import CuspQuadrature

pytestmark = pytest.mark.fast


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch: pytest.MonkeyPatch):
    """Every test starts and ends with an empty cache, default budgets and the switch unset."""
    monkeypatch.delenv(geometry_cache.ENV_GEOMETRY_CACHE, raising=False)
    geometry_cache.clear()
    geometry_cache.reset_limits()
    yield
    geometry_cache.clear()
    geometry_cache.reset_limits()


def _coarse_numerics():
    """All-electron numerics on a small fixed grid with a short eigensolve."""
    # Bit-identity needs no converged solve, so the filter stops after three degree-6 iterations.
    from cdft.config import ALL_ELECTRON_NUMERICS

    base = ALL_ELECTRON_NUMERICS
    return dataclasses.replace(
        base,
        grid=dataclasses.replace(base.grid, spacing=0.5, box_lengths=(9.0, 9.0, 9.0)),
        eigen=dataclasses.replace(base.eigen, cross_check=False, max_iterations=3, chebyshev_degree=6),
        device=type(base.device)("cpu"),
    )


def _fingerprint(artifact) -> dict:
    """The quantities ``scripts/fingerprint.py`` stores, as exact Python values."""
    result = artifact.result
    assert result is not None, artifact.error_message
    return {
        "status": artifact.status.value,
        "energies": {f.name: repr(getattr(result.energies, f.name)) for f in dataclasses.fields(result.energies)},
        "eigenvalues": [repr(float(x)) for x in result.eigenvalues.flatten()],
        "occupations": [repr(float(x)) for x in result.occupations.flatten()],
        "density": result.density.detach().cpu().numpy().tobytes(),
        "orbitals": None if result.orbitals is None else result.orbitals.detach().cpu().numpy().tobytes(),
        "iterations": result.n_iterations,
        "trajectory": list(result.trajectory.energies),
    }


def _measurements(artifact) -> str:
    """``repr`` of the measurements less the cache record, for a NaN-safe exact comparison."""
    return repr({k: v for k, v in artifact.measurements.items() if k != "geometry_cache"})


def _solve(scenario, numerics):
    from cdft.scf.solve import solve_scenario

    return solve_scenario(scenario, numerics, derive_grid=False, device=torch.device("cpu"))


def _versions(bundle: geometry_cache.GeometryBundle) -> dict[int, int]:
    """``_version`` of every strided tensor reachable from a bundle's parts, by id."""
    found: dict[int, torch.Tensor] = {}
    seen: set[int] = set()

    def walk(obj, depth=0):
        if depth > 8 or id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, torch.Tensor):
            if obj.layout == torch.strided:
                found[id(obj)] = obj
            else:
                found[id(obj)] = obj.values()
            return
        if isinstance(obj, dict):
            children = list(obj.values())
        elif isinstance(obj, (list, tuple)):
            children = list(obj)
        else:
            children = list(getattr(obj, "__dict__", {}).values())
            for cls in type(obj).__mro__:
                children += [getattr(obj, n) for n in getattr(cls, "__slots__", ()) if hasattr(obj, n)]
        for child in children:
            if not callable(child) or isinstance(child, torch.Tensor):
                walk(child, depth + 1)

    walk(list(bundle.parts.values()))
    return {key: tensor._version for key, tensor in found.items()}


class TestRepeatSolves:
    def test_noninteracting_second_solve_is_a_bitwise_hit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from cdft.physics_config import REGISTRY

        scenario, numerics = REGISTRY["h_atom"], _coarse_numerics()
        first = _solve(scenario, numerics)
        record = first.measurements["geometry_cache"]
        assert record["hit"] is False and record["enabled"] is True and record["bypassed"] is False
        assert record["parts"] == {"factor": False, "quadrature": False}
        assert record["entries"] == 1 and record["bytes"] > 0 and record["retained"] is True
        (bundle,) = geometry_cache._CACHE.entries.values()
        versions = _versions(bundle)

        second = _solve(scenario, numerics)
        again = second.measurements["geometry_cache"]
        assert again["hit"] is True and again["key"] == record["key"]
        assert again["parts"] == {"factor": True, "quadrature": True}
        assert _fingerprint(second) == _fingerprint(first)
        assert _measurements(second) == _measurements(first)
        assert _versions(bundle) == versions  # nothing wrote into a cached tensor

        monkeypatch.setenv(geometry_cache.ENV_GEOMETRY_CACHE, "0")
        off = _solve(scenario, numerics)
        assert off.measurements["geometry_cache"]["hit"] is False
        assert off.measurements["geometry_cache"]["enabled"] is False
        assert _fingerprint(off) == _fingerprint(first)
        assert geometry_cache.stats()["entries"] == 1  # the switch stores nothing new

    def test_interacting_second_solve_is_a_bitwise_hit(self) -> None:
        from cdft.physics_config import helium, lda_functional

        base = _coarse_numerics()
        numerics = dataclasses.replace(
            base,
            scf=dataclasses.replace(base.scf, max_iterations=2),
        )
        scenario = dataclasses.replace(helium(), xc=lda_functional("vwn"))
        first = _solve(scenario, numerics)
        record = first.measurements["geometry_cache"]
        assert record["hit"] is False
        assert set(record["parts"]) == {"factor", "quadrature", "step_geometry", f"('poisson', {numerics.grid.poisson_pad_factor!r})"}
        (bundle,) = geometry_cache._CACHE.entries.values()
        versions = _versions(bundle)

        second = _solve(scenario, numerics)
        assert second.measurements["geometry_cache"]["hit"] is True
        assert second.measurements["geometry_cache"]["key"] == record["key"]
        assert _fingerprint(second) == _fingerprint(first)
        assert _measurements(second) == _measurements(first)
        assert _versions(bundle) == versions

        with geometry_cache.bypass():
            rebuilt = _solve(scenario, numerics)
        assert rebuilt.measurements["geometry_cache"]["hit"] is False
        assert rebuilt.measurements["geometry_cache"]["bypassed"] is True
        assert _fingerprint(rebuilt) == _fingerprint(first)
        assert geometry_cache.stats()["entries"] == 1
        assert _solve(scenario, numerics).measurements["geometry_cache"]["hit"] is True


class TestKeys:
    @staticmethod
    def _grid(spacing: float = 0.5):
        from cdft.grid import UniformGrid
        from contract import BoundaryMode, DomainMode, GridConfig

        config = GridConfig(
            spacing=spacing, domain=DomainMode.BOX, box_lengths=(9.0, 9.0, 9.0),
            boundary=BoundaryMode.ZERO, fd_order=8, fd_gradient_order=8,
        )
        return UniformGrid.from_config(config, device="cpu", dtype=torch.float64)

    def test_translated_nuclei_and_other_spacing_miss(self) -> None:
        at_origin = torch.zeros((1, 3), dtype=torch.float64)
        shifted = torch.tensor([[0.0, 0.0, 1e-12]], dtype=torch.float64)
        key = geometry_cache.geometry_key(self._grid(), (1.0,), at_origin)
        assert key == geometry_cache.geometry_key(self._grid(), (1.0,), at_origin.clone())
        assert key != geometry_cache.geometry_key(self._grid(), (1.0,), shifted)
        assert key != geometry_cache.geometry_key(self._grid(0.4), (1.0,), at_origin)
        assert key != geometry_cache.geometry_key(self._grid(), (2.0,), at_origin)
        assert geometry_cache.short_hash(key) != geometry_cache.short_hash(
            geometry_cache.geometry_key(self._grid(), (1.0,), shifted)
        )

    def test_a_patched_quadrature_constant_misses(self, monkeypatch: pytest.MonkeyPatch) -> None:
        grid, positions = self._grid(), torch.zeros((1, 3), dtype=torch.float64)
        key = geometry_cache.geometry_key(grid, (1.0,), positions)
        monkeypatch.setattr(CuspQuadrature, "node_spacing_overlapping", 0.5)
        assert geometry_cache.geometry_key(grid, (1.0,), positions) != key
        monkeypatch.undo()
        monkeypatch.setattr(geometry_cache, "CACHE_VERSION", geometry_cache.CACHE_VERSION + 1)
        assert geometry_cache.geometry_key(grid, (1.0,), positions) != key

    def test_translated_structure_solve_misses(self) -> None:
        from cdft.physics_config import REGISTRY
        from cdft.scf.noninteracting import build_hamiltonian

        scenario, numerics = REGISTRY["h_atom"], _coarse_numerics()
        grid, _, _ = build_hamiltonian(scenario, numerics, torch.device("cpu"), derive_grid=False)
        assert grid.geometry_cache_record["hit"] is False
        grid, _, _ = build_hamiltonian(scenario, numerics, torch.device("cpu"), derive_grid=False)
        assert grid.geometry_cache_record["hit"] is True
        structure = scenario.structure
        moved = dataclasses.replace(
            structure, positions=tuple((float(p[0]) + 0.25, float(p[1]), float(p[2])) for p in structure.positions)
        )
        grid, _, _ = build_hamiltonian(
            dataclasses.replace(scenario, structure=moved), numerics, torch.device("cpu"), derive_grid=False
        )
        assert grid.geometry_cache_record["hit"] is False
        assert geometry_cache.stats()["entries"] == 2


class TestBudget:
    @staticmethod
    def _fill(name: str, n_bytes: int) -> geometry_cache.GeometryLease:
        key = (("device", "cpu"), ("name", name))
        lease = geometry_cache.GeometryLease(key)
        lease.part("blob", lambda: torch.zeros(n_bytes, dtype=torch.uint8))
        return lease

    def test_lru_eviction(self) -> None:
        geometry_cache.set_cache_limit_bytes("cpu", 2500)
        self._fill("a", 1000)
        self._fill("b", 1000)
        assert geometry_cache.GeometryLease((("device", "cpu"), ("name", "a"))).bundle is not None  # a is now newest
        self._fill("c", 1000)
        names = [dict(k)["name"] for k in geometry_cache._CACHE.entries]
        assert names == ["a", "c"]
        stats = geometry_cache.stats()
        assert stats["entries"] == 2 and stats["bytes"] == 2000 and stats["evictions"] == 1
        assert stats["misses"] == 3 and stats["hits"] == 0
        geometry_cache.set_cache_limit_bytes("cpu", 1500)
        assert [dict(k)["name"] for k in geometry_cache._CACHE.entries] == ["c"]
        assert geometry_cache.cache_limit_bytes(torch.device("cpu")) == 1500
        assert geometry_cache.cache_limit_bytes("cuda") == geometry_cache.DEFAULT_CUDA_LIMIT_BYTES

    def test_a_bundle_over_budget_is_used_not_kept(self) -> None:
        geometry_cache.set_cache_limit_bytes("cpu", 500)
        lease = self._fill("big", 1000)
        assert geometry_cache.stats()["entries"] == 0
        record = lease.record()
        assert record["bytes"] == 1000 and record["hit"] is False and record["retained"] is False

    def test_tiny_budget_forces_solve_misses(self) -> None:
        from cdft.physics_config import REGISTRY

        geometry_cache.set_cache_limit_bytes("cpu", 1024)
        scenario, numerics = REGISTRY["h_atom"], _coarse_numerics()
        first = _solve(scenario, numerics)
        second = _solve(scenario, numerics)
        assert first.measurements["geometry_cache"]["hit"] is False
        assert second.measurements["geometry_cache"]["hit"] is False
        assert second.measurements["geometry_cache"]["entries"] == 0
        assert _fingerprint(first) == _fingerprint(second)

    def test_negative_limit_and_bad_switch_are_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(ValueError):
            geometry_cache.set_cache_limit_bytes("cuda", -1)
        monkeypatch.setenv(geometry_cache.ENV_GEOMETRY_CACHE, "maybe")
        with pytest.raises(ValueError):
            geometry_cache.enabled()

    def test_tensor_bytes_counts_shared_storage_once(self) -> None:
        base = torch.zeros(100, dtype=torch.float64)
        assert geometry_cache.tensor_bytes([base, base[10:], {"x": base}]) == 800
        csr = torch.eye(4, dtype=torch.float64).to_sparse_csr()
        parts = (csr.crow_indices(), csr.col_indices(), csr.values())
        assert geometry_cache.tensor_bytes(csr) == sum(p.untyped_storage().nbytes() for p in parts) >= 5 * 8 + 4 * 8 + 4 * 8


class TestTransposeAsCsc:
    def test_lift_is_bitwise_the_csr_transpose(self, monkeypatch: pytest.MonkeyPatch) -> None:
        grid = TestKeys._grid(0.5)
        positions = torch.tensor([[0.1, 0.2, -0.3]], dtype=torch.float64)  # off the lattice
        factor = CuspFactor(grid, (1.0,), positions)
        reference = CuspQuadrature(grid, factor)
        monkeypatch.setattr(quadrature_module, "TRANSPOSE_AS_CSC", True)
        view = CuspQuadrature(grid, factor)
        assert reference.describe()["transpose_layout"] == "csr"
        assert view.describe()["transpose_layout"] == "csc_view"
        assert view._operator_t.layout == torch.sparse_csc
        assert view._operator_t.values().data_ptr() == view._operator.values().data_ptr()
        generator = torch.Generator().manual_seed(3)
        nodes = torch.randn((3, reference.n_nodes), generator=generator, dtype=torch.float64)
        assert torch.equal(view.lift(nodes), reference.lift(nodes))
        assert torch.equal(view.lift_gradient(nodes), reference.lift_gradient(nodes))
        assert torch.equal(view.mass_weights, reference.mass_weights)
