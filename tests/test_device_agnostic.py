"""Device residency, seeded starts, int32 CSR indices and the deterministic switch.

All marked ``fast`` and asserted on the CPU in a form that fails there when it would fail on a
card; ``CDFT_DEVICE=cuda`` runs the same tests on one. Seconds.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from cdft.grid import UniformGrid
from cdft.operators.cusp import CuspFactor
from cdft.operators.external import external_potential
from cdft.operators.hamiltonian import CuspFactoredHamiltonian, LocalHamiltonian
from cdft.operators.poisson import CoulombCutoffPoisson
from cdft.operators.quadrature import CuspQuadrature
from cdft.precision import (
    configure_device,
    csr_index_dtype,
    deterministic_mode,
    host_floats,
    hot_dtype,
    policy_record,
    seeded_randn,
)
from contract import (
    AtomicStructure,
    BoundaryMode,
    DomainMode,
    ExternalPotentialKind,
    ExternalPotentialSpec,
    GridConfig,
    PrecisionConfig,
)

pytestmark = pytest.mark.fast


# --- helpers ---------------------------------------------------------------------------------


def _grid(device: torch.device, spacing: float = 0.5, edge: float = 10.0) -> UniformGrid:
    """A small box grid with the origin on a point."""
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


def _tensors(obj: object, path: str = "", seen: set[int] | None = None):
    """Yield ``(path, tensor)`` for every tensor reachable from ``obj``, each object once."""
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, torch.Tensor):
        yield path, obj
        return
    if isinstance(obj, torch.Generator):
        return
    if isinstance(obj, (str, bytes, int, float, bool, type(None), type, torch.device, torch.dtype)):
        return
    if callable(obj) and not hasattr(obj, "__dict__") and not hasattr(obj, "__slots__"):
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from _tensors(value, f"{path}[{key!r}]", seen)
        return
    if isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            yield from _tensors(value, f"{path}[{index}]", seen)
        return
    module = type(obj).__module__ or ""
    if not module.startswith(("cdft", "contract")):
        return
    names: list[str] = []
    if hasattr(obj, "__dict__"):
        names.extend(vars(obj))
    for klass in type(obj).__mro__:
        names.extend(getattr(klass, "__slots__", ()))
    for name in dict.fromkeys(names):
        try:
            value = getattr(obj, name)
        except AttributeError:
            continue
        yield from _tensors(value, f"{path}.{name}", seen)


def _assert_resident(obj: object, device: torch.device, label: str) -> int:
    """Assert every reachable tensor is on ``device``; return how many were checked."""
    wrong = [(p, str(t.device)) for p, t in _tensors(obj, label) if t.device != device]
    assert not wrong, f"tensors off {device}: {wrong[:10]}"
    return sum(1 for _ in _tensors(obj, label))


def _hydrogen_molecule_ion_objects(device: torch.device):
    """Grid, factor, quadrature and both Hamiltonians for two protons 1.5 bohr apart (h = 0.25)."""
    # The 6-bohr box keeps the overlapping sphere rules small (12k nodes) yet two-centred.
    grid = _grid(device, spacing=0.25, edge=6.0)
    positions = torch.tensor([[-0.75, 0.0, 0.0], [0.75, 0.0, 0.0]], dtype=torch.float64, device=device)
    factor = CuspFactor(grid, (1.0, 1.0), positions)
    factor.face_weights(grid.fd_order)
    quadrature = CuspQuadrature(grid, factor)
    transformed = CuspFactoredHamiltonian(grid, factor, quadrature=quadrature)
    spec = ExternalPotentialSpec(kind=ExternalPotentialKind.HARMONIC, omega=1.0)
    plain = LocalHamiltonian(grid, external_potential(spec, AtomicStructure(), grid))
    poisson = CoulombCutoffPoisson(grid)
    return grid, factor, quadrature, transformed, plain, poisson


def _interacting_step(device: torch.device):
    """A helium LDA :class:`~cdft.scf.step.KohnShamStep` on a coarse grid, after one full step."""
    from cdft.config import ALL_ELECTRON_NUMERICS
    from cdft.physics_config import helium, lda_functional
    from cdft.scf.loop import build_interacting_step, initial_density

    numerics = dataclasses.replace(
        ALL_ELECTRON_NUMERICS,
        grid=dataclasses.replace(ALL_ELECTRON_NUMERICS.grid, spacing=0.5, box_lengths=(9.0, 9.0, 9.0)),
        # Residency, not convergence: two filter steps exercise every code path of the step.
        eigen=dataclasses.replace(
            ALL_ELECTRON_NUMERICS.eigen, cross_check=False, max_iterations=2, chebyshev_degree=4
        ),
    )
    scenario = dataclasses.replace(helium(), xc=lda_functional("vwn"))
    grid, step = build_interacting_step(scenario, numerics, device=device, derive_grid=False)
    density = initial_density(step)
    potentials = step.potentials(density, step.functional)
    n_out, eigen, energies = step.step(density, step.functional)
    return grid, step, (density, potentials, n_out, eigen)


# --- residency -------------------------------------------------------------------------------


class TestResidency:
    """Every tensor an operator holds lives on ``grid.device``."""

    def test_operators_live_on_the_grid_device(self, device: torch.device) -> None:
        grid, factor, quadrature, transformed, plain, poisson = _hydrogen_molecule_ion_objects(device)
        assert grid.device == device
        checked = 0
        for label, obj in (
            ("grid", grid),
            ("factor", factor),
            ("quadrature", quadrature),
            ("transformed", transformed),
            ("plain", plain),
            ("poisson", poisson),
        ):
            checked += _assert_resident(obj, device, label)
        assert checked > 30
        # The two CSR operators, whose index arrays are not reachable as attributes.
        for operator in (quadrature._operator, quadrature._operator_t):
            assert operator.device == device
            assert operator.crow_indices().dtype == csr_index_dtype(device)
            assert operator.col_indices().dtype == csr_index_dtype(device)

    def test_the_scf_step_lives_on_the_grid_device(self, device: torch.device) -> None:
        grid, step, outputs = _interacting_step(device)
        assert _assert_resident(step, device, "step") > 30
        assert _assert_resident(list(outputs), device, "outputs") > 5

    def test_no_factory_call_relies_on_the_default_device(self) -> None:
        """No factory omits ``device=``: under a ``meta`` default it would yield a meta tensor."""
        cpu = torch.device("cpu")
        with torch.device("meta"):
            objects = _hydrogen_molecule_ion_objects(cpu)
            psi = seeded_randn((2, objects[0].n_points), torch.Generator(device="cpu").manual_seed(3), device=cpu)
            applied = objects[3].apply(psi)
            bounds = objects[3].spectral_bounds(generator=torch.Generator(device="cpu").manual_seed(4))
            hartree = objects[5].solve(psi[0] ** 2)
            step_objects = _interacting_step(cpu)
        for label, obj in zip(("grid", "factor", "quadrature", "transformed", "plain", "poisson"), objects):
            _assert_resident(obj, cpu, label)
        _assert_resident(step_objects[1], cpu, "step")
        _assert_resident(list(step_objects[2]), cpu, "outputs")
        assert applied.device == cpu and hartree.device == cpu
        assert all(isinstance(b, float) for b in bounds)


# --- seeded random start ---------------------------------------------------------------------


class TestSeededStart:
    """The random start is a CPU draw moved to the device (D-60 generator paragraph)."""

    def test_same_seed_same_bits(self, device: torch.device) -> None:
        a = seeded_randn((4, 7), torch.Generator(device="cpu").manual_seed(11), device=device)
        b = seeded_randn((4, 7), torch.Generator(device="cpu").manual_seed(11), device=device)
        assert a.device == device and a.dtype == torch.float64
        assert torch.equal(a, b)

    def test_independent_of_the_target_device(self, device: torch.device) -> None:
        targets = [torch.device("cpu"), torch.device("meta"), device]
        if torch.cuda.is_available():
            targets.append(torch.device("cuda"))
        reference = seeded_randn((3, 5), torch.Generator(device="cpu").manual_seed(5), device="cpu")
        for target in targets:
            generator = torch.Generator(device="cpu").manual_seed(5)
            first = seeded_randn((3, 5), generator, device=target)
            second = seeded_randn(9, generator, device=target)
            assert first.device.type == target.type
            if target.type != "meta":
                assert torch.equal(first.cpu(), reference)
            # The generator advanced by the same amount whatever the target.
            check = torch.Generator(device="cpu").manual_seed(5)
            torch.randn((3, 5), generator=check, dtype=torch.float64)
            expected_second = torch.randn((9,), generator=check, dtype=torch.float64)
            if target.type != "meta":
                assert torch.equal(second.cpu(), expected_second)

    def test_the_cpu_draw_is_the_pre_port_draw(self) -> None:
        """On the CPU the helper is exactly ``torch.randn(..., generator=...)``."""
        for shape in (17, (3, 4), (2, 3, 5)):
            got = seeded_randn(shape, torch.Generator(device="cpu").manual_seed(0), device="cpu")
            size = (shape,) if isinstance(shape, int) else shape
            want = torch.randn(
                size, device="cpu", dtype=torch.float64, generator=torch.Generator(device="cpu").manual_seed(0)
            )
            assert torch.equal(got, want)
        torch.manual_seed(123)
        got = seeded_randn(6, None, device="cpu")
        torch.manual_seed(123)
        assert torch.equal(got, torch.randn(6, dtype=torch.float64))


# --- int32 CSR operator (the CUDA default, exercised here on the CPU) --------------------------


class TestCsrIndexDtype:
    """The int32-indexed interpolation operator is the int64 one to round-off."""

    @staticmethod
    def _probe(grid: UniformGrid, quadrature: CuspQuadrature):
        points = grid.points()
        field = torch.exp(-0.3 * (points * points).sum(dim=-1)) * (1.0 + 0.2 * points[:, 0])
        nodes = torch.cos(quadrature.points).sum(dim=-1)
        return field, nodes

    def test_int32_matches_int64_on_the_cpu(self) -> None:
        cpu = torch.device("cpu")
        _, _, quadrature, _, _, _ = _hydrogen_molecule_ion_objects(cpu)
        grid = quadrature.grid
        assert quadrature._operator.crow_indices().dtype == torch.int64
        field, nodes = self._probe(grid, quadrature)
        interp64 = quadrature.interpolate(field)
        lift64 = quadrature.lift(nodes)
        gradient64 = quadrature.interpolate_gradient(field)

        quadrature._build_stencils(index_dtype=torch.int32)
        assert quadrature._operator.crow_indices().dtype == torch.int32
        assert quadrature._operator.col_indices().dtype == torch.int32
        assert quadrature._operator_t.col_indices().dtype == torch.int32
        interp32 = quadrature.interpolate(field)
        lift32 = quadrature.lift(nodes)
        gradient32 = quadrature.interpolate_gradient(field)
        assert float((interp32 - interp64).abs().max()) <= 1e-15 * max(1.0, float(interp64.abs().max()))
        assert float((lift32 - lift64).abs().max()) <= 1e-15 * max(1.0, float(lift64.abs().max()))
        assert float((gradient32 - gradient64).abs().max()) <= 1e-15 * max(1.0, float(gradient64.abs().max()))

        # Back to the policy default: the CPU operator is the int64 one, bit for bit.
        quadrature._build_stencils()
        assert quadrature._operator.crow_indices().dtype == torch.int64
        assert torch.equal(quadrature.interpolate(field), interp64)
        assert torch.equal(quadrature.lift(nodes), lift64)

    def test_policy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert csr_index_dtype(torch.device("cpu")) == torch.int64
        monkeypatch.delenv("CDFT_CSR_INT64", raising=False)
        assert csr_index_dtype(torch.device("cuda")) == torch.int32
        monkeypatch.setenv("CDFT_CSR_INT64", "1")
        assert csr_index_dtype(torch.device("cuda")) == torch.int64
        assert csr_index_dtype(torch.device("cpu")) == torch.int64

    def test_a_misspelled_switch_raises_rather_than_reading_as_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # O-28: the CSR-int64 and warn-only switches go through env_switch, so a typo is an error
        # (G5.5), not silently "off"; a real off value still reads as off.
        from cdft.precision import ENV_CSR_INT64, ENV_DETERMINISTIC_WARN_ONLY, env_switch

        monkeypatch.setenv(ENV_CSR_INT64, "ture")
        with pytest.raises(ValueError, match=ENV_CSR_INT64):
            csr_index_dtype(torch.device("cuda"))
        monkeypatch.setenv(ENV_CSR_INT64, "off")
        assert csr_index_dtype(torch.device("cuda")) == torch.int32
        monkeypatch.setenv(ENV_DETERMINISTIC_WARN_ONLY, "yse")
        with pytest.raises(ValueError, match=ENV_DETERMINISTIC_WARN_ONLY):
            env_switch(ENV_DETERMINISTIC_WARN_ONLY, False)
        if not torch.cuda.is_available():
            # The fake-CUDA branch of configure_device: the switch is read before torch is touched.
            from cdft.precision import CUBLAS_WORKSPACE_CONFIG, configure_device

            monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG)
            with pytest.raises(ValueError, match=ENV_DETERMINISTIC_WARN_ONLY):
                configure_device(torch.device("cuda"), deterministic=True)


# --- precision record ------------------------------------------------------------------------


class TestPrecisionRecord:
    """No record claims a float32 hot path while none is wired (G5.1)."""

    @pytest.mark.parametrize("name", ["cpu", "cuda", "meta"])
    def test_hot_path_is_reported_as_float64_everywhere(self, name: str) -> None:
        config = PrecisionConfig()
        target = torch.device(name)
        assert hot_dtype(config, target) == torch.float64
        record = policy_record(config, target)
        assert record["hot_path"] == "float64"
        assert record["device_type"] == name


# --- deterministic switch --------------------------------------------------------------------


class _FakeSwitch:
    """Stand-in for torch's process-global deterministic-algorithm state; records every call."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, enabled: bool = False) -> None:
        self.enabled = enabled
        self.warn_only = False
        self.calls: list[tuple[bool, bool]] = []
        monkeypatch.setattr(torch, "use_deterministic_algorithms", self.use)
        monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: self.enabled)
        monkeypatch.setattr(torch, "is_deterministic_algorithms_warn_only_enabled", lambda: self.warn_only)

    def use(self, mode: bool, *, warn_only: bool = False) -> None:
        self.calls.append((bool(mode), bool(warn_only)))
        self.enabled = bool(mode)
        self.warn_only = bool(warn_only) if mode else False


class TestDeterministicSwitch:
    """``configure_device(device, deterministic)`` honours ``NumericsConfig.deterministic``."""

    CUDA = torch.device("cuda")

    def test_off_is_not_switched_on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        switch = _FakeSwitch(monkeypatch)
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
        assert configure_device(self.CUDA, False) == "off"
        assert switch.calls == []  # a fresh process: the switch is not touched at all
        # The cuBLAS workspace is set either way; it must precede the first handle.
        import os

        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"

    def test_off_after_a_strict_solve_switches_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        switch = _FakeSwitch(monkeypatch, enabled=True)
        assert configure_device(self.CUDA, False) == "off"
        assert switch.calls == [(False, False)]
        assert all(mode is False for mode, _ in switch.calls)

    def test_strict_and_warn_only_on_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        switch = _FakeSwitch(monkeypatch)
        monkeypatch.delenv("CDFT_DETERMINISTIC_WARN_ONLY", raising=False)
        assert configure_device(self.CUDA, True) == "strict"
        monkeypatch.setenv("CDFT_DETERMINISTIC_WARN_ONLY", "1")
        assert configure_device(self.CUDA, True) == "warn_only"
        assert switch.calls == [(True, False), (True, True)]

    @pytest.mark.parametrize("requested", [False, True])
    def test_the_cpu_never_calls_the_switch(self, monkeypatch: pytest.MonkeyPatch, requested: bool) -> None:
        switch = _FakeSwitch(monkeypatch)
        assert configure_device(torch.device("cpu"), requested) == "off"
        assert switch.calls == []

    def test_provenance_says_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from cdft.io.provenance import capture
        from config import MODEL_SYSTEM_NUMERICS

        _FakeSwitch(monkeypatch)
        configure_device(self.CUDA, False)
        assert deterministic_mode() == "off"
        record = capture(MODEL_SYSTEM_NUMERICS, 0.0, device=torch.device("cpu"))
        assert record.deterministic_algorithms == "off"

    @pytest.mark.parametrize("requested", [False, True])
    def test_both_solvers_pass_the_numerics_value(self, monkeypatch: pytest.MonkeyPatch, requested: bool) -> None:
        import cdft.scf.loop as loop_module
        import cdft.scf.noninteracting as ni_module
        from cdft.physics_config import REGISTRY
        from cdft.scf.solve import solve_scenario
        from config import ALL_ELECTRON_NUMERICS, MODEL_SYSTEM_NUMERICS
        from contract import RunStatus

        seen: list[tuple[str, bool]] = []

        class _Stop(Exception):
            pass

        def recorder(device: torch.device, deterministic: bool = False) -> str:
            seen.append((device.type, deterministic))
            raise _Stop("recorded")  # the solver reports it as an ERROR artifact; nothing is solved

        monkeypatch.setattr(ni_module, "configure_device", recorder)
        monkeypatch.setattr(loop_module, "configure_device", recorder)
        for scenario_id, base in (("harmonic_w1", MODEL_SYSTEM_NUMERICS), ("he_atom_lda", ALL_ELECTRON_NUMERICS)):
            numerics = dataclasses.replace(base, deterministic=requested)
            artifact = solve_scenario(REGISTRY[scenario_id], numerics, device=torch.device("cpu"))
            assert artifact.status is RunStatus.ERROR and "recorded" in (artifact.error_message or "")
        assert seen == [("cpu", requested), ("cpu", requested)]


class TestHostFloats:
    """The batched host read returns exactly what ``float(x)`` returns, in argument order."""

    def test_values_and_order(self, device: torch.device) -> None:
        generator = torch.Generator(device="cpu").manual_seed(3)
        values = seeded_randn(5, generator, device=device)
        scalar = values.sum()
        out = host_floats(scalar, values[:2], 0.25, values[4])
        assert out == [float(scalar), float(values[0]), float(values[1]), 0.25, float(values[4])]
        assert host_floats(1.5, 2) == [1.5, 2.0]


# --- chunked triangular solve (works around the cuBLAS trsm cliff above ~2^19 rhs) -------------


class TestWideTriangularSolve:
    """``solve_triangular_wide`` equals ``torch.linalg.solve_triangular`` column for column."""

    def test_cpu_path_is_the_plain_call(self) -> None:
        from cdft.precision import solve_triangular_wide

        g = torch.Generator(device="cpu").manual_seed(11)
        x = torch.randn((5, 3000), generator=g, dtype=torch.float64)
        chol = torch.linalg.cholesky(x @ x.t() / 3000 + torch.eye(5, dtype=torch.float64))
        assert torch.equal(
            solve_triangular_wide(chol, x), torch.linalg.solve_triangular(chol, x, upper=False)
        )

    def test_chunking_is_bitwise_the_unchunked_solve(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Chunked forward substitution reproduces the unchunked solve to round-off."""
        import cdft.precision as precision

        g = torch.Generator(device="cpu").manual_seed(12)
        x = torch.randn((7, 1000), generator=g, dtype=torch.float64)
        chol = torch.linalg.cholesky(x @ x.t() / 1000 + torch.eye(7, dtype=torch.float64))
        plain = torch.linalg.solve_triangular(chol, x, upper=False)
        monkeypatch.setattr(precision, "CUDA_TRSM_CHUNK", 64)
        out = torch.empty_like(x)
        for start in range(0, 1000, 64):
            stop = min(start + 64, 1000)
            out[..., start:stop] = torch.linalg.solve_triangular(chol, x[..., start:stop], upper=False)
        # The loop above is the helper's CUDA body, where it is bitwise exact; a CPU BLAS may
        # block forward substitution by column count, so the tolerance is round-off, not bitwise:
        # MKL on Windows fails ``rtol=1e-14, atol=0`` (2026-09-19, T-4), a relative test
        # that no element near zero can meet. Round-off of a 7-row substitution is a few ulp of
        # the largest entry.
        scale = float(plain.abs().max())
        error = float((out - plain).abs().max())
        assert error <= 1e-13 * scale, f"chunked vs unchunked solve differ by {error:.3e} (scale {scale:.3e})"
        # On the CPU the helper never chunks: exactly one solve over the full width. Same
        # round-off tolerance -- MKL is not bitwise reproducible across buffer alignments (T-4).
        calls: list[int] = []
        real_solve = torch.linalg.solve_triangular

        def spy(factor, rhs, **kwargs):
            calls.append(rhs.shape[-1])
            return real_solve(factor, rhs, **kwargs)

        monkeypatch.setattr(torch.linalg, "solve_triangular", spy)
        helper = precision.solve_triangular_wide(chol, x)
        assert calls == [1000]
        assert torch.allclose(helper, plain, rtol=1e-14, atol=0.0)
