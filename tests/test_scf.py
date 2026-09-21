"""The self-consistent field machinery: mixers, scenario dispatch, the pure step and the driver.

Mixers and dispatch are ``fast``; the helium LDA solves are ``slow`` and ``physics``. Tens of
seconds with ``--runslow``, instant without.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from contract import MixingConfig, MixingScheme, SCFStepProtocol
from cdft.config import ALL_ELECTRON_NUMERICS
from cdft.physics_config import helium, lda_functional, NONINTERACTING_XC
from cdft.scf.mixing import LinearMixer, PulayMixer, make_mixer
from cdft.scf.solve import is_interacting


def _numerics(h: float, edge: float):
    return dataclasses.replace(
        ALL_ELECTRON_NUMERICS,
        grid=dataclasses.replace(ALL_ELECTRON_NUMERICS.grid, spacing=h, box_lengths=(edge, edge, edge)),
        eigen=dataclasses.replace(ALL_ELECTRON_NUMERICS.eigen, cross_check=False),
    )


@pytest.mark.fast
class TestMixers:
    """On a contractive linear map every mixer reaches the same fixed point."""

    @staticmethod
    def _map(n: torch.Tensor) -> torch.Tensor:
        # F(n) = A n + b with spectral radius 0.8: the fixed point is (I - A)^-1 b.
        a = torch.tensor([[0.5, 0.3], [0.1, 0.6]], dtype=torch.float64)
        b = torch.tensor([1.0, 2.0], dtype=torch.float64)
        return (a @ n.reshape(-1) + b).reshape(1, 2)

    def _run(self, mixer, iterations: int) -> torch.Tensor:
        n = torch.zeros((1, 2), dtype=torch.float64)
        for k in range(1, iterations + 1):
            n = mixer.mix(n, self._map(n), k)
        return n

    def test_linear_and_pulay_agree_on_the_fixed_point(self) -> None:
        inner = lambda a, b: (a * b).sum()  # noqa: E731
        exact = torch.linalg.solve(torch.eye(2, dtype=torch.float64) - torch.tensor([[0.5, 0.3], [0.1, 0.6]], dtype=torch.float64), torch.tensor([1.0, 2.0], dtype=torch.float64))
        linear = self._run(LinearMixer(0.5), 200)
        pulay = self._run(PulayMixer(0.5, history=4, period=1, inner=inner), 12)
        periodic = self._run(PulayMixer(0.5, history=4, period=2, inner=inner), 20)
        for result in (linear, pulay, periodic):
            assert torch.allclose(result.reshape(-1), exact, atol=1e-9)

    def test_make_mixer_honours_the_scheme(self) -> None:
        from cdft.grid import UniformGrid
        from contract import BoundaryMode, DomainMode, GridConfig

        grid = UniformGrid.from_config(GridConfig(spacing=1.0, fd_order=2, domain=DomainMode.BOX, boundary=BoundaryMode.ZERO, box_lengths=(4.0, 4.0, 4.0), use_double_grid=False, fourier_filter_projectors=False))
        inner = lambda a, b: (a * b).sum()  # noqa: E731
        assert isinstance(make_mixer(MixingConfig(scheme=MixingScheme.LINEAR), grid, inner), LinearMixer)
        periodic = make_mixer(MixingConfig(scheme=MixingScheme.PERIODIC_PULAY, pulay_period=3), grid, inner)
        assert isinstance(periodic, PulayMixer) and periodic.period == 3
        plain = make_mixer(MixingConfig(scheme=MixingScheme.PULAY), grid, inner)
        assert isinstance(plain, PulayMixer) and plain.period == 1
        kerker = make_mixer(MixingConfig(scheme=MixingScheme.PERIODIC_PULAY, kerker=True), grid, inner)
        assert kerker.preconditioner is not None
        assert make_mixer(MixingConfig(kerker=False), grid, inner).preconditioner is None

    def test_pulay_survives_a_dependent_history(self) -> None:
        inner = lambda a, b: (a * b).sum()  # noqa: E731
        mixer = PulayMixer(0.5, history=6, period=1, inner=inner)
        n = torch.ones((1, 2), dtype=torch.float64)
        for k in range(1, 8):
            n = mixer.mix(n, n + 1e-3, k)  # identical residuals every time: a singular Gram matrix
        assert torch.isfinite(n).all()


@pytest.mark.fast
class TestDispatchToTheSolver:
    def test_the_seven_increment_one_scenarios_are_non_interacting(self) -> None:
        from cdft.physics_config import REGISTRY

        for scenario_id in ("harmonic_w1", "box_L10", "h_atom", "he_plus", "h2plus_R2", "he_atom", "h2_R1.4"):
            assert REGISTRY[scenario_id].xc == NONINTERACTING_XC
            assert not is_interacting(REGISTRY[scenario_id])
        assert is_interacting(REGISTRY["he_atom_lda"])

    def test_the_quick_profile_selection_is_a_subset_of_the_registry(self) -> None:
        from cdft.physics_config import QUICK_PROFILE_SCENARIOS, REGISTRY

        assert set(QUICK_PROFILE_SCENARIOS) <= set(REGISTRY.ids())
        assert any(is_interacting(REGISTRY[s]) for s in QUICK_PROFILE_SCENARIOS)


@pytest.mark.slow
@pytest.mark.physics
class TestPureStep:
    """``KohnShamStep`` implements ``SCFStepProtocol`` and is a pure function of its inputs (D-49)."""

    def test_step_is_stateless_and_deterministic(self) -> None:
        from cdft.scf.loop import build_interacting_step, initial_density

        scenario = dataclasses.replace(helium(), xc=lda_functional("vwn"))
        grid, step = build_interacting_step(scenario, _numerics(0.5, 12.0), derive_grid=False)
        assert isinstance(step, SCFStepProtocol)
        n_in = initial_density(step)
        snapshot = n_in.clone()
        before = {k: v for k, v in vars(step).items()}
        n_out_1, eigen_1, energies_1 = step.step(n_in, step.functional)
        n_out_2, eigen_2, energies_2 = step.step(n_in, step.functional)
        assert torch.equal(n_in, snapshot), "the input density was mutated"
        if n_out_1.device.type == "cuda" and not torch.are_deterministic_algorithms_enabled():
            # Statelessness is the claim; bitwise repetition is not, on a card with
            # non-deterministic reductions allowed (D-61: G5.2 judges the same pair at 1e-9 Ha,
            # 2.7e-13 measured). Two steps from one density agree to that noise (D-83).
            assert torch.allclose(n_out_1, n_out_2, rtol=0.0, atol=1e-11)
            assert torch.allclose(eigen_1.eigenvalues, eigen_2.eigenvalues, rtol=0.0, atol=1e-10)
            for key, value in dataclasses.asdict(energies_1).items():
                other = dataclasses.asdict(energies_2)[key]
                if isinstance(value, float):
                    assert abs(value - other) < 1e-9, f"energy {key}: {value!r} vs {other!r}"
                else:
                    assert value == other
        else:
            assert torch.equal(n_out_1, n_out_2)
            assert torch.equal(eigen_1.eigenvalues, eigen_2.eigenvalues)
            assert energies_1 == energies_2
        for key, value in before.items():
            assert vars(step)[key] is value, f"attribute {key} was rebound by step()"
        residual = step.residual(n_in, step.functional)
        assert torch.allclose(residual, n_out_1 - n_in)

    def test_scf_converges_and_records_the_standard_energy_form(self) -> None:
        from cdft.scf.loop import solve_self_consistent

        scenario = dataclasses.replace(helium(), xc=lda_functional("vwn"))
        artifact = solve_self_consistent(scenario, _numerics(0.5, 12.0), derive_grid=False)
        assert artifact.result is not None and artifact.result.converged
        energies = artifact.result.energies
        assert energies.closure_error() < 1e-12
        assert abs(energies.total - energies.harris_foulkes) < 1e-6
        assert artifact.measurements["external_method"] == "band_energy_complement"
        assert artifact.measurements["charge_error"] < 1e-12
        assert artifact.measurements["external_consistency"] < 1e-9
        # 3.9e-3 Ha from the NIST value at h = 0.5, and on the right side of it.
        assert -2.84 < energies.total < -2.82
