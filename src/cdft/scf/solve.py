"""One entry point for solving a scenario: non-interacting or self-consistent, decided by the scenario.

``xc.name == "none"`` (:data:`~cdft.physics_config.NONINTERACTING_XC`) is a bare-potential
eigenproblem and goes to :func:`~cdft.scf.noninteracting.solve_noninteracting`; anything else
carries a Hartree term and the named functional and goes to :mod:`cdft.scf.loop`. The choice is
made from the scenario alone, so a caller cannot run an interacting scenario through the
non-interacting solver by picking the wrong function (D-53).
"""

from __future__ import annotations

import torch

from contract import NumericsConfig, RunArtifact, ScenarioSpec

from .noninteracting import solve_noninteracting

__all__ = ["solve_scenario", "is_interacting"]


def is_interacting(scenario: ScenarioSpec) -> bool:
    """Whether the scenario's Hamiltonian carries Hartree and exchange--correlation terms.

    ``xc.name == "none"`` is the non-interacting marker; any other name is resolved by
    :mod:`cdft.xc.dispatch` and switches the Hartree term on with it. There is no Hartree-only mode.
    """
    return scenario.xc.name.lower() != "none"


def solve_scenario(
    scenario: ScenarioSpec,
    numerics: NumericsConfig,
    n_states: int | None = None,
    cross_check: bool | None = None,
    *,
    derive_grid: bool = True,
    device: torch.device | None = None,
) -> RunArtifact:
    """Solve one scenario and return a fully gated artifact, by whichever path it needs.

    ``device`` is forwarded to the solver, ``None`` resolving ``numerics.device``; the device used
    is recorded in the artifact's provenance.
    """
    if not is_interacting(scenario):
        return solve_noninteracting(
            scenario,
            numerics,
            n_states=n_states,
            cross_check=cross_check,
            derive_grid=derive_grid,
            device=device,
        )
    from .loop import solve_self_consistent

    return solve_self_consistent(
        scenario,
        numerics,
        n_states=n_states,
        cross_check=cross_check,
        derive_grid=derive_grid,
        device=device,
    )
