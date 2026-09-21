"""Physical quantities pinned at 1e-9 relative: this layer asks only whether a number moved.

Grouped by discretisation identities, converged physics, the oracles (the rulers) and the
known-open gate measurements. Marked ``golden``, the solves also ``slow``; minutes.
Accept a change with ``pytest --rebless`` and record why in ``docs/05_DECISION_LOG.md``.
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import os

import pytest
import torch

from cdft.config import ALL_ELECTRON_NUMERICS
from cdft.physics_config import REGISTRY
from cdft.scf.noninteracting import solve_noninteracting

#: Markers for every test in this module (see ``tests/conftest.py``). The pins are the CPU audit
#: path's (D-60, D-83): under ``CDFT_DEVICE=cuda`` the card reproduces them to 4e-7 relative at
#: most (G1.13 2.740499e-9 against 2.7405002e-9, 2026-09-19), which is D-60's device agreement,
#: not a pin, so the module is skipped there and the card is judged by G5.2 and the fingerprints.
pytestmark = [
    pytest.mark.golden,
    pytest.mark.skipif(
        os.environ.get("CDFT_DEVICE", "cpu") == "cuda",
        reason="golden pins are the CPU audit path; the card is compared by scripts/fingerprint.py --diff (D-60)",
    ),
]


@functools.lru_cache(maxsize=None)
def _solve(scenario_id: str, n_states: int = 1):
    """Solve one registered scenario at production settings, without the cross-check.

    Memoised per scenario: the solve is deterministic and the value tests only read from the
    artifact, so five solves serve every assertion here. The one mutator,
    ``GateSuite.run_on_artifact``, is handed a copy instead of the cached object.
    """
    scenario = {s.scenario_id: s for s in REGISTRY}[scenario_id]
    numerics = dataclasses.replace(
        ALL_ELECTRON_NUMERICS,
        eigen=dataclasses.replace(ALL_ELECTRON_NUMERICS.eigen, cross_check=False),
    )
    return solve_noninteracting(scenario, numerics, n_states=n_states)


#: Tolerance for a *fitted* convergence order, looser than the global 1e-9 pin.
#:
#: A fitted slope divides differences of nearly equal float64 reductions, so its last digits follow
#: the BLAS and the thread count: measured cross-platform spread at order 8 is 5.3e-6 relative.
#: 1e-4 is twenty times that spread and still 150 times tighter than the 1.5e-2 deviation from the
#: nominal order that gate G0.1 thresholds, so a real stencil change moves it by orders.
_FITTED_ORDER_RTOL = 1.0e-4


# --- round-off identities: properties of the discretisation --------------------------------------


@pytest.mark.fast
class TestDiscretisationIdentities:
    """Numbers true to round-off; a move here means the operator itself changed."""

    @pytest.mark.parametrize("charges", [(1.0,), (2.0,), (1.0, 1.0)])
    def test_self_adjointness_in_the_weighted_measure(self, golden, charges) -> None:
        """``<a|A|b>_w = <b|A|a>_w`` stays at round-off rather than the 1e-2 seen before D-38."""
        from tests.test_physics_invariants import _cusp_operator

        grid, hamiltonian = _cusp_operator(charges)
        generator = torch.Generator().manual_seed(7)
        a, b = torch.randn((2, 1, grid.n_points), dtype=torch.float64, generator=generator)
        left = float(hamiltonian.measure.cross(a, hamiltonian.apply(b))[0, 0])
        right = float(hamiltonian.measure.cross(b, hamiltonian.apply(a))[0, 0])
        scale = max(abs(left), abs(right), 1e-300)
        golden.check(
            f"selfadjoint.asymmetry.Z{'_'.join(f'{c:g}' for c in charges)}",
            abs(left - right) / scale,
            rtol=1e-3,
            note=(
                "relative asymmetry of the cusp operator in its own measure. Pinned loosely at "
                "1e-3 because the value is at round-off and its last digits are BLAS noise; what "
                "is being pinned is the ORDER, which was 1e-2 before D-38 and is 1e-16 after"
            ),
        )

    @pytest.mark.parametrize("charges", [(1.0,), (2.0,)])
    def test_checkerboard_rayleigh_quotient(self, golden, charges) -> None:
        """The checkerboard Rayleigh quotient stays high in the spectrum, not near ``-Z^2/2``."""
        from tests.test_physics_invariants import _cusp_operator

        grid, hamiltonian = _cusp_operator(charges)
        alternating = ((-1.0) ** grid._box_index.sum(dim=-1)).to(torch.float64).unsqueeze(0)
        alternating = alternating / hamiltonian.measure.norm(alternating)[0]
        quotient = float(
            hamiltonian.measure.cross(alternating, hamiltonian.apply(alternating))[0, 0]
        )
        golden.check(
            f"checkerboard.rayleigh.Z{charges[0]:g}",
            quotient,
            note=(
                "must stay large and positive; a value near -Z^2/2 means the kinetic form has "
                "regained a null space and every eigenvector is contaminated"
            ),
        )

    @pytest.mark.parametrize("order", [2, 4, 6, 8])
    def test_measured_laplacian_order(self, golden, order: int) -> None:
        """The fitted convergence order of the Laplacian stencil (G0.1)."""
        from cdft.gates.tier0 import LaplacianOrderGate

        numerics = dataclasses.replace(
            ALL_ELECTRON_NUMERICS,
            grid=dataclasses.replace(ALL_ELECTRON_NUMERICS.grid, fd_order=order),
        )
        result = LaplacianOrderGate().evaluate(numerics)
        golden.check(
            f"stencil.laplacian.measured_order.p{order}",
            float(result.detail["measured_order"]),
            rtol=_FITTED_ORDER_RTOL,
            note=(
                f"nominal {order}; fitted from four refinements of a Gaussian. Pinned loosely at "
                f"1e-4 because a fitted slope is not a measured value: see _FITTED_ORDER_RTOL"
            ),
        )


# --- converged physics at production settings ----------------------------------------------------


@pytest.mark.slow
@pytest.mark.physics
class TestGroundStates:
    """Ground states for every Phase 1 system, at production settings so the pin is one."""

    @pytest.mark.parametrize(
        "scenario_id", ["h_atom", "he_plus", "he_atom", "h2plus_R2", "h2_R1.4"]
    )
    def test_ground_state_eigenvalue(self, golden, scenario_id: str) -> None:
        """The lowest eigenvalue of the converged run."""
        artifact = _solve(scenario_id)
        golden.check(
            f"eigenvalue.ground.{scenario_id}",
            float(artifact.result.eigenvalues[0][0]),
            note="ground-state eigenvalue at the D-42 derived grid",
        )

    @pytest.mark.parametrize(
        "scenario_id", ["h_atom", "he_plus", "he_atom", "h2plus_R2", "h2_R1.4"]
    )
    def test_kinetic_energy(self, golden, scenario_id: str) -> None:
        """The kinetic energy of the converged run; it moves where the eigenvalue would not."""
        artifact = _solve(scenario_id)
        golden.check(
            f"kinetic.{scenario_id}",
            float(artifact.result.energies.kinetic),
            note="T_s at the D-42 derived grid; moves independently of the eigenvalue",
        )

    @pytest.mark.parametrize("scenario_id", ["h_atom", "he_plus"])
    def test_grid_derivation_is_stable(self, golden, scenario_id: str) -> None:
        """The spacing the D-42 derivation rule produces, so a change to the rule cannot pass."""
        artifact = _solve(scenario_id)
        described = artifact.measurements["grid"]
        golden.check(
            f"grid.derived_spacing.{scenario_id}",
            float(described["spacing_bohr"]),
            note="derived by cdft.grid.coulomb_spacing; a change here changes every number above",
        )


# --- the oracles: pinning the rulers -------------------------------------------------------------


@pytest.mark.physics
class TestOracles:
    """The reference solvers, pinned apart from what they measure: a drifting ruler is silent."""

    @pytest.mark.parametrize("charge", [1.0, 2.0, 8.0])
    def test_radial_oracle(self, golden, charge: float) -> None:
        """The radial oracle's hydrogenic ground state against ``-Z^2/2``."""
        from cdft.reference.radial import RadialGrid, solve_radial

        result = solve_radial(lambda r: -charge / r, n_states=1, grid=RadialGrid())
        golden.check(
            f"oracle.radial.Z{charge:g}",
            float(result.eigenvalues[0]),
            note=f"exact -{charge**2 / 2:g}; logarithmic grid, direct banded diagonalisation",
        )

    @pytest.mark.parametrize("bond_length", [1.4, 2.0])
    def test_two_centre_oracle(self, golden, bond_length: float) -> None:
        """The two-centre oracle's H2+ electronic energy (the total less the 1/R repulsion)."""
        from cdft.reference.two_centre import h2_plus_energy

        result = h2_plus_energy(bond_length=bond_length)
        golden.check(
            f"oracle.two_centre.R{bond_length:g}.electronic",
            float(result.electronic_energy),
            note="Richardson-extrapolated over three nested grids, order re-measured every call",
        )

    def test_two_centre_oracle_matches_the_literature(self, golden) -> None:
        """The oracle's distance from the published H2+ total, which is what makes it an oracle."""
        from cdft.reference.two_centre import h2_plus_energy

        published = -0.6026342144949
        result = h2_plus_energy(bond_length=2.0)
        golden.check(
            "oracle.two_centre.R2.literature_delta",
            abs(float(result.total_energy) - published),
            rtol=1e-2,
            note=(
                "Madsen and Peek, Atomic Data 2, 171 (1971). Pinned loosely at 1e-2 relative "
                "because the last digits of a 1e-9 agreement are extrapolation noise; what is "
                "pinned is that the agreement stays at 1e-9 and does not become 1e-6"
            ),
        )


# --- known-open failures: pinned so they cannot quietly get worse ---------------------------------


@pytest.mark.slow
@pytest.mark.physics
class TestKnownOpenFailures:
    """Molecular gate measurements for O-13 and O-19, pinned red or green, against drift.

    A red row that was already red draws no attention, so the value is pinned as well as the
    verdict; rows closed by D-54, D-58 and D-75 stay pinned under ``closed.*``. No molecular row is
    known-open since D-75 (G1.13 at 2.7e-9 and 2.9e-9 against 1e-8).
    """

    #: Rows that still fail (known-open, ``cdft.gates.known_open``) and rows that were closed.
    ROWS = {
        "h2plus_R2": {"known_open": (), "closed": ("G1.13", "G2.7", "G3.1")},
        "h2_R1.4": {"known_open": (), "closed": ("G1.13", "G2.7", "G3.1")},
    }

    @pytest.mark.parametrize("scenario_id", sorted(ROWS))
    def test_molecular_gate_measurements(self, golden, scenario_id: str) -> None:
        """Every O-13/O-19 gate on one molecule, from one solve, pinned row by row."""
        from cdft.gates.runner import GateSuite

        artifact = GateSuite().run_on_artifact(copy.deepcopy(_solve(scenario_id)))
        results = {r.gate_id: r for r in artifact.gates.results}
        for gate_id in self.ROWS[scenario_id]["known_open"]:
            golden.check(
                f"known_open.{scenario_id}.{gate_id}",
                float(results[gate_id].measured),
                rtol=1e-6,
                note=(
                    "an expected failure (O-13 or O-19). Pinned so that a regression cannot hide "
                    "behind a row that was already red; retiring this pin is what fixing the item "
                    "looks like"
                ),
            )
        for gate_id in self.ROWS[scenario_id]["closed"]:
            golden.check(
                f"closed.{scenario_id}.{gate_id}",
                float(results[gate_id].measured),
                rtol=1e-6,
                note=(
                    "a gate that failed before D-54 (O-13) or D-75 (O-19) and passes since; pinned "
                    "so that a regression is caught as a moved value before it is caught as a red row"
                ),
            )
