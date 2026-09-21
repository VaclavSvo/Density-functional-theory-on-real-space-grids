"""The radial reference oracle against closed forms, and its independence from the 3-D solver.

An oracle that shares code with what it checks is not an oracle, so its imports are inspected.
Marked ``fast`` and ``physics``; a second or two.
"""

from __future__ import annotations

import numpy as np
import pytest

from cdft.reference.radial import RadialGrid, hydrogenic_reference, solve_radial

#: Markers for every test in this module (see ``tests/conftest.py``).
pytestmark = [pytest.mark.fast, pytest.mark.physics]



class TestAgainstClosedForms:
    """The oracle reproduces every spectrum that is known analytically."""

    @pytest.mark.parametrize(
        ("charge", "tolerance"), [(1.0, 1e-10), (2.0, 1e-9), (8.0, 1e-8)]
    )
    def test_hydrogenic_spectrum(self, charge: float, tolerance: float) -> None:
        """``eps_n = -Z^2 / (2 n^2)`` for three principal quantum numbers."""
        result = hydrogenic_reference(charge, n_states=3)
        exact = np.array([-0.5 * charge**2 / n**2 for n in (1, 2, 3)])
        assert np.max(np.abs(result.eigenvalues - exact)) < tolerance

    def test_hydrogenic_p_states(self) -> None:
        """The centrifugal term is right: the 2p level of hydrogen is also -1/8."""
        result = solve_radial(lambda r: -1.0 / r, n_states=1, angular_momentum=1)
        assert float(result.eigenvalues[0]) == pytest.approx(-0.125, abs=1e-9)

    def test_isotropic_harmonic_oscillator(self) -> None:
        """A potential with no singularity: ``eps = (2k + l + 3/2) omega``."""
        omega = 1.0
        # A harmonic well has no cusp, so the tolerance is set by the logarithmic spacing at large
        # r rather than by r_min; measured 1.1e-8 on this grid.
        grid = RadialGrid(r_min=1e-8, r_max=25.0, n_points=1200)
        s_states = solve_radial(lambda r: 0.5 * omega**2 * r**2, n_states=2, grid=grid)
        assert float(s_states.eigenvalues[0]) == pytest.approx(1.5 * omega, abs=1e-7)
        assert float(s_states.eigenvalues[1]) == pytest.approx(3.5 * omega, abs=1e-7)
        p_state = solve_radial(
            lambda r: 0.5 * omega**2 * r**2, n_states=1, angular_momentum=1, grid=grid
        )
        assert float(p_state.eigenvalues[0]) == pytest.approx(2.5 * omega, abs=1e-7)

    def test_radial_function_is_normalised(self) -> None:
        result = hydrogenic_reference(1.0, n_states=1)
        integral = float((result.radial_functions[0] ** 2 * result.radii * result.grid.dx).sum())
        assert integral == pytest.approx(1.0, abs=1e-10)

    def test_radial_function_has_the_right_cusp(self) -> None:
        """``u = r R``, so ``ln(u/r)`` has slope ``-Z`` near the origin: the cusp is resolved."""
        charge = 2.0
        result = hydrogenic_reference(charge, n_states=1)
        radii = result.radii
        window = (radii > 1e-3) & (radii < 1e-2)
        log_r_function = np.log(np.abs(result.radial_functions[0][window] / radii[window]))
        r = radii[window]
        slope = np.polyfit(r, log_r_function, 1)[0]
        assert slope == pytest.approx(-charge, abs=1e-3)


class TestGridBehaviour:
    """What limits the accuracy of the radial grid, measured rather than assumed."""

    def test_inner_cutoff_sets_the_accuracy_not_the_point_count(self) -> None:
        """The error tracks ``r_min``, not the point count, which is why it defaults to 1e-12."""
        # u(r_min) is forced to zero while the true value is of order r_min.
        coarse_cut = hydrogenic_reference(1.0, grid=RadialGrid(r_min=1e-7, n_points=1600))
        more_points = hydrogenic_reference(1.0, grid=RadialGrid(r_min=1e-7, n_points=2400))
        deeper_cut = hydrogenic_reference(1.0, grid=RadialGrid(r_min=1e-12, n_points=1600))

        coarse_error = abs(float(coarse_cut.eigenvalues[0]) + 0.5)
        points_error = abs(float(more_points.eigenvalues[0]) + 0.5)
        deep_error = abs(float(deeper_cut.eigenvalues[0]) + 0.5)

        assert points_error > 0.3 * coarse_error  # tripling the points barely moves it
        assert deep_error < 1e-3 * coarse_error  # lowering the cutoff moves it by orders

    def test_default_grid_resolves_the_core(self) -> None:
        """The default grid puts over a thousand points inside one bohr, resolving the cusp."""
        assert RadialGrid().points_inside(1.0) > 1000

    def test_degenerate_grid_is_refused(self) -> None:
        with pytest.raises(ValueError):
            RadialGrid(r_min=1.0, r_max=0.5)
        with pytest.raises(ValueError):
            RadialGrid(n_points=10)


def test_oracle_does_not_depend_on_what_it_checks() -> None:
    """The oracle imports no grid, cusp transform or eigensolver, so it stays independent."""
    import ast
    import pathlib

    source = pathlib.Path(
        pathlib.Path(__file__).resolve().parents[1], "src", "cdft", "reference", "radial.py"
    ).read_text(encoding="utf-8")
    imported: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)

    forbidden = ("grid", "cusp", "hamiltonian", "chefsi", "lobpcg", "rayleigh")
    offenders = [name for name in imported for bad in forbidden if bad in name]
    assert not offenders, f"the radial oracle must stay independent; it imports {offenders}"
