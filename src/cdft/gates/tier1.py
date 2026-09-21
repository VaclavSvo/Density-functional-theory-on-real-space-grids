"""Tier 1 -- exact physical limits: each gate compares against a code-independent closed form."""

from __future__ import annotations

import math

import torch

from contract import (
    BoundaryMode,
    DomainMode,
    EigenConfig,
    GateKind,
    GateResult,
    GridConfig,
    NumericsConfig,
    RunArtifact,
)

from ..eigen.chefsi import ChebyshevFilteredSubspace
from ..grid import UniformGrid
from ..operators.external import external_potential
from ..operators.hamiltonian import LocalHamiltonian
from .base import ArtifactGate, ComponentGate, GateSpec, artifact_device, artifact_grid, make_result, skipped

__all__ = [
    "HarmonicOscillatorGate",
    "ParticleInBoxGate",
    "GridIsotropyGate",
    "ChargeNormalisationGate",
    "KatoCuspGate",
    "RadialReferenceGate",
    "NonInteractingLimitGate",
    "CuspWeightQuadratureGate",
    "TwoCentreReferenceGate",
    "KineticScalingGate",
    "VirialTheoremGate",
    "LiebOxfordGate",
]


class HarmonicOscillatorGate(ArtifactGate):
    """G1.2 -- lowest harmonic-well eigenvalues against ``(n + 3/2) omega``; 1e-6 Ha (analytic).

    Degeneracies included, so an accidentally anisotropic operator fails too.
    """

    spec = GateSpec(
        gate_id="G1.2",
        name="3D harmonic oscillator spectrum",
        threshold=1.0e-6,
        kind=GateKind.DERIVED,
        citation="analytic",
        units="Ha",
        min_states=10,
    )

    def __init__(self, n_levels: int = 10) -> None:
        """Configure how many of the lowest eigenvalues are compared."""
        self.n_levels = n_levels

    def applicable(self, artifact: RunArtifact) -> bool:
        """Applies to any run whose external potential is the harmonic well."""
        return artifact.result is not None and artifact.scenario.external.kind.value == "harmonic"

    def evaluate(self, artifact: RunArtifact) -> GateResult:
        """Compare the computed spectrum with ``(n + 3/2) omega``, including degeneracies."""
        assert artifact.result is not None
        omega = artifact.scenario.external.omega
        # ``.cpu()``: the reference levels are a CPU tensor.
        computed = artifact.result.eigenvalues[0].to(torch.float64).cpu()
        available = min(self.n_levels, computed.numel())
        exact = _harmonic_levels(omega, available)
        deltas = (computed[:available] - exact).abs()
        return make_result(
            self.spec,
            float(deltas.max()),
            {
                "omega": omega,
                "n_levels": available,
                "computed": computed[:available].tolist(),
                "exact": exact.tolist(),
                "per_level_error": deltas.tolist(),
            },
        )


def _harmonic_levels(omega: float, count: int) -> torch.Tensor:
    """Return the lowest ``count`` harmonic-oscillator eigenvalues with their degeneracies."""
    levels: list[float] = []
    shell = 0
    while len(levels) < count:
        degeneracy = (shell + 1) * (shell + 2) // 2
        levels.extend([(shell + 1.5) * omega] * degeneracy)
        shell += 1
    return torch.tensor(levels[:count], dtype=torch.float64)


class ParticleInBoxGate(ArtifactGate):
    """G1.3 -- cubic-well eigenvalues against ``pi^2 (nx^2+ny^2+nz^2)/(2 L^2)``; 1e-5 relative.

    The only scenario that isolates the boundary treatment, and the reason
    :class:`~contract.BoundaryMode` exists: a zero-filled halo costs the stencil its nominal rate.
    """

    spec = GateSpec(
        gate_id="G1.3",
        name="Particle in a cubic box spectrum",
        threshold=1.0e-5,
        kind=GateKind.DERIVED,
        citation="analytic",
        units="relative",
        min_states=4,
    )

    def __init__(self, n_levels: int = 4) -> None:
        """Configure how many of the lowest eigenvalues are compared."""
        self.n_levels = n_levels

    def applicable(self, artifact: RunArtifact) -> bool:
        """Applies to any run whose external potential is the infinite cubic well."""
        return (
            artifact.result is not None
            and artifact.scenario.external.kind.value == "particle_in_box"
        )

    def evaluate(self, artifact: RunArtifact) -> GateResult:
        """Compare the computed spectrum with the analytic sine-series eigenvalues."""
        assert artifact.result is not None
        length = artifact.scenario.external.box_length
        computed = artifact.result.eigenvalues[0].to(torch.float64).cpu()
        available = min(self.n_levels, computed.numel())
        exact = _box_levels(length, available)
        relative = ((computed[:available] - exact) / exact).abs()
        return make_result(
            self.spec,
            float(relative.max()),
            {
                "box_length": length,
                "computed": computed[:available].tolist(),
                "exact": exact.tolist(),
                "relative_error": relative.tolist(),
                "boundary": artifact.numerics.grid.boundary.value,
            },
        )


def _box_levels(length: float, count: int) -> torch.Tensor:
    """Return the lowest ``count`` eigenvalues of a cubic infinite well of edge ``length``."""
    base = math.pi**2 / (2.0 * length**2)
    triples = sorted(
        nx * nx + ny * ny + nz * nz
        for nx in range(1, 8)
        for ny in range(1, 8)
        for nz in range(1, 8)
    )
    return torch.tensor([base * t for t in triples[:count]], dtype=torch.float64)


class GridIsotropyGate(ArtifactGate):
    """G1.4 -- splitting of the exactly three-fold p-like level; 1e-5 Ha (D2; D6).

    A Cartesian stencil breaks rotational symmetry; the splitting must also fall under refinement.
    """

    spec = GateSpec(
        gate_id="G1.4",
        name="Grid isotropy (p-level splitting)",
        threshold=1.0e-5,
        kind=GateKind.DERIVED,
        citation="D2; D6",
        units="Ha",
        min_states=4,
    )

    def applicable(self, artifact: RunArtifact) -> bool:
        """Applies to the harmonic well, whose first excited shell is exactly three-fold."""
        return (
            artifact.result is not None
            and artifact.scenario.external.kind.value == "harmonic"
            and artifact.result.eigenvalues.shape[-1] >= 4
        )

    def evaluate(self, artifact: RunArtifact) -> GateResult:
        """Measure the spread of the three states of the first excited shell."""
        assert artifact.result is not None
        triplet = artifact.result.eigenvalues[0, 1:4].to(torch.float64)
        splitting = float(triplet.max() - triplet.min())
        return make_result(
            self.spec,
            splitting,
            {"triplet": triplet.tolist(), "spacing": artifact.numerics.grid.spacing},
        )


class ChargeNormalisationGate(ArtifactGate):
    """G1.5 -- integrated density against the electron count; 1e-12 electrons.

    The quadrature of a normalised orbital block is exact to round-off, hence the tight threshold.
    SKIPs when the run recorded no charge error.
    """

    spec = GateSpec(
        gate_id="G1.5",
        name="Charge normalisation",
        threshold=1.0e-12,
        kind=GateKind.EXACT,
        units="electrons",
    )

    def evaluate(self, artifact: RunArtifact) -> GateResult:
        """Read the charge error the solve recorded."""
        value = artifact.measurements.get("charge_error")
        if value is None:
            return skipped(self.spec, "the run recorded no charge normalisation measurement")
        return make_result(self.spec, float(value))


class CuspWeightQuadratureGate(ArtifactGate):
    """G1.13 -- quadrature of the cusp weight ``f^2`` against its closed form; 1e-8 relative.

    Closed forms: ``pi / Z^3`` for one centre, prolate-spheroidal for two equal charges; the tail
    outside the box is subtracted first (D-55 item 6). Other geometries SKIP. Threshold is what a
    cell-by-cell quadrature delivers; the plain ``h^3`` rule fails it, known-open against A-9.
    """

    spec = GateSpec(
        gate_id="G1.13",
        name="Cusp-weight quadrature",
        threshold=1.0e-8,
        kind=GateKind.EXACT,
        citation="closed-form integrals of exp(-2 Z r) and exp(-2 Z (s_a + s_b)); A-9",
        units="relative",
    )

    def applicable(self, artifact: RunArtifact) -> bool:
        """Only a cusp-factorised run carries a weight to integrate."""
        return artifact.result is not None and bool(artifact.measurements.get("cusp_factorisation"))

    def evaluate(self, artifact: RunArtifact) -> GateResult:
        """Read the relative quadrature error the solve recorded, or say why there is none."""
        value = artifact.measurements.get("cusp_weight_quadrature_error")
        if value is None:
            reason = artifact.measurements.get(
                "cusp_weight_quadrature_note", "the run recorded no cusp-weight quadrature measurement"
            )
            return skipped(self.spec, str(reason))
        return make_result(
            self.spec,
            float(value),
            detail={
                "exact_integral": artifact.measurements.get("cusp_weight_integral_exact"),
                "grid_integral": artifact.measurements.get("cusp_weight_integral_grid"),
                "density_amplitude_ratio": artifact.measurements.get("density_amplitude_ratio"),
            },
        )


class KatoCuspGate(ArtifactGate):
    """G1.11 -- fitted ``d(ln n)/dr`` at the nucleus against Kato's ``-2Z``; 0.10 relative (D-32).

    ``ln n`` is fitted over a shell from half a spacing out -- the on-site point carries the
    cell-averaged potential -- to three spacings, before the exponential stops dominating. SKIPs
    with fewer than 8 points in the window, and for rung >= 2, where a diverging ``v_xc`` makes
    ``-2Z`` false for the converged density (D-52 item 3; D-55).
    """

    spec = GateSpec(
        gate_id="G1.11",
        name="Kato cusp condition",
        threshold=0.10,
        kind=GateKind.EMPIRICAL,
        citation="Kato cusp condition; D-32",
        units="relative deviation of d(ln n)/dr from -2Z",
    )

    def applicable(self, artifact: RunArtifact) -> bool:
        """Applies to all-electron runs with a bare Coulomb nucleus and a stored density."""
        return (
            artifact.result is not None
            and artifact.result.density is not None
            and artifact.scenario.external.kind.value == "nuclear_coulomb"
            and artifact.scenario.structure.n_atoms > 0
        )

    def evaluate(self, artifact: RunArtifact) -> GateResult:
        """Fit d(ln n)/dr near the first nucleus and compare with -2Z."""
        assert artifact.result is not None
        from ..structure import nuclear_charges, positions_tensor

        from ..scf.solve import is_interacting

        interacting = is_interacting(artifact.scenario)
        # The grid the run recorded, not one re-derived: a derived lattice is anchored on the first
        # nucleus (D-64), so a rebuild would read the density half a cell from where it lives.
        grid = artifact_grid(artifact, artifact_device(artifact))
        charges = nuclear_charges(artifact.scenario.structure, artifact.scenario.external)
        positions = positions_tensor(artifact.scenario.structure, device=grid.device)
        density = artifact.result.density
        total = density.sum(dim=0) if density.dim() > 1 else density

        radius = (grid.points() - positions[0]).norm(dim=-1)
        h = grid.spacing

        # A window that reaches towards a neighbouring nucleus fits the bonding density instead of
        # the cusp. A fifth of the nearest internuclear distance keeps that below the tolerance.
        outer = 3.0 * h
        if len(positions) > 1:
            separations = (positions[1:] - positions[0]).norm(dim=-1)
            nearest = float(separations.min())
            outer = min(outer, 0.2 * nearest)

        window = (radius > 0.5 * h) & (radius < outer) & (total > 0.0)
        if int(window.sum()) < 8:
            return skipped(
                self.spec,
                f"only {int(window.sum())} grid points lie in the usable fit window "
                f"({0.5 * h:.3g} to {outer:.3g} bohr at h = {h:g}), which needs at least 8. The "
                f"outer edge is capped at a fifth of the nearest internuclear distance so the fit "
                f"does not read the neighbouring density instead of the cusp; a finer spacing "
                f"would make this measurable",
            )

        r_window = radius[window].to(torch.float64)
        log_n = torch.log(total[window].to(torch.float64))
        # Quadratic fit ln n = a + b r + c r^2; the cusp is b. A linear fit reads the curvature of a
        # self-consistent density as slope (D-55); for a bare one-electron atom both agree.
        # The quadratic needs four distinct radii, else it fits the lattice shells rather than the
        # density, so a window capped by a neighbour falls back to linear and the result says so.
        # Shells are counted at a tenth of a spacing: a nucleus a few thousandths of a bohr off the
        # lattice (O-23) splits each shell into a near-collinear cluster of radii.
        distinct_radii = int(torch.unique(torch.round(r_window / (0.1 * h))).numel())
        fit_degree = 2 if distinct_radii >= 4 else 1
        offsets = artifact.measurements.get("nucleus_offgrid_bohr", []) if artifact.measurements else []
        nucleus_offset = float(offsets[0]) if offsets else 0.0
        columns = [torch.ones_like(r_window), r_window]
        if fit_degree == 2:
            columns.append(r_window * r_window)
        design = torch.stack(columns, dim=1)
        coefficients = torch.linalg.lstsq(design, log_n.unsqueeze(1)).solution.squeeze(1)
        slope = float(coefficients[1])
        curvature = float(coefficients[2]) if fit_degree == 2 else float("nan")
        exact = -2.0 * charges[0]
        if interacting and artifact.scenario.xc.rung.value >= 2:
            return skipped(
                self.spec,
                f"GGA level ({artifact.scenario.xc.name}): the exact Kohn-Sham density of a GGA does "
                f"not satisfy d(ln n)/dr = -2Z at the nucleus because v_xc diverges there (D-52 item 3, "
                f"D-55); measured slope {slope:.6f} against the bare -2Z = {exact:g} (relative "
                f"deviation {abs(slope - exact) / abs(exact):.3e}), recorded and not asserted",
            )
        return make_result(
            self.spec,
            abs(slope - exact) / abs(exact),
            {
                "measured_slope": slope,
                "fitted_curvature": curvature,
                "fit_degree": fit_degree,
                "distinct_radii_in_window": distinct_radii,
                "nucleus_offgrid_bohr": nucleus_offset,
                "exact_slope": exact,
                "charge": charges[0],
                "n_points_in_window": int(window.sum()),
                "window_bohr": [0.5 * h, outer],
                "window_capped_by_geometry": outer < 3.0 * h,
                "spacing": h,
            },
        )


class NonInteractingLimitGate(ComponentGate):
    """G1.6 -- iterative eigenvalues against a dense diagonalisation of the operator; 1e-10 Ha.

    The matrix-free ``apply`` is the only description of the Hamiltonian, so nothing else confirms
    that the operator being iterated is the one intended. The SCF one-iteration form of this gate is
    not implemented; the ``form`` entry of the result says so.
    """

    spec = GateSpec(
        gate_id="G1.6",
        name="Matrix-free operator vs dense diagonalisation",
        threshold=1.0e-10,
        kind=GateKind.EXACT,
        citation="A2",
        units="Ha",
    )

    def __init__(self, points_per_axis: int = 12, box: float = 8.0, n_states: int = 6) -> None:
        """Configure the small grid on which the dense matrix is affordable."""
        self.points_per_axis = points_per_axis
        self.box = box
        self.n_states = n_states

    def evaluate(self, numerics: NumericsConfig) -> GateResult:
        """Build a small harmonic-well Hamiltonian, densify it, and compare the two solvers."""
        n = self.points_per_axis
        h = self.box / (n - 1)
        config = GridConfig(
            spacing=h,
            fd_order=min(numerics.grid.fd_order, 4),
            fd_gradient_order=min(numerics.grid.fd_gradient_order, 4),
            domain=DomainMode.BOX,
            boundary=BoundaryMode.ZERO,
            box_lengths=(self.box, self.box, self.box),
            use_double_grid=False,
            fourier_filter_projectors=False,
        )
        grid = UniformGrid.from_config(config)
        from contract import AtomicStructure, ExternalPotentialKind, ExternalPotentialSpec

        spec = ExternalPotentialSpec(kind=ExternalPotentialKind.HARMONIC, omega=1.0)
        v_ext = external_potential(spec, AtomicStructure(), grid)
        hamiltonian = LocalHamiltonian(grid, v_ext)

        identity = torch.eye(grid.n_points, dtype=torch.float64, device=grid.device)
        dense = hamiltonian.apply(identity)
        dense = 0.5 * (dense + dense.transpose(-1, -2))
        dense_evals = torch.linalg.eigvalsh(dense)[: self.n_states]

        solver = ChebyshevFilteredSubspace(
            grid,
            EigenConfig(
                n_extra_states=6,
                chebyshev_degree=numerics.eigen.chebyshev_degree,
                max_iterations=300,
                residual_tol=1.0e-11,
            ),
        )
        result = solver.solve(
            hamiltonian, self.n_states, generator=torch.Generator(device="cpu").manual_seed(0)
        )
        delta = (result.eigenvalues - dense_evals).abs()
        return make_result(
            self.spec,
            float(delta.max()),
            {
                "n_points": grid.n_points,
                "dense": dense_evals.tolist(),
                "iterative": result.eigenvalues.tolist(),
                "iterations": result.n_iterations,
                "form": "Increment 1 form: operator identity, not the SCF one-iteration form",
            },
        )


class RadialReferenceGate(ArtifactGate):
    """G4.8 -- the 3-D solver against an independent radial solution; 1.5936e-3 Ha (chemical).

    The oracle separates the physics rather than discretising it in three dimensions, resolves the
    cusp on a logarithmic grid -- so it checks the transform of D-35 without using it -- and
    diagonalises directly. SKIPs if the oracle is unavailable, declines or fails to converge (D-05).
    """

    spec = GateSpec(
        gate_id="G4.8",
        name="Radial reference agreement",
        threshold=1.5936e-3,
        kind=GateKind.DERIVED,
        citation="separation of variables; D-05 oracle policy",
        units="Ha",
    )

    def __init__(self, n_states: int = 1) -> None:
        """Configure how many eigenvalues are compared."""
        self.n_states = n_states

    def applicable(self, artifact: RunArtifact) -> bool:
        """Applies where the external potential is spherically symmetric about a single centre."""
        external = artifact.scenario.external
        if artifact.result is None:
            return False
        if external.kind.value in ("harmonic", "gaussian_well"):
            return artifact.scenario.structure.n_atoms == 0
        if external.kind.value in ("nuclear_coulomb", "soft_coulomb"):
            return artifact.scenario.structure.n_atoms == 1
        return False

    def _potential(self, artifact: RunArtifact):
        """Return ``v(r)`` for this scenario as a vectorised callable."""
        import numpy as np

        from ..structure import nuclear_charges

        external = artifact.scenario.external
        kind = external.kind.value
        if kind == "nuclear_coulomb":
            charge = nuclear_charges(artifact.scenario.structure, external)[0]
            return lambda r: -charge / r
        if kind == "soft_coulomb":
            charge = nuclear_charges(artifact.scenario.structure, external)[0]
            softening = external.softening
            return lambda r: -charge / np.sqrt(r**2 + softening**2)
        if kind == "harmonic":
            omega = external.omega
            return lambda r: 0.5 * omega**2 * r**2
        depth, width = external.depth, external.width
        return lambda r: -depth * np.exp(-(r**2) / (2.0 * width**2))

    def evaluate(self, artifact: RunArtifact) -> GateResult:
        """Solve the same physics radially and compare eigenvalues, or the total if interacting."""
        assert artifact.result is not None
        try:
            from ..reference.radial import RadialGrid, solve_radial
        except ImportError as exc:  # pragma: no cover - scipy is a hard dependency
            return skipped(self.spec, f"the radial oracle is unavailable: {exc}")
        from ..scf.solve import is_interacting

        if is_interacting(artifact.scenario):
            return self._evaluate_interacting(artifact)

        computed = artifact.result.eigenvalues[0]
        available = min(self.n_states, int(computed.shape[-1]))
        mesh = RadialGrid()
        reference = solve_radial(self._potential(artifact), n_states=available, grid=mesh)

        got = torch.as_tensor(computed[:available], dtype=torch.float64).cpu()
        want = torch.as_tensor(reference.eigenvalues[:available], dtype=torch.float64)
        deltas = (got - want).abs()
        return make_result(
            self.spec,
            float(deltas.max()),
            {
                "three_dimensional": got.tolist(),
                "radial_reference": want.tolist(),
                "per_state_difference": deltas.tolist(),
                "radial_points": mesh.n_points,
                "radial_r_min": mesh.r_min,
                "radial_points_inside_1_bohr": mesh.points_inside(1.0),
                "cusp_factorisation": artifact.measurements.get("cusp_factorisation"),
                "note": (
                    "the radial solver uses a logarithmic grid and resolves the cusp directly, so "
                    "this comparison does not depend on the transform of D-35"
                ),
            },
        )



    def _evaluate_interacting(self, artifact: RunArtifact) -> GateResult:
        """Compare the 3-D total energy with the self-consistent radial oracle.

        The radial atom uses the same functional object, converged to 1e-10 Ha; the oracle
        reproduces NIST SRD 141 to 4e-7 Ha at LDA. Threshold as for the bare path. SKIPs for
        spin-polarised runs and beyond the 1s shell.
        """
        from ..reference.radial_ks import solve_radial_ks
        from ..structure import nuclear_charges
        from ..xc.dispatch import functional_from_spec

        assert artifact.result is not None
        charge = nuclear_charges(artifact.scenario.structure, artifact.scenario.external)[0]
        n_electrons = float(artifact.measurements.get("n_electrons", float("nan")))
        if artifact.scenario.electrons.spin_polarised or n_electrons > 2.0 + 1.0e-9:
            return skipped(self.spec, "the radial oracle covers spin-restricted 1s-shell atoms only in this increment")
        try:
            functional = functional_from_spec(artifact.scenario.xc)
            reference = solve_radial_ks(charge, n_electrons, functional)
        except NotImplementedError as exc:
            return skipped(self.spec, f"the radial oracle declined: {exc}")
        if not reference.converged:
            return skipped(self.spec, f"the radial oracle did not converge in {reference.iterations} iterations")
        got = float(artifact.result.energies.total)
        want = float(reference.total)
        return make_result(
            self.spec,
            abs(got - want),
            {
                "three_dimensional_total": got,
                "radial_total": want,
                "three_dimensional_eigenvalue": float(artifact.result.eigenvalues[0, 0]),
                "radial_eigenvalue": float(reference.eigenvalues[0]),
                "radial_decomposition": {
                    "kinetic": reference.kinetic, "external": reference.external,
                    "hartree": reference.hartree, "xc": reference.xc,
                },
                "radial_iterations": reference.iterations,
                "functional": functional.name,
                "note": (
                    "self-consistent radial Kohn-Sham on a logarithmic grid with the same functional "
                    "object; independent of the 3-D discretisation, the cusp factorisation and the "
                    "sphere quadrature (D-36 extended)"
                ),
            },
        )


class TwoCentreReferenceGate(ArtifactGate):
    """G4.9 -- the 3-D solver against a prolate-spheroidal two-centre solution; 2.5e-3 Ha (D-41).

    :mod:`cdft.reference.two_centre` separates the problem, handles both cusps analytically -- so
    the transform of D-35 is not used to check itself -- and closes the eigenvalue by a Wronskian
    root find; it reproduces Madsen and Peek (1971) to 2.2e-9 Ha at R = 2. The threshold sits above
    the known multi-centre error (1.0e-3 Ha at R = 2, 2.0e-3 at R = 1.4; D-42, O-13) and tightens
    with it. SKIPs when the oracle declines the geometry.
    """

    spec = GateSpec(
        gate_id="G4.9",
        name="Two-centre reference agreement",
        threshold=2.5e-3,
        kind=GateKind.DERIVED,
        citation="prolate-spheroidal separation; Madsen and Peek, Atomic Data 2, 171 (1971); D-41",
        units="Ha",
    )

    def applicable(self, artifact: RunArtifact) -> bool:
        """Applies to a two-centre bare-nucleus scenario with a computed spectrum.

        The oracle is one-electron, so on a self-consistent run the comparison would report the
        interaction energy as solver error; interacting two-centre scenarios go to G4.7 instead.
        """
        from ..scf.solve import is_interacting

        if artifact.result is None:
            return False
        if artifact.scenario.external.kind.value != "nuclear_coulomb":
            return False
        if is_interacting(artifact.scenario):
            return False
        return artifact.scenario.structure.n_atoms == 2

    def evaluate(self, artifact: RunArtifact) -> GateResult:
        """Solve the same two-centre problem by separation and compare the ground eigenvalue."""
        assert artifact.result is not None
        try:
            from ..reference.two_centre import solve_two_centre_extrapolated
        except ImportError as exc:  # pragma: no cover - scipy is a hard dependency
            return skipped(self.spec, f"the two-centre oracle is unavailable: {exc}")

        from ..structure import nuclear_charges, positions_tensor

        structure = artifact.scenario.structure
        charges = nuclear_charges(structure, artifact.scenario.external)
        positions = positions_tensor(structure)
        bond_length = float(torch.linalg.norm(positions[0] - positions[1]))
        pair = (float(charges[0]), float(charges[1]))

        try:
            reference = solve_two_centre_extrapolated(bond_length=bond_length, charges=pair)
        except Exception as exc:  # the oracle refuses rather than returning a bad number
            return skipped(
                self.spec,
                f"the two-centre oracle declined this geometry (R = {bond_length:g} bohr, "
                f"Z = {pair}): {type(exc).__name__}: {exc}. A refused extrapolation is the oracle "
                f"working as designed -- it reports no number rather than an unconverged one.",
            )

        computed = float(artifact.result.eigenvalues[0][0])
        want = float(reference.electronic_energy)
        return make_result(
            self.spec,
            abs(computed - want),
            {
                "three_dimensional": computed,
                "two_centre_reference": want,
                "difference": computed - want,
                "bond_length": bond_length,
                "charges": list(pair),
                "oracle_residual": float(reference.residual),
                "oracle_convergence_order": float(reference.convergence_order),
                "oracle_extrapolation_shift": float(reference.extrapolation_shift),
                "chemical_accuracy": 1.5936e-3,
                "note": (
                    "the reference is the ELECTRONIC energy; the published total for H2+ at R = 2, "
                    "-0.6026342144949 Ha, includes the 1/R nuclear repulsion this solver does not "
                    "carry. Comparing against the total is the easiest way to manufacture a 0.5 Ha "
                    "discrepancy out of a correct run."
                ),
            },
        )


class KineticScalingGate(ArtifactGate):
    """G1.12 -- ``T[psi_lambda] = lambda^2 T[psi]`` under coordinate scaling; 1e-12 relative.

    A statement about the operator alone, so it needs no reference value. Sampling ``psi`` at ``h``
    and ``psi_lambda`` at ``h / lambda`` gives an identical array, so the relation carries no
    truncation error and the threshold is round-off; it catches a wrong power of ``h`` in ``T``.
    """

    spec = GateSpec(
        gate_id="G1.12",
        name="Kinetic energy coordinate scaling",
        threshold=1.0e-12,
        kind=GateKind.EXACT,
        citation="uniform coordinate scaling; Levy and Perdew, PRA 32, 2010 (1985)",
        units="relative",
    )

    #: Scaling factors on either side of unity: a bug invariant under one rarely survives its
    #: reciprocal.
    FACTORS = (0.5, 2.0, 3.0)

    #: Points per edge of the probe grid; small, since the identity is exact at any size.
    N_POINTS = 25

    def applicable(self, artifact: RunArtifact) -> bool:
        """Applies wherever a run exists; the check is on the operator, not on the system."""
        return artifact.result is not None

    def evaluate(self, artifact: RunArtifact) -> GateResult:
        """Scale the spacing with the sampled array held fixed; ``T`` must scale as ``lambda^2``."""
        from ..grid import UniformGrid as _Grid

        order = artifact.numerics.grid.fd_order
        base_spacing = 0.25
        n = self.N_POINTS

        def kinetic_at(spacing: float) -> float:
            """Return ``T`` for the fixed sampled amplitude on a grid of this spacing."""
            edge = (n - 1) * spacing
            config = GridConfig(
                spacing=spacing,
                fd_order=order,
                domain=DomainMode.BOX,
                boundary=BoundaryMode.ZERO,
                box_lengths=(edge, edge, edge),
                use_double_grid=False,
                fourier_filter_projectors=False,
            )
            grid = _Grid.from_config(config, structure=None, dtype=torch.float64)
            points = grid.points()
            centre = points.mean(dim=0)
            # In grid-index units, not bohr, so the array is identical at every spacing; that
            # identity is what makes the relation exact.
            index_r2 = (((points - centre) / spacing) ** 2).sum(dim=-1)
            psi = torch.exp(-index_r2 / (2.0 * (n / 6.0) ** 2))
            psi = psi / torch.sqrt(grid.integrate(psi * psi))
            return float(-0.5 * grid.integrate(psi * grid.laplacian(psi)))

        reference = kinetic_at(base_spacing)
        deviations: dict[str, float] = {}
        values: dict[str, float] = {}
        worst = 0.0
        for factor in self.FACTORS:
            scaled = kinetic_at(base_spacing / factor)
            expected = factor**2 * reference
            relative = abs(scaled - expected) / abs(expected)
            deviations[f"lambda_{factor:g}"] = relative
            values[f"lambda_{factor:g}"] = scaled
            worst = max(worst, relative)

        return make_result(
            self.spec,
            worst,
            {
                "base_kinetic": reference,
                "base_spacing": base_spacing,
                "scaled_kinetic": values,
                "relative_deviation": deviations,
                "fd_order": order,
                "points_per_edge": n,
                "note": (
                    "psi sampled at spacing h and psi_lambda sampled at spacing h/lambda are the "
                    "same array, so the relation carries no interpolation or truncation error and "
                    "the threshold is round-off rather than a convergence argument"
                ),
            },
        )


class VirialTheoremGate(ArtifactGate):
    """G1.7 -- ``-V / T`` against 2 for a Coulombic Hamiltonian; 1e-3 dimensionless (empirical).

    Kinetic and potential terms are assembled independently of each other, so this catches a defect
    that scales both and that the total energy cannot see. Applies to one bare centre only: the
    theorem needs a stationary geometry, and a pseudopotential is not homogeneous of degree -1.
    """

    spec = GateSpec(
        gate_id="G1.7",
        name="Virial theorem",
        threshold=1.0e-3,
        kind=GateKind.EMPIRICAL,
        citation="virial theorem for a degree -1 homogeneous potential",
        units="dimensionless",
    )

    def applicable(self, artifact: RunArtifact) -> bool:
        """Applies to a single all-electron Coulomb centre, where every geometry is stationary."""
        if artifact.result is None or artifact.result.energies is None:
            return False
        if artifact.scenario.external.kind.value != "nuclear_coulomb":
            return False
        if artifact.scenario.pseudo is not None:
            return False
        return artifact.scenario.structure.n_atoms == 1

    def evaluate(self, artifact: RunArtifact) -> GateResult:
        """Measure the scaling virial from the recorded energy decomposition.

        Non-interacting: ``|-E_ext / T - 2|``. Interacting: Levy and Perdew, PRA 32, 2010 (1985)
        give ``2 T_s + E_ext + E_H - integral n r.grad v_xc = 0``, the last term being the recorded
        ``virial_xc_scaling_term``, and the gate reports ``|2T + E_ext + E_H + S_xc| / |T|``. A GGA
        also needs the gradient dependence of ``v_xc`` under scaling, and SKIPs until written.
        """
        assert artifact.result is not None
        energies = artifact.result.energies
        kinetic = float(energies.kinetic)
        potential = float(energies.external)
        if kinetic == 0.0:
            return skipped(
                self.spec,
                "the kinetic energy is exactly zero, so -V/T is undefined; this is a degenerate "
                "run rather than a virial violation",
            )
        interacting = energies.hartree != 0.0 or energies.xc != 0.0
        if not interacting:
            ratio = -potential / kinetic
            return make_result(
                self.spec,
                abs(ratio - 2.0),
                {
                    "kinetic": kinetic,
                    "potential": potential,
                    "minus_v_over_t": ratio,
                    "expected": 2.0,
                    "total": float(energies.total),
                    "external_consistency": artifact.measurements.get("external_consistency"),
                    "form": "non-interacting: -E_ext / T = 2",
                },
            )
        scaling = artifact.measurements.get("virial_xc_scaling_term")
        if scaling is None:
            return skipped(
                self.spec,
                f"the run recorded no exchange-correlation scaling term (functional "
                f"{artifact.measurements.get('functional')}); the interacting virial is implemented "
                f"for local functionals only in this increment",
            )
        virial = 2.0 * kinetic + potential + float(energies.hartree) + float(scaling)
        return make_result(
            self.spec,
            abs(virial) / abs(kinetic),
            {
                "kinetic": kinetic,
                "external": potential,
                "hartree": float(energies.hartree),
                "xc": float(energies.xc),
                "xc_scaling_term": float(scaling),
                "residual_2T_plus_V": virial,
                "form": "interacting: |2T + E_ext + E_H - int n r.grad v_xc| / T (Levy-Perdew scaling)",
                "external_consistency": artifact.measurements.get("external_consistency"),
            },
        )


class LiebOxfordGate(ArtifactGate):
    """G1.8 -- ``max(0, -E_xc - C_LO integral n^(4/3))`` on a converged density; 0 Ha [A16].

    The integrated form, not the pointwise one: B88 [A5] violates the local bound at large reduced
    gradient while every real system stays inside the integrated one. The Chan--Handy constant is
    reported alongside. The integral uses the same cusp-aware quadrature as ``E_xc``.
    """

    spec = GateSpec(
        gate_id="G1.8",
        name="Lieb-Oxford bound",
        threshold=0.0,
        kind=GateKind.EXACT,
        citation="Lieb and Oxford, Int. J. Quantum Chem. 19, 427 (1981); A16",
        units="Ha",
    )

    C_LO = 2.273
    C_CHAN_HANDY = 1.6358

    def applicable(self, artifact: RunArtifact) -> bool:
        """Applies to a run with an exchange--correlation energy and the recorded integral."""
        return (
            artifact.result is not None
            and artifact.result.energies.xc != 0.0
            and "lieb_oxford_integral" in artifact.measurements
        )

    def evaluate(self, artifact: RunArtifact) -> GateResult:
        """Compare ``-E_xc`` with ``C_LO integral n^(4/3)``."""
        assert artifact.result is not None
        e_xc = float(artifact.result.energies.xc)
        integral = float(artifact.measurements["lieb_oxford_integral"])
        bound = self.C_LO * integral
        return make_result(
            self.spec,
            max(0.0, -e_xc - bound),
            {
                "e_xc": e_xc,
                "integral_n_4_3": integral,
                "lieb_oxford_bound": -bound,
                "ratio_to_bound": (-e_xc) / bound if bound else float("nan"),
                "chan_handy_violation": max(0.0, -e_xc - self.C_CHAN_HANDY * integral),
            },
        )
