"""The non-interacting solve: one diagonalisation of ``T + v_ext``, no self-consistency.

A complete solver for three reference families: the analytic model systems, the one-electron
all-electron systems of Phase 1 and the non-interacting limit of G1.6. The SCF loop wraps it rather
than replacing it. Rationale, including why ``E_ext`` is the band-energy complement:
``docs/03_METHOD.md (Part E)``.
"""

from __future__ import annotations

import time
import uuid

import torch

from contract import (
    EigenResult,
    ExternalPotentialKind,
    EnergyBreakdown,
    NumericsConfig,
    RunArtifact,
    RunStatus,
    ScenarioSpec,
    SCFResult,
    SCFTrajectory,
)

from ..eigen.chefsi import ChebyshevFilteredSubspace, graphed_filter_for
from ..eigen.lobpcg import LOBPCG
from ..eigen.rayleigh_ritz import orthonormality_error, self_adjointness_error
from ..grid import UniformGrid, grid_config_for_scenario, lattice_anchor
from ..io.provenance import capture
from ..operators.external import external_potential
from ..operators import geometry_cache
from ..operators.cusp import CuspFactor
from ..operators.fused import kernel_record
from ..operators.hamiltonian import CuspFactoredHamiltonian, LocalHamiltonian
from ..operators.quadrature import CuspQuadrature
from ..precision import (
    configure_device,
    peak_memory_bytes,
    reset_peak_memory,
    resolve_device,
    seeded_randn,
)
from ..structure import (
    electron_count,
    nuclear_charges,
    nuclear_repulsion,
    positions_tensor,
)
from .occupations import aufbau_occupations

__all__ = ["solve_noninteracting", "build_hamiltonian"]


def build_hamiltonian(
    scenario: ScenarioSpec,
    numerics: NumericsConfig,
    device: torch.device | None = None,
    *,
    derive_grid: bool = True,
) -> tuple[UniformGrid, LocalHamiltonian, torch.Tensor]:
    """Build the grid, the external potential and the fixed Hamiltonian for a scenario.

    A triple rather than solver-internal state, so a gate can rebuild the exact operator a run used
    (G0.4, G2.5 need the operator, not a summary). Factor and quadrature come from
    :mod:`cdft.operators.geometry_cache`; whether they were cached is in
    ``grid.geometry_cache_record``.
    """
    dev = device or resolve_device(numerics.device)
    grid_config, overrides = grid_config_for_scenario(
        scenario, numerics.grid, derive_grid=derive_grid
    )
    grid = UniformGrid.from_config(
        grid_config,
        structure=scenario.structure if scenario.structure.n_atoms else None,
        device=dev,
        dtype=torch.float64,
        anchor=lattice_anchor(scenario, derive_grid=derive_grid),
    )
    grid.resolution_overrides = overrides
    v_ext = external_potential(scenario.external, scenario.structure, grid)

    if _use_cusp_factorisation(scenario):
        charges = nuclear_charges(scenario.structure, scenario.external)
        positions = positions_tensor(scenario.structure, device=grid.device)
        # Geometry-only: served by the cache when an earlier solve built them for the same key.
        cache = geometry_cache.lease(grid, charges, positions)
        factor = cache.part("factor", lambda: CuspFactor(grid, charges, positions))
        # The cusp-aware quadrature (D-54) is the operator's measure, not post-processing, so the
        # density it normalises is the one the Hamiltonian saw.
        quadrature = cache.part("quadrature", lambda: CuspQuadrature(grid, factor))
        hamiltonian = CuspFactoredHamiltonian(grid, factor, quadrature=quadrature)
        grid.geometry_cache_record = cache.record()
        return grid, hamiltonian, v_ext
    grid.geometry_cache_record = geometry_cache.not_applicable_record(
        "plain operator: no cusp factor or quadrature to cache"
    )
    return grid, LocalHamiltonian(grid, v_ext), v_ext


def _use_cusp_factorisation(scenario: ScenarioSpec) -> bool:
    """Whether this scenario is solved in the cusp-transformed representation (D-35).

    True for ``NUCLEAR_COULOMB`` with at least one nucleus and the flag at its default; elsewhere
    the factor would be the identity, so the untransformed operator is used directly.
    """
    if not scenario.external.cusp_factorisation:
        return False
    if scenario.structure.n_atoms == 0:
        return False
    return scenario.external.kind is ExternalPotentialKind.NUCLEAR_COULOMB


def solve_noninteracting(
    scenario: ScenarioSpec,
    numerics: NumericsConfig,
    n_states: int | None = None,
    cross_check: bool | None = None,
    *,
    derive_grid: bool = True,
    device: torch.device | None = None,
) -> RunArtifact:
    """Solve one fixed-potential eigenproblem and return a fully gated artifact.

    ``n_states`` defaults to what the electron count needs; ``cross_check`` also runs LOBPCG and
    records the agreement (G2.5); ``derive_grid`` allows the D-42 Coulomb sizing rule to rewrite
    the spacing and the box, and is ``False`` only in the convergence gates G3.1, G3.2 and G3.5,
    whose job is to vary exactly those two; ``device`` defaults to ``numerics.device``.

    An artifact is always returned -- a failed convergence is ``UNCONVERGED``, an exception
    ``ERROR`` with a message -- so that one bad scenario cannot lose a whole batch.
    """
    started = time.perf_counter()
    run_id = f"run_{uuid.uuid4().hex[:16]}"
    device = device if device is not None else resolve_device(numerics.device)
    measurements: dict[str, object] = {}

    try:
        configure_device(device, numerics.deterministic)
        reset_peak_memory(device)
        grid, hamiltonian, v_ext = build_hamiltonian(
            scenario, numerics, device=device, derive_grid=derive_grid
        )
        n_electrons = electron_count(scenario.structure, scenario.external, scenario.electrons.n_electrons)
        n_spin = scenario.electrons.n_spin
        capacity = 2.0 if n_spin == 1 else 1.0
        needed = max(1, int(-(-n_electrons // capacity)))
        requested = n_states if n_states is not None else needed
        requested = max(requested, needed)

        generator = torch.Generator(device="cpu").manual_seed(numerics.seed)
        # CUDA-graph replay of the filter; None -- the eager filter -- on the CPU.
        graphs = graphed_filter_for(device)
        solver = ChebyshevFilteredSubspace(grid, numerics.eigen, graphed_filter=graphs)
        result: EigenResult = solver.solve(hamiltonian, requested, generator=generator)
        if graphs is not None:
            measurements["cuda_graphs"] = graphs.record()

        orth = orthonormality_error(result.eigenvectors, hamiltonian.measure)
        measurements["orthonormality_error"] = orth
        measurements["eigen_residual_max"] = float(result.residuals.max())
        measurements["eigen_iterations"] = float(result.n_iterations)
        measurements["eigen_eigenvalue_drift"] = float(getattr(solver, "eigenvalue_drift", float("nan")))
        # Which criterion stopped the solver: "converged" alone cannot tell a residual that reached
        # its tolerance from Ritz values that merely stopped moving (G5.5, D-44).
        measurements["eigen_stop_reason"] = str(getattr(solver, "stop_reason", "unrecorded"))
        measurements["divergence_form"] = bool(getattr(hamiltonian, "uses_divergence_form", False))
        # The fused-kernel tier is chosen at the first stencil apply on a device.
        measurements["fused_kernels"] = kernel_record(grid.device)
        measurements["hamiltonian_applications"] = float(hamiltonian.apply_count)
        measurements["hermiticity_error"] = _hermiticity_error(hamiltonian, grid, generator)
        measurements["grid"] = grid.describe()
        measurements["geometry_cache"] = grid.geometry_cache_record
        measurements["grid_overrides"] = getattr(grid, "resolution_overrides", {})
        measurements["n_electrons"] = n_electrons
        measurements["nucleus_offgrid_bohr"] = _nucleus_grid_offsets(scenario, grid)

        do_cross = numerics.eigen.cross_check if cross_check is None else cross_check
        if do_cross:
            reference = LOBPCG(grid, numerics.eigen)
            second = reference.solve(hamiltonian, requested, generator=torch.Generator(device="cpu").manual_seed(numerics.seed + 1))
            delta = (result.eigenvalues - second.eigenvalues).abs()
            measurements["two_solver_max_delta"] = float(delta.max())
            measurements["two_solver_converged"] = bool(second.converged)
            measurements["lobpcg_iterations"] = float(second.n_iterations)

        eigenvalues = result.eigenvalues.reshape(1, -1).repeat(n_spin, 1)
        occupations = aufbau_occupations(
            eigenvalues, n_electrons, n_spin=n_spin, magnetisation=scenario.electrons.magnetisation
        )

        orbitals = result.eigenvectors.reshape(1, requested, grid.n_points).repeat(n_spin, 1, 1)
        transformed = isinstance(hamiltonian, CuspFactoredHamiltonian)
        measurements["cusp_factorisation"] = transformed
        if transformed:
            # The eigenvectors are phi, not psi: the density is f^2 |phi|^2 and the kinetic energy
            # comes from the factor's gradient form (cdft.operators.cusp).
            density = torch.stack(
                [hamiltonian.factor.density(orbitals[s], occupations[s]) for s in range(n_spin)]
            )
            measurements["weight_dynamic_range"] = hamiltonian.measure.dynamic_range
            quadrature = hamiltonian.quadrature
            rho = (occupations[..., None] * orbitals.pow(2)).sum(dim=(0, 1))
            # G1.13: the rule the solver normalises in (cusp-aware mass weights, D-54) against the
            # closed form of integral f^2. The plain h^3 value, which misses by ~15 % on derived
            # atomic grids (A-9), stays recorded beside it.
            exact_weight = hamiltonian.factor.weight_integral_exact()
            plain_weight = hamiltonian.factor.weight_integral_grid()
            grid_weight = float(quadrature.mass_weights.sum())
            measurements["cusp_weight_integral_grid"] = grid_weight
            measurements["cusp_weight_integral_plain"] = plain_weight
            measurements["cusp_quadrature"] = quadrature.describe()
            if exact_weight is None:
                measurements["cusp_weight_quadrature_note"] = (
                    "no closed form for integral f^2 with this geometry (only one centre, or two "
                    "centres of equal charge, are covered); the quadrature error is unmeasured here"
                )
            else:
                # The closed form is over all space, the grid over its box, so the tail beyond the
                # box is subtracted and G1.13 measures the quadrature and not the box.
                outside = hamiltonian.factor.weight_integral_outside_box()
                in_box = exact_weight - outside
                measurements["cusp_weight_integral_exact"] = exact_weight
                measurements["cusp_weight_integral_outside_box"] = outside
                measurements["cusp_weight_quadrature_error"] = abs(grid_weight - in_box) / exact_weight
                measurements["cusp_weight_quadrature_error_plain"] = abs(plain_weight - in_box) / exact_weight
                # The normalised density is scaled by (in-box exact)/grid against the true density.
                measurements["density_amplitude_ratio"] = in_box / grid_weight
            measurements["self_adjointness_error"] = self_adjointness_error(
                hamiltonian, hamiltonian.measure, n_vectors=16, generator=generator
            )
            # Charge in the solver's own measure (round-off by construction) and in the plain rule
            # (the A-9 defect, kept visible).
            charge_quadrature = float((quadrature.mass_weights * rho).sum())
            measurements["charge_error"] = abs(charge_quadrature - n_electrons)
            measurements["charge_error_plain"] = abs(float(grid.integrate(density.sum(dim=0))) - n_electrons)
            measurements["external_direct"] = hamiltonian.external_energy_direct(rho)
        else:
            density = (occupations[..., None] * orbitals.pow(2)).sum(dim=1)
            measurements["charge_error"] = abs(float(grid.integrate(density.sum(dim=0))) - n_electrons)

        kinetic = float(
            sum(
                hamiltonian.kinetic_energy(orbitals[s], occupations[s])
                for s in range(n_spin)
            )
        )
        ion_ion = nuclear_repulsion(scenario.structure, scenario.external)
        band = float((occupations * eigenvalues).sum())

        # The electronic total is exactly the band energy here, so E_ext is its complement. The
        # direct value beside it is the bounded by-parts integral of D-54 on the cusp-factorised
        # path (no 1/r pole is sampled), the plain sum of n v_ext otherwise.
        if not transformed:
            measurements["external_direct"] = float(grid.integrate(density.sum(dim=0) * v_ext))
        external_direct = measurements["external_direct"]
        external = band - kinetic
        total = kinetic + external + ion_ion
        measurements["band_energy"] = band
        measurements["external_consistency"] = abs(external - external_direct)
        # Which quantity E_ext holds, recorded: the two definitions differ by ~1.6e-1 Ha for H at
        # the derived grid, and the corpus is append-only.
        measurements["external_method"] = "band_energy_complement"
        measurements["external_method_note"] = (
            "E_ext = band - T, exact for a non-interacting solve. external_direct is the "
            "cusp-aware by-parts integral (D-54) and external_consistency their difference, a "
            "measurement of the quadrature; the self-consistent path defines E_ext as the "
            "complement E - T - E_H - E_xc - E_ii (D-52 item 2), also recorded under this key."
        )

        energies = EnergyBreakdown(
            total=total,
            kinetic=kinetic,
            external=external,
            hartree=0.0,
            xc=0.0,
            nonlocal_ps=0.0,
            ion_ion=ion_ion,
            harris_foulkes=band + ion_ion,
        )
        finite = bool(torch.isfinite(density).all() and torch.isfinite(result.eigenvalues).all())
        measurements["all_finite"] = finite

        scf_result = SCFResult(
            converged=result.converged,
            n_iterations=result.n_iterations,
            density=density,
            eigenvalues=eigenvalues,
            occupations=occupations,
            energies=energies,
            trajectory=SCFTrajectory(
                energies=[total],
                residual_norms=[float(result.residuals.max())],
                density_changes=[0.0],
                mixing_events=["none: non-interacting solve, no density mixing"],
            ),
            v_hartree=None,
            v_xc=None,
            orbitals=orbitals if numerics.output.store_orbitals else None,
        )
        status = RunStatus.VALID if result.converged else RunStatus.UNCONVERGED
        peak = _record_peak(measurements, device)
        artifact = RunArtifact(
            run_id=run_id,
            scenario=scenario,
            numerics=numerics,
            provenance=capture(
                numerics, time.perf_counter() - started, device=device, gpu_peak_bytes=peak
            ),
            result=scf_result,
            measurements=measurements,
            status=status,
        )
        return artifact

    except Exception as exc:  # noqa: BLE001 - a physics run must report, not crash the batch
        return RunArtifact(
            run_id=run_id,
            scenario=scenario,
            numerics=numerics,
            provenance=_capture_after_error(numerics, started, device, measurements),
            result=None,
            measurements=measurements,
            status=RunStatus.ERROR,
            error_message=f"{type(exc).__name__}: {exc}",
        )


def _record_peak(measurements: dict[str, object], device: torch.device) -> int | None:
    """Write ``gpu_peak_bytes`` into the measurements on CUDA and return it; ``None`` elsewhere.

    Nothing is written off CUDA: a peak that was not measured is absent, not zero.
    """
    peak = peak_memory_bytes(device)
    if peak is not None:
        measurements["gpu_peak_bytes"] = peak
    return peak


def _capture_after_error(
    numerics: NumericsConfig,
    started: float,
    device: torch.device,
    measurements: dict[str, object],
):
    """Provenance for a run that raised: the device and, where it can be read, the peak so far."""
    try:
        peak = _record_peak(measurements, device)
    except Exception:  # noqa: BLE001 - a broken device must not hide the original error
        peak = None
    return capture(numerics, time.perf_counter() - started, device=device, gpu_peak_bytes=peak)


def _nucleus_grid_offsets(scenario: ScenarioSpec, grid: UniformGrid) -> list[float]:
    """Return each nucleus's distance to the nearest grid point, in bohr.

    Recorded on every all-electron run: the ``-Z/r`` regularisation applies only at a coincident
    point, so an off-point nucleus sits elsewhere on the convergence curve than its neighbours in a
    scan. An offset above about a tenth of a spacing deserves suspicion.
    """
    if scenario.structure.n_atoms == 0:
        return []
    positions = positions_tensor(scenario.structure, device=grid.device, dtype=torch.float64)
    origin = torch.tensor(grid.origin, device=grid.device, dtype=torch.float64)
    fractional = (positions - origin) / grid.spacing
    return (fractional - fractional.round()).mul(grid.spacing).norm(dim=-1).tolist()


def _hermiticity_error(
    hamiltonian: LocalHamiltonian, grid: UniformGrid, generator: torch.Generator, n_vectors: int = 32
) -> float:
    """Return ``max |<phi|H|psi> - <psi|H|phi>|`` over random pairs (G0.4).

    Random rather than smooth probes: a smooth probe has decayed at the domain boundary, which is
    where a masked domain is most likely to be wrong.
    """
    probes = seeded_randn((n_vectors, grid.n_points), generator, device=grid.device, dtype=torch.float64)
    applied = hamiltonian.apply(probes)
    left = (probes @ applied.transpose(-1, -2)) * grid.volume_element
    return float((left - left.transpose(-1, -2)).abs().max())
