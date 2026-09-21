"""Tier 0 -- algebraic gates. No physics; each localises a bug to one module."""

from __future__ import annotations

import math

import torch

from contract import (
    BoundaryMode,
    DomainMode,
    GateKind,
    GateResult,
    GridConfig,
    NumericsConfig,
    RunArtifact,
)

from ..grid import GridGeometry, UniformGrid
from ..operators.poisson import (
    CoulombCutoffPoisson,
    analytic_gaussian_hartree_energy,
    analytic_gaussian_potential,
)
from .base import ArtifactGate, ComponentGate, GateSpec, artifact_device, make_result, skipped

__all__ = [
    "LaplacianOrderGate",
    "GradientOrderGate",
    "PoissonAnalyticGate",
    "OrthonormalityGate",
    "HermiticityGate",
    "WeightedSelfAdjointnessGate",
    "FiniteValuesGate",
]


class LaplacianOrderGate(ComponentGate):
    """G0.1 -- fitted convergence order of the Laplacian against the nominal; 0.05 relative (D2).

    ``log ||err||_inf`` against ``log h`` over four spacings, on a Gaussian of fixed
    ``sigma = 4 h_coarsest``: ``docs/03_METHOD.md (Part C)`` reads ``sigma = 4h``, which would
    narrow the feature with the grid and make no order measurable. Catches a mistyped stencil
    weight, which is wrong by a small amount everywhere and converges smoothly.
    """

    spec = GateSpec(
        gate_id="G0.1",
        name="Laplacian convergence order",
        threshold=0.05,
        kind=GateKind.DERIVED,
        citation="D2",
        units="relative deviation from nominal order",
    )

    #: Refinement ladder in bohr. Ratios of 1.5, not 2: four halvings of a 3-D grid is 4096 times
    #: the points, while a factor of 3 overall already moves an order-8 error by six decades.
    DEFAULT_SPACINGS = (0.45, 0.30, 0.225, 0.15)

    def __init__(self, spacings: tuple[float, ...] | None = None, domain_in_sigma: float = 5.0) -> None:
        """Configure the refinement ladder and the box size in units of the Gaussian width."""
        self.spacings = spacings or self.DEFAULT_SPACINGS
        self.coarse_spacing = max(self.spacings)
        self.domain_in_sigma = domain_in_sigma

    def evaluate(self, numerics: NumericsConfig) -> GateResult:
        """Fit the convergence order and compare it with the nominal order of the configuration."""
        order = numerics.grid.fd_order
        sigma = 4.0 * self.coarse_spacing
        half_width = self.domain_in_sigma * sigma
        spacings: list[float] = []
        errors: list[float] = []
        boundary_errors: list[float] = []

        for h in self.spacings:
            n = int(round(2.0 * half_width / h)) + 1
            geometry = GridGeometry(
                shape=(n, n, n), spacing=h, origin=(-half_width, -half_width, -half_width)
            )
            mask = torch.ones((n, n, n), dtype=torch.bool)
            grid = UniformGrid(
                geometry, mask, fd_order=order, gradient_order=order, boundary=BoundaryMode.ZERO
            )
            points = grid.points()
            r_sq = (points * points).sum(dim=-1)
            field = torch.exp(-r_sq / (2.0 * sigma**2))
            exact = field * (r_sq / sigma**4 - 3.0 / sigma**2)
            measured = grid.laplacian(field)
            deviation = (measured - exact).abs()
            # Measured where the test function lives, |r| <= 3 sigma, with the whole-domain error
            # reported beside it: with a zero-filled halo the edge error is the neglected Gaussian
            # tail over h**2, so it grows under refinement and measures the boundary, not the
            # stencil (about order -2 for a correct order-8 operator).
            interior = r_sq <= (3.0 * sigma) ** 2
            errors.append(float(deviation[interior].max()))
            boundary_errors.append(float(deviation.max()))
            spacings.append(h)

        log_h = torch.log(torch.tensor(spacings, dtype=torch.float64))
        log_e = torch.log(torch.tensor(errors, dtype=torch.float64))
        centred_h = log_h - log_h.mean()
        slope = float((centred_h * (log_e - log_e.mean())).sum() / (centred_h * centred_h).sum())
        deviation = abs(slope - order) / order
        return make_result(
            self.spec,
            deviation,
            {
                "measured_order": slope,
                "nominal_order": float(order),
                "spacings": spacings,
                "errors_interior": errors,
                "errors_whole_domain": boundary_errors,
                "sigma": sigma,
                "measured_region": "|r| <= 3 sigma",
                "interpretation": "sigma fixed at 4 * h_coarsest; see the gate docstring",
            },
        )


class GradientOrderGate(ComponentGate):
    """G0.9 -- fitted convergence order of the gradient against the nominal one; 0.05 relative (D2).

    Same test function and interior norm as G0.1. Every GGA and meta-GGA consumes ``|grad n|**2``,
    so a gradient a quarter of an order down contaminates rung-2 and rung-3 energies while G0.1
    stays green.
    """

    spec = GateSpec(
        gate_id="G0.9",
        name="Gradient convergence order",
        threshold=0.05,
        kind=GateKind.DERIVED,
        citation="D2",
        units="relative deviation from nominal order",
    )

    DEFAULT_SPACINGS = (0.45, 0.30, 0.225, 0.15)

    def __init__(self, spacings: tuple[float, ...] | None = None, domain_in_sigma: float = 5.0) -> None:
        """Configure the refinement ladder and the box size in units of the Gaussian width."""
        self.spacings = spacings or self.DEFAULT_SPACINGS
        self.coarse_spacing = max(self.spacings)
        self.domain_in_sigma = domain_in_sigma

    def evaluate(self, numerics: NumericsConfig) -> GateResult:
        """Fit the gradient's convergence order against the nominal order of the configuration."""
        order = numerics.grid.fd_gradient_order
        sigma = 4.0 * self.coarse_spacing
        half_width = self.domain_in_sigma * sigma
        spacings: list[float] = []
        errors: list[float] = []

        for h in self.spacings:
            n = int(round(2.0 * half_width / h)) + 1
            geometry = GridGeometry(
                shape=(n, n, n), spacing=h, origin=(-half_width, -half_width, -half_width)
            )
            mask = torch.ones((n, n, n), dtype=torch.bool)
            grid = UniformGrid(
                geometry, mask, fd_order=order, gradient_order=order, boundary=BoundaryMode.ZERO
            )
            points = grid.points()
            r_sq = (points * points).sum(dim=-1)
            field = torch.exp(-r_sq / (2.0 * sigma**2))
            exact = -(points.transpose(0, 1) / sigma**2) * field
            deviation = (grid.gradient(field) - exact).abs()
            interior = r_sq <= (3.0 * sigma) ** 2
            errors.append(float(deviation[:, interior].max()))
            spacings.append(h)

        log_h = torch.log(torch.tensor(spacings, dtype=torch.float64))
        log_e = torch.log(torch.tensor(errors, dtype=torch.float64))
        centred = log_h - log_h.mean()
        slope = float((centred * (log_e - log_e.mean())).sum() / (centred * centred).sum())
        return make_result(
            self.spec,
            abs(slope - order) / order,
            {
                "measured_order": slope,
                "nominal_order": float(order),
                "spacings": spacings,
                "errors_interior": errors,
                "measured_region": "|r| <= 3 sigma",
            },
        )


class PoissonAnalyticGate(ComponentGate):
    """G0.2 -- open-boundary Hartree solver against an analytic Gaussian charge; 1e-8 Ha (F15; D14).

    For ``rho(r) = Q (2 pi s^2)^(-3/2) exp(-r^2 / 2 s^2)`` the potential is
    ``V(r) = Q erf(r / (sqrt(2) s)) / r`` and the energy ``Q^2 / (2 sqrt(pi) s)``. The energy error
    is the verdict; the potential's relative infinity-norm error is enforced beside it at 1e-6,
    scaled into the same units, since a boundary artefact shows there long before it moves E_H.
    """

    spec = GateSpec(
        gate_id="G0.2",
        name="Poisson solver against analytic Gaussian",
        threshold=1.0e-8,
        kind=GateKind.DERIVED,
        citation="F15; D14; F16",
        units="Ha (absolute error in E_H)",
    )

    def __init__(self, sigma: float | None = None, box: float = 16.0, charge: float = 1.0) -> None:
        """Configure the test charge. ``sigma`` defaults to four times the configured spacing."""
        self.sigma = sigma
        self.box = box
        self.charge = charge

    def evaluate(self, numerics: NumericsConfig) -> GateResult:
        """Solve for a Gaussian charge and compare energy and potential against the closed form."""
        h = numerics.grid.spacing
        sigma = self.sigma if self.sigma is not None else 4.0 * h
        config = GridConfig(
            spacing=h,
            fd_order=numerics.grid.fd_order,
            domain=DomainMode.BOX,
            boundary=BoundaryMode.ZERO,
            box_lengths=(self.box, self.box, self.box),
            poisson_pad_factor=numerics.grid.poisson_pad_factor,
            use_double_grid=False,
            fourier_filter_projectors=False,
        )
        grid = UniformGrid.from_config(config)
        points = grid.points()
        radius = points.norm(dim=-1)
        density = (
            self.charge
            * (2.0 * math.pi * sigma**2) ** -1.5
            * torch.exp(-radius**2 / (2.0 * sigma**2))
        )
        # Renormalise to the exact charge: the quadrature of a Gaussian on a finite grid carries its
        # own error, which would otherwise be charged to the Poisson solver.
        density = density * (self.charge / float(grid.integrate(density)))

        solver = CoulombCutoffPoisson(grid, pad_factor=config.poisson_pad_factor)
        v_numeric = solver.solve(density)
        v_exact = analytic_gaussian_potential(radius, sigma, self.charge)
        e_numeric = solver.energy(density, v_numeric)
        e_exact = analytic_gaussian_hartree_energy(sigma, self.charge)

        potential_error = float((v_numeric - v_exact).abs().max() / v_exact.abs().max())
        energy_error = abs(e_numeric - e_exact)
        # The potential error is scaled into the energy threshold's units so the report keeps one
        # measured value: on the energy alone a far-field potential wrong by 1e-4 would pass.
        POTENTIAL_THRESHOLD = 1.0e-6
        scaled_potential_error = potential_error * (self.spec.threshold / POTENTIAL_THRESHOLD)
        binding = "energy" if energy_error >= scaled_potential_error else "potential"
        return make_result(
            self.spec,
            max(energy_error, scaled_potential_error),
            {
                "binding_quantity": binding,
                "energy_error": energy_error,
                "potential_rel_inf_error_threshold": POTENTIAL_THRESHOLD,
                "pad_factor": config.poisson_pad_factor,
                "sigma": sigma,
                "spacing": h,
                "sigma_over_h": sigma / h,
                "energy_numeric": e_numeric,
                "energy_exact": e_exact,
                "potential_rel_inf_error": potential_error,
                "potential_threshold": 1.0e-6,
                "cutoff_radius": solver.cutoff_radius,
                "charge_outside_cutoff": solver.charge_outside_cutoff(density),
                "padded_shape": list(solver.padded_shape),
            },
        )


class OrthonormalityGate(ArtifactGate):
    """G0.3 -- ``max |S - I|`` for the orbital block after Rayleigh--Ritz; 1e-12 [I8].

    Measured inside the solve, since recomputing it from a fresh diagonalisation would certify a
    different run. The threshold leaves three orders over float64 round-off on a million points.
    SKIPs when the run recorded no orthonormality measurement.
    """

    spec = GateSpec(
        gate_id="G0.3",
        name="Orbital block orthonormality",
        threshold=1.0e-12,
        kind=GateKind.EXACT,
        citation="I8",
        units="max |S - I|",
    )

    def evaluate(self, artifact: RunArtifact) -> GateResult:
        """Read the orthonormality error the solve recorded."""
        value = artifact.measurements.get("orthonormality_error")
        if value is None:
            return skipped(self.spec, "the run recorded no orthonormality measurement")
        return make_result(self.spec, float(value))


class HermiticityGate(ArtifactGate):
    """G0.4 -- Hermiticity of the Hamiltonian under random probes; 1e-10 Ha, 1e-5 below float64.

    Random rather than smooth probes: a smooth trial function has already decayed at the domain
    boundary, where a masked-domain stencil is most likely wrong. SKIPs with no recorded probe.
    """

    spec = GateSpec(
        gate_id="G0.4",
        name="Hamiltonian Hermiticity",
        threshold=1.0e-10,
        kind=GateKind.EXACT,
        units="Ha",
    )

    def applicable(self, artifact: RunArtifact) -> bool:
        """Not applicable on the cusp-transformed path, where the operator is not plainly symmetric.

        ``A = f^-1 H f`` is self-adjoint in the weighted measure ``f^2`` and deliberately not in the
        plain one; G0.12 is the weighted replacement.
        """
        return artifact.result is not None and not bool(
            artifact.measurements.get("cusp_factorisation", False)
        )

    def evaluate(self, artifact: RunArtifact) -> GateResult:
        """Read the Hermiticity error the solve recorded."""
        value = artifact.measurements.get("hermiticity_error")
        if value is None:
            return skipped(self.spec, "the run recorded no Hermiticity probe")
        policy = artifact.provenance.precision_policy.get("hot_path", "float64")
        spec = self.spec if policy == "float64" else GateSpec(
            gate_id=self.spec.gate_id,
            name=self.spec.name,
            threshold=1.0e-5,
            kind=self.spec.kind,
            citation=self.spec.citation,
            units=self.spec.units,
        )
        return make_result(spec, float(value), {"hot_path_precision": policy})


class WeightedSelfAdjointnessGate(ArtifactGate):
    """G0.12 -- relative asymmetry of ``A`` in ``<a,b>_w = integral(f^2 a b)``; 1e-12 (D-35; D-38).

    The weighted replacement for G0.4 on the cusp-transformed path. It probes the same operator that
    generates the search subspace, which the staggered divergence form makes symmetric by
    construction -- hence a round-off threshold. A Chebyshev filter on a non-symmetric operator does
    not converge.
    """

    spec = GateSpec(
        gate_id="G0.12",
        name="Weighted self-adjointness of the operator",
        threshold=1.0e-12,
        kind=GateKind.EXACT,
        citation="D-35; D-38",
        units="relative asymmetry",
    )

    def applicable(self, artifact: RunArtifact) -> bool:
        """Applies only where a weighted measure is in use."""
        return artifact.result is not None and bool(
            artifact.measurements.get("cusp_factorisation", False)
        )

    def evaluate(self, artifact: RunArtifact) -> GateResult:
        """Rebuild the operator and measure ``max|<a|A|b>_w - <b|A|a>_w|`` over random probes."""
        import torch

        from ..scf.noninteracting import build_hamiltonian

        from ..precision import seeded_randn

        _, hamiltonian, _ = build_hamiltonian(
            artifact.scenario, artifact.numerics, device=artifact_device(artifact)
        )
        generator = torch.Generator(device="cpu").manual_seed(artifact.numerics.seed)
        probes = seeded_randn(
            (8, hamiltonian.grid.n_points), generator, device=hamiltonian.grid.device, dtype=torch.float64
        )
        matrix = hamiltonian.measure.cross(probes, hamiltonian.apply(probes))
        scale = float(matrix.abs().max())
        asymmetry = float((matrix - matrix.transpose(-1, -2)).abs().max())
        return make_result(
            self.spec,
            asymmetry / scale if scale > 0.0 else asymmetry,
            {
                "asymmetry_absolute": asymmetry,
                "matrix_scale": scale,
                "weight_dynamic_range": artifact.measurements.get("weight_dynamic_range"),
                "collocation_asymmetry_recorded": artifact.measurements.get(
                    "self_adjointness_error"
                ),
            },
        )


class FiniteValuesGate(ArtifactGate):
    """G0.8 -- no NaN and no Inf anywhere in the produced fields."""

    spec = GateSpec(
        gate_id="G0.8",
        name="No NaN or Inf",
        threshold=0.0,
        kind=GateKind.EXACT,
        units="count of non-finite values",
    )

    def evaluate(self, artifact: RunArtifact) -> GateResult:
        """Report zero when every recorded field is finite, and one otherwise."""
        flag = artifact.measurements.get("all_finite")
        if flag is None:
            return skipped(self.spec, "the run recorded no finiteness check")
        return make_result(self.spec, 0.0 if bool(flag) else 1.0)
