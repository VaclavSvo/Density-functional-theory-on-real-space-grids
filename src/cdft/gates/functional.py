"""Component gates on the exchange--correlation layer: G0.5, G0.6, G1.1, G1.8, G1.9.

Each iterates over every functional in :data:`cdft.xc.dispatch.NATIVE`, so a new one is gated as
soon as it is registered, and each reports the worst functional rather than a mean. Random points
are fixed-seed, log-uniform in density over ``[1e-4, 10]`` and uniform in the reduced gradient over
``[0, 3]`` plus a second set over ``[0, 8]``, where functionals that coincide at small ``s``
separate. Polarised sets use unequal channels and an arbitrary angle, so ``sigma_ud`` is exercised.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from contract import GateKind, GateResult, NumericsConfig, XCRung

from .base import ComponentGate, GateSpec, make_result, skipped

__all__ = [
    "FunctionalDerivativeGate",
    "LibxcAgreementGate",
    "UniformGasGate",
    "ExchangeScalingGate",
    "random_ingredients",
]


def random_ingredients(
    n_points: int, polarised: bool, s_max: float, seed: int, channel_ratio_max: float | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(density, sigma, tau)`` in contract shapes at random but physical points.

    ``tau = tau_W + alpha tau_unif`` with ``alpha`` in ``[0, 6]``, so the von Weizsaecker bound
    holds everywhere and no meta-GGA is evaluated outside its domain. ``channel_ratio_max`` caps
    ``n_up / n_dn`` and its inverse, which a finite-difference gate needs and a pointwise one does
    not.
    """
    rng = np.random.default_rng(seed)
    n_spin = 2 if polarised else 1
    n = np.exp(rng.uniform(math.log(1.0e-4), math.log(10.0), size=(n_spin, n_points)))
    if polarised and channel_ratio_max is not None:
        ratio = np.exp(rng.uniform(-math.log(channel_ratio_max), math.log(channel_ratio_max), size=n_points))
        n[1] = n[0] * ratio
    if not polarised:
        k_f = (3.0 * math.pi**2 * n) ** (1.0 / 3.0)
        s = rng.uniform(0.0, s_max, size=(1, n_points))
        sigma = (2.0 * k_f * n * s) ** 2
        tau_w = sigma / (8.0 * n)
        tau_unif = 0.3 * (3.0 * math.pi**2) ** (2.0 / 3.0) * n ** (5.0 / 3.0)
        tau = tau_w + rng.uniform(0.0, 6.0, size=(1, n_points)) * tau_unif
        return n, sigma, tau
    # Per-channel reduced gradient of the spin-scaled density 2 n_s, then the gradient magnitude.
    s = rng.uniform(0.0, s_max, size=(2, n_points))
    k_f2 = (3.0 * math.pi**2 * 2.0 * n) ** (1.0 / 3.0)
    grad = 2.0 * k_f2 * (2.0 * n) * s / 2.0
    cos = rng.uniform(-1.0, 1.0, size=n_points)
    sigma = np.stack([grad[0] ** 2, grad[0] * grad[1] * cos, grad[1] ** 2])
    tau_w = grad**2 / (8.0 * n)
    tau_unif = 0.3 * (6.0 * math.pi**2) ** (2.0 / 3.0) * n ** (5.0 / 3.0)
    tau = tau_w + rng.uniform(0.0, 6.0, size=(2, n_points)) * tau_unif
    return n, sigma, tau


def _ingredients_for(
    functional, polarised: bool, n_points: int, s_max: float, seed: int, channel_ratio_max=None
):
    """Trim the random ingredients to what the functional's rung consumes."""
    n, sigma, tau = random_ingredients(n_points, polarised, s_max, seed, channel_ratio_max)
    rung = functional.rung.value
    return (
        torch.tensor(n),
        torch.tensor(sigma) if rung >= XCRung.GGA.value else None,
        torch.tensor(tau) if rung >= XCRung.META_GGA.value else None,
    )


class FunctionalDerivativeGate(ComponentGate):
    """G0.5 -- autograd potentials against central differences of ``sum(e_xc)``; 1e-4 relative.

    The step is relative, ``1e-4 |x|``: an absolute step would swamp a reduced gradient of 1e-9 and
    measure the functional's nonlinearity. The threshold sits above the differencing noise of a
    badly conditioned functional such as LYP and below what a wrong derivative produces; G0.6 holds
    the same potentials' values. SKIPs when too few points are resolvable.
    """

    spec = GateSpec(
        gate_id="G0.5",
        name="v_xc is the functional derivative of E_xc",
        threshold=1.0e-4,
        kind=GateKind.EXACT,
        citation="definition of the functional derivative; autograd vs central difference",
        units="relative",
    )

    N_POINTS = 2000
    #: Largest ``n_up / n_dn`` in the polarised set: a channel much smaller than its partner
    #: contributes below the cancellation noise, and its finite difference measures round-off.
    CHANNEL_RATIO_MAX = 100.0
    #: A channel-point is judged only where the differencing floor ``eps |e| / step`` is below this
    #: fraction of the slope's scale, independently of the threshold, which cannot admit noise.
    FLOOR_FRACTION = 1.0e-8

    def evaluate(self, numerics: NumericsConfig) -> GateResult:
        """Compare autograd potentials with finite differences for every native functional."""
        from ..xc.dispatch import NATIVE, functional_by_name

        worst = 0.0
        judged = unresolved = 0
        detail: dict[str, object] = {}
        for name in NATIVE:
            functional = functional_by_name(name)
            for polarised in (False, True):
                density, sigma, tau = _ingredients_for(
                    functional, polarised, self.N_POINTS, 3.0, 11, channel_ratio_max=self.CHANNEL_RATIO_MAX
                )
                out = functional.evaluate(density, sigma, tau)
                # PZ81 [A9] as published is discontinuous at r_s = 1, so a difference straddling
                # the seam measures the jump. Excluded for every functional, to keep it uniform.
                r_s =(3.0 / (4.0 * math.pi * density.sum(dim=0))) ** (1.0 / 3.0)
                seam = (r_s - 1.0).abs() < 0.01
                deviation = 0.0
                for label, tensor, grad in (("rho", density, out.v_xc), ("sigma", sigma, out.v_sigma), ("tau", tau, out.v_tau)):
                    if tensor is None:
                        continue
                    for channel in range(tensor.shape[0]):
                        step = 1.0e-4 * tensor[channel].abs().clamp_min(1.0e-30)
                        plus = tensor.clone()
                        minus = tensor.clone()
                        plus[channel] += step
                        minus[channel] -= step
                        args_p = [density, sigma, tau]
                        args_m = [density, sigma, tau]
                        index = {"rho": 0, "sigma": 1, "tau": 2}[label]
                        args_p[index] = plus
                        args_m[index] = minus
                        e_p = functional.energy(*args_p)
                        e_m = functional.energy(*args_m)
                        fd = (e_p - e_m) / (2.0 * step)
                        scale = grad[channel].abs().clamp_min(1.0)
                        # Round-off floor: ``e`` is known to ``eps |e|``, so the slope is known to
                        # ``2 eps |e| / (2 step)``. Above it a point is unresolved, not failed.
                        floor =2.0 * torch.finfo(torch.float64).eps * out.e_xc.abs() / (2.0 * step)
                        resolvable = (floor < self.FLOOR_FRACTION * scale) & ~seam
                        unresolved += int((~resolvable).sum())
                        judged += int(resolvable.sum())
                        if bool(resolvable.any()):
                            gap = ((fd - grad[channel]).abs() / scale)[resolvable]
                            deviation = max(deviation, float(gap.max()))
                detail[f"{name}/{'polarised' if polarised else 'unpolarised'}"] = deviation
                worst = max(worst, deviation)
        detail["n_points"] = self.N_POINTS
        detail["points_judged"] = judged
        detail["points_unresolved_by_round_off"] = unresolved
        if judged < 20 * unresolved:
            return skipped(
                self.spec,
                f"only {judged} channel-points were resolvable by finite differences against "
                f"{unresolved} below the round-off floor; the random set is degenerate",
            )
        return make_result(self.spec, worst, detail)


class LibxcAgreementGate(ComponentGate):
    """G0.6 -- ``e_xc`` and every potential against libxc at 1e5 random points; 1e-10 (D-05).

    Scaled by ``max(1, |libxc|)``: absolute where the quantity is O(1), relative where it is large,
    since ``v_sigma`` reaches several hundred and an absolute demand there would ask for eleven
    significant figures. Both conventions are in the detail, as are the deferred functionals, which
    are listed with their reason rather than silently passed. Off the plan unless the oracles are
    requested (O-24); requested without libxc it SKIPs with the reason.
    """

    spec = GateSpec(
        gate_id="G0.6",
        name="libxc agreement, pointwise",
        threshold=1.0e-10,
        kind=GateKind.EXACT,
        citation="I1 (libxc), through PySCF; D-05",
        units="max |native - libxc| / max(1, |libxc|)",
        requires_oracle=True,  # off the plan unless CDFT_ORACLES=1 (O-24)
    )

    N_POINTS = 100_000

    def evaluate(self, numerics: NumericsConfig) -> GateResult:
        """Run the comparison for every native functional and both spin cases."""
        from ..xc.dispatch import DEFERRED_FUNCTIONALS, NATIVE, functional_by_name, libxc_code
        from ..xc.oracle import evaluate_libxc, libxc_available

        if not libxc_available():
            return skipped(
                self.spec,
                "PySCF (which carries libxc) is not importable; install the 'oracles' extra. The "
                "native functionals are unverified against libxc in this environment",
            )
        worst = 0.0
        detail: dict[str, object] = {"deferred": dict(DEFERRED_FUNCTIONALS)}
        version = None
        for name in NATIVE:
            functional = functional_by_name(name)
            code = libxc_code(functional)
            for polarised in (False, True):
                for s_max in (3.0, 8.0):
                    density, sigma, tau = _ingredients_for(functional, polarised, self.N_POINTS, s_max, 7)
                    reference = evaluate_libxc(
                        code,
                        density.numpy(),
                        None if sigma is None else sigma.numpy(),
                        None if tau is None else tau.numpy(),
                    )
                    version = reference.libxc_version
                    out = functional.evaluate(density, sigma, tau)
                    pairs = [("e_xc", out.e_xc.cpu().numpy(), reference.e_xc), ("v_rho", out.v_xc.cpu().numpy(), reference.v_rho)]
                    if sigma is not None:
                        pairs.append(("v_sigma", out.v_sigma.cpu().numpy(), reference.v_sigma))
                    if tau is not None:
                        pairs.append(("v_tau", out.v_tau.cpu().numpy(), reference.v_tau))
                    for label, native, libxc_values in pairs:
                        diff = np.abs(native - libxc_values)
                        scaled = float((diff / np.maximum(1.0, np.abs(libxc_values))).max())
                        key = f"{name}/{'pol' if polarised else 'unpol'}/s<{s_max:g}/{label}"
                        detail[key] = {"scaled": scaled, "absolute": float(diff.max())}
                        worst = max(worst, scaled)
        detail["libxc_version"] = version
        detail["n_points"] = self.N_POINTS
        return make_result(self.spec, worst, detail)


class UniformGasGate(ComponentGate):
    """G1.1 -- uniform-gas limits of every functional; 1e-12 Ha per electron [A7], [A8], [A9].

    Exchange must equal ``-(3/4)(3/pi)^(1/3) n^(1/3)`` and a GGA must reduce to its LDA at
    ``sigma = 0``; both are exact identities, hence the round-off threshold. The LDA correlations
    are compared with the Ceperley--Alder data they fit on :attr:`FIT_TOLERANCE` instead, which
    makes the gate a check on the physics rather than on the code agreeing with itself.
    """

    spec = GateSpec(
        gate_id="G1.1",
        name="Uniform electron gas exchange and correlation",
        threshold=1.0e-12,
        kind=GateKind.EXACT,
        citation="A8 (PW92 Table II), A7, A9; PBE reduces to PW92 at sigma = 0",
        units="Ha per electron",
    )

    #: PW92 [A8], Table II: ``-e_c(r_s, zeta = 0)`` in Hartree from the Ceperley--Alder Monte Carlo
    #: data the LDA correlations are fitted to (four digits).
    CEPERLEY_ALDER = {2.0: 0.04479, 10.0: 0.01861}
    #: Tolerance of the fit-vs-data comparison; the parametrisations are fitted to about 1 %.
    FIT_TOLERANCE = 1.0e-3

    def evaluate(self, numerics: NumericsConfig) -> GateResult:
        """Evaluate every functional on uniform densities and compare with the uniform-gas limits."""
        from ..xc.dispatch import functional_by_name

        detail: dict[str, object] = {}
        worst = 0.0
        rs_values = torch.tensor([0.5, 1.0, 2.0, 5.0, 10.0], dtype=torch.float64)
        n = 3.0 / (4.0 * math.pi * rs_values**3)
        density = n.reshape(1, -1)
        zero_sigma = torch.zeros_like(density)
        # Exchange on a uniform gas is Dirac exchange exactly, for every exchange functional.
        dirac = -0.75 * (3.0 / math.pi) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
        for name in ("lda_x", "gga_x_pbe", "gga_x_b88"):
            functional = functional_by_name(name)
            out = functional.evaluate(density, None if functional.rung is XCRung.LDA else zero_sigma)
            deviation = float(((out.e_xc / n) - dirac).abs().max())
            detail[f"{name}/dirac_exchange"] = deviation
            worst = max(worst, deviation)
        # GGA correlation at zero gradient reduces to its LDA.
        pbe_c = functional_by_name("gga_c_pbe").evaluate(density, zero_sigma).e_xc
        pw_mod = functional_by_name("lda_c_pw_mod").evaluate(density).e_xc
        deviation = float(((pbe_c - pw_mod) / n).abs().max())
        detail["gga_c_pbe/reduces_to_pw92_mod"] = deviation
        worst = max(worst, deviation)
        # The LDA correlations against the Monte Carlo data they fit, on FIT_TOLERANCE.
        fit_worst = 0.0
        for name in ("lda_c_vwn", "lda_c_pw", "lda_c_pz"):
            functional = functional_by_name(name)
            for rs, minus_ec in self.CEPERLEY_ALDER.items():
                dens = torch.tensor([[3.0 / (4.0 * math.pi * rs**3)]], dtype=torch.float64)
                e_per_particle = float(functional.evaluate(dens).e_xc[0] / dens[0, 0])
                gap = abs(e_per_particle + minus_ec)
                detail[f"{name}/vs_ceperley_alder_rs{rs:g}"] = {"fit": e_per_particle, "data": -minus_ec, "gap": gap}
                fit_worst = max(fit_worst, gap)
        detail["ceperley_alder_worst_gap"] = fit_worst
        detail["ceperley_alder_tolerance"] = self.FIT_TOLERANCE
        if fit_worst > self.FIT_TOLERANCE:
            worst = max(worst, fit_worst)
        return make_result(self.spec, worst, detail)


class ExchangeScalingGate(ComponentGate):
    """G1.9 -- ``E_x[n_lambda] = lambda E_x[n]`` under coordinate scaling; 1e-12 relative [A16].

    Sampled at spacing ``h / lambda`` the scaled density is the same array times ``lambda^3``, with
    ``sigma`` times ``lambda^8``, so the only error is round-off (the trick of G1.12). Exchange
    functionals only; correlation does not obey the relation.
    """

    spec = GateSpec(
        gate_id="G1.9",
        name="Uniform coordinate scaling of exchange",
        threshold=1.0e-12,
        kind=GateKind.EXACT,
        citation="Levy and Perdew, PRA 32, 2010 (1985); A16",
        units="relative",
    )

    FACTORS = (0.5, 2.0, 3.0)

    def evaluate(self, numerics: NumericsConfig) -> GateResult:
        """Scale a sampled density and require the exchange energy to scale linearly."""
        from ..xc.dispatch import functional_by_name

        rng = np.random.default_rng(5)
        n_points = 4096
        h = 0.25
        n = torch.tensor(np.exp(rng.uniform(math.log(1e-3), math.log(2.0), size=(1, n_points))))
        sigma = torch.tensor(rng.uniform(0.0, 1.0, size=(1, n_points))) * n**2
        detail: dict[str, object] = {}
        worst = 0.0
        for name in ("lda_x", "gga_x_pbe", "gga_x_b88"):
            functional = functional_by_name(name)
            needs_sigma = functional.rung is not XCRung.LDA
            base = float(functional.evaluate(n, sigma if needs_sigma else None).e_xc.sum() * h**3)
            for factor in self.FACTORS:
                scaled = functional.evaluate(
                    n * factor**3, (sigma * factor**8) if needs_sigma else None
                ).e_xc.sum() * (h / factor) ** 3
                relative = abs(float(scaled) - factor * base) / abs(factor * base)
                detail[f"{name}/lambda_{factor:g}"] = relative
                worst = max(worst, relative)
        return make_result(self.spec, worst, detail)
