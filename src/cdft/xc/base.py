"""The semi-local functional base class: spin-polarised throughout, derivatives by autograd.

Every native functional is written once, as a function of the per-spin ingredients
``(n_up, n_dn, sigma_uu, sigma_ud, sigma_dd, tau_up, tau_dn)`` returning the energy density per
unit volume. The spin-polarised form is primary and the restricted call is the special case
``n_up = n_dn = n/2`` (D-51). Potentials come from ``torch.autograd`` through the energy density, so
no functional carries a hand-transcribed derivative that could disagree with its energy (G0.5
against finite differences, G0.6 against libxc pointwise at 1e-10).

A point below :data:`DENSITY_THRESHOLD` contributes nothing and its potential is zero, as in libxc.
Rationale: ``docs/03_METHOD.md`` (Part E).
"""

from __future__ import annotations

import abc

import torch

from contract import XCOutput, XCRung

__all__ = [
    "SemiLocalFunctional",
    "DENSITY_THRESHOLD",
    "SIGMA_THRESHOLD",
    "TAU_THRESHOLD",
    "split_spin",
]

#: Total density below which a point is empty: ``e_xc = 0`` and every potential is zero. 1e-14, not
#: the float64 floor, because the LDA correlation parametrisations contain ``ln r_s`` and
#: ``r_s^(1/2)``, whose derivatives amplify round-off as ``n -> 0``; at 1e-14 a point's energy
#: density is below 1e-19 Ha/bohr^3.
DENSITY_THRESHOLD = 1.0e-14

#: Floor on a reduced-gradient argument, so ``sqrt(sigma)`` has a finite derivative at zero
#: gradient. Far below anything a grid produces; it only keeps autograd from seeing ``0 * inf``.
SIGMA_THRESHOLD = 1.0e-40

#: Floor on the kinetic energy density, for the same reason as :data:`SIGMA_THRESHOLD`.
TAU_THRESHOLD = 1.0e-40


def split_spin(
    density: torch.Tensor,
    sigma: torch.Tensor | None,
    tau: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Map the contract's ``(n_spin, ...)`` arrays onto the seven per-spin ingredients.

    ``n_spin == 1``: ``n_up = n_dn = n/2``, ``sigma_* = sigma/4``, ``tau_* = tau/2``. ``n_spin ==
    2``: as given, ``sigma`` ordered ``(uu, ud, dd)`` as the contract and libxc define it.
    """
    n_spin = density.shape[0]
    if n_spin == 1:
        n_up = n_dn = 0.5 * density[0]
        if sigma is None:
            s_uu = s_ud = s_dd = None
        else:
            s_uu = s_ud = s_dd = 0.25 * sigma[0]
        if tau is None:
            t_up = t_dn = None
        else:
            t_up = t_dn = 0.5 * tau[0]
        return n_up, n_dn, s_uu, s_ud, s_dd, t_up, t_dn
    if n_spin != 2:
        raise ValueError(f"density must have a leading spin axis of extent 1 or 2, got {n_spin}")
    n_up, n_dn = density[0], density[1]
    if sigma is None:
        s_uu = s_ud = s_dd = None
    else:
        if sigma.shape[0] != 3:
            raise ValueError(f"a spin-polarised sigma has three components (uu, ud, dd), got {sigma.shape[0]}")
        s_uu, s_ud, s_dd = sigma[0], sigma[1], sigma[2]
    if tau is None:
        t_up = t_dn = None
    else:
        t_up, t_dn = tau[0], tau[1]
    return n_up, n_dn, s_uu, s_ud, s_dd, t_up, t_dn


class SemiLocalFunctional(abc.ABC):
    """Base class implementing :class:`~contract.XCFunctionalProtocol` for rungs 1--3.

    Subclasses implement :meth:`energy_density` and declare :attr:`rung`, :attr:`name` and
    :attr:`libxc_reference`; spin splitting, thresholds, autograd potentials and the
    :class:`~contract.XCOutput` are here.
    """

    #: Rung of Jacob's ladder; decides which ingredients :meth:`evaluate` requires.
    rung: XCRung
    #: Identifier written into the record and used by :mod:`cdft.xc.dispatch`.
    name: str
    #: libxc functional identifiers this implementation must reproduce (gate G0.6).
    libxc_reference: tuple[str, ...] = ()

    @abc.abstractmethod
    def energy_density(
        self,
        n_up: torch.Tensor,
        n_dn: torch.Tensor,
        sigma_uu: torch.Tensor | None,
        sigma_ud: torch.Tensor | None,
        sigma_dd: torch.Tensor | None,
        tau_up: torch.Tensor | None,
        tau_dn: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return ``e_xc`` per unit volume, shape ``(n_pts,)``, for the per-spin ingredients.

        Inputs are already clamped to the thresholds above, so division is safe. Torch operations
        only, so autograd can differentiate it.
        """

    def evaluate(
        self,
        density: torch.Tensor,
        sigma: torch.Tensor | None = None,
        tau: torch.Tensor | None = None,
        lapl: torch.Tensor | None = None,
        nldf: torch.Tensor | None = None,
    ) -> XCOutput:
        """Evaluate the energy density and its derivatives (:class:`~contract.XCFunctionalProtocol`).

        ``v_xc`` and ``v_sigma`` are ``torch.autograd.grad`` of ``sum(e_xc)`` against the
        contract-shaped inputs, so the chain rule through the spin splitting is autograd's and the
        derivatives are the contract's: ``de/dn`` per channel, ``de/dsigma`` per component. The call
        builds its own graph; the inputs need not carry gradients.
        """
        if self.rung.value >= XCRung.GGA.value and sigma is None:
            raise ValueError(f"{self.name} is a rung-{self.rung.value} functional and needs sigma")
        if self.rung.value >= XCRung.META_GGA.value and tau is None:
            raise ValueError(f"{self.name} is a meta-GGA and needs tau")
        if nldf is not None:
            raise ValueError(f"{self.name} is semi-local and consumes no non-local features")

        dens = density.detach().to(torch.float64).requires_grad_(True)
        sig = None if sigma is None else sigma.detach().to(torch.float64).requires_grad_(True)
        kin = None if tau is None else tau.detach().to(torch.float64).requires_grad_(True)

        with torch.enable_grad():
            e_xc = self._energy_from_contract_inputs(dens, sig, kin)
            inputs = [dens] + ([sig] if sig is not None else []) + ([kin] if kin is not None else [])
            grads = torch.autograd.grad(e_xc.sum(), inputs, allow_unused=True)
        v_rho = grads[0]
        v_sigma = grads[1] if sig is not None else None
        v_tau = grads[-1] if kin is not None else None
        zeros = lambda t: torch.zeros_like(t) if t is None else t  # noqa: E731
        return XCOutput(
            e_xc=e_xc.detach(),
            v_xc=zeros(v_rho).detach() if v_rho is not None else torch.zeros_like(dens),
            v_sigma=None if sig is None else (v_sigma.detach() if v_sigma is not None else torch.zeros_like(sig)),
            v_tau=None if kin is None else (v_tau.detach() if v_tau is not None else torch.zeros_like(kin)),
            v_lapl=None,
            v_nldf=None,
        )

    def energy(
        self,
        density: torch.Tensor,
        sigma: torch.Tensor | None = None,
        tau: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return ``e_xc`` per volume for contract-shaped inputs, without detaching.

        For callers differentiating through the functional themselves (G0.5's finite-difference
        check, or implicit differentiation downstream).
        """
        return self._energy_from_contract_inputs(density.to(torch.float64), sigma, tau)

    def _energy_from_contract_inputs(
        self, density: torch.Tensor, sigma: torch.Tensor | None, tau: torch.Tensor | None
    ) -> torch.Tensor:
        """Spin-split, threshold, evaluate, and zero the empty points."""
        n_up, n_dn, s_uu, s_ud, s_dd, t_up, t_dn = split_spin(density, sigma, tau)
        total = n_up + n_dn
        empty = total < DENSITY_THRESHOLD
        # Clamp inside the formula so no division or fractional power sees zero, then mask: masked
        # points carry a zero energy and a zero gradient rather than a NaN.
        floor = 0.5 * DENSITY_THRESHOLD
        n_up_c = n_up.clamp_min(floor)
        n_dn_c = n_dn.clamp_min(floor)
        clamp_s = lambda s: None if s is None else s.clamp_min(SIGMA_THRESHOLD)  # noqa: E731
        clamp_t = lambda t: None if t is None else t.clamp_min(TAU_THRESHOLD)  # noqa: E731
        s_ud_c = None if s_ud is None else s_ud  # the cross term may be negative; not clamped
        e = self.energy_density(
            n_up_c, n_dn_c, clamp_s(s_uu), s_ud_c, clamp_s(s_dd), clamp_t(t_up), clamp_t(t_dn)
        )
        return torch.where(empty, torch.zeros_like(e), e)

    def __repr__(self) -> str:
        """Name and rung, for logs."""
        return f"{type(self).__name__}(name={self.name!r}, rung={self.rung.name})"


# -- shared building blocks --

#: ``(3/4)(3/pi)^(1/3)``, the Dirac/Slater exchange constant of a spin-unpolarised gas.
_SLATER = 0.75 * (3.0 / torch.pi) ** (1.0 / 3.0)


def slater_exchange_density(n_up: torch.Tensor, n_dn: torch.Tensor) -> torch.Tensor:
    """Return the Dirac exchange energy density of a spin-polarised gas, per unit volume.

    Spin scaling of exchange: ``E_x[n_up, n_dn] = (E_x[2 n_up] + E_x[2 n_dn]) / 2`` with the
    unpolarised ``e_x = -(3/4)(3/pi)^(1/3) n^(4/3)``, so per channel ``-2^(1/3) (3/4)(3/pi)^(1/3)
    n_s^(4/3)``. Reproduces libxc ``LDA_X`` to 5.6e-17 (O-10).
    """
    factor = 2.0 ** (1.0 / 3.0) * _SLATER
    return -factor * (n_up ** (4.0 / 3.0) + n_dn ** (4.0 / 3.0))


def wigner_seitz_radius(n: torch.Tensor) -> torch.Tensor:
    """``r_s = (3 / (4 pi n))^(1/3)``."""
    return (3.0 / (4.0 * torch.pi * n)) ** (1.0 / 3.0)


def spin_polarisation(n_up: torch.Tensor, n_dn: torch.Tensor) -> torch.Tensor:
    """``zeta = (n_up - n_dn) / (n_up + n_dn)``, clamped to the open interval (-1, 1).

    The clamp is for autograd: ``(1 - zeta)^(2/3)`` has an infinite derivative at ``zeta = 1``, so
    a fully polarised point would return NaN potentials. The margin is below anything a
    threshold-clamped density pair can produce.
    """
    zeta = (n_up - n_dn) / (n_up + n_dn)
    return zeta.clamp(-1.0 + 1.0e-15, 1.0 - 1.0e-15)


def f_zeta(zeta: torch.Tensor) -> torch.Tensor:
    """The spin-interpolation function ``f(zeta) = [(1+zeta)^(4/3) + (1-zeta)^(4/3) - 2] / (2^(4/3) - 2)``."""
    return ((1.0 + zeta) ** (4.0 / 3.0) + (1.0 - zeta) ** (4.0 / 3.0) - 2.0) / (2.0 ** (4.0 / 3.0) - 2.0)


#: ``f''(0) = 4 / (9 (2^(1/3) - 1))``, exact.
F_ZETA_SECOND_DERIVATIVE_EXACT = 4.0 / (9.0 * (2.0 ** (1.0 / 3.0) - 1.0))
