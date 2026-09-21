"""Rung-1 functionals: Slater exchange and the PW92 [A8], VWN5 [A7] and PZ81 [A9] correlations.

Every parameter is transcribed from the cited paper and cross-checked against libxc (G0.6, 1e-10
pointwise, both spin cases). Both PW92 parameter sets are carried because PBE correlation [A4] is
defined on the modified one (libxc ``LDA_C_PW_MOD``); they differ by a few 1e-6 Ha per electron,
the size of the NIST comparison tolerance, so confusing them fails G4.7 invisibly.

Spin interpolation follows each paper: PW92 and VWN5 carry the spin-stiffness term
``alpha_c f(zeta)(1 - zeta^4)/f''(0)``, PZ81 interpolates linearly in ``f(zeta)``.
"""

from __future__ import annotations

import math

import torch

from contract import XCRung

from .base import (
    F_ZETA_SECOND_DERIVATIVE_EXACT,
    SemiLocalFunctional,
    f_zeta,
    slater_exchange_density,
    spin_polarisation,
    wigner_seitz_radius,
)

__all__ = [
    "SlaterExchange",
    "PW92Correlation",
    "VWN5Correlation",
    "PZ81Correlation",
    "LDA",
    "pw92_epsilon",
]


class SlaterExchange(SemiLocalFunctional):
    """Dirac--Slater exchange (libxc ``LDA_X``)."""

    rung = XCRung.LDA
    name = "lda_x"
    libxc_reference = ("lda_x",)

    def energy_density(self, n_up, n_dn, sigma_uu, sigma_ud, sigma_dd, tau_up, tau_dn):
        """``-2^(1/3) (3/4)(3/pi)^(1/3) sum_s n_s^(4/3)``."""
        return slater_exchange_density(n_up, n_dn)


# -- PW92 --

#: Perdew--Wang 1992 [A8], Table I, as printed: ``(A, alpha_1, beta_1, beta_2, beta_3, beta_4)`` for
#: the paramagnetic energy, the ferromagnetic energy and minus the spin stiffness; ``p = 1``.
_PW92_ORIGINAL = {
    "para": (0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294),
    "ferro": (0.015545, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517),
    "stiffness": (0.016887, 0.11125, 10.357, 3.6231, 0.88026, 0.49671),
    "fpp0": 1.709921,
}
#: The "modified" set used inside PBE [A4] and by libxc ``LDA_C_PW_MOD``: ``A`` to more digits and
#: the exact ``f''(0)``.
_PW92_MODIFIED = {
    "para": (0.0310907, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294),
    "ferro": (0.01554535, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517),
    "stiffness": (0.0168869, 0.11125, 10.357, 3.6231, 0.88026, 0.49671),
    "fpp0": F_ZETA_SECOND_DERIVATIVE_EXACT,
}


def _pw92_g(rs: torch.Tensor, params: tuple[float, ...]) -> torch.Tensor:
    """PW92 eq. (10): ``G(r_s) = -2A(1 + a1 r_s) ln[1 + 1 / (2A (b1 r_s^1/2 + b2 r_s + b3 r_s^3/2 + b4 r_s^2))]``."""
    a, alpha1, beta1, beta2, beta3, beta4 = params
    sqrt_rs = rs.sqrt()
    denominator = 2.0 * a * (beta1 * sqrt_rs + beta2 * rs + beta3 * rs * sqrt_rs + beta4 * rs * rs)
    return -2.0 * a * (1.0 + alpha1 * rs) * torch.log1p(1.0 / denominator)


def pw92_epsilon(rs: torch.Tensor, zeta: torch.Tensor, modified: bool) -> torch.Tensor:
    """PW92 correlation energy per particle, eq. (8): ``e_c(r_s, zeta)``.

    ``e_c = e_c(r_s,0) + alpha_c f(zeta)(1 - zeta^4)/f''(0) + [e_c(r_s,1) - e_c(r_s,0)] f(zeta) zeta^4``
    with ``alpha_c = -G(r_s; stiffness parameters)``.
    """
    table = _PW92_MODIFIED if modified else _PW92_ORIGINAL
    para = _pw92_g(rs, table["para"])
    ferro = _pw92_g(rs, table["ferro"])
    alpha_c = -_pw92_g(rs, table["stiffness"])
    fz = f_zeta(zeta)
    z4 = zeta**4
    return para + alpha_c * fz * (1.0 - z4) / table["fpp0"] + (ferro - para) * fz * z4


class PW92Correlation(SemiLocalFunctional):
    """Perdew--Wang 1992 correlation [A8] (libxc ``LDA_C_PW`` or ``LDA_C_PW_MOD``)."""

    rung = XCRung.LDA

    def __init__(self, modified: bool = False) -> None:
        """``modified=True`` selects the PBE-internal parameter set (``LDA_C_PW_MOD``)."""
        self.modified = modified
        self.name = "lda_c_pw_mod" if modified else "lda_c_pw"
        self.libxc_reference = (self.name,)

    def energy_density(self, n_up, n_dn, sigma_uu, sigma_ud, sigma_dd, tau_up, tau_dn):
        """``n e_c(r_s, zeta)``."""
        n = n_up + n_dn
        return n * pw92_epsilon(wigner_seitz_radius(n), spin_polarisation(n_up, n_dn), self.modified)


# -- VWN5 --

#: Vosko--Wilk--Nusair 1980 [A7], the "VWN5" fit to the Ceperley--Alder data: ``(A, x0, b, c)`` for
#: the paramagnetic energy, the ferromagnetic energy and the spin stiffness.
_VWN5 = {
    "para": (0.0310907, -0.10498, 3.72744, 12.9352),
    "ferro": (0.01554535, -0.32500, 7.06042, 18.0578),
    "stiffness": (-1.0 / (6.0 * math.pi**2), -0.0047584, 1.13107, 13.0045),
}


def _vwn_aux(x: torch.Tensor, params: tuple[float, ...]) -> torch.Tensor:
    """VWN eq. (4.4): the Pade-in-``x = sqrt(r_s)`` fit ``e(x)``."""
    a, x0, b, c = params
    big_x = x * x + b * x + c
    big_x0 = x0 * x0 + b * x0 + c
    q = math.sqrt(4.0 * c - b * b)
    atan_term = torch.atan(q / (2.0 * x + b))
    return a * (
        torch.log(x * x / big_x)
        + (2.0 * b / q) * atan_term
        - (b * x0 / big_x0)
        * (torch.log((x - x0) ** 2 / big_x) + (2.0 * (b + 2.0 * x0) / q) * atan_term)
    )


class VWN5Correlation(SemiLocalFunctional):
    """Vosko--Wilk--Nusair correlation, parametrisation V [A7] (libxc ``LDA_C_VWN``).

    The functional NIST SRD 141 uses, so the one G4.7 compares at. Spin interpolation with the
    stiffness term and the exact ``f''(0)``, as in VWN eq. (3.2) and libxc.
    """

    rung = XCRung.LDA
    name = "lda_c_vwn"
    libxc_reference = ("lda_c_vwn",)

    def energy_density(self, n_up, n_dn, sigma_uu, sigma_ud, sigma_dd, tau_up, tau_dn):
        """``n e_c`` with the three-fit spin interpolation."""
        n = n_up + n_dn
        x = wigner_seitz_radius(n).sqrt()
        zeta = spin_polarisation(n_up, n_dn)
        para = _vwn_aux(x, _VWN5["para"])
        ferro = _vwn_aux(x, _VWN5["ferro"])
        alpha_c = _vwn_aux(x, _VWN5["stiffness"])
        fz = f_zeta(zeta)
        z4 = zeta**4
        epsilon = para + alpha_c * fz * (1.0 - z4) / F_ZETA_SECOND_DERIVATIVE_EXACT + (ferro - para) * fz * z4
        return n * epsilon


# -- PZ81 --

#: Perdew--Zunger 1981 [A9], Table XII: ``(gamma, beta_1, beta_2, A, B, C, D)`` for the
#: unpolarised and fully polarised gas.
_PZ81 = {
    "para": (-0.1423, 1.0529, 0.3334, 0.0311, -0.048, 0.0020, -0.0116),
    "ferro": (-0.0843, 1.3981, 0.2611, 0.01555, -0.0269, 0.0007, -0.0048),
}


def _pz81_epsilon(rs: torch.Tensor, params: tuple[float, ...]) -> torch.Tensor:
    """PZ81 eqs. (C3)--(C5): the two-branch fit, ``r_s >= 1`` Pade and ``r_s < 1`` logarithmic."""
    gamma, beta1, beta2, a, b, c, d = params
    high = gamma / (1.0 + beta1 * rs.sqrt() + beta2 * rs)
    log_rs = torch.log(rs)
    low = a * log_rs + b + c * rs * log_rs + d * rs
    return torch.where(rs >= 1.0, high, low)


class PZ81Correlation(SemiLocalFunctional):
    """Perdew--Zunger 1981 correlation [A9] (libxc ``LDA_C_PZ``)."""

    rung = XCRung.LDA
    name = "lda_c_pz"
    libxc_reference = ("lda_c_pz",)

    def energy_density(self, n_up, n_dn, sigma_uu, sigma_ud, sigma_dd, tau_up, tau_dn):
        """``n [e_U + f(zeta)(e_P - e_U)]``, PZ81 eq. (C2)."""
        n = n_up + n_dn
        rs = wigner_seitz_radius(n)
        zeta = spin_polarisation(n_up, n_dn)
        para = _pz81_epsilon(rs, _PZ81["para"])
        ferro = _pz81_epsilon(rs, _PZ81["ferro"])
        return n * (para + f_zeta(zeta) * (ferro - para))


# -- combined exchange + correlation --


class LDA(SemiLocalFunctional):
    """Slater exchange plus one LDA correlation: the rung-1 functional the SCF consumes."""

    rung = XCRung.LDA

    def __init__(self, correlation: SemiLocalFunctional, name: str) -> None:
        """Pair Slater exchange with ``correlation``; ``name`` is the dispatch identifier."""
        self.exchange = SlaterExchange()
        self.correlation = correlation
        self.name = name
        self.libxc_reference = self.exchange.libxc_reference + correlation.libxc_reference

    def energy_density(self, n_up, n_dn, sigma_uu, sigma_ud, sigma_dd, tau_up, tau_dn):
        """Exchange plus correlation."""
        return self.exchange.energy_density(
            n_up, n_dn, sigma_uu, sigma_ud, sigma_dd, tau_up, tau_dn
        ) + self.correlation.energy_density(n_up, n_dn, sigma_uu, sigma_ud, sigma_dd, tau_up, tau_dn)
