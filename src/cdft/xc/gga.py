"""Rung-2 functionals: PBE [A4], B88 exchange [A5] and LYP correlation [A6].

Per-spin form, verified polarised against libxc (G0.6). Exchange uses the exact spin scaling
``E_x[n_up, n_dn] = (E_x[2 n_up] + E_x[2 n_dn]) / 2``, so the unpolarised enhancement factor is
evaluated on ``2 n_s`` and ``4 sigma_ss``. The reduced gradient is formed as ``s^2``, never ``s``,
so no square root of a vanishing gradient enters autograd.
"""

from __future__ import annotations

import math

import torch

from contract import XCRung

from .base import SemiLocalFunctional, spin_polarisation, wigner_seitz_radius
from .lda import pw92_epsilon

__all__ = ["PBEExchange", "PBECorrelation", "B88Exchange", "LYPCorrelation", "GGA"]

#: ``(3 pi^2)^(1/3)``: ``k_F = this * n^(1/3)``.
_KF = (3.0 * math.pi**2) ** (1.0 / 3.0)
#: Unpolarised Dirac exchange constant ``(3/4)(3/pi)^(1/3)``.
_SLATER = 0.75 * (3.0 / math.pi) ** (1.0 / 3.0)


def _reduced_gradient_squared(n: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """``s^2 = sigma / (4 k_F^2 n^2)`` for an unpolarised density ``n`` with ``sigma = |grad n|^2``."""
    return sigma / (4.0 * _KF**2 * n ** (8.0 / 3.0))


def _exchange_spin_scaled(enhancement, n_up, n_dn, sigma_uu, sigma_dd) -> torch.Tensor:
    """``(1/2) sum_s e_x^unif(2 n_s) F_x(s^2[2 n_s, 4 sigma_ss])``."""
    total = torch.zeros_like(n_up)
    for n_s, sig_s in ((n_up, sigma_uu), (n_dn, sigma_dd)):
        n2 = 2.0 * n_s
        s2 = _reduced_gradient_squared(n2, 4.0 * sig_s)
        total = total + 0.5 * (-_SLATER * n2 ** (4.0 / 3.0)) * enhancement(s2)
    return total


class PBEExchange(SemiLocalFunctional):
    """PBE exchange [A4] (libxc ``GGA_X_PBE``): ``F_x = 1 + kappa - kappa / (1 + mu s^2 / kappa)``."""

    rung = XCRung.GGA
    name = "gga_x_pbe"
    libxc_reference = ("gga_x_pbe",)
    #: PBE [A4] eq. (14): ``kappa = 0.804`` from the Lieb--Oxford bound, ``mu = beta pi^2 / 3``
    #: with ``beta = 0.066725``, to the digits libxc uses so G0.6 holds at 1e-10.
    kappa = 0.804
    mu = 0.2195149727645171

    def energy_density(self, n_up, n_dn, sigma_uu, sigma_ud, sigma_dd, tau_up, tau_dn):
        """Spin-scaled PBE exchange energy density."""
        kappa, mu = self.kappa, self.mu
        return _exchange_spin_scaled(
            lambda s2: 1.0 + kappa - kappa / (1.0 + mu * s2 / kappa), n_up, n_dn, sigma_uu, sigma_dd
        )


class PBECorrelation(SemiLocalFunctional):
    """PBE correlation [A4] (libxc ``GGA_C_PBE``): ``e_c^PW92mod + H(r_s, zeta, t)``."""

    rung = XCRung.GGA
    name = "gga_c_pbe"
    libxc_reference = ("gga_c_pbe",)
    #: PBE [A4] eqs. (4), (8): ``beta = 0.066725`` (high-density gradient coefficient),
    #: ``gamma = (1 - ln 2) / pi^2``.
    beta = 0.06672455060314922
    gamma = (1.0 - math.log(2.0)) / math.pi**2

    def energy_density(self, n_up, n_dn, sigma_uu, sigma_ud, sigma_dd, tau_up, tau_dn):
        """``n [e_c^unif(r_s, zeta) + H]``, PBE eqs. (3), (7), (8)."""
        n = n_up + n_dn
        rs = wigner_seitz_radius(n)
        zeta = spin_polarisation(n_up, n_dn)
        eps_unif = pw92_epsilon(rs, zeta, modified=True)
        phi = 0.5 * ((1.0 + zeta) ** (2.0 / 3.0) + (1.0 - zeta) ** (2.0 / 3.0))
        sigma = sigma_uu + 2.0 * sigma_ud + sigma_dd
        k_f = _KF * n ** (1.0 / 3.0)
        k_s_sq = 4.0 * k_f / math.pi
        # t^2 = sigma / (4 phi^2 k_s^2 n^2)
        t2 = sigma / (4.0 * phi * phi * k_s_sq * n * n)
        beta, gamma = self.beta, self.gamma
        phi3 = phi**3
        a = (beta / gamma) / (torch.exp(-eps_unif / (gamma * phi3)) - 1.0)
        at2 = a * t2
        h = gamma * phi3 * torch.log1p((beta / gamma) * t2 * (1.0 + at2) / (1.0 + at2 + at2 * at2))
        return n * (eps_unif + h)


class B88Exchange(SemiLocalFunctional):
    """Becke 1988 exchange [A5] (libxc ``GGA_X_B88``).

    Per spin: ``e_x = -C n_s^(4/3) - beta n_s^(4/3) x^2 / (1 + 6 beta x asinh x)`` with
    ``x = |grad n_s| / n_s^(4/3)`` and ``C = (3/2)(3/(4 pi))^(1/3)``. ``x asinh x`` is formed from
    ``x^2`` so that the zero-gradient limit has a finite autograd derivative.
    """

    rung = XCRung.GGA
    name = "gga_x_b88"
    libxc_reference = ("gga_x_b88",)
    #: B88 [A5]: ``beta = 0.0042`` fitted to Hartree--Fock atomic exchange energies.
    beta = 0.0042

    def energy_density(self, n_up, n_dn, sigma_uu, sigma_ud, sigma_dd, tau_up, tau_dn):
        """Sum over spin channels of the B88 form."""
        c_lda = 1.5 * (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)
        total = torch.zeros_like(n_up)
        for n_s, sig_s in ((n_up, sigma_uu), (n_dn, sigma_dd)):
            n43 = n_s ** (4.0 / 3.0)
            x2 = sig_s / (n43 * n43)
            x = x2.sqrt()
            total = total - c_lda * n43 - self.beta * n43 * x2 / (1.0 + 6.0 * self.beta * x * torch.asinh(x))
        return total


class LYPCorrelation(SemiLocalFunctional):
    """Lee--Yang--Parr correlation [A6] (libxc ``GGA_C_LYP``), in the Laplacian-free form.

    The Laplacian-free rewriting of Miehlich, Savin, Stoll and Preuss [A27] (Chem. Phys. Lett. 157,
    200, 1989), with the LYP constants ``a = 0.04918, b = 0.132, c = 0.2533, d = 0.349`` and
    ``C_F = (3/10)(3 pi^2)^(2/3)``.
    """

    rung = XCRung.GGA
    name = "gga_c_lyp"
    libxc_reference = ("gga_c_lyp",)
    a = 0.04918
    b = 0.132
    c = 0.2533
    d = 0.349

    def energy_density(self, n_up, n_dn, sigma_uu, sigma_ud, sigma_dd, tau_up, tau_dn):
        """Miehlich et al. eq. (2)."""
        a, b, c, d = self.a, self.b, self.c, self.d
        c_f = 0.3 * (3.0 * math.pi**2) ** (2.0 / 3.0)
        n = n_up + n_dn
        n_m13 = n ** (-1.0 / 3.0)
        denom = 1.0 + d * n_m13
        omega = torch.exp(-c * n_m13) / denom * n ** (-11.0 / 3.0)
        delta = c * n_m13 + d * n_m13 / denom
        sigma = sigma_uu + 2.0 * sigma_ud + sigma_dd
        nab = n_up * n_dn
        first = -a * 4.0 / denom * nab / n
        bracket = (
            nab
            * (
                2.0 ** (11.0 / 3.0) * c_f * (n_up ** (8.0 / 3.0) + n_dn ** (8.0 / 3.0))
                + (47.0 / 18.0 - 7.0 * delta / 18.0) * sigma
                - (2.5 - delta / 18.0) * (sigma_uu + sigma_dd)
                - (delta - 11.0) / 9.0 * (n_up / n * sigma_uu + n_dn / n * sigma_dd)
            )
            - (2.0 / 3.0) * n * n * sigma
            + ((2.0 / 3.0) * n * n - n_up * n_up) * sigma_dd
            + ((2.0 / 3.0) * n * n - n_dn * n_dn) * sigma_uu
        )
        return first - a * b * omega * bracket


class GGA(SemiLocalFunctional):
    """An exchange and a correlation part combined: the rung-2 functional the SCF consumes."""

    rung = XCRung.GGA

    def __init__(self, exchange: SemiLocalFunctional, correlation: SemiLocalFunctional, name: str) -> None:
        """Pair ``exchange`` with ``correlation``; ``name`` is the dispatch identifier."""
        self.exchange = exchange
        self.correlation = correlation
        self.name = name
        self.libxc_reference = exchange.libxc_reference + correlation.libxc_reference

    def energy_density(self, n_up, n_dn, sigma_uu, sigma_ud, sigma_dd, tau_up, tau_dn):
        """Exchange plus correlation."""
        args = (n_up, n_dn, sigma_uu, sigma_ud, sigma_dd, tau_up, tau_dn)
        return self.exchange.energy_density(*args) + self.correlation.energy_density(*args)
