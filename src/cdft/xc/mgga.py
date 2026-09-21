"""Rung-3 functionals -- r2SCAN [A11], [A12]: deferred, deliberately not transcribed.

The reference definitions were unreachable and nothing is filled in from memory (D-29), so
requesting ``r2scan`` raises :data:`R2SCAN_DEFERRAL_REASON` rather than returning a stub. The
meta-GGA SCF terms -- ``-(1/2) div(v_tau grad psi)`` in ``Hamiltonian.apply`` and
``(1/2) integral v_tau sum_i f_i |grad psi_i|^2`` in the energy -- are deferred with it, so the
whole rung is deferred, not only the transcription.
"""

from __future__ import annotations

from contract import XCRung

from .base import SemiLocalFunctional

__all__ = ["R2SCAN", "R2SCAN_DEFERRAL_REASON"]

R2SCAN_DEFERRAL_REASON = (
    "r2SCAN is DEFERRED (I2 follow-up): its reference definitions (libxc Maple source, "
    "gitlab.com/libxc and tddft.org) were unreachable when the functional was scheduled, and the project "
    "does not transcribe fitted constants from memory (D-29). The meta-GGA SCF terms "
    "(orbital-dependent v_tau operator) are deferred with it."
)


class R2SCAN(SemiLocalFunctional):
    """Placeholder that refuses to evaluate, naming the deferral (never a silent fallback, G5.5)."""

    rung = XCRung.META_GGA
    name = "r2scan"
    libxc_reference = ("mgga_x_r2scan", "mgga_c_r2scan")

    def energy_density(self, n_up, n_dn, sigma_uu, sigma_ud, sigma_dd, tau_up, tau_dn):
        """Raise: this functional is not implemented."""
        raise NotImplementedError(R2SCAN_DEFERRAL_REASON)
