"""libxc as a pointwise oracle, reached through PySCF -- **never imported by the solver** (D-05).

G0.6 asks whether a native functional reproduces libxc [I1] to 1e-10 at random points. This is the
only place libxc is called, and only the gate and the tests import it; a static check in
``tests/test_xc.py`` asserts nothing else under ``cdft.xc`` touches ``pyscf``.

Conversions to the contract's conventions happen here: libxc returns the energy per particle and
``e_xc`` is per volume, so the return is multiplied by the total density; PySCF's ``vrho``
``(n_pts, 2)`` and ``vsigma`` ``(n_pts, 3)``, ordered ``(uu, ud, dd)``, are transposed to
``(n_spin, n_pts)`` and ``(3, n_pts)``. Inputs are ``sigma`` and ``tau``, from which a gradient
along ``x`` of magnitude ``sqrt(sigma_ss)`` is built, ``sigma_ud`` entering as the angle between
the two spin gradients.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["LibxcReference", "libxc_available", "evaluate_libxc"]


def libxc_available() -> bool:
    """Whether PySCF (and hence libxc) can be imported."""
    try:
        from pyscf.dft import libxc  # noqa: F401
    except Exception:  # pragma: no cover - environment dependent
        return False
    return True


@dataclass(slots=True)
class LibxcReference:
    """libxc's answer at a set of points, in the contract's conventions."""

    e_xc: np.ndarray
    """Energy density per volume, ``(n_pts,)``."""
    v_rho: np.ndarray
    """``(n_spin, n_pts)``."""
    v_sigma: np.ndarray | None
    """``(n_sigma, n_pts)`` or ``None`` for LDA."""
    v_tau: np.ndarray | None
    """``(n_spin, n_pts)`` or ``None`` below meta-GGA."""
    xc_code: str
    libxc_version: str


def _gradients_from_sigma(
    sigma_uu: np.ndarray, sigma_ud: np.ndarray, sigma_dd: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return two gradient vectors ``(3, n)`` realising the given sigma triple.

    ``grad_up`` along ``x`` with ``|grad_up|^2 = sigma_uu``; ``grad_dn`` in the ``xy`` plane with
    ``|grad_dn|^2 = sigma_dd`` and ``grad_up . grad_dn = sigma_ud``. Requires
    ``sigma_ud^2 <= sigma_uu sigma_dd`` (Cauchy--Schwarz), which any physical triple satisfies.
    """
    g_up = np.sqrt(sigma_uu)
    g_dn = np.sqrt(sigma_dd)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.where(g_up * g_dn > 0.0, sigma_ud / np.where(g_up * g_dn > 0.0, g_up * g_dn, 1.0), 1.0)
    cos = np.clip(cos, -1.0, 1.0)
    sin = np.sqrt(np.clip(1.0 - cos * cos, 0.0, 1.0))
    grad_up = np.stack([g_up, np.zeros_like(g_up), np.zeros_like(g_up)])
    grad_dn = np.stack([g_dn * cos, g_dn * sin, np.zeros_like(g_dn)])
    return grad_up, grad_dn


def evaluate_libxc(
    xc_code: str,
    density: np.ndarray,
    sigma: np.ndarray | None = None,
    tau: np.ndarray | None = None,
) -> LibxcReference:
    """Evaluate libxc for contract-shaped inputs.

    ``xc_code`` is comma-separated libxc identifiers, e.g. ``"GGA_X_PBE,GGA_C_PBE"``; ``density``
    is ``(n_spin, n_pts)``, ``sigma`` ``(1, n_pts)`` or ``(3, n_pts)`` ordered ``(uu, ud, dd)`` and
    required from GGA up, ``tau`` ``(n_spin, n_pts)`` and required for meta-GGA.
    """
    from pyscf.dft import libxc

    density = np.asarray(density, dtype=np.float64)
    n_spin, n_pts = density.shape
    kind = libxc.xc_type(xc_code)
    n_components = {"LDA": 1, "GGA": 4, "MGGA": 6}[kind]
    if kind in ("GGA", "MGGA") and sigma is None:
        raise ValueError(f"{xc_code} needs sigma")
    if kind == "MGGA" and tau is None:
        raise ValueError(f"{xc_code} needs tau")

    def channel(n, grad, tau_s):
        rho = np.zeros((n_components, n_pts))
        rho[0] = n
        if n_components >= 4:
            rho[1:4] = grad
        if n_components == 6:
            rho[5] = tau_s
        return rho

    if n_spin == 1:
        if sigma is not None:
            grad = np.zeros((3, n_pts))
            grad[0] = np.sqrt(np.asarray(sigma, dtype=np.float64)[0])
        else:
            grad = np.zeros((3, n_pts))
        rho = channel(density[0], grad, None if tau is None else np.asarray(tau)[0])
        exc, vxc, _, _ = libxc.eval_xc(xc_code, rho, spin=0, deriv=1)
        v_rho = np.asarray(vxc[0]).reshape(1, n_pts)
        v_sigma = None if kind == "LDA" else np.asarray(vxc[1]).reshape(1, n_pts)
        v_tau = None if kind != "MGGA" else np.asarray(vxc[3]).reshape(1, n_pts)
        total = density[0]
    else:
        if sigma is not None:
            s = np.asarray(sigma, dtype=np.float64)
            grad_up, grad_dn = _gradients_from_sigma(s[0], s[1], s[2])
        else:
            grad_up = grad_dn = np.zeros((3, n_pts))
        t = None if tau is None else np.asarray(tau, dtype=np.float64)
        rho_up = channel(density[0], grad_up, None if t is None else t[0])
        rho_dn = channel(density[1], grad_dn, None if t is None else t[1])
        exc, vxc, _, _ = libxc.eval_xc(xc_code, (rho_up, rho_dn), spin=1, deriv=1)
        v_rho = np.asarray(vxc[0]).reshape(n_pts, 2).T
        v_sigma = None if kind == "LDA" else np.asarray(vxc[1]).reshape(n_pts, 3).T
        v_tau = None if kind != "MGGA" else np.asarray(vxc[3]).reshape(n_pts, 2).T
        total = density[0] + density[1]

    version = getattr(libxc, "__version__", None) or str(libxc.libxc_version())
    return LibxcReference(
        e_xc=np.asarray(exc) * total,
        v_rho=v_rho,
        v_sigma=v_sigma,
        v_tau=v_tau,
        xc_code=xc_code,
        libxc_version=version,
    )
