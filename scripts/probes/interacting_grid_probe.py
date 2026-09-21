#!/usr/bin/env python3
"""The interacting grid rule, by measurement (D-53, D-54).

    python scripts/probes/interacting_grid_probe.py [--out reports/interacting_grid_probe.json]

On the hydrogenic 1s density ``n = (Z^3/pi) f^2`` (the exact Phase-1 density of H and He+, every
integral closed-form) at a ladder of spacings, compare the plain ``h^3`` sum with the cusp-aware
quadrature of ``cdft.operators.quadrature`` on ``int n - N`` (exact 0), ``E_H`` (exact ``5Z/16``),
``int n v_H`` (exact ``5Z/8``) and ``E_xc^LDA(VWN)`` -- the last against a 400001-point radial
trapezoid of the same functional on the same density. ``v_H`` is the closed-form 1s potential.

Evidence for D-53: on the D-42 atomic grid (``h = 1/Z``) the plain rule is wrong by 15 % in the
charge and 0.19--0.37 Ha in ``E_H``, while the cusp-aware rule is already exact to 1e-8 there, so
the interacting spacing is set by the interpolation of the self-consistent density, not by the
quadrature of the cusp. Writes the table as JSON to ``--out``.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
for entry in (REPO_ROOT, REPO_ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from contract import BoundaryMode, DomainMode, GridConfig  # noqa: E402

CASES = {
    1.0: [(1.0, 32.0), (0.5, 24.0), (0.25, 20.0), (0.2, 20.0)],
    2.0: [(0.5, 16.0), (0.25, 12.0), (0.2, 12.0), (0.16, 12.0), (0.125, 12.0)],
}


def radial_xc(functional, charge: float) -> float:
    """``int e_xc`` of the 1s density by a fine one-dimensional trapezoid in ``r``."""
    c = charge**3 / math.pi
    r = np.linspace(0.0, 40.0 / charge, 400001)
    n = c * np.exp(-2.0 * charge * r)
    e = functional.evaluate(torch.tensor(n[None, :], dtype=torch.float64)).e_xc.numpy()
    return float(np.trapezoid(4.0 * math.pi * r**2 * e, r))


def probe(charge: float, h: float, edge: float, functional) -> dict:
    """One rung: every integral, plain and cusp-aware, with its error."""
    from cdft.grid import UniformGrid
    from cdft.operators.cusp import CuspFactor
    from cdft.operators.quadrature import CuspQuadrature, hydrogenic_1s_potential

    config = GridConfig(
        spacing=h, fd_order=8, domain=DomainMode.BOX, boundary=BoundaryMode.ZERO,
        box_lengths=(edge, edge, edge), use_double_grid=False, fourier_filter_projectors=False,
    )
    grid = UniformGrid.from_config(config)
    factor = CuspFactor(grid, (charge,), torch.zeros((1, 3), dtype=torch.float64))
    started = time.perf_counter()
    quadrature = CuspQuadrature(grid, factor)
    build_seconds = time.perf_counter() - started

    c = charge**3 / math.pi
    r_nodes = quadrature.points.norm(dim=-1)
    r_grid = grid.points().norm(dim=-1)
    n_nodes = c * quadrature.f2
    n_grid = c * factor.weight
    v_nodes = c * hydrogenic_1s_potential(charge, r_nodes)
    v_grid = c * hydrogenic_1s_potential(charge, r_grid)

    charge_plain = float(grid.integrate(n_grid))
    charge_quad = float(quadrature.mass_weights.sum()) * c
    nv_plain = float(grid.integrate(n_grid * v_grid))
    nv_quad = float(quadrature.integrate(n_nodes * v_nodes, n_grid * v_grid))
    e_nodes = functional.evaluate(n_nodes[None, :]).e_xc
    e_grid = functional.evaluate(n_grid[None, :]).e_xc
    xc_plain = float(grid.integrate(e_grid))
    xc_quad = float(quadrature.integrate(e_nodes, e_grid))
    xc_radial = radial_xc(functional, charge)
    return {
        "Z": charge, "h": h, "edge": edge, "n_points": grid.n_points, "n_nodes": quadrature.n_nodes,
        "sphere_radius": float(quadrature.spheres[0].radius_sphere), "build_seconds": build_seconds,
        "min_mass_weight": float(quadrature.mass_weights.min()),
        "charge_error": {"plain": charge_plain - 1.0, "quadrature": charge_quad - 1.0},
        "hartree_error": {"plain": 0.5 * nv_plain - 5.0 * charge / 16.0, "quadrature": 0.5 * nv_quad - 5.0 * charge / 16.0},
        "n_vh_error": {"plain": nv_plain - 5.0 * charge / 8.0, "quadrature": nv_quad - 5.0 * charge / 8.0},
        "xc_error": {"plain": xc_plain - xc_radial, "quadrature": xc_quad - xc_radial, "radial_reference": xc_radial},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(REPO_ROOT / "reports" / "interacting_grid_probe.json"))
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    from cdft.xc import functional_by_name

    functional = functional_by_name("lda_vwn")
    rows = []
    print(f"{'Z':>3} {'h':>6} {'edge':>5} | {'int n - N':^23} | {'E_H - 5Z/16':^23} | {'E_xc - radial':^23}")
    print(f"{'':>3} {'':>6} {'':>5} | {'plain':>11} {'quad':>11} | {'plain':>11} {'quad':>11} | {'plain':>11} {'quad':>11}")
    for charge, cases in CASES.items():
        for h, edge in cases:
            row = probe(charge, h, edge, functional)
            rows.append(row)
            print(
                f"{charge:3.0f} {h:6.3f} {edge:5.1f} | {row['charge_error']['plain']:+11.2e} {row['charge_error']['quadrature']:+11.2e}"
                f" | {row['hartree_error']['plain']:+11.2e} {row['hartree_error']['quadrature']:+11.2e}"
                f" | {row['xc_error']['plain']:+11.2e} {row['xc_error']['quadrature']:+11.2e}",
                flush=True,
            )
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"functional": "lda_vwn", "rows": rows}, indent=1))
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
