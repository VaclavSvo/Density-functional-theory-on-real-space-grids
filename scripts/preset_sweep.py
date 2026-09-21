#!/usr/bin/env python3
"""Measure eigenvalue error against box size and grid spacing, for the preset retune (D-38).

Run as ``python scripts/preset_sweep.py atoms`` / ``molecules``. Writes a JSON table to
``reports/preset_sweep_<leg>.json``. The table, not an argument, is what the retune is derived from.
"""
from __future__ import annotations

import dataclasses
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from cdft.config import ALL_ELECTRON_NUMERICS
from cdft.physics_config import REGISTRY
from cdft.scf.noninteracting import solve_noninteracting

BY_ID = {s.scenario_id: s for s in REGISTRY}

# Electronic eigenvalue references, NOT total energies: exact hydrogenic -Z^2/2 for H and He+;
# H2+ at R = 2 from Madsen & Peek 1971 (total -0.6026342144949) minus the 1/R = 0.5 repulsion.
EXACT = {"h_atom": -0.5, "he_plus": -2.0, "h2plus_R2": -0.6026342144949 - 0.5}


def measure(sid: str, h: float, edge: float) -> dict:
    """Solve one scenario at this spacing and box edge and report the error against a reference."""
    scenario = BY_ID[sid]
    npts = int(round(edge / h)) + 1
    grid = dataclasses.replace(ALL_ELECTRON_NUMERICS.grid, spacing=h, box_lengths=(edge, edge,
    edge))
    numerics = dataclasses.replace(
        ALL_ELECTRON_NUMERICS, grid=grid,
        eigen=dataclasses.replace(ALL_ELECTRON_NUMERICS.eigen, cross_check=False),
    )
    t0 = time.perf_counter()
    artifact = solve_noninteracting(scenario, numerics, n_states=1)
    dt = time.perf_counter() - t0
    e = float(artifact.result.eigenvalues.reshape(-1)[0])
    err = abs(e - EXACT[sid]) if sid in EXACT else float("nan")
    row = dict(scenario=sid, h=h, box=edge, npts=npts, energy=e, error=err,
               seconds=dt, status=artifact.status.value)
    print(
        f"{sid:10s} h={h:<5g} L={edge:<5g} {npts:3d}^3  E={e:+.12f}  err={err:.2e}  "
        f"{dt:7.2f}s  {artifact.status.value}",
        flush=True,
    )
    return row


def main() -> None:
    """Run the requested sweep leg and write its table to reports/."""
    leg = sys.argv[1] if len(sys.argv) > 1 else "atoms"
    rows: list[dict] = []
    if leg == "atoms":
        print("### box sweep at fixed h  (is accuracy bought with box size?)")
        for sid, h in (("h_atom", 0.8), ("he_plus", 0.8)):
            for edge in (8.0, 12.0, 16.0, 20.0, 24.0, 32.0, 40.0):
                rows.append(measure(sid, h, edge))
        print("\n### spacing sweep at a generous box  (does resolution buy anything?)")
        for sid, edge in (("h_atom", 32.0), ("he_plus", 16.0)):
            for h in (1.6, 1.0, 0.8, 0.5, 0.4):
                rows.append(measure(sid, h, edge))
    else:
        print("### molecular spacing sweep at a fixed generous box")
        for h in (1.0, 0.8, 0.5, 0.4, 0.32, 0.25):
            rows.append(measure("h2plus_R2", h, 20.0))
        print("\n### molecular box sweep at fixed h")
        for edge in (10.0, 14.0, 20.0, 24.0):
            rows.append(measure("h2plus_R2", 0.5, edge))
    out = pathlib.Path(__file__).resolve().parents[1] / "reports" / f"preset_sweep_{leg}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
