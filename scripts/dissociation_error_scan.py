#!/usr/bin/env python3
"""Error against the two-centre oracle as a function of bond length (ledger entry A-2).

    python scripts/dissociation_error_scan.py     # writes reports/dissociation_error_scan.json

Sets the priority of A-1 by answering whether the multi-centre error is a smooth systematic --
which largely cancels in the energy differences this project reports, making a 1e-3 Ha absolute
error far less serious than it looks -- or has structure, which does not cancel and would make
every dissociation curve unusable.
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
from cdft.physics_config import h2_plus
from cdft.reference.two_centre import solve_two_centre_extrapolated
from cdft.scf.noninteracting import solve_noninteracting

BONDS = (0.8, 1.0, 1.2, 1.4, 1.6, 2.0, 2.4, 3.0, 4.0, 5.0, 6.0)

rows = []
print(f"{'R':>5s} {'computed':>15s} {'oracle':>15s} {'error':>11s} {'x chem acc':>11s} {'s':>7s}")
for R in BONDS:
    scenario = h2_plus(bond_length=R, scenario_id=f"h2plus_R{R:g}")
    numerics = dataclasses.replace(
        ALL_ELECTRON_NUMERICS,
        eigen=dataclasses.replace(ALL_ELECTRON_NUMERICS.eigen, cross_check=False),
    )
    t0 = time.perf_counter()
    artifact = solve_noninteracting(scenario, numerics, n_states=1)
    dt = time.perf_counter() - t0
    computed = float(artifact.result.eigenvalues[0][0])
    reference = solve_two_centre_extrapolated(bond_length=R, charges=(1.0, 1.0))
    exact = float(reference.electronic_energy)
    err = computed - exact
    rows.append(dict(bond_length=R, computed=computed, oracle=exact, error=err,
                     seconds=dt, oracle_residual=float(reference.residual)))
    print(
        f"{R:5.1f} {computed:+15.9f} {exact:+15.9f} {err:+11.3e} "
        f"{err / 1.5936e-3:+11.2f} {dt:7.1f}",
        flush=True,
    )

out = pathlib.Path(__file__).resolve().parents[1] / "reports" / "dissociation_error_scan.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

# Second differences on a near-uniform sample are the cheapest smoothness test.
print("\nsecond differences of the error (smoothness probe):")
for i in range(1, len(rows) - 1):
    d2 = rows[i-1]["error"] - 2*rows[i]["error"] + rows[i+1]["error"]
    print(f"  R={rows[i]['bond_length']:4.1f}  d2 = {d2:+.3e}")
print(f"\nwrote {out}")
