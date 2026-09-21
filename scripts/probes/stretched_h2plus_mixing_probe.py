#!/usr/bin/env python3
"""O-22 probe: solve ``h2plus_R8_lda`` with one mixing change at a time and print the SCF trajectory.

    python scripts/probes/stretched_h2plus_mixing_probe.py --kerker --max-iter 40 [--q0 1.5] [--alpha 0.15] [--device cuda]

Needs more than 6 GB of host RAM on the derived 116^3 box (D-81). Writes
``reports/h2plus_R8_mixing_<tag>.json`` with the energies, residuals and fallback events; a diagnostic, never a
production path (the mixing change that converges it becomes a decision, D-nn).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
for entry in (REPO_ROOT, REPO_ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kerker", action="store_true", help="switch the Kerker preconditioner on [F11]")
    parser.add_argument("--q0", type=float, default=None, help="Kerker q0 (default from MixingConfig)")
    parser.add_argument("--alpha", type=float, default=None, help="mixing alpha (default from MixingConfig)")
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument("--no-ladder", action="store_true", help="disable the fallback ladder [F12]")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()

    from cdft.config import numerics_for_spec
    from cdft.physics_config import REGISTRY
    from cdft.scf.solve import solve_scenario
    from contract import Device

    scenario = REGISTRY["h2plus_R8_lda"]
    base, _ = numerics_for_spec(scenario)
    mixing = base.mixing
    if args.kerker:
        mixing = dataclasses.replace(mixing, kerker=True, **({"kerker_q0": args.q0} if args.q0 else {}))
    if args.alpha:
        mixing = dataclasses.replace(mixing, alpha=args.alpha)
    scf = dataclasses.replace(base.scf, max_iterations=args.max_iter, fallback_ladder=not args.no_ladder)
    numerics = dataclasses.replace(
        base, mixing=mixing, scf=scf, device=Device(args.device),
        eigen=dataclasses.replace(base.eigen, cross_check=False),
    )
    tag = ("kerker" if args.kerker else "plain") + (f"_q{args.q0:g}" if args.q0 else "") + (f"_a{args.alpha:g}" if args.alpha else "")
    print("mixing:", mixing, "\nscf:", scf, flush=True)
    started = time.perf_counter()
    artifact = solve_scenario(scenario, numerics)
    elapsed = time.perf_counter() - started
    measurements = artifact.measurements
    print(f"status={artifact.status.value} wall={elapsed:.0f}s iterations={measurements.get('scf_iterations')} "
          f"stop={measurements.get('scf_stop_reason')}")
    print("fallbacks:", measurements.get("scf_fallbacks"))
    result = artifact.result
    if result is None:
        print("error:", artifact.error_message)
        return 1
    trajectory = result.trajectory
    for i, (energy, residual) in enumerate(zip(trajectory.energies, trajectory.residual_norms), 1):
        print(f"  it {i:3d}  E={energy:+.10f}  res={residual:.3e}")
    print("total energy", repr(result.energies.total), "eigenvalues", [repr(float(x)) for x in result.eigenvalues.flatten()])
    out = REPO_ROOT / "reports" / f"h2plus_R8_mixing_{tag}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "args": vars(args), "status": artifact.status.value, "wall_s": elapsed,
        "energies": [float(x) for x in trajectory.energies], "residuals": [float(x) for x in trajectory.residual_norms],
        "fallbacks": measurements.get("scf_fallbacks"), "total": float(result.energies.total),
        "eigenvalues": [float(x) for x in result.eigenvalues.flatten()], "grid": measurements.get("grid"),
        "gpu_peak_bytes": measurements.get("gpu_peak_bytes"),
    }, indent=1))
    print("written", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
