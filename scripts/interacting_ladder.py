#!/usr/bin/env python3
"""Refinement ladders of the self-consistent all-electron atoms against NIST SRD 141.

    python scripts/interacting_ladder.py [--systems he_lda,h_lda,he_plus_lda,he_pbe,h_pbe] \\
        [--spacings 0.5,0.35,0.25,0.2,0.16,0.125] [--box 12] [--out reports/interacting_ladder.json]

Every rung is one converged SCF at fixed box and spacing with the D-42 derivation bypassed
(``derive_grid=False``), so the sweep varies exactly the number it claims to. The JSON records each
rung's energy terms, eigenvalue, SCF and eigensolver diagnostics, wall time, and the difference from
the transcribed NIST value where one exists (``cdft.reference.literature``, read there, never
retyped). The fitted order between consecutive rungs is evidence, never a verdict (D-48).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pathlib
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))


def systems():
    from cdft.physics_config import helium, hydrogenic, lda_functional, pbe_functional

    return {
        "he_lda": (dataclasses.replace(helium("he_lda"), xc=lda_functional("vwn")), "He/lda/total_energy"),
        "he_plus_lda": (dataclasses.replace(hydrogenic(2.0, "He", "he_plus_lda"), xc=lda_functional("vwn")), "He+/lda/total_energy"),
        "h_lda": (dataclasses.replace(hydrogenic(1.0, "H", "h_lda"), xc=lda_functional("vwn")), "H/lda/total_energy"),
        "he_pbe": (dataclasses.replace(helium("he_pbe"), xc=pbe_functional()), None),
        "h_pbe": (dataclasses.replace(hydrogenic(1.0, "H", "h_pbe"), xc=pbe_functional()), None),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--systems", default="he_lda,h_lda,he_plus_lda")
    parser.add_argument("--spacings", default="0.5,0.35,0.25,0.2,0.16,0.125")
    parser.add_argument("--box", type=float, default=12.0)
    parser.add_argument("--out", default=str(REPO_ROOT / "reports" / "interacting_ladder.json"))
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)

    import torch

    torch.set_num_threads(args.threads)
    from cdft.config import ALL_ELECTRON_NUMERICS
    from cdft.reference.literature import lookup
    from cdft.scf.loop import solve_self_consistent

    catalogue = systems()
    out_path = pathlib.Path(args.out)
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for name in args.systems.split(","):
        scenario, nist_key = catalogue[name]
        reference = lookup(nist_key).value if nist_key else None
        rows = results.setdefault(name, {"reference": reference, "reference_key": nist_key, "rungs": []})
        previous = None
        for h in (float(x) for x in args.spacings.split(",")):
            numerics = dataclasses.replace(
                ALL_ELECTRON_NUMERICS,
                grid=dataclasses.replace(ALL_ELECTRON_NUMERICS.grid, spacing=h, box_lengths=(args.box,) * 3),
                eigen=dataclasses.replace(ALL_ELECTRON_NUMERICS.eigen, cross_check=False),
            )
            started = time.perf_counter()
            artifact = solve_self_consistent(scenario, numerics, derive_grid=False)
            elapsed = time.perf_counter() - started
            if artifact.result is None:
                row = {"h": h, "box": args.box, "status": artifact.status.value, "error": artifact.error_message}
            else:
                e = artifact.result.energies
                row = {
                    "h": h,
                    "box": args.box,
                    "status": artifact.status.value,
                    "total": e.total,
                    "harris_foulkes": e.harris_foulkes,
                    "kinetic": e.kinetic,
                    "external": e.external,
                    "hartree": e.hartree,
                    "xc": e.xc,
                    "eigenvalue": float(artifact.result.eigenvalues[0, 0]),
                    "scf_iterations": artifact.result.n_iterations,
                    "n_points": artifact.measurements["grid"]["n_points"],
                    "external_consistency": artifact.measurements["external_consistency"],
                    "charge_error": artifact.measurements["charge_error"],
                    "wall_seconds": elapsed,
                    "error_vs_reference": (e.total - reference) if reference is not None else None,
                }
                if previous is not None and reference is not None and previous.get("total") is not None:
                    e_prev, e_now = previous["total"] - reference, e.total - reference
                    if e_prev * e_now > 0 and previous["h"] != h:
                        row["fitted_order_vs_previous"] = math.log(abs(e_prev) / abs(e_now)) / math.log(previous["h"] / h)
            rows["rungs"] = [r for r in rows["rungs"] if not (r["h"] == h and r["box"] == args.box)] + [row]
            previous = row
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(results, indent=2))
            summary = f"E={row.get('total')}" if "total" in row else row.get("error")
            err = row.get("error_vs_reference")
            print(f"{name:12s} h={h:<6} box={args.box:<5} {summary} err={err if err is None else f'{err:+.2e}'} it={row.get('scf_iterations')} {elapsed:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
