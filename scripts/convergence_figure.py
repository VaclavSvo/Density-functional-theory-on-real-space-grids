#!/usr/bin/env python3
"""Energy error against grid spacing: the two refinement ladders drawn in the README.

    python scripts/convergence_figure.py run  [--out docs/figures/convergence_ladders.json]
    python scripts/convergence_figure.py draw [--data docs/figures/convergence_ladders.json]

``run`` solves the two ladders and writes their rungs: bare H2+ at R = 2 bohr against the
two-centre oracle (spacing-limited, A-1), and He at LDA (VWN) against NIST SRD 141 on a 16-bohr box
(interpolation-limited to ~O(h^3), A-10). ``draw`` reads the JSON and writes
``docs/figures/convergence.svg`` and ``.png``. The data file is versioned so that the figure can be
redrawn without the solves (about 40 minutes on two CPU cores).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
for entry in (REPO_ROOT, REPO_ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

DATA_DEFAULT = REPO_ROOT / "docs" / "figures" / "convergence_ladders.json"
FIGURE_STEM = REPO_ROOT / "docs" / "figures" / "convergence"

#: Bare H2+ spacings, each an integer fraction of the bond length so both protons stay on the lattice.
H2PLUS_SPACINGS = (0.5, 0.4, 1 / 3, 0.25, 0.2, 1 / 6)
#: Helium at LDA on the 16-bohr box (an integer number of cells at every spacing).
HE_SPACINGS = (0.5, 0.4, 0.32, 0.25, 0.2)
HE_BOX = 16.0


def run_h2plus() -> dict:
    """Solve bare H2+ at R = 2 on the derived box at each spacing of the lattice rule."""
    from cdft.config import ALL_ELECTRON_NUMERICS
    from cdft.grid import grid_rule
    from cdft.physics_config import REGISTRY
    from cdft.reference.two_centre import h2_plus_energy
    from cdft.scf.solve import solve_scenario

    scenario = REGISTRY["h2plus_R2"]
    reference = h2_plus_energy(2.0).electronic_energy
    rungs = []
    for h in H2PLUS_SPACINGS:
        started = time.perf_counter()
        with grid_rule(molecular_spacing=h):
            artifact = solve_scenario(scenario, ALL_ELECTRON_NUMERICS, cross_check=False)
        wall = time.perf_counter() - started
        eigenvalue = float(artifact.result.eigenvalues[0, 0])
        grid = artifact.measurements["grid"]
        rung = {
            "h": grid["spacing_bohr"],
            "n_points": grid["n_points"],
            "value": eigenvalue,
            "error": eigenvalue - reference,
            "wall_seconds": wall,
        }
        rungs.append(rung)
        print(f"h2plus_R2  h={rung['h']:.4f}  err={rung['error']:+.3e}  {wall:.0f} s", flush=True)
    return {
        "label": "H₂⁺, R = 2 bohr, bare nuclei vs the two-centre oracle",
        "reference": reference,
        "reference_source": "prolate-spheroidal two-centre solver (cdft.reference.two_centre)",
        "quantity": "electronic ground-state eigenvalue",
        "rungs": rungs,
    }


def run_helium() -> dict:
    """Solve He at LDA (VWN) on a fixed 16-bohr box at each spacing, against NIST SRD 141."""
    from cdft.config import ALL_ELECTRON_NUMERICS
    from cdft.physics_config import helium, lda_functional
    from cdft.reference.literature import lookup
    from cdft.scf.loop import solve_self_consistent

    scenario = dataclasses.replace(helium("he_lda_ladder"), xc=lda_functional("vwn"))
    reference = lookup("He/lda/total_energy").value
    rungs = []
    for h in HE_SPACINGS:
        numerics = dataclasses.replace(
            ALL_ELECTRON_NUMERICS,
            grid=dataclasses.replace(ALL_ELECTRON_NUMERICS.grid, spacing=h, box_lengths=(HE_BOX,) * 3),
            eigen=dataclasses.replace(ALL_ELECTRON_NUMERICS.eigen, cross_check=False),
        )
        started = time.perf_counter()
        artifact = solve_self_consistent(scenario, numerics, derive_grid=False)
        wall = time.perf_counter() - started
        total = artifact.result.energies.total
        rung = {
            "h": h,
            "box": HE_BOX,
            "n_points": artifact.measurements["grid"]["n_points"],
            "value": total,
            "error": total - reference,
            "scf_iterations": artifact.result.n_iterations,
            "wall_seconds": wall,
        }
        rungs.append(rung)
        print(f"he_lda     h={h:.4f}  err={rung['error']:+.3e}  it={rung['scf_iterations']}  {wall:.0f} s", flush=True)
    return {
        "label": "He, LDA (VWN), 16-bohr box vs NIST SRD 141",
        "reference": reference,
        "reference_source": "NIST SRD 141, He/lda/total_energy",
        "quantity": "total energy",
        "rungs": rungs,
    }


def draw(data_path: pathlib.Path) -> None:
    """Draw the two ladders on one log-log axis with an O(h^3) guide line."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    data = json.loads(data_path.read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(5.4, 3.7), dpi=200)
    styles = {"h2plus_R2": ("o-", "#1f5fa8"), "he_lda": ("s-", "#c0392b")}
    for key, (marker, colour) in styles.items():
        rungs = sorted(data[key]["rungs"], key=lambda r: r["h"])
        h = [r["h"] for r in rungs]
        err = [abs(r["error"]) for r in rungs]
        ax.loglog(h, err, marker, color=colour, ms=4.5, lw=1.4, label=data[key]["label"])
        # Mark a change of sign between neighbouring rungs: |error| dips there for no numerical reason.
        for a, b in zip(rungs, rungs[1:]):
            if a["error"] * b["error"] < 0:
                x = (a["h"] * b["h"]) ** 0.5
                ax.annotate("error changes sign", xy=(x, min(abs(a["error"]), abs(b["error"]))),
                            xytext=(x * 1.04, 1.35e-6), fontsize=6.5, color=colour, ha="left",
                            arrowprops=dict(arrowstyle="-", color=colour, lw=0.6))
    he = {round(r["h"], 4): abs(r["error"]) for r in data["he_lda"]["rungs"]}
    anchor_h, anchor_e = 0.25, he[0.25]
    hs = [0.16, 0.5]
    ax.loglog(hs, [anchor_e * (x / anchor_h) ** 3 for x in hs], "--", color="0.45", lw=1.0,
              label="O(h³) through the He point at h = 0.25")
    ax.axhline(1.5936e-3, color="0.6", lw=0.8, ls=":")
    ax.text(0.545, 1.5936e-3 * 0.72, "chemical accuracy, 1 kcal/mol", fontsize=6.5, color="0.4", ha="right")
    ax.axvline(0.25, color="0.8", lw=0.8)
    ax.text(0.245, 1.3e-6, "production h", fontsize=6.5, color="0.4", rotation=90, ha="right", va="bottom")
    ax.set_xlabel("grid spacing h (bohr)")
    ax.set_ylabel("|energy error| (Ha)")
    ax.set_xlim(0.155, 0.56)
    ax.set_ylim(1e-6, 1e-2)
    ax.set_xticks([0.16, 0.2, 0.25, 0.3, 0.4, 0.5])
    ax.set_xticklabels(["0.16", "0.2", "0.25", "0.3", "0.4", "0.5"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", lw=0.3, alpha=0.6)
    ax.legend(fontsize=6.8, loc="upper left", frameon=False)
    ax.set_title("Energy error against grid spacing", fontsize=9)
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(f"{FIGURE_STEM}.{ext}")
    print(f"wrote {FIGURE_STEM}.svg and .png")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="solve both ladders and write the JSON")
    run_p.add_argument("--out", default=str(DATA_DEFAULT))
    run_p.add_argument("--threads", type=int, default=2)
    run_p.add_argument("--only", choices=["h2plus_R2", "he_lda"], default=None)
    draw_p = sub.add_parser("draw", help="draw the figure from the JSON")
    draw_p.add_argument("--data", default=str(DATA_DEFAULT))
    args = parser.parse_args(argv)

    if args.command == "run":
        import torch

        torch.set_num_threads(args.threads)
        out = pathlib.Path(args.out)
        data = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
        if args.only in (None, "h2plus_R2"):
            data["h2plus_R2"] = run_h2plus()
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(data, indent=2), encoding="utf-8")
        if args.only in (None, "he_lda"):
            data["he_lda"] = run_helium()
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(data, indent=2), encoding="utf-8")
        data["provenance"] = {"torch": torch.__version__, "threads": args.threads, "device": "cpu"}
        out.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"wrote {out}")
        return 0
    draw(pathlib.Path(args.data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
