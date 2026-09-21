#!/usr/bin/env python3
"""Compute the Gaussian-basis all-electron references NIST does not cover (O-12).

    python scripts/pyscf_reference.py            # writes src/cdft/reference/computed.py

PySCF spin-restricted Kohn--Sham, all-electron, point nuclei, at cc-pVQZ, cc-pV5Z and aug-cc-pV5Z
(the largest correlation-consistent sets PySCF ships) with ``grids.level = 9`` and
``conv_tol = 1e-12``, for He at LDA (VWN) and PBE and H2 at R = 1.4 bohr at LDA and PBE. The stored
value is the cc-pV5Z total and the stated uncertainty is ``max(|V5Z - VQZ|, |aug-V5Z - V5Z|)`` --
the basis-set truncation these Gaussian numbers carry and a grid code does not (D-03), so gate G4.7
compares within that uncertainty and never tighter. The functional strings are libxc identifiers, so
the reference uses exactly the functional the native implementation reproduces (G0.6). Every entry
is a ``COMPUTED_ORACLE`` (``ReferenceKind.CROSS_CODE``), not literature (D-29).
"""

from __future__ import annotations

import datetime
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "src" / "cdft" / "reference" / "computed.py"

SYSTEMS = {
    # scenario_id: (atom string in bohr, charge, xc code)
    "he_atom_lda": ("He 0 0 0", 0, "LDA_X,LDA_C_VWN"),
    "he_atom_pbe": ("He 0 0 0", 0, "GGA_X_PBE,GGA_C_PBE"),
    "h2_R1.4_lda": ("H 0 0 -0.7; H 0 0 0.7", 0, "LDA_X,LDA_C_VWN"),
    "h2_R1.4_pbe": ("H 0 0 -0.7; H 0 0 0.7", 0, "GGA_X_PBE,GGA_C_PBE"),
}
BASES = ("cc-pVQZ", "cc-pV5Z", "aug-cc-pV5Z")
VALUE_BASIS = "cc-pV5Z"


def compute(atom: str, charge: int, xc: str, basis: str) -> tuple[float, float]:
    """Return ``(total energy, HOMO eigenvalue)`` in Hartree."""
    from pyscf import dft, gto

    mol = gto.M(atom=atom, unit="Bohr", basis=basis, charge=charge, spin=0, verbose=0)
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.grids.level = 9
    mf.conv_tol = 1.0e-12
    mf.conv_tol_grad = 1.0e-8
    energy = mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"PySCF did not converge for {atom} / {xc} / {basis}")
    occupied = mf.mo_energy[mf.mo_occ > 0]
    return float(energy), float(occupied.max())


def main() -> int:
    import pyscf

    rows = {}
    for scenario_id, (atom, charge, xc) in SYSTEMS.items():
        values = {basis: compute(atom, charge, xc, basis) for basis in BASES}
        value = values[VALUE_BASIS][0]
        uncertainty = max(abs(values[b][0] - value) for b in BASES if b != VALUE_BASIS)
        rows[scenario_id] = {
            "value": value,
            "uncertainty": uncertainty,
            "homo": values[VALUE_BASIS][1],
            "per_basis": {basis: values[basis][0] for basis in BASES},
            "method": f"PySCF {pyscf.__version__} RKS all-electron, point nuclei, xc={xc}, "
                      f"basis {VALUE_BASIS} (uncertainty = max |other basis - {VALUE_BASIS}| over {BASES}), "
                      f"grids.level=9, conv_tol=1e-12",
            "xc": xc,
        }
        print(f"{scenario_id:14s} " + "  ".join(f"{b} {values[b][0]:.9f}" for b in BASES) + f"  unc {uncertainty:.1e}  HOMO {values[VALUE_BASIS][1]:.6f}")

    today = datetime.date.today().isoformat()
    lines = [
        '"""Computed Gaussian-basis references (COMPUTED_ORACLE, ``ReferenceKind.CROSS_CODE``) -- GENERATED.',
        "",
        f"Written by ``scripts/pyscf_reference.py`` on {today}; do not edit by hand, re-run the script.",
        "PySCF spin-restricted all-electron Kohn--Sham totals at cc-pV5Z with the largest basis-set",
        "difference to cc-pVQZ / aug-cc-pV5Z as the stated uncertainty (O-12). Gate G4.7 compares against ``value`` and reports",
        "``uncertainty`` beside it; it is a cross-code oracle, not literature (D-29).",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        f'GENERATED = "{today}"',
        "",
        "COMPUTED: dict[str, dict] = " + _pretty(rows),
        "",
        "",
        "def lookup_computed(scenario_id: str) -> dict | None:",
        '    """Return the computed reference row for a scenario id, or ``None``."""',
        "    return COMPUTED.get(scenario_id)",
        "",
    ]
    OUTPUT.write_text("\n".join(lines))
    print(f"written {OUTPUT}")
    return 0


def _pretty(rows: dict) -> str:
    import pprint

    return pprint.pformat(rows, width=100, sort_dicts=False)


if __name__ == "__main__":
    raise SystemExit(main())
