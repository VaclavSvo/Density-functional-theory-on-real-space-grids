"""Computed Gaussian-basis references (COMPUTED_ORACLE, ``ReferenceKind.CROSS_CODE``) -- GENERATED.

Written by ``scripts/pyscf_reference.py`` on 2026-09-14; do not edit by hand, re-run the script.
PySCF spin-restricted all-electron Kohn--Sham totals at cc-pV5Z with the largest basis-set
difference to cc-pVQZ / aug-cc-pV5Z as the stated uncertainty (O-12). Gate G4.7 compares against ``value`` and reports
``uncertainty`` beside it; it is a cross-code oracle, not literature (D-29).
"""

from __future__ import annotations

GENERATED = "2026-09-14"

COMPUTED: dict[str, dict] = {'he_atom_lda': {'value': -2.8347276033566433,
                 'uncertainty': 0.0001628604466605843,
                 'homo': -0.5699405024548679,
                 'per_basis': {'cc-pVQZ': -2.8345647429099827,
                               'cc-pV5Z': -2.8347276033566433,
                               'aug-cc-pV5Z': -2.8347859194318965},
                 'method': 'PySCF 2.14.0 RKS all-electron, point nuclei, xc=LDA_X,LDA_C_VWN, basis '
                           "cc-pV5Z (uncertainty = max |other basis - cc-pV5Z| over ('cc-pVQZ', "
                           "'cc-pV5Z', 'aug-cc-pV5Z')), grids.level=9, conv_tol=1e-12",
                 'xc': 'LDA_X,LDA_C_VWN'},
 'he_atom_pbe': {'value': -2.8928294617067456,
                 'uncertainty': 0.0001791448769417059,
                 'homo': -0.5788016926060725,
                 'per_basis': {'cc-pVQZ': -2.892650316829804,
                               'cc-pV5Z': -2.8928294617067456,
                               'aug-cc-pV5Z': -2.8928830913638093},
                 'method': 'PySCF 2.14.0 RKS all-electron, point nuclei, xc=GGA_X_PBE,GGA_C_PBE, '
                           'basis cc-pV5Z (uncertainty = max |other basis - cc-pV5Z| over '
                           "('cc-pVQZ', 'cc-pV5Z', 'aug-cc-pV5Z')), grids.level=9, conv_tol=1e-12",
                 'xc': 'GGA_X_PBE,GGA_C_PBE'},
 'h2_R1.4_lda': {'value': -1.1374628148182775,
                 'uncertainty': 0.0001360968842614163,
                 'homo': -0.3773638621112064,
                 'per_basis': {'cc-pVQZ': -1.137326717934016,
                               'cc-pV5Z': -1.1374628148182775,
                               'aug-cc-pV5Z': -1.1374640058436678},
                 'method': 'PySCF 2.14.0 RKS all-electron, point nuclei, xc=LDA_X,LDA_C_VWN, basis '
                           "cc-pV5Z (uncertainty = max |other basis - cc-pV5Z| over ('cc-pVQZ', "
                           "'cc-pV5Z', 'aug-cc-pV5Z')), grids.level=9, conv_tol=1e-12",
                 'xc': 'LDA_X,LDA_C_VWN'},
 'h2_R1.4_pbe': {'value': -1.1666733967754421,
                 'uncertainty': 0.0001385058995717614,
                 'homo': -0.38155518665982885,
                 'per_basis': {'cc-pVQZ': -1.1665348908758704,
                               'cc-pV5Z': -1.1666733967754421,
                               'aug-cc-pV5Z': -1.1666741013271849},
                 'method': 'PySCF 2.14.0 RKS all-electron, point nuclei, xc=GGA_X_PBE,GGA_C_PBE, '
                           'basis cc-pV5Z (uncertainty = max |other basis - cc-pV5Z| over '
                           "('cc-pVQZ', 'cc-pV5Z', 'aug-cc-pV5Z')), grids.level=9, conv_tol=1e-12",
                 'xc': 'GGA_X_PBE,GGA_C_PBE'}}


def lookup_computed(scenario_id: str) -> dict | None:
    """Return the computed reference row for a scenario id, or ``None``."""
    return COMPUTED.get(scenario_id)
