"""Transcription checks on the published values in :mod:`cdft.reference.literature`.

No physics: every assertion is a property the source already guarantees, so a failure means a
mistyped digit. A transcribed reference is the one input no gate can check. Marked ``fast``;
under a second. Closes defect D-5.
"""

from __future__ import annotations

import math

import pytest

from cdft.reference.literature import (
    EXACT_NONRELATIVISTIC,
    NIST_SRD141,
    TWO_CENTRE_TABULATED,
    LiteratureValue,
    decomposition_residual,
    lookup,
)

pytestmark = [pytest.mark.fast]

#: The four terms NIST SRD 141 publishes alongside each total energy.
_DECOMPOSITION = ("kinetic_energy", "hartree_energy", "external_energy", "xc_energy")

#: Largest residual the published rounding can produce in ``|E_tot - sum(parts)|``.
#:
#: Not the database's stated 1e-6 accuracy, which is the accuracy of one printed value: the residual
#: sums five numbers printed to six decimals, each carrying up to 0.5e-6, hence 5 * 0.5e-6. Anything
#: above this bound is a typo rather than rounding. Measured on the rows as transcribed: at most
#: 1.0e-6, one unit in the last published place.
_ROUNDING_BOUND = 2.5e-6


def _complete_rows() -> list[str]:
    """Return ``system/functional`` for every NIST row that publishes a full decomposition."""
    prefixes = sorted({key.rsplit("/", 1)[0] for key in NIST_SRD141})
    return [
        prefix
        for prefix in prefixes
        if all(f"{prefix}/{part}" in NIST_SRD141 for part in _DECOMPOSITION)
        and f"{prefix}/total_energy" in NIST_SRD141
    ]


class TestNistTranscription:
    """``E_tot = E_kin + E_coul + E_enuc + E_xc`` holds for every complete NIST row."""

    @pytest.mark.parametrize("row", _complete_rows())
    def test_decomposition_closes(self, row: str) -> None:
        """The published total equals the published parts, to the published rounding."""
        system, functional = row.split("/")
        residual = decomposition_residual(system, functional)
        assert residual <= _ROUNDING_BOUND, (
            f"{row}: |E_tot - sum(parts)| = {residual:.3e} Ha, above the {_ROUNDING_BOUND:.1e} that "
            f"rounding five six-decimal numbers can produce. This is a TRANSCRIPTION error in "
            f"cdft/reference/literature.py, not a solver defect -- open the citation on that entry "
            f"and re-read the digits."
        )

    def test_at_least_the_phase_1_rows_are_complete(self) -> None:
        """The Phase 1 rows carry a decomposition, so the check above cannot cover nothing."""
        assert set(_complete_rows()) >= {"H/lda", "H/lsd", "He/lda", "He/lsd", "He+/lda", "He+/lsd"}

    def test_he_lsd_equals_he_lda_term_by_term(self) -> None:
        """A closed shell is unpolarised: NIST prints the same six numbers in both tables (D-5)."""
        for quantity in ("total_energy", "kinetic_energy", "hartree_energy", "external_energy", "xc_energy", "eigenvalue_1s"):
            assert NIST_SRD141[f"He/lsd/{quantity}"].value == NIST_SRD141[f"He/lda/{quantity}"].value


class TestEveryEntryIsUsable:
    """Citations, units, accuracies and the lookup, for every table."""

    @pytest.mark.parametrize(
        "table", [NIST_SRD141, EXACT_NONRELATIVISTIC, TWO_CENTRE_TABULATED], ids=["nist", "exact", "two_centre"]
    )
    def test_entries_are_cited_and_in_hartree(self, table: dict[str, LiteratureValue]) -> None:
        """Every value names a source and is in Hartree."""
        assert table, "an empty table would make every test in this file vacuous"
        for key, entry in table.items():
            assert entry.citation, f"{key} has no citation"
            assert entry.units == "Ha", f"{key} is in {entry.units!r}, not Hartree"

    def test_stated_accuracy_is_a_number_or_explicitly_absent(self) -> None:
        """``accuracy`` is the floor on any tolerance built from a value, so it must be readable."""
        for table in (NIST_SRD141, EXACT_NONRELATIVISTIC, TWO_CENTRE_TABULATED):
            for key, entry in table.items():
                assert isinstance(entry.accuracy, float), f"{key}: accuracy is not a float"
                assert entry.accuracy >= 0.0 or math.isnan(entry.accuracy), key

    def test_lookup_raises_with_the_available_keys(self) -> None:
        """A missing reference raises, and the message lists the keys that do exist."""
        with pytest.raises(KeyError) as excinfo:
            lookup("Ne/lda/total_energy")
        assert "H/lda/total_energy" in str(excinfo.value)

    def test_lookup_reaches_all_three_tables(self) -> None:
        """One entry from each table, so a table dropped from ``_ALL`` cannot go unnoticed."""
        assert lookup("H/lda/total_energy").value == pytest.approx(-0.445671)
        assert lookup("He/total_energy").value == pytest.approx(-2.903724377034, abs=1e-12)
        assert lookup("H2+/R2.0/total_energy").value == pytest.approx(-0.6026342144949, abs=1e-13)


class TestClosedFormsAreExact:
    """The entries that are algebra rather than transcription are exactly right."""

    @pytest.mark.parametrize(("key", "charge"), [("H/total_energy", 1.0), ("He+/total_energy", 2.0)])
    def test_hydrogenic_totals(self, key: str, charge: float) -> None:
        """A one-electron total is ``-Z^2/2`` exactly, and is stored with accuracy 0."""
        entry = EXACT_NONRELATIVISTIC[key]
        assert entry.value == -0.5 * charge**2
        assert entry.accuracy == 0.0

    def test_two_centre_electronic_energy_is_the_total_minus_one_over_r(self) -> None:
        """The H2+ entry's note is arithmetically right: the tabulated total includes 1/R."""
        entry = TWO_CENTRE_TABULATED["H2+/R2.0/total_energy"]
        electronic = entry.value - 1.0 / 2.0
        assert electronic == pytest.approx(-1.1026342144949, abs=1e-13)
        assert "-1.1026342144949" in entry.note
