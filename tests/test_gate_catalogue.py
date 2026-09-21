"""The gate table in ``docs/03_METHOD.md`` (Part C) defines the catalogue's gates.

G5.8 cross-checks the catalogue against the evaluators and the registry at run time but cannot read
a document, so the markdown is parsed here. Identity only, never thresholds: those live on each
``GateSpec`` and a threshold asserted twice eventually differs twice. Marked ``fast``; instant.
"""

from __future__ import annotations

import pathlib
import re

import pytest

#: Markers for every test in this module (see ``tests/conftest.py``).
pytestmark = [pytest.mark.fast]

SPEC_PATH = (
    pathlib.Path(__file__).resolve().parents[1] / "docs" / "03_METHOD.md"
)

#: A gate row opens with ``| **G<tier>.<n>** |``. Continuation rows carrying a note start with
#: ``| |`` and are intentionally not matched.
_ROW = re.compile(r"^\|\s*\*\*(G\d+\.\d+)\*\*\s*\|", re.MULTILINE)


def specification_ids() -> set[str]:
    """Return every gate identifier the specification document defines a row for."""
    if not SPEC_PATH.exists():  # pragma: no cover - the document is part of the repository
        pytest.skip(f"the specification document is not present at {SPEC_PATH}")
    return set(_ROW.findall(SPEC_PATH.read_text(encoding="utf-8")))


def catalogue_ids() -> set[str]:
    """Return every gate identifier the catalogue defines."""
    from cdft.gates.catalogue import CATALOGUE

    return {entry.gate_id for entry in CATALOGUE}


class TestSpecificationAndCatalogueAgree:
    """The document and the catalogue define the same set of gate identifiers."""

    def test_every_catalogue_gate_appears_in_the_specification(self) -> None:
        """Every catalogued gate has a specification row; one that does not is never reviewed."""
        missing = sorted(catalogue_ids() - specification_ids())
        assert not missing, (
            f"{len(missing)} gate(s) exist in cdft.gates.catalogue and have no row in "
            f"{SPEC_PATH.name}: {', '.join(missing)}.\n"
            f"Add a row for each, or remove it from the catalogue. A gate that exists only in code "
            f"is one nobody reviews."
        )

    def test_every_specified_gate_appears_in_the_catalogue(self) -> None:
        """Every specified gate is catalogued, so each one carries a lifecycle (D-37)."""
        missing = sorted(specification_ids() - catalogue_ids())
        assert not missing, (
            f"{len(missing)} gate(s) have a row in {SPEC_PATH.name} and no catalogue entry: "
            f"{', '.join(missing)}.\n"
            f"Add each to cdft.gates.catalogue with a lifecycle, so that the suite can say whether "
            f"it is implemented, deferred with an owner, or out of Phase 1."
        )

    def test_the_specification_defines_a_plausible_number_of_gates(self) -> None:
        """The regex parses a plausible number of rows, so the two tests above are not vacuous."""
        found = specification_ids()
        assert len(found) >= 40, (
            f"only {len(found)} gate rows were parsed from {SPEC_PATH.name}, which is too few to "
            f"be the real table -- the row format has probably changed and this file's regex has "
            f"not. Both agreement tests above would pass vacuously in that state."
        )


class TestCatalogueSelfConsistency:
    """Invariants of the catalogue itself: unique ids, owners, tiers."""

    def test_gate_ids_are_unique(self) -> None:
        """No identifier appears twice; the second entry would be silently ignored."""
        from cdft.gates.catalogue import CATALOGUE

        seen: dict[str, int] = {}
        for entry in CATALOGUE:
            seen[entry.gate_id] = seen.get(entry.gate_id, 0) + 1
        duplicates = sorted(gate_id for gate_id, n in seen.items() if n > 1)
        assert not duplicates, f"duplicate catalogue entries: {', '.join(duplicates)}"

    def test_every_deferred_gate_names_an_owner(self) -> None:
        """Every DEFERRED entry names the increment that owes the work (D-37)."""
        from cdft.gates.catalogue import CATALOGUE, Lifecycle

        orphans = sorted(
            entry.gate_id
            for entry in CATALOGUE
            if entry.lifecycle is Lifecycle.DEFERRED and not entry.owner
        )
        assert not orphans, (
            f"deferred with no owning increment: {', '.join(orphans)}. A deferral with nobody "
            f"attached is indistinguishable from having forgotten."
        )

    def test_every_out_of_phase_gate_names_its_phase(self) -> None:
        """Every OUT_OF_PHASE_1 entry names the phase it belongs to."""
        from cdft.gates.catalogue import CATALOGUE, Lifecycle

        orphans = sorted(
            entry.gate_id
            for entry in CATALOGUE
            if entry.lifecycle is Lifecycle.OUT_OF_PHASE_1 and not entry.owner
        )
        assert not orphans, f"out of Phase 1 with no phase named: {', '.join(orphans)}"

    def test_gate_tier_matches_its_identifier(self) -> None:
        """Each entry's ``tier`` matches the tier digit of its identifier."""
        from cdft.gates.catalogue import CATALOGUE

        wrong = [
            f"{entry.gate_id} is recorded as tier {entry.tier}"
            for entry in CATALOGUE
            if int(entry.gate_id[1: entry.gate_id.index(".")]) != entry.tier
        ]
        assert not wrong, "; ".join(wrong)


#: The kind column of a gate row, where the tier's table has one (tier 5 has none).
_KINDS = ("EXACT", "DERIVED", "EMPIRICAL")


def specification_kinds() -> dict[str, str]:
    """Return ``gate_id -> kind`` for every specification row that states a kind cell."""
    if not SPEC_PATH.exists():  # pragma: no cover - the document is part of the repository
        pytest.skip(f"the specification document is not present at {SPEC_PATH}")
    kinds: dict[str, str] = {}
    for line in SPEC_PATH.read_text(encoding="utf-8").splitlines():
        match = _ROW.match(line)
        if match is None:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        stated = [cell for cell in cells if cell in _KINDS]
        if stated:
            kinds[match.group(1)] = stated[0]
    return kinds


class TestThresholdKindsAgree:
    """B3: the kind is part of a gate's identity and is stated three times; they must agree."""

    def test_every_wired_evaluator_carries_the_catalogue_kind(self) -> None:
        from cdft.gates.catalogue import CATALOGUE
        from cdft.gates.runner import default_artifact_gates, default_component_gates
        from cdft.gates.tier5 import GateReportGate

        catalogued = {entry.gate_id: entry.kind for entry in CATALOGUE}
        wired = (*default_artifact_gates(), *default_component_gates(), GateReportGate())
        wrong = [
            f"{gate.spec.gate_id}: GateSpec {gate.spec.kind.value}, catalogue "
            f"{catalogued[gate.spec.gate_id].value}"
            for gate in wired
            if gate.spec.gate_id in catalogued
            and gate.spec.kind is not catalogued[gate.spec.gate_id]
        ]
        assert not wrong, "; ".join(wrong)

    def test_every_stated_specification_kind_matches_the_catalogue(self) -> None:
        from cdft.gates.catalogue import CATALOGUE

        catalogued = {entry.gate_id: entry.kind.value.upper() for entry in CATALOGUE}
        stated = specification_kinds()
        assert len(stated) >= 40, "the kind column parsed too few rows to be the real table"
        wrong = [
            f"{gate_id}: {SPEC_PATH.name} {kind}, catalogue {catalogued[gate_id]}"
            for gate_id, kind in sorted(stated.items())
            if gate_id in catalogued and catalogued[gate_id] != kind
        ]
        assert not wrong, "; ".join(wrong)

    def test_g52_is_exact_in_all_three_places(self) -> None:
        # B3: the tier-5 table has no kind column, so the G5.2 row states its kind in prose.
        from cdft.gates.catalogue import by_id
        from cdft.gates.tier5 import DeterminismGate
        from contract import GateKind

        assert by_id("G5.2").kind is GateKind.EXACT
        assert DeterminismGate.spec.kind is GateKind.EXACT
        row = next(
            line for line in SPEC_PATH.read_text(encoding="utf-8").splitlines()
            if line.startswith("| **G5.2** |")
        )
        assert "**EXACT**" in row and "not DERIVED" in row
