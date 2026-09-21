"""Failures that are known, measured, owned and scheduled -- and must not hide a new one.

A known-open failure keeps its FAIL verdict in the record, so the corpus still marks the run
INVALID; what it loses is the power to set the suite's exit code, and only while it does not get
worse. Thresholds are never raised to fit a measurement. Each entry is keyed to one gate on one
scenario, expires upward when the measurement grows past ``recorded * drift``, and is reported as
stale when it drops well below ``recorded``, so a quiet fix cannot leave a gate excused forever.
"""

from __future__ import annotations

from dataclasses import dataclass

from .runner import UNEVALUABLE_GATE_NAME

__all__ = [
    "KnownOpen",
    "KNOWN_OPEN",
    "COMPONENT",
    "Partition",
    "classify",
    "lookup",
    "partition_failures",
]

#: Scenario id used for a *component* gate, which verifies an operator rather than a run and so
#: belongs to no scenario.
COMPONENT = "<component>"


@dataclass(frozen=True, slots=True)
class KnownOpen:
    """One gate failing on one scenario for a reason that is written down and owned."""

    scenario_id: str
    gate_id: str
    open_item: str
    """Identifier of the open item, e.g. ``O-13``. Never blank: without one this is an excuse."""

    recorded: float
    """The measured value when this entry was written."""

    drift: float
    """Multiplicative tolerance on ``recorded``; beyond it the failure is new and blocks."""

    note: str = ""


#: Every currently-accepted failure. Deliberately short, and shortening it is the work. A retired
#: row's measurement stays pinned in ``tests/golden/values.json`` under a ``closed.*`` key. The
#: three molecular G1.13 rows (O-19, C8) were retired by D-75: 2.29e-8 / 2.03e-8 / 2.42e-5 became
#: 2.7e-9 / 2.9e-9 / 2.9e-9 against the unchanged 1e-8 threshold. The G5.9 row (O-31, recorded 2.0
#: on a 2-core host) was retired by D-82: the census read the 27 memoised
#: ``gauss_legendre._rule`` builds -- a CPU computation on every device -- as a loop site; classed
#: host-side, both probes meet their 2026-09-16 budgets exactly (9/7 and 80/12 reads). The card
#: host's 4.0 of the same night was recorded without detail; its next full run carries the detail
#: and decides whether anything remains there.
KNOWN_OPEN: tuple[KnownOpen, ...] = (
    KnownOpen(
        scenario_id="h2_R1.4_lda",
        gate_id="G2.3",
        open_item="G2.3",
        recorded=1.0,
        drift=1.5,
        note=(
            "one residual rise (5.2e-6 after 4.4e-6, iteration 22 of 25) inside the five-iteration "
            "window; the energies fall to 4e-12 Ha. Mechanism (D-75 follow-up): the periodic Pulay mixer's "
            "extrapolation step every third iteration overshoots by ~20 % in the residual at the "
            "converged tail, which the gate's strict residual rule (D-55 item 8, meant for too few "
            "filter steps) counts; h2_R1.4_pbe has the same rise one iteration earlier, outside its "
            "window. A path change (mixer period, or a restart before the last iterations) would "
            "move every SCF fingerprint and is not taken here; the threshold stays 0"
        ),
    ),
)

_INDEX = {(entry.scenario_id, entry.gate_id): entry for entry in KNOWN_OPEN}


def lookup(scenario_id: str, gate_id: str) -> KnownOpen | None:
    """Return the known-open entry for this gate on this scenario, if there is one."""
    return _INDEX.get((scenario_id, gate_id))


def classify(scenario_id: str, gate_id: str, measured: float) -> tuple[str, KnownOpen | None]:
    """Decide what a failing gate's measurement means for the build.

    Returns
    -------
    tuple
        ``(kind, entry)`` where ``kind`` is one of:

        ``"new"``
            No entry, or the measurement has grown past ``recorded * drift``; fails the build.
        ``"known"``
            Within the recorded band; reported, does not fail the build.
        ``"improved"``
            Under half of what was recorded; reported as stale so the entry gets retired.

    """
    entry = _INDEX.get((scenario_id, gate_id))
    if entry is None:
        return "new", None
    if measured > entry.recorded * entry.drift:
        return "new", entry
    if measured < 0.5 * entry.recorded:
        return "improved", entry
    return "known", entry


@dataclass(frozen=True, slots=True)
class Partition:
    """How one report's failures split into the three kinds that mean different things."""

    new: int
    """Failures that set the exit code: a defect, or a known one that got worse."""

    known: int
    """Failures registered in :data:`KNOWN_OPEN` and still inside their recorded band."""

    unimplemented: int
    """Gates a scenario named that this build cannot evaluate."""

    rows: tuple[str, ...]
    """One human-readable line per known or regressed failure, for the report."""

    @property
    def blocking(self) -> bool:
        """Whether this report should fail the build."""
        return self.new > 0


def partition_failures(scenario_id: str, results) -> Partition:
    """Split a gate report's failures into new, known-open, and not-yet-implemented.

    The single implementation of the rule: ``test_suite.py`` and ``cdft.run`` both derive an exit
    code from a gate report and must derive the same one.
    """
    new = known = unimplemented = 0
    rows: list[str] = []
    for result in results:
        if result.verdict.value != "fail":
            continue
        if result.name == UNEVALUABLE_GATE_NAME:  # the runner's own string (O-26)
            unimplemented += 1
            continue
        kind, entry = classify(scenario_id, result.gate_id, result.measured)
        if kind == "new":
            new += 1
            if entry is not None:
                rows.append(
                    f"  {scenario_id}/{result.gate_id}: {result.measured:.3e} is WORSE than the "
                    f"recorded {entry.recorded:.3e} ({entry.open_item}) by more than the "
                    f"{entry.drift:g}x drift allowance -- this is a regression"
                )
        else:
            known += 1
            assert entry is not None
            stale = (
                " (now well below what was recorded -- retire the entry)"
                if kind == "improved"
                else ""
            )
            rows.append(
                f"  {scenario_id}/{result.gate_id}: {result.measured:.3e} vs recorded "
                f"{entry.recorded:.3e}, open item {entry.open_item}{stale}"
            )
    return Partition(new=new, known=known, unimplemented=unimplemented, rows=tuple(rows))
