"""Shared fixtures, markers and the golden-value machinery for the whole test suite.

Markers, so a run can be sliced by cost: ``fast`` (under a second; algebra, weights, contract
invariants, 1-D oracles), ``physics`` (against a closed form, an oracle or a published number),
``golden`` (against the pins in ``tests/golden/values.json``) and ``slow`` (tens of seconds; 3-D
solves at production settings). ``slow`` is deselected unless ``--runslow`` is given, so ``pytest``
stays an inner-loop command; it is also never deselected during a re-bless, or the expensive pins
would go stale.

``oracle`` marks a test that needs PySCF/libxc; it is *deselected* (not skipped, so it is off every
summary) unless ``CDFT_ORACLES=1`` (O-24). Requested without PySCF, the test's own skip says so.

``pytest --rebless`` rewrites ``tests/golden/values.json`` from the current run and prints a diff
of every value that changed. Separate and explicit on purpose: a suite that updates its own
expectations cannot detect a regression, and one that cannot be updated loses its assertions.

``CDFT_DEVICE=cpu`` (the default) or ``cuda`` sets the device of the presets every solve starts
from (``config.MODEL_SYSTEM_NUMERICS``, ``config.ALL_ELECTRON_NUMERICS``). ``cuda`` without a card
is a usage error before collection, never a skip or a fall back to the CPU (G5.5). Tests that build
a grid directly stay on the CPU; the ``device`` fixture carries the device-specific invariants.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import pathlib
from typing import Any, Iterator

import pytest

GOLDEN_PATH = pathlib.Path(__file__).parent / "golden" / "values.json"

#: Relative tolerance for every pinned value.
#:
#: 1e-9, not 1e-15: BLAS reductions are not associative, so the same eigenvalue differs in its last
#: digits with the thread count. Still tight enough that losing one significant digit of physics
#: trips it.
GOLDEN_RTOL = 1.0e-9

#: Absolute floor, so a pinned value that is legitimately near zero does not demand infinite
#: relative precision.
GOLDEN_ATOL = 1.0e-14


#: Environment variable selecting the device of the numerics presets for the whole session.
DEVICE_ENV = "CDFT_DEVICE"

#: Accepted values of :data:`DEVICE_ENV`; ``auto`` is not one, a run must say which leg it is.
DEVICE_CHOICES = ("cpu", "cuda")

#: The marker of a test that calls PySCF/libxc (O-24).
ORACLE_MARKER = "oracle"


def oracles_requested() -> bool:
    """Whether ``CDFT_ORACLES`` asks for the oracle tests; a typo is a usage error, never off."""
    from cdft.reference.oracles import ENV_ORACLES, oracles_enabled

    try:
        return oracles_enabled()
    except ValueError as exc:
        raise pytest.UsageError(f"{ENV_ORACLES}: {exc}") from None


def selected_device_name() -> str:
    """Return the device named by ``CDFT_DEVICE`` (default ``cpu``), validated but not resolved."""
    value = os.environ.get(DEVICE_ENV, "cpu").strip().lower() or "cpu"
    if value not in DEVICE_CHOICES:
        raise pytest.UsageError(
            f"{DEVICE_ENV}={value!r} is not one of {', '.join(DEVICE_CHOICES)}"
        )
    return value


def _apply_device(name: str) -> None:
    """Rebind the solve presets of the root ``config`` module to the selected device.

    Called from :func:`pytest_configure`, before any test module is imported, so every
    ``from cdft.config import ...`` sees the rebound object (same module object, D-57). ``cpu``
    rebinds nothing.
    """
    import torch

    import config as numerics_presets
    from contract import Device

    if name == "cuda" and not torch.cuda.is_available():
        raise pytest.UsageError(
            f"{DEVICE_ENV}=cuda but torch reports no CUDA device (torch {torch.__version__}, "
            f"built for CUDA {torch.version.cuda}). Refusing to run the CUDA leg on the CPU "
            f"(gate G5.5); unset {DEVICE_ENV} or set it to cpu."
        )
    if name == "cuda":
        # Before any test can create a cuBLAS handle (cdft.precision.configure_device, D-61).
        from cdft.precision import CUBLAS_WORKSPACE_CONFIG

        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG)
    target = Device(name)
    for attribute in ("MODEL_SYSTEM_NUMERICS", "ALL_ELECTRON_NUMERICS"):
        preset = getattr(numerics_presets, attribute)
        if preset.device is not target:
            setattr(numerics_presets, attribute, dataclasses.replace(preset, device=target))


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register ``--rebless`` and ``--runslow``."""
    parser.addoption(
        "--rebless",
        action="store_true",
        default=False,
        help="rewrite tests/golden/values.json from this run and print what changed",
    )
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="also run tests marked slow (three-dimensional solves at production settings)",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Declare the markers so an unknown-marker warning means a typo, not a new category."""
    for name, description in (
        ("fast", "under a second; algebra, weights, contract invariants, 1-D oracles"),
        ("physics", "compares against a closed form, an oracle, or a published value"),
        ("golden", "compares against the pinned values in tests/golden/values.json"),
        ("slow", "tens of seconds; three-dimensional solves at production settings"),
        (ORACLE_MARKER, "needs PySCF/libxc; deselected unless CDFT_ORACLES=1 (O-24)"),
    ):
        config.addinivalue_line("markers", f"{name}: {description}")
    _apply_device(selected_device_name())
    oracles_requested()  # fail the session here, before collection, on a mis-set switch


def pytest_report_header(config: pytest.Config) -> str:
    """Name the device leg and the oracle switch in the session header, so a log says which ran."""
    oracles = "on" if oracles_requested() else "off"
    return f"cdft device leg: {DEVICE_ENV}={selected_device_name()}   oracles: {oracles}"


def deselect_oracle_tests(items: list[pytest.Item], oracles: bool) -> list[pytest.Item]:
    """Remove the ``oracle``-marked items in place when ``oracles`` is off; return the removed."""
    if oracles:
        return []
    removed = [item for item in items if ORACLE_MARKER in item.keywords]
    if removed:
        items[:] = [item for item in items if ORACLE_MARKER not in item.keywords]
    return removed


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Deselect ``oracle`` tests unless requested; skip ``slow`` tests unless asked for.

    The slow skip is never applied during a re-bless, or its pins go stale. Oracle tests are
    deselected, not skipped, so they are off the summary line unless ``CDFT_ORACLES=1`` (O-24).
    """
    deselected = deselect_oracle_tests(items, oracles_requested())
    if deselected:
        config.hook.pytest_deselected(items=deselected)
    if config.getoption("--runslow") or config.getoption("--rebless"):
        return
    skip = pytest.mark.skip(reason="slow: run with --runslow or via test_suite.py --profile gpu")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


class GoldenStore:
    """Loads the pinned values, compares against them, and collects re-blessed replacements.

    One instance per session. Comparison is relative with an absolute floor, and a failure prints
    the stored value, the new one, the relative change and the command that accepts it.
    """

    def __init__(self, path: pathlib.Path, reblessing: bool) -> None:
        """Load ``path`` if it exists; an absent file is an empty store, not an error."""
        self.path = path
        self.reblessing = reblessing
        self.stored: dict[str, Any] = {}
        if path.exists():
            self.stored = json.loads(path.read_text(encoding="utf-8"))
        self.updated: dict[str, Any] = dict(self.stored)
        self.changes: list[tuple[str, Any, Any]] = []
        self.missing: list[str] = []

    def check(
        self,
        key: str,
        value: float,
        *,
        rtol: float = GOLDEN_RTOL,
        atol: float = GOLDEN_ATOL,
        note: str = "",
    ) -> None:
        """Assert ``value`` matches the pinned entry for ``key``, or record it when re-blessing."""
        entry = self.stored.get(key)
        record: dict[str, Any] = {"value": float(value)}
        if note:
            record["note"] = note

        if self.reblessing:
            old = None if entry is None else float(entry["value"])
            if old is None or not _close(float(value), old, rtol, atol):
                self.changes.append((key, old, float(value)))
            self.updated[key] = record
            return

        if entry is None:
            self.missing.append(key)
            pytest.fail(
                f"no pinned value for {key!r}. This is a new measurement, so nothing is wrong with "
                f"the physics -- run `pytest --rebless` to record it, and read the printed diff "
                f"before committing the file."
            )

        expected = float(entry["value"])
        if not _close(float(value), expected, rtol, atol):
            relative = (
                abs(float(value) - expected) / abs(expected) if expected else float("inf")
            )
            pytest.fail(
                f"{key}: pinned {expected!r}, measured {float(value)!r} "
                f"(relative change {relative:.3e}, tolerance {rtol:.1e}).\n"
                f"  {entry.get('note', '')}\n"
                f"  If this change is CORRECT -- you improved the method on purpose -- accept it "
                f"with `pytest --rebless` and say why in docs/05_DECISION_LOG.md.\n"
                f"  If it is NOT, you have changed the physics without meaning to, which is the "
                f"reason this layer exists."
            )

    def write(self) -> None:
        """Write the re-blessed file, sorted, with a trailing newline so diffs stay small."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        ordered = {key: self.updated[key] for key in sorted(self.updated)}
        self.path.write_text(json.dumps(ordered, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _close(got: float, want: float, rtol: float, atol: float) -> bool:
    """Relative comparison with an absolute floor; a NaN never matches anything, including itself."""
    if math.isnan(got) or math.isnan(want):
        return False
    return abs(got - want) <= max(atol, rtol * abs(want))


@pytest.fixture(scope="session")
def golden(request: pytest.FixtureRequest) -> Iterator[GoldenStore]:
    """Session-scoped access to the pinned values."""
    store = GoldenStore(GOLDEN_PATH, reblessing=bool(request.config.getoption("--rebless")))
    yield store
    if store.reblessing:
        store.write()
        terminal = request.config.pluginmanager.get_plugin("terminalreporter")
        lines = [f"\nre-blessed {GOLDEN_PATH}"]
        if store.changes:
            lines.append(f"  {len(store.changes)} value(s) changed:")
            for key, old, new in sorted(store.changes):
                if old is None:
                    lines.append(f"    + {key}: {new!r}   (new)")
                else:
                    relative = abs(new - old) / abs(old) if old else float("inf")
                    lines.append(f"    ~ {key}: {old!r} -> {new!r}   (relative {relative:.3e})")
            lines.append(
                "  Every line above is a change in what this code computes. Record why in "
                "docs/05_DECISION_LOG.md before committing."
            )
        else:
            lines.append("  no values changed")
        if terminal is not None:  # pragma: no branch - the reporter is always present under pytest
            terminal.write_line("\n".join(lines))


# --- shared physics fixtures --------------------------------------------------------------


@pytest.fixture(scope="session")
def device():
    """The torch device selected by ``CDFT_DEVICE`` for this session."""
    # Indexed (``cuda:0``), because a tensor made on ``cuda`` reports ``cuda:0`` and the two
    # ``torch.device`` values compare unequal; residency tests need the indexed form.
    import torch

    name = selected_device_name()
    if name == "cuda":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device(name)
