"""The oracle switch: whether PySCF/libxc-backed checks run at all (O-24).

Off by default. Off means the oracle-dependent tests are deselected and the oracle-dependent gates
leave the plan and the record, rather than appearing as SKIPPED rows on every diagnostic surface;
the mode that ran is written into the record (``measurements["oracles"]``). The stored references
of :mod:`cdft.reference.computed` are data, not an oracle call, and are never switched.
"""

from __future__ import annotations

import os

from ..precision import env_switch

__all__ = ["ENV_ORACLES", "oracles_enabled", "oracles_mode", "enable_oracles"]

#: ``CDFT_ORACLES=1`` turns the PySCF/libxc oracles on; unset or ``0`` leaves them off. Any other
#: value raises through :func:`cdft.precision.env_switch`, never reads as either setting (G5.5).
ENV_ORACLES = "CDFT_ORACLES"


def oracles_enabled() -> bool:
    """Whether oracle-backed tests and gates are requested for this process (default off)."""
    return env_switch(ENV_ORACLES, False)


def oracles_mode() -> str:
    """The switch as it goes into a record: ``"on"`` or ``"off"``."""
    return "on" if oracles_enabled() else "off"


def enable_oracles() -> None:
    """Turn the oracles on for this process and its children (the ``--oracles`` flags)."""
    os.environ[ENV_ORACLES] = "1"
