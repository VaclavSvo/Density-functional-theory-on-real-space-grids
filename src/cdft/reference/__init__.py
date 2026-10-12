"""Independent reference solvers used as oracles. Never on the solver hot path.

The PySCF/libxc oracles run only when :func:`oracles_enabled` says so (``CDFT_ORACLES=1``); the
radial and two-centre references are in-house and always on.

Design: D-05 in ``docs/05_DECISION_LOG.md``.
"""

from __future__ import annotations

from .oracles import ENV_ORACLES, enable_oracles, oracles_enabled, oracles_mode

__all__ = ["ENV_ORACLES", "enable_oracles", "oracles_enabled", "oracles_mode"]
