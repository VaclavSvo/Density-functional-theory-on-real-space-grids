"""Shim: the canonical module is ``run.py`` at the repository root.

Keeps the import path ``cdft.run`` working while the edited file sits beside ``contract.py``.
This ``sys.modules`` entry *is* the root module, so there is one set of module-level registries and
``isinstance`` never sees two classes of the same name. Design: D-57 in ``docs/05_DECISION_LOG.md``.
"""

from __future__ import annotations

import sys as _sys

import run as _canonical

_sys.modules[__name__] = _canonical

if __name__ == "__main__":  # pragma: no cover  -- ``python -m cdft.run``
    raise SystemExit(_canonical.main())
