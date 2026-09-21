"""Shim: the canonical module is ``run.py`` at the repository root (D-57).

Kept so that ``cdft.run`` -- the import path every test, script and gate uses -- keeps working
while the file that is edited sits beside ``contract.py``. The module object is shared: this entry
in ``sys.modules`` *is* the root module, so there is exactly one ``REGISTRY``/``NUMERICS`` and
``isinstance`` never sees two classes of the same name.
"""

from __future__ import annotations

import sys as _sys

import run as _canonical

_sys.modules[__name__] = _canonical

if __name__ == "__main__":  # pragma: no cover  -- ``python -m cdft.run``
    raise SystemExit(_canonical.main())
