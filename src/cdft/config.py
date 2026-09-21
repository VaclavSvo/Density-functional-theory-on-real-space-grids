"""Shim: the canonical module is ``config.py`` at the repository root (D-57).

Kept so that ``cdft.config`` -- the import path every test, script and gate uses -- keeps working
while the file that is edited sits beside ``contract.py``. The module object is shared: this entry
in ``sys.modules`` *is* the root module, so there is exactly one ``REGISTRY``/``NUMERICS`` and
``isinstance`` never sees two classes of the same name.
"""

from __future__ import annotations

import sys as _sys

import config as _canonical

_sys.modules[__name__] = _canonical
