"""``cdft`` -- the Classical DFT Solver: real-space, ground-state Kohn-Sham, non-neural.

No machine learning in this package; G5.7 enforces it by static import scan. The frozen API is
``contract.py`` at the repository root, put on the path by the block below (D-15, D-24).
"""

from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if (_REPO_ROOT / "contract.py").is_file() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from contract import CONTRACT_VERSION  # noqa: E402  (import follows the path bootstrap by design)

__version__ = "0.5.0"

#: Contract version this package was written against. :func:`cdft.io.provenance.capture` records
#: the version actually imported; a mismatch is reported, never tolerated.
CONTRACT_VERSION_EXPECTED = "1.6.1"

__all__ = ["CONTRACT_VERSION", "CONTRACT_VERSION_EXPECTED", "__version__"]
