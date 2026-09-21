"""Name -> functional. The one place a :class:`~contract.XCSpec` is turned into an object.

Names are lower-case and match the ``XCSpec.name`` written into every record.
:func:`functional_from_spec` refuses a spec whose ``libxc_reference`` disagrees with the
implementation's, so a record cannot say "PBE" while having been checked against something else.
``module_path``, a learned functional supplied from outside (D-24), is resolved here too and is the
only non-native entry.
"""

from __future__ import annotations

import importlib

from contract import XCFunctionalProtocol, XCSpec

from .base import SemiLocalFunctional
from .gga import GGA, B88Exchange, LYPCorrelation, PBECorrelation, PBEExchange
from .lda import LDA, PW92Correlation, PZ81Correlation, SlaterExchange, VWN5Correlation
from .mgga import R2SCAN_DEFERRAL_REASON

__all__ = ["NATIVE", "functional_from_spec", "functional_by_name", "DEFERRED_FUNCTIONALS", "libxc_code"]


def _builders() -> dict[str, callable]:
    return {
        "lda_vwn": lambda: LDA(VWN5Correlation(), "lda_vwn"),
        "lda_pw92": lambda: LDA(PW92Correlation(modified=False), "lda_pw92"),
        "lda_pw92_mod": lambda: LDA(PW92Correlation(modified=True), "lda_pw92_mod"),
        "lda_pz81": lambda: LDA(PZ81Correlation(), "lda_pz81"),
        "pbe": lambda: GGA(PBEExchange(), PBECorrelation(), "pbe"),
        "blyp": lambda: GGA(B88Exchange(), LYPCorrelation(), "blyp"),
        # Component functionals, for the pointwise gates and for anyone composing their own pair.
        "lda_x": SlaterExchange,
        "lda_c_vwn": VWN5Correlation,
        "lda_c_pw": lambda: PW92Correlation(modified=False),
        "lda_c_pw_mod": lambda: PW92Correlation(modified=True),
        "lda_c_pz": PZ81Correlation,
        "gga_x_pbe": PBEExchange,
        "gga_c_pbe": PBECorrelation,
        "gga_x_b88": B88Exchange,
        "gga_c_lyp": LYPCorrelation,
    }


#: Every native functional name this build can evaluate.
NATIVE: tuple[str, ...] = tuple(_builders())

#: Names the specification lists and this build declines, each with the recorded reason.
DEFERRED_FUNCTIONALS: dict[str, str] = {"r2scan": R2SCAN_DEFERRAL_REASON}


def functional_by_name(name: str) -> SemiLocalFunctional:
    """Return a fresh instance of the native functional called ``name`` (case-insensitive)."""
    key = name.lower()
    builders = _builders()
    if key in builders:
        return builders[key]()
    if key in DEFERRED_FUNCTIONALS:
        raise NotImplementedError(DEFERRED_FUNCTIONALS[key])
    raise KeyError(f"unknown functional {name!r}. Native: {', '.join(NATIVE)}; deferred: {', '.join(DEFERRED_FUNCTIONALS)}")


def libxc_code(functional: SemiLocalFunctional) -> str:
    """The comma-separated libxc identifier string gate G0.6 evaluates for this functional."""
    return ",".join(code.upper() for code in functional.libxc_reference)


def functional_from_spec(spec: XCSpec) -> XCFunctionalProtocol:
    """Resolve an :class:`~contract.XCSpec` to an object implementing the functional protocol.

    Raises ``ValueError`` for ``spec.name == "none"`` (a non-interacting scenario: the caller took
    the wrong path) or a ``libxc_reference`` that disagrees with the implementation's, and
    ``NotImplementedError`` for a deferred functional, with the reason.
    """
    if spec.name.lower() == "none":
        raise ValueError("a non-interacting scenario (xc.name == 'none') has no functional to resolve")
    if spec.module_path:
        module_name, _, attribute = spec.module_path.rpartition(":")
        module = importlib.import_module(module_name or spec.module_path)
        candidate = getattr(module, attribute) if attribute else getattr(module, "functional")
        functional = candidate() if callable(candidate) and not hasattr(candidate, "evaluate") else candidate
        if not isinstance(functional, XCFunctionalProtocol):
            raise TypeError(f"{spec.module_path} does not implement XCFunctionalProtocol")
        return functional
    functional = functional_by_name(spec.name)
    expected = tuple(code.lower() for code in functional.libxc_reference)
    declared = tuple(code.lower() for code in spec.libxc_reference)
    if declared and declared != expected:
        raise ValueError(
            f"XCSpec {spec.name!r} declares libxc_reference={declared} but the native "
            f"implementation is verified against {expected}; one of the two is wrong"
        )
    if functional.rung is not spec.rung:
        raise ValueError(f"XCSpec {spec.name!r} declares rung {spec.rung.name}, implementation is {functional.rung.name}")
    return functional

