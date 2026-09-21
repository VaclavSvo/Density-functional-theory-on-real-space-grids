"""Exchange--correlation functionals, native in torch and differentiable.

``dispatch.functional_from_spec`` turns an :class:`~contract.XCSpec` into an object implementing
:class:`~contract.XCFunctionalProtocol`; ``oracle`` reaches libxc for G0.6 only (D-05).
"""

from .dispatch import DEFERRED_FUNCTIONALS, NATIVE, functional_by_name, functional_from_spec

__all__ = ["DEFERRED_FUNCTIONALS", "NATIVE", "functional_by_name", "functional_from_spec"]
