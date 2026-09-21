"""Test package marker, so one test module can import a helper from another.

``test_golden_regression`` and ``test_throughput`` reuse helpers from
``test_physics_invariants`` and ``test_device_agnostic`` rather than keeping second copies.
"""
