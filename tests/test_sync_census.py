"""The sync census behind G5.9 (``scripts/sync_census.py``): what counts as a device sync.

``fast``; milliseconds. The census itself is measured by the gate, not here.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest
import torch

from cdft.operators import gauss_legendre as gl

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _census_module():
    name = "cdft_scripts_sync_census"
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "sync_census.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@pytest.mark.fast
class TestHostSideSites:
    def test_gauss_legendre_rule_is_a_host_side_site_for_reads_and_implicit_syncs(self) -> None:
        """A fresh rule order builds on a CPU tensor: its ``.numpy()`` read and its ``eigvalsh``
        are reported as host-side, not as syncs (O-31: 27 orders once each were read as a loop
        site of a 7-iteration probe)."""
        census = _census_module()
        gl._rule.cache_clear()
        with census.SyncCensus(root=ROOT) as c:
            gl.gauss_legendre(37)
            gl.gauss_legendre(37)  # memoised: no second build
        s = c.summary()
        assert s["reads_total"] == 0
        assert s["host_side_reads_total"] == 1
        assert s["implicit_total"] == 0
        assert s["host_side_implicit_total"] >= 1
        assert s["syncs_total"] == 0
        assert s["read_sites"] == [] and s["implicit_sites"] == []
        assert census.loop_syncs(s, iterations=1)["loop_sites"] == []

    def test_a_device_read_still_counts(self) -> None:
        census = _census_module()
        with census.SyncCensus(root=ROOT) as c:
            float(torch.ones(3, dtype=torch.float64).sum())
        s = c.summary()
        # The read is attributed to the innermost project frame: this test file.
        assert s["reads_total"] == 1 and s["host_side_reads_total"] == 0
        assert s["read_sites"][0]["site"].startswith("tests/test_sync_census.py")
