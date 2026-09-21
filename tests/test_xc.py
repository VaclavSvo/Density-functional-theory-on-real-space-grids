"""The exchange--correlation layer: shapes, spin splitting, dispatch and libxc agreement.

All marked ``fast``; the libxc comparison is marked ``oracle``, so it is deselected unless
``CDFT_ORACLES=1`` and skips if PySCF is then missing (O-24). Under a second.
"""

from __future__ import annotations

import ast
import pathlib

import numpy as np
import pytest
import torch

from contract import XCOutput, XCRung, XCSpec
from cdft.gates.functional import random_ingredients
from cdft.xc import DEFERRED_FUNCTIONALS, NATIVE, functional_by_name, functional_from_spec
from cdft.xc.base import DENSITY_THRESHOLD

pytestmark = [pytest.mark.fast]

XC_DIR = pathlib.Path(__file__).resolve().parents[1] / "src" / "cdft" / "xc"


def _libxc():
    try:
        from cdft.xc.oracle import evaluate_libxc, libxc_available
    except ImportError:  # pragma: no cover
        return None
    return evaluate_libxc if libxc_available() else None


class TestContractSurface:
    """Every native functional returns exactly what the contract defines."""

    @pytest.mark.parametrize("name", NATIVE)
    def test_shapes_unpolarised_and_polarised(self, name: str) -> None:
        functional = functional_by_name(name)
        for polarised in (False, True):
            n, sigma, tau = random_ingredients(50, polarised, 2.0, seed=1)
            needs_sigma = functional.rung.value >= XCRung.GGA.value
            out = functional.evaluate(torch.tensor(n), torch.tensor(sigma) if needs_sigma else None)
            assert isinstance(out, XCOutput)
            n_spin = 2 if polarised else 1
            assert out.e_xc.shape == (50,)
            assert out.v_xc.shape == (n_spin, 50)
            if needs_sigma:
                assert out.v_sigma.shape == (3 if polarised else 1, 50)
            else:
                assert out.v_sigma is None
            assert out.v_tau is None and out.v_lapl is None and out.v_nldf is None
            assert torch.isfinite(out.e_xc).all() and torch.isfinite(out.v_xc).all()

    @pytest.mark.parametrize("name", ["lda_vwn", "pbe"])
    def test_restricted_call_equals_equal_channel_polarised_call(self, name: str) -> None:
        """``evaluate(n)`` equals ``evaluate((n/2, n/2))`` in energy and potential."""
        functional = functional_by_name(name)
        n, sigma, _ = random_ingredients(40, False, 2.0, seed=2)
        needs_sigma = functional.rung is not XCRung.LDA
        restricted = functional.evaluate(torch.tensor(n), torch.tensor(sigma) if needs_sigma else None)
        pol_n = torch.tensor(np.concatenate([0.5 * n, 0.5 * n]))
        pol_sigma = torch.tensor(np.concatenate([0.25 * sigma, 0.25 * sigma, 0.25 * sigma])) if needs_sigma else None
        polarised = functional.evaluate(pol_n, pol_sigma)
        assert torch.allclose(restricted.e_xc, polarised.e_xc, rtol=0, atol=1e-13)
        assert torch.allclose(restricted.v_xc[0], polarised.v_xc[0], rtol=0, atol=1e-12)
        assert torch.allclose(restricted.v_xc[0], polarised.v_xc[1], rtol=0, atol=1e-12)

    def test_empty_points_return_zero_energy_and_zero_potential(self) -> None:
        functional = functional_by_name("pbe")
        n = torch.tensor([[0.0, 0.1 * DENSITY_THRESHOLD, 1.0]], dtype=torch.float64)
        sigma = torch.tensor([[0.0, 1e-30, 0.3]], dtype=torch.float64)
        out = functional.evaluate(n, sigma)
        assert out.e_xc[0] == 0.0 and out.e_xc[1] == 0.0 and out.e_xc[2] != 0.0
        assert out.v_xc[0, 0] == 0.0 and out.v_xc[0, 1] == 0.0
        assert torch.isfinite(out.v_sigma).all()

    def test_gga_requires_sigma_and_lda_ignores_it(self) -> None:
        with pytest.raises(ValueError):
            functional_by_name("pbe").evaluate(torch.rand(1, 5, dtype=torch.float64) + 0.1)
        out = functional_by_name("lda_vwn").evaluate(torch.rand(1, 5, dtype=torch.float64) + 0.1)
        assert out.v_sigma is None


class TestDispatch:
    def test_every_native_name_resolves(self) -> None:
        for name in NATIVE:
            assert functional_by_name(name).name == name

    def test_spec_resolution_and_libxc_reference_check(self) -> None:
        functional = functional_from_spec(XCSpec(name="pbe", rung=XCRung.GGA, libxc_reference=("gga_x_pbe", "gga_c_pbe")))
        assert functional.name == "pbe"
        with pytest.raises(ValueError):
            functional_from_spec(XCSpec(name="pbe", rung=XCRung.GGA, libxc_reference=("gga_x_b88", "gga_c_pbe")))
        with pytest.raises(ValueError):
            functional_from_spec(XCSpec(name="pbe", rung=XCRung.LDA, libxc_reference=("gga_x_pbe", "gga_c_pbe")))
        with pytest.raises(ValueError):
            functional_from_spec(XCSpec(name="none", rung=XCRung.LDA, libxc_reference=()))

    def test_deferred_functional_raises_with_its_reason(self) -> None:
        assert "r2scan" in DEFERRED_FUNCTIONALS
        with pytest.raises(NotImplementedError, match="DEFERRED"):
            functional_by_name("r2scan")
        with pytest.raises(KeyError):
            functional_by_name("no_such_functional")

    def test_only_the_oracle_module_touches_pyscf(self) -> None:
        """Only ``oracle.py`` imports pyscf: libxc is off the hot path (D-05)."""
        for path in XC_DIR.glob("*.py"):
            if path.name == "oracle.py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                assert not any(name.split(".")[0] == "pyscf" for name in names), f"{path.name} imports pyscf"


@pytest.mark.oracle
class TestAgainstLibxc:
    """A pointwise check on every functional; gate G0.6 does 1e5 points in the suite."""

    @pytest.mark.parametrize("name", NATIVE)
    @pytest.mark.parametrize("polarised", [False, True])
    def test_energy_and_potentials_match_libxc(self, name: str, polarised: bool) -> None:
        evaluate_libxc = _libxc()
        if evaluate_libxc is None:
            pytest.skip("PySCF (libxc) is not installed")
        from cdft.xc.dispatch import libxc_code

        functional = functional_by_name(name)
        n, sigma, tau = random_ingredients(2000, polarised, 6.0, seed=3)
        needs_sigma = functional.rung.value >= XCRung.GGA.value
        reference = evaluate_libxc(libxc_code(functional), n, sigma if needs_sigma else None, None)
        out = functional.evaluate(torch.tensor(n), torch.tensor(sigma) if needs_sigma else None)
        scaled = lambda a, b: float((np.abs(a - b) / np.maximum(1.0, np.abs(b))).max())  # noqa: E731
        assert scaled(out.e_xc.numpy(), reference.e_xc) < 1e-10
        assert scaled(out.v_xc.numpy(), reference.v_rho) < 1e-10
        if needs_sigma:
            assert scaled(out.v_sigma.numpy(), reference.v_sigma) < 1e-10


class TestKnownProperties:
    def test_pz81_is_discontinuous_at_rs_one_as_published(self) -> None:
        """PZ81's two branches meet with the published 3.2e-5 Ha jump at ``r_s = 1`` (para)."""
        from cdft.xc.lda import _PZ81, _pz81_epsilon

        rs = torch.tensor([1.0 - 1e-12, 1.0 + 1e-12], dtype=torch.float64)
        jump = float(_pz81_epsilon(rs, _PZ81["para"])[1] - _pz81_epsilon(rs, _PZ81["para"])[0])
        assert 3.0e-5 < abs(jump) < 3.4e-5

    def test_pbe_correlation_reduces_to_pw92_mod_at_zero_gradient(self) -> None:
        n = torch.tensor([[0.02, 0.3, 4.0]], dtype=torch.float64)
        pbe = functional_by_name("gga_c_pbe").evaluate(n, torch.zeros_like(n)).e_xc
        pw = functional_by_name("lda_c_pw_mod").evaluate(n).e_xc
        assert torch.allclose(pbe, pw, rtol=0, atol=1e-14)
