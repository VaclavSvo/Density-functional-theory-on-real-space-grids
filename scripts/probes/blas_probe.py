"""Which BLAS/LAPACK entry points survive in this interpreter, alone and after ``import torch``.

    python scripts/probes/blas_probe.py

Each probe runs in a fresh subprocess, so an abort (``Fatal Python error: Aborted``; exit 3221226505
/ -6 / 134) is reported instead of killing the caller. Prints one line per probe with its exit code,
plus the numpy/scipy/torch versions and BLAS build info, so a record can name the offending
combination. Use it when a solve dies inside ``numpy.linalg`` or ``scipy.linalg``: the production
path needs neither (``cdft.operators.gauss_legendre``), but the oracle legs in ``cdft.reference``
still use scipy's, which this probes separately.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

PROBES: dict[str, str] = {
    "numpy matmul (BLAS), numpy alone": "import numpy as np; a = np.eye(64); a.T @ a",
    "numpy matmul (BLAS), after torch": "import torch, numpy as np; a = np.eye(64); a.T @ a",
    "numpy eigvalsh, numpy alone": "import numpy as np; np.linalg.eigvalsh(np.eye(4))",
    "numpy eigvalsh, after torch": "import torch, numpy as np; np.linalg.eigvalsh(np.eye(4))",
    "numpy eigvalsh, torch after numpy": "import numpy as np, torch; np.linalg.eigvalsh(np.eye(4))",
    "numpy leggauss(4), after torch": "import torch, numpy as np; np.polynomial.legendre.leggauss(4)",
    "numpy solve, after torch": "import torch, numpy as np; np.linalg.solve(np.eye(4), np.ones(4))",
    "scipy eigh_tridiagonal, after torch": (
        "import torch, numpy as np; from scipy.linalg import eigh_tridiagonal; "
        "eigh_tridiagonal(np.arange(5.0), np.ones(4))"
    ),
    "scipy eig_banded, after torch": (
        "import torch, numpy as np; from scipy.linalg import eig_banded; "
        "eig_banded(np.vstack([np.ones(5), np.arange(5.0)]))"
    ),
    "torch eigvalsh": "import torch; torch.linalg.eigvalsh(torch.eye(4, dtype=torch.float64))",
    "cdft gauss_legendre(4), after torch": (
        "import torch; from cdft.operators.gauss_legendre import gauss_legendre; gauss_legendre(4)"
    ),
}


def _run(code: str) -> tuple[int, str]:
    """Run ``code`` in a fresh interpreter; return the exit code and the first stderr line."""
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    lines = [line for line in proc.stderr.splitlines() if line.strip()]
    # The BLAS runtime prints its reason (OpenBLAS, MKL, or OpenMP's "OMP: Error #15") before
    # Python's own "Fatal Python error", so the first stderr line is the informative one.
    return proc.returncode, (lines[0] if lines else "")


def _versions() -> dict[str, str]:
    """Versions and BLAS build info of numpy, scipy and torch, each read in its own process."""
    code = textwrap.dedent(
        """
        import json, sys
        out = {"python": sys.version.split()[0], "platform": sys.platform}
        try:
            import numpy as np
            out["numpy"] = np.__version__
            try:
                info = np.show_config(mode="dicts")
                out["numpy_blas"] = info.get("Build Dependencies", {}).get("blas", {}).get("name", "?")
                out["numpy_lapack"] = info.get("Build Dependencies", {}).get("lapack", {}).get("name", "?")
            except Exception as exc:  # pragma: no cover - old numpy
                out["numpy_blas"] = f"unavailable ({exc})"
        except Exception as exc:
            out["numpy"] = f"import failed: {exc}"
        try:
            import scipy
            out["scipy"] = scipy.__version__
        except Exception as exc:
            out["scipy"] = f"import failed: {exc}"
        try:
            import torch
            out["torch"] = torch.__version__
            out["torch_cuda"] = str(torch.version.cuda)
            out["torch_mkl"] = str(torch.backends.mkl.is_available())
        except Exception as exc:
            out["torch"] = f"import failed: {exc}"
        print(json.dumps(out))
        """
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:  # pragma: no cover - the version probe itself died
        return {"error": proc.stderr.strip()[-400:], "code": str(proc.returncode)}


def main() -> int:
    """Run every probe and print a table; exit 1 if any probe did not return 0."""
    for key, value in _versions().items():
        print(f"{key:>14}: {value}")
    print()
    failures = 0
    for name, code in PROBES.items():
        rc, tail = _run(code)
        status = "ok" if rc == 0 else f"EXIT {rc}"
        failures += rc != 0
        print(f"{status:<18} {name}" + (f"   [{tail[:120]}]" if rc != 0 and tail else ""))
    print()
    if failures:
        print(
            f"{failures} probe(s) failed. If 'numpy alone' passes and 'after torch' aborts, the two\n"
            "BLAS runtimes conflict in this process; the production path avoids numpy's LAPACK\n"
            "(cdft.operators.gauss_legendre), but numpy matmul is everywhere. If 'numpy alone' aborts\n"
            "too, numpy's BLAS is broken in this env (a conda/pip mix): rebuild the env with pip only\n"
            "(README, Windows). PySCF is not needed in it -- its references are stored in\n"
            "cdft/reference/computed.py. An 'OMP: Error #15' reason line means two OpenMP runtimes\n"
            "(conda MKL numpy + pip torch): KMP_DUPLICATE_LIB_OK=TRUE only silences it and is unsupported\n"
            "by Intel; rebuild pip-only instead."
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
