#!/usr/bin/env python3
"""Isolate the triangular solve of ``orthonormalise`` on the GPU (H4, second probe).

    python scripts/probes/trsm_probe.py

``scripts/probes/gemm_probe.py`` cleared the Gram GEMM, ``eigh``, Cholesky and the rotation at every size
around 2^19 points, leaving the one call it did not time:
``torch.linalg.solve_triangular(L, X, upper=False)`` with ``L`` of shape ``(k, k)`` and ``X`` of
shape ``(k, n)`` -- cuBLAS ``trsm`` against ``n`` right-hand sides. This times it for ``n`` on both
sides of 2^19 beside its candidate replacements: ``chunked`` (the same call on column chunks of at
most 2^18, each column solved independently so the arithmetic per column is unchanged), ``inverse``
(``L^{-1} @ X``, a GEMM, rounding differently from a forward substitution), ``right`` (the
transposed formulation) and ``torch.linalg.solve`` (LU) for reference. Medians of 5 after 2
warm-ups, synchronised, in float64 (the subspace dtype) and float32.
"""

from __future__ import annotations

import json
import os
import statistics
import time

import torch

SIZES = (262144, 373248, 524287, 524288, 524289, 531441, 592704, 1048576)
STATES = (5, 7, 10)
CHUNK = 2**18


def _time(fn, device, repeats=5, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize(device)
        samples.append(1e3 * (time.perf_counter() - t0))
    return statistics.median(samples)


def chunked_solve(chol: torch.Tensor, flat: torch.Tensor, chunk: int = CHUNK) -> torch.Tensor:
    """``solve_triangular`` over column chunks of at most ``chunk`` right-hand sides."""
    n = flat.shape[-1]
    if n <= chunk:
        return torch.linalg.solve_triangular(chol, flat, upper=False)
    out = torch.empty_like(flat)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        out[:, start:stop] = torch.linalg.solve_triangular(chol, flat[:, start:stop], upper=False)
    return out


def main() -> int:
    device = torch.device("cuda")
    print(f"torch {torch.__version__}  CUDA {torch.version.cuda}  device {torch.cuda.get_device_name(0)}", flush=True)
    results = {}
    for dtype in (torch.float64, torch.float32):
        print(f"\n=== {dtype} ===", flush=True)
        print(f"{'k':>3} {'n':>8} {'trsm':>10} {'chunked':>10} {'inverse':>10} {'right':>10} {'lu solve':>10}  max|chunk-trsm| max|inv-trsm|", flush=True)
        rows = []
        for k in STATES:
            for n in SIZES:
                g = torch.Generator(device="cpu").manual_seed(0)
                x = torch.randn((k, n), generator=g, dtype=dtype).to(device)
                gram = x @ x.t() / n + torch.eye(k, dtype=dtype, device=device)
                chol = torch.linalg.cholesky(gram)
                eye = torch.eye(k, dtype=dtype, device=device)
                ref = torch.linalg.solve_triangular(chol, x, upper=False)
                row = {
                    "k": k, "n": n,
                    "trsm_ms": _time(lambda: torch.linalg.solve_triangular(chol, x, upper=False), device),
                    "chunked_ms": _time(lambda: chunked_solve(chol, x), device),
                    "inverse_ms": _time(lambda: torch.linalg.solve_triangular(chol, eye, upper=False) @ x, device),
                    "right_ms": _time(lambda: torch.linalg.solve_triangular(chol.t(), x.t(), upper=True, left=False).t(), device),
                    "lu_ms": _time(lambda: torch.linalg.solve(chol, x), device),
                    "chunked_vs_trsm": float((chunked_solve(chol, x) - ref).abs().max()),
                    "inverse_vs_trsm": float((torch.linalg.solve_triangular(chol, eye, upper=False) @ x - ref).abs().max()),
                }
                rows.append(row)
                print(
                    f"{k:>3} {n:>8} {row['trsm_ms']:>10.3f} {row['chunked_ms']:>10.3f} {row['inverse_ms']:>10.3f} "
                    f"{row['right_ms']:>10.3f} {row['lu_ms']:>10.3f}  {row['chunked_vs_trsm']:.2e}  {row['inverse_vs_trsm']:.2e}",
                    flush=True,
                )
        results[str(dtype)] = rows
    out = os.path.join("reports", "benchmarks", "trsm_probe.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nwritten {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
