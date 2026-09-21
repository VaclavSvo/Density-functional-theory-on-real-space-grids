#!/usr/bin/env python3
"""Isolate the Rayleigh--Ritz cliff at 2^19 points on the GPU (H4).

    python scripts/probes/gemm_probe.py        # every op, every size, three cuBLAS workspace settings

``rayleigh_ritz`` jumps from 21 ms at 72^3 points to 17 s at 81^3 on a GTX 1660 Ti while every
stencil, FFT and CSR kernel scales linearly, and 2^19 = 524 288 lies between the two. This times
the pieces of one step in isolation for ``n`` on both sides of 2^19: the Gram ``X @ X^T`` in both
layouts, ``eigh`` and ``cholesky`` on the ``k x k`` matrix, and the rotation ``C @ X``. Each
``CUBLAS_WORKSPACE_CONFIG`` setting runs in a fresh subprocess, since cuBLAS reads the variable
when it creates its handle -- the solver sets ``:4096:8`` for deterministic mode (D-61), and that
workspace limit is the suspect. Medians of 5 after 2 warm-ups, synchronised.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time

SIZES = (262144, 373248, 524287, 524288, 524289, 531441, 592704, 1048576)
STATES = (5, 10)
WORKSPACES = ("unset", ":4096:8", ":16:8", ":4096:2")


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


def _worker(dtype_name: str) -> None:
    global torch
    import torch

    device = torch.device("cuda")
    dtype = getattr(torch, dtype_name)
    rows = []
    for k in STATES:
        for n in SIZES:
            g = torch.Generator(device="cpu").manual_seed(0)
            x = torch.randn((k, n), generator=g, dtype=dtype).to(device)
            xt = x.t().contiguous()  # (n, k) layout
            gram = x @ x.t()
            gram = gram + k * torch.eye(k, dtype=dtype, device=device)
            c = torch.randn((k, k), generator=g, dtype=dtype).to(device)
            row = {
                "k": k,
                "n": n,
                "gram_x_xT_ms": _time(lambda: x @ x.t(), device),
                "gram_xtT_xt_ms": _time(lambda: xt.t() @ xt, device),
                "gram_mm_out_ms": _time(lambda: torch.mm(x, x.t()), device),
                "rotate_C_x_ms": _time(lambda: c @ x, device),
                "eigh_kxk_ms": _time(lambda: torch.linalg.eigh(gram), device),
                "cholesky_kxk_ms": _time(lambda: torch.linalg.cholesky(gram), device),
                "sum_sq_ms": _time(lambda: (x * x).sum(dim=1), device),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    print(json.dumps({"done": True, "dtype": dtype_name}), flush=True)


def main() -> int:
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        _worker(sys.argv[2])
        return 0
    import torch as _torch

    print(f"torch {_torch.__version__}  CUDA {_torch.version.cuda}  device {_torch.cuda.get_device_name(0)}", flush=True)
    results: dict[str, list[dict]] = {}
    for dtype_name in ("float64", "float32"):
        for ws in WORKSPACES:
            env = dict(os.environ)
            env.pop("CUBLAS_WORKSPACE_CONFIG", None)
            if ws != "unset":
                env["CUBLAS_WORKSPACE_CONFIG"] = ws
            label = f"{dtype_name} / CUBLAS_WORKSPACE_CONFIG={ws}"
            print(f"\n=== {label} ===", flush=True)
            proc = subprocess.run(
                [sys.executable, __file__, "--worker", dtype_name],
                env=env, capture_output=True, text=True, check=False,
            )
            rows = [json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")]
            rows = [r for r in rows if "n" in r]
            results[label] = rows
            if proc.returncode != 0:
                print(proc.stderr[-2000:], flush=True)
            print(f"{'k':>3} {'n':>8} {'gram X.XT':>10} {'gram XtT.Xt':>11} {'mm':>9} {'rotate':>9} {'eigh':>8} {'chol':>8} {'sumsq':>8}", flush=True)
            for r in rows:
                print(
                    f"{r['k']:>3} {r['n']:>8} {r['gram_x_xT_ms']:>10.3f} {r['gram_xtT_xt_ms']:>11.3f} "
                    f"{r['gram_mm_out_ms']:>9.3f} {r['rotate_C_x_ms']:>9.3f} {r['eigh_kxk_ms']:>8.3f} "
                    f"{r['cholesky_kxk_ms']:>8.3f} {r['sum_sq_ms']:>8.3f}",
                    flush=True,
                )
    out = os.path.join("reports", "benchmarks", "gemm_probe.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nwritten {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
