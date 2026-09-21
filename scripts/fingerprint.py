#!/usr/bin/env python3
"""Full-precision fingerprint of every registered scenario, for before/after comparison.

    python scripts/fingerprint.py reports/fingerprint_<tag>.json
    python scripts/fingerprint.py --device cuda reports/fingerprint_gpu64.json
    python scripts/fingerprint.py --scenarios h_atom he_atom_lda reports/fp_subset.json
    python scripts/fingerprint.py --diff reports/fingerprint_a.json reports/fingerprint_b.json

Every energy term at ``repr()`` precision, every eigenvalue, every occupation and a SHA-256 of the
density and orbital arrays -- the numerical standard of ``docs/02_STATUS.md`` (Part A §2).
Solved at the solver's own state count and without the LOBPCG cross-check, so the fingerprint is
the production solve and nothing else; ``cuda`` without a CUDA device is an error (G5.5). Writes
the named JSON, rewritten after every scenario so an interrupted run keeps what it finished.

``--diff`` reports the largest ``|delta|`` of each quantity class and fails when an energy or
eigenvalue moves by more than ``--tol`` (default 1e-10 Ha: a CPU/GPU difference is summation order
only), an occupation by more than ``--occupation-tol``, or a status, grid or scenario set differs;
hashes and iteration counts are informational. ``--exact`` fails on any difference at all.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import pathlib
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

#: Keys of a scenario entry compared as floating-point lists / mappings.
_FLOAT_CLASSES = ("energies", "eigenvalues", "occupations")
#: Keys compared for equality only and never failing the diff.
_HASH_KEYS = ("density_sha256", "orbitals_sha256")
#: Per-run facts that are not results: ignored by ``--exact``. ``n_states`` is absent from older
#: files; where both sides carry it a mismatch is reported.
_INFORMATIONAL_KEYS = ("wall_time_s", "gpu_peak_bytes", "n_states")


def _sha(tensor) -> str:
    import numpy as np

    array = np.ascontiguousarray(tensor.detach().cpu().numpy().astype(np.float64))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _write(target: pathlib.Path, data: dict) -> None:
    """Write the fingerprint atomically (temporary file, then replace)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(target)


def fingerprint(
    device: str = "cpu",
    scenario_ids: list[str] | None = None,
    target: pathlib.Path | None = None,
    deterministic: bool = False,
) -> dict:
    """Solve the registered scenarios at production settings and collect the fingerprint.

    ``scenario_ids`` restricts the set (default: all); with ``target`` the file is rewritten after
    every scenario.
    """
    import platform

    import torch

    from cdft.config import numerics_for_spec
    from cdft.physics_config import REGISTRY
    from cdft.precision import deterministic_mode, resolve_device
    from cdft.scf.solve import solve_scenario
    from contract import Device

    requested = Device(device)
    resolved = resolve_device(requested)
    out: dict = {
        "_meta": {
            "device_requested": device,
            "device": str(resolved),
            "device_name": torch.cuda.get_device_name(resolved) if resolved.type == "cuda" else platform.machine(),
            "torch": torch.__version__,
            "torch_cuda": str(torch.version.cuda),
            "threads": int(torch.get_num_threads()),
            "hostname": platform.node(),
            "deterministic_requested": bool(deterministic),
        }
    }
    selected = [s for s in REGISTRY if scenario_ids is None or s.scenario_id in scenario_ids]
    if scenario_ids is not None:
        unknown = sorted(set(scenario_ids) - {s.scenario_id for s in selected})
        if unknown:
            raise SystemExit(f"unknown scenario id(s): {', '.join(unknown)}")
    for scenario in selected:
        base, _ = numerics_for_spec(scenario)
        numerics = dataclasses.replace(
            base,
            eigen=dataclasses.replace(base.eigen, cross_check=False),
            output=dataclasses.replace(base.output, store_orbitals=True),
            device=requested,
            deterministic=bool(deterministic),
        )
        started = time.perf_counter()
        print(f"[fingerprint] {scenario.scenario_id} on {resolved} ...", flush=True)
        artifact = solve_scenario(scenario, numerics)
        elapsed = time.perf_counter() - started
        n_states = None
        if artifact.result is None:
            out[scenario.scenario_id] = {"status": artifact.status.value, "error": artifact.error_message}
        else:
            result = artifact.result
            # The solver's own count (``needed``): one state for every model system.
            n_states = int(result.eigenvalues.shape[-1])
            energies = {
                field.name: repr(getattr(result.energies, field.name))
                for field in dataclasses.fields(result.energies)
            }
            out[scenario.scenario_id] = {
                "status": artifact.status.value,
                "energies": energies,
                "eigenvalues": [repr(float(x)) for x in result.eigenvalues.flatten()],
                "occupations": [repr(float(x)) for x in result.occupations.flatten()],
                "density_sha256": _sha(result.density),
                "orbitals_sha256": _sha(result.orbitals) if result.orbitals is not None else None,
                "grid": artifact.measurements.get("grid"),
                "n_iterations": result.n_iterations,
                "n_states": n_states,
                "wall_time_s": round(elapsed, 3),
                "gpu_peak_bytes": artifact.measurements.get("gpu_peak_bytes"),
            }
        print(
            f"[fingerprint] {scenario.scenario_id}: {artifact.status.value} in {elapsed:.1f} s"
            f"  (n_states {n_states})",
            flush=True,
        )
        out["_meta"]["deterministic_algorithms"] = deterministic_mode()
        if target is not None:
            _write(target, out)
        del artifact
        if resolved.type == "cuda":
            torch.cuda.empty_cache()
    return out


def _as_float(value) -> float:
    """Parse a stored ``repr`` (or a number) back to a float; ``None`` (or ``"None"``) becomes NaN."""
    if value is None or value == "None":
        return math.nan
    return float(value)


def _max_delta(left, right) -> float:
    """Largest ``|a - b|`` over two equally shaped lists or mappings of numbers; ``inf`` on a shape mismatch."""
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return math.inf
        pairs = [(left[k], right[k]) for k in left]
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return math.inf
        pairs = list(zip(left, right))
    else:
        return 0.0 if left == right else math.inf
    worst = 0.0
    for a, b in pairs:
        fa, fb = _as_float(a), _as_float(b)
        if math.isnan(fa) and math.isnan(fb):
            continue
        delta = abs(fa - fb)
        worst = max(worst, delta if not math.isnan(delta) else math.inf)
    return worst


def diff(
    a: dict,
    b: dict,
    tol: float = 1.0e-10,
    occupation_tol: float = 0.0,
    exact: bool = False,
) -> int:
    """Print the differences between two fingerprints; return the number of failing scenarios.

    See the module docstring for what fails. With ``exact`` any difference in any key fails.
    """
    for label, fp in (("left", a), ("right", b)):
        meta = fp.get("_meta")
        if meta:
            print(f"{label:<6} {json.dumps(meta, sort_keys=True)}")
    failing = 0
    overall = {name: 0.0 for name in _FLOAT_CLASSES}
    hashes_equal = {key: 0 for key in _HASH_KEYS}
    hashes_total = {key: 0 for key in _HASH_KEYS}
    scenario_ids = sorted(k for k in set(a) | set(b) if not k.startswith("_"))
    for scenario_id in scenario_ids:
        left, right = a.get(scenario_id), b.get(scenario_id)
        if left is None or right is None:
            failing += 1
            print(f"{scenario_id:<16} FAIL   present on one side only")
            continue
        if exact:
            comparable_left = {k: v for k, v in left.items() if k not in _INFORMATIONAL_KEYS}
            comparable_right = {k: v for k, v in right.items() if k not in _INFORMATIONAL_KEYS}
            if None not in (left.get("n_states"), right.get("n_states")) and left["n_states"] != right["n_states"]:
                comparable_left["n_states"], comparable_right["n_states"] = left["n_states"], right["n_states"]
            if comparable_left == comparable_right:
                for key in _HASH_KEYS:
                    if left.get(key) is not None:
                        hashes_total[key] += 1
                        hashes_equal[key] += 1
                print(f"{scenario_id:<16} identical")
                continue
        problems: list[str] = []
        deltas: dict[str, float] = {}
        for name in _FLOAT_CLASSES:
            if name not in left and name not in right:
                continue
            delta = _max_delta(left.get(name), right.get(name))
            deltas[name] = delta
            overall[name] = max(overall[name], delta)
            limit = occupation_tol if name == "occupations" else tol
            if exact and left.get(name) != right.get(name):
                problems.append(f"{name} differ")
            elif not exact and delta > limit:
                problems.append(f"{name} |delta| {delta:.3e} > {limit:g}")
        for key in ("status", "grid"):
            if left.get(key) != right.get(key):
                problems.append(f"{key}: {left.get(key)} -> {right.get(key)}")
        if left.get("error") or right.get("error"):
            problems.append(f"error: {left.get('error')!r} / {right.get('error')!r}")
        equal_hashes = []
        for key in _HASH_KEYS:
            if left.get(key) is None and right.get(key) is None:
                continue
            hashes_total[key] += 1
            same = left.get(key) == right.get(key)
            hashes_equal[key] += int(same)
            equal_hashes.append(f"{key.split('_')[0]}={'equal' if same else 'unequal'}")
            if exact and not same:
                problems.append(f"{key} differs")
        iterations = f"iter {left.get('n_iterations')}/{right.get('n_iterations')}"
        if None not in (left.get("n_states"), right.get("n_states")) and left["n_states"] != right["n_states"]:
            problems.append(f"n_states {left['n_states']} -> {right['n_states']}")
        if exact and left.get("n_iterations") != right.get("n_iterations"):
            problems.append("n_iterations differ")
        verdict = "FAIL" if problems else "ok"
        failing += bool(problems)
        numbers = "  ".join(f"{k}={v:.3e}" for k, v in deltas.items())
        print(f"{scenario_id:<16} {verdict:<6} {numbers}  {' '.join(equal_hashes)}  {iterations}")
        for problem in problems:
            print(f"   {problem}")
        if exact and problems:
            for key in sorted(set(left) | set(right)):
                lv, rv = left.get(key), right.get(key)
                if lv == rv or key in ("wall_time_s", "gpu_peak_bytes"):
                    continue
                if key == "n_states" and None in (lv, rv):
                    continue
                if isinstance(lv, dict) and isinstance(rv, dict):
                    for sub in sorted(set(lv) | set(rv)):
                        if lv.get(sub) != rv.get(sub):
                            print(f"   {key}.{sub}: {lv.get(sub)} -> {rv.get(sub)}")
                else:
                    print(f"   {key}: {lv} -> {rv}")
    print("-" * 78)
    print(
        "max |delta| over all scenarios: "
        + "  ".join(f"{k}={v:.3e}" for k, v in overall.items())
        + ("  (exact mode)" if exact else f"  (tol {tol:g}, occupation tol {occupation_tol:g})")
    )
    print(
        "hashes equal: "
        + "  ".join(f"{k}={hashes_equal[k]}/{hashes_total[k]}" for k in _HASH_KEYS)
        + ("" if exact else "  (informational: summation order changes them)")
    )
    print(f"{failing} failing scenario(s) of {len(scenario_ids)}")
    return failing


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and either fingerprint or diff; return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", help="output path, or two paths with --diff")
    parser.add_argument("--diff", action="store_true", help="compare two stored fingerprints")
    parser.add_argument(
        "--device", choices=("cpu", "cuda", "auto"), default="cpu",
        help="device to solve on (default cpu; cuda without a device is an error)",
    )
    parser.add_argument("--scenarios", nargs="+", default=None, help="restrict to these scenario ids")
    parser.add_argument("--tol", type=float, default=1.0e-10, help="energy/eigenvalue tolerance for --diff")
    parser.add_argument(
        "--occupation-tol", type=float, default=0.0, help="occupation tolerance for --diff (default exact)"
    )
    parser.add_argument("--exact", action="store_true", help="--diff: any difference at all fails")
    parser.add_argument(
        "--deterministic", action="store_true",
        help="solve with NumericsConfig.deterministic=True (strict torch kernels on CUDA; recorded in _meta)",
    )
    args = parser.parse_args(argv)
    if args.diff:
        if len(args.paths) != 2:
            parser.error("--diff takes exactly two paths")
        a = json.loads(pathlib.Path(args.paths[0]).read_text(encoding="utf-8"))
        b = json.loads(pathlib.Path(args.paths[1]).read_text(encoding="utf-8"))
        return 1 if diff(a, b, tol=args.tol, occupation_tol=args.occupation_tol, exact=args.exact) else 0
    target = pathlib.Path(args.paths[0])
    data = fingerprint(
        device=args.device, scenario_ids=args.scenarios, target=target, deterministic=args.deterministic
    )
    _write(target, data)
    print(f"fingerprint of {len(data) - 1} scenarios written to {target}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
