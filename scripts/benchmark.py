#!/usr/bin/env python3
"""Throughput benchmark of the solver kernels and the registered scenarios.

    python scripts/benchmark.py --device cpu --quick                  # validate the script (64^3)
    python scripts/benchmark.py --device cuda                         # kernels + every scenario
    python scripts/benchmark.py --device cuda --sizes 64 96 128 --scenarios h_atom he_atom_lda
    python scripts/benchmark.py --device cuda --no-scenarios --fused eager

Kernel layer (synthetic cubes, ``--sizes``) and scenario layer (``--scenarios``). Each kernel is
the median of ``--repeats`` calls after ``--warmup``, synchronised on CUDA; effective GB/s is
reported only where a byte count is well defined and carries the definition that produced it.
Writes ``reports/benchmarks/<hostname>_<device>.json`` (``--output``) after every measurement, so
an interrupted run keeps what it measured, and prints Markdown tables at the end. ``--device
cuda`` without a CUDA device is an error, never a fallback (G5.5).
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import os
import pathlib
import platform
import statistics
import subprocess
import sys
import time
from typing import Callable

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

print = functools.partial(print, flush=True)  # noqa: A001 - every progress line is flushed

#: Spacing of the synthetic cubes, bohr. Z h = 0.25 keeps the cusp quadrature's mass weights
#: positive (they need Z h <= 1) and matches the interacting production spacing (D-53).
SYNTHETIC_SPACING = 0.25
#: Block size of the apply / filter kernels (the CheFSI block of a two-electron system plus extras).
BLOCK = 5
#: Chebyshev degree of the filter kernel.
FILTER_DEGREE = 16
#: Scenario grids whose interpolation operators are timed.
CSR_SCENARIOS = ("h_atom_lda", "h2_R1.4_lda")


class Recorder:
    """Accumulates results and rewrites the JSON file after each one."""

    def __init__(self, path: pathlib.Path, header: dict[str, object]) -> None:
        """Start a report at ``path``."""
        self.path = path
        self.data: dict[str, object] = {"header": header, "kernels": [], "scenarios": [], "profiles": []}
        self.write()

    def add(self, layer: str, entry: dict[str, object]) -> None:
        """Append ``entry`` to ``layer`` and write."""
        self.data[layer].append(entry)  # type: ignore[union-attr]
        self.write()

    def write(self) -> None:
        """Write atomically."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".partial")
        temporary.write_text(json.dumps(self.data, indent=2, default=str), encoding="utf-8")
        temporary.replace(self.path)


def header_for(device) -> dict[str, object]:
    """Machine and build description the results are keyed by."""
    import torch

    from cdft.precision import deterministic_mode

    header: dict[str, object] = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": str(torch.version.cuda),
        "device": str(device),
        "threads": int(torch.get_num_threads()),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "deterministic_algorithms": deterministic_mode(),
    }
    if device.type == "cuda":
        header["device_name"] = torch.cuda.get_device_name(device)
        header["device_capability"] = ".".join(str(v) for v in torch.cuda.get_device_capability(device))
        header["device_memory_bytes"] = int(torch.cuda.get_device_properties(device).total_memory)
    else:
        header["device_name"] = platform.processor() or platform.machine()
    # Additive keys, so older readers of the file keep working.
    from cdft.device_policy import apply_policy, capture_profile

    settings = apply_policy(device)
    header["device_profile"] = capture_profile(device).as_dict()
    header["derived_settings"] = settings.as_dict()
    return header


def timed(fn: Callable[[], object], device, repeats: int, warmup: int) -> dict[str, float]:
    """Median / min / max wall time of ``fn`` over ``repeats`` calls after ``warmup`` calls."""
    import torch

    cuda = device.type == "cuda"
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize(device)
    samples = []
    for _ in range(repeats):
        if cuda:
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize(device)
        samples.append(time.perf_counter() - start)
    return {
        "median_s": statistics.median(samples),
        "min_s": min(samples),
        "max_s": max(samples),
        "repeats": repeats,
        "warmup": warmup,
    }


def _laplacian_taps(order: int) -> int:
    """Number of non-zero terms the Cartesian Laplacian stencil applies (three axes)."""
    from cdft.operators.laplacian import fd_coefficients

    return 3 * sum(1 for c in fd_coefficients(2, order) if c != 0.0)


def _synthetic_grid(n: int, device, dtype):
    """An ``n^3`` box grid at :data:`SYNTHETIC_SPACING` with the origin on the central point."""
    import torch

    from cdft.grid import UniformGrid
    from contract import BoundaryMode, DomainMode, GridConfig

    edge = (n - 1) * SYNTHETIC_SPACING
    config = GridConfig(
        spacing=SYNTHETIC_SPACING,
        fd_order=8,
        domain=DomainMode.BOX,
        boundary=BoundaryMode.ZERO,
        box_lengths=(edge, edge, edge),
        use_double_grid=False,
        fourier_filter_projectors=False,
    )
    return UniformGrid.from_config(config, device=device, dtype=dtype if dtype is not None else torch.float64)


def kernel_layer(recorder: Recorder, device, sizes: list[int], repeats: int, warmup: int, csr_scenarios) -> None:
    """Time every synthetic-cube kernel at each size and record it."""
    import torch

    from cdft.eigen.chefsi import chebyshev_filter
    from cdft.eigen.rayleigh_ritz import rayleigh_ritz
    from cdft.operators.cusp import CuspFactor
    from cdft.operators import fused
    from cdft.operators.divergence import _weighted_divergence_eager, weighted_divergence
    from cdft.operators.external import external_potential
    from cdft.operators.hamiltonian import CuspFactoredHamiltonian, LocalHamiltonian
    from cdft.operators.poisson import CoulombCutoffPoisson
    from cdft.operators.quadrature import CuspQuadrature
    from cdft.precision import configure_device, peak_memory_bytes, reset_peak_memory, seeded_randn
    from contract import AtomicStructure, ExternalPotentialKind, ExternalPotentialSpec

    configure_device(device)
    taps = _laplacian_taps(8)

    def record(kernel: str, size, dtype, timing: dict[str, float], **extra) -> None:
        entry = {"kernel": kernel, "size": size, "dtype": str(dtype).removeprefix("torch."), **timing, **extra}
        if "bytes_per_call" in extra:
            entry["effective_GB_per_s"] = extra["bytes_per_call"] / timing["median_s"] / 1e9
        recorder.add("kernels", entry)
        rate = f"  {entry['effective_GB_per_s']:.3f} GB/s" if "effective_GB_per_s" in entry else ""
        print(f"  {kernel:<18} {str(size):<14} {entry['dtype']:<8} {timing['median_s'] * 1e3:10.3f} ms{rate}")

    for n in sizes:
        print(f"[kernels] {n}^3 on {device}")
        for dtype in (torch.float64, torch.float32):
            reset_peak_memory(device)
            grid = _synthetic_grid(n, device, dtype)
            spec = ExternalPotentialSpec(kind=ExternalPotentialKind.HARMONIC, omega=0.2)
            local = LocalHamiltonian(grid, external_potential(spec, AtomicStructure(), grid, dtype=dtype))
            generator = torch.Generator(device="cpu").manual_seed(0)
            block = seeded_randn((BLOCK, grid.n_points), generator, device=device, dtype=dtype)
            itemsize = block.element_size()
            apply_bytes = (taps + 1) * itemsize * BLOCK * grid.n_points
            record(
                "apply_local", f"{n}^3", dtype, timed(lambda: local.apply(block), device, repeats, warmup),
                block=BLOCK, n_points=grid.n_points, bytes_per_call=apply_bytes,
                bytes_definition="(stencil taps + 1) * itemsize * block * n_points",
            )
            lo, hi = local.spectral_bounds(generator=torch.Generator(device="cpu").manual_seed(1))
            lower = lo + 0.05 * (hi - lo)
            record(
                "chebyshev_local", f"{n}^3", dtype,
                timed(lambda: chebyshev_filter(local, block, FILTER_DEGREE, lower, hi, lo), device, repeats, warmup),
                block=BLOCK, degree=FILTER_DEGREE, bytes_per_call=FILTER_DEGREE * apply_bytes,
                bytes_definition="degree * apply bytes",
                gpu_peak_bytes=peak_memory_bytes(device),
            )
            del local, block, grid

        dtype = torch.float64
        reset_peak_memory(device)
        grid = _synthetic_grid(n, device, dtype)
        positions = torch.zeros((1, 3), dtype=dtype, device=device)
        compile_before = _compile_s(device)
        build_start = time.perf_counter()
        factor = CuspFactor(grid, (1.0,), positions)
        quadrature = CuspQuadrature(grid, factor)
        cusp = CuspFactoredHamiltonian(grid, factor, quadrature=quadrature)
        cusp.apply(seeded_randn((1, grid.n_points), None, device=device))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        build_s = time.perf_counter() - build_start
        block = seeded_randn((BLOCK, grid.n_points), torch.Generator(device="cpu").manual_seed(0), device=device)
        itemsize = block.element_size()
        # The divergence form: two staggered stencils per axis, each of ``order`` taps.
        cusp_bytes = (3 * 2 * 8 + 1) * itemsize * BLOCK * grid.n_points
        compile_mid = _compile_s(device)
        timing = timed(lambda: cusp.apply(block), device, repeats, warmup)
        record(
            "apply_cusp", f"{n}^3", dtype, timing,
            block=BLOCK, n_points=grid.n_points, build_s=build_s, n_nodes=quadrature.n_nodes,
            bytes_per_call=cusp_bytes,
            bytes_definition="(3 axes * 2 staggered stencils * 8 taps + 1) * itemsize * block * n_points",
            fused_tier=fused.kernel_record(device)["tier"],
            fused_compile_s=_compile_s(device) - compile_before,
            fused_compile_in_build_s=compile_mid - compile_before,
        )
        # The stencil alone, eager against the device dispatch (the same call on the CPU).
        stencil_bytes = (3 * 2 * 8 + 3) * itemsize * BLOCK * grid.n_points
        box = grid.scatter_to_box(block)
        stencil_args = (box, factor.face_weights(grid.fd_order), grid.spacing, grid.fd_order, grid.shape)
        for name, kernel in (("divergence_eager", _weighted_divergence_eager), ("divergence_fused", weighted_divergence)):
            before = _compile_s(device)
            record(
                name, f"{n}^3", dtype, timed(functools.partial(kernel, *stencil_args), device, repeats, warmup),
                block=BLOCK, bytes_per_call=stencil_bytes,
                bytes_definition="(3 axes * 2 stencils * 8 taps + 3 weights) * itemsize * block * n_points",
                fused_tier=fused.kernel_record(device)["tier"] if name.endswith("fused") else "eager",
                fused_compile_s=_compile_s(device) - before,
            )
        del box, stencil_args
        lo, hi = cusp.spectral_bounds(generator=torch.Generator(device="cpu").manual_seed(1))
        lower = lo + 0.05 * (hi - lo)
        before = _compile_s(device)
        record(
            "chebyshev_cusp", f"{n}^3", dtype,
            timed(lambda: chebyshev_filter(cusp, block, FILTER_DEGREE, lower, hi, lo), device, repeats, warmup),
            block=BLOCK, degree=FILTER_DEGREE, bytes_per_call=FILTER_DEGREE * cusp_bytes,
            bytes_definition="degree * apply bytes",
            fused_tier=fused.kernel_record(device)["tier"],
            fused_compile_s=_compile_s(device) - before,
        )
        # One recurrence step: today's expression against fused.chebyshev_update (same c, a, b).
        step_generator = torch.Generator(device="cpu").manual_seed(3)
        y = seeded_randn((BLOCK, grid.n_points), step_generator, device=device)
        previous = seeded_randn((BLOCK, grid.n_points), step_generator, device=device)
        applied = cusp.apply(y)
        e, c = 0.5 * (hi - lower), 0.5 * (hi + lower)
        sigma = e / (lo - c)
        sigma2 = 1.0 / (2.0 / sigma - sigma)
        a, b = 2.0 * sigma2 / e, sigma * sigma2
        step_bytes = 4 * itemsize * BLOCK * grid.n_points
        step_definition = "4 * itemsize * block * n_points (three inputs read, one output written)"
        record(
            "chebyshev_step_eager", f"{n}^3", dtype,
            timed(lambda: (applied - c * y) * a - b * previous, device, repeats, warmup),
            block=BLOCK, bytes_per_call=step_bytes, bytes_definition=step_definition, fused_tier="eager",
        )
        before = _compile_s(device)
        record(
            "chebyshev_step_fused", f"{n}^3", dtype,
            timed(lambda: fused.chebyshev_update(applied, y, previous, c, a, b), device, repeats, warmup),
            block=BLOCK, bytes_per_call=step_bytes, bytes_definition=step_definition,
            fused_tier=fused.kernel_record(device).get("chebyshev", "eager"),
            fused_compile_s=_compile_s(device) - before,
        )
        del y, previous, applied
        for states in (5, 10):
            trial = seeded_randn((states, grid.n_points), torch.Generator(device="cpu").manual_seed(2), device=device)
            record(
                "rayleigh_ritz", f"{n}^3/{states}", dtype,
                timed(lambda: rayleigh_ritz(cusp, trial, cusp.measure), device, repeats, warmup),
                states=states,
            )
        points = grid.points()
        density = torch.exp(-(points * points).sum(dim=-1))
        poisson = CoulombCutoffPoisson(grid)
        record(
            "poisson", f"{n}^3", dtype, timed(lambda: poisson.solve(density), device, repeats, warmup),
            padded_shape=list(poisson.padded_shape),
            gpu_peak_bytes=peak_memory_bytes(device),
        )
        del grid, factor, quadrature, cusp, block, poisson, density, points
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for scenario_id in csr_scenarios:
        csr_kernel(recorder, device, scenario_id, repeats, warmup, record)


def _compile_s(device) -> float:
    """Compile time the fused-kernel registry has recorded on ``device`` so far (0 on the CPU)."""
    from cdft.operators import fused

    return float(fused.kernel_record(device).get("compile_s", 0.0))


def csr_kernel(recorder: Recorder, device, scenario_id: str, repeats: int, warmup: int, record) -> None:
    """Time ``L`` and ``L^T`` on the quadrature of one registered scenario's production grid."""
    import torch

    from cdft.config import ALL_ELECTRON_NUMERICS
    from cdft.grid import UniformGrid, grid_config_for_scenario, lattice_anchor
    from cdft.operators.cusp import CuspFactor
    from cdft.operators.quadrature import CuspQuadrature
    from cdft.physics_config import REGISTRY
    from cdft.precision import peak_memory_bytes, reset_peak_memory
    from cdft.scf.solve import is_interacting
    from cdft.structure import nuclear_charges, positions_tensor

    scenario = REGISTRY[scenario_id]
    print(f"[kernels] L / L^T on the {scenario_id} grid")
    reset_peak_memory(device)
    config, _ = grid_config_for_scenario(scenario, ALL_ELECTRON_NUMERICS.grid, interacting=is_interacting(scenario))
    grid = UniformGrid.from_config(
        config, structure=scenario.structure, device=device, dtype=torch.float64,
        anchor=lattice_anchor(scenario, derive_grid=True),
    )
    start = time.perf_counter()
    factor = CuspFactor(
        grid,
        nuclear_charges(scenario.structure, scenario.external),
        positions_tensor(scenario.structure, device=device),
    )
    quadrature = CuspQuadrature(grid, factor)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    build_s = time.perf_counter() - start
    operator = quadrature._operator
    nnz = int(operator.values().numel())
    index_bytes = operator.col_indices().element_size()
    points = grid.points()
    field = torch.exp(-0.5 * (points * points).sum(dim=-1))
    nodes = torch.cos(quadrature.points).sum(dim=-1)
    common = {
        "scenario": scenario_id,
        "grid_shape": list(grid.shape),
        "n_points": grid.n_points,
        "n_nodes": quadrature.n_nodes,
        "nnz": nnz,
        "csr_index_dtype": str(operator.col_indices().dtype).removeprefix("torch."),
        "transpose_layout": getattr(quadrature, "transpose_layout", None),
        "build_s": build_s,
        "bytes_per_call": nnz * (8 + index_bytes),
        "bytes_definition": "nnz * (value + index itemsize)",
        "operator_bytes_each": nnz * (8 + index_bytes) + int(operator.crow_indices().numel()) * index_bytes,
    }
    size = f"{quadrature.n_nodes // 1000}k nodes"
    record("interp_L", size, torch.float64, timed(lambda: quadrature.interpolate(field), device, repeats, warmup), **common)
    record(
        "lift_LT", size, torch.float64, timed(lambda: quadrature.lift(nodes), device, repeats, warmup),
        **common, gpu_peak_bytes=peak_memory_bytes(device),
    )
    del grid, factor, quadrature, operator, field, nodes, points
    if device.type == "cuda":
        torch.cuda.empty_cache()


def scenario_layer(recorder: Recorder, device, scenario_ids: list[str]) -> None:
    """Solve each scenario at the production presets (no cross-check) and record the cost."""
    import torch

    from cdft.config import numerics_for_spec
    from cdft.operators import fused
    from cdft.physics_config import REGISTRY
    from cdft.scf.solve import solve_scenario
    from contract import Device

    for scenario_id in scenario_ids:
        scenario = REGISTRY[scenario_id]
        base, _ = numerics_for_spec(scenario)
        numerics = dataclasses.replace(
            base, eigen=dataclasses.replace(base.eigen, cross_check=False), device=Device(device.type)
        )
        print(f"[scenario] {scenario_id} on {device} ...")
        compile_before = _compile_s(device)
        start = time.perf_counter()
        artifact = solve_scenario(scenario, numerics, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        measurements = artifact.measurements
        entry = {
            "scenario": scenario_id,
            "status": artifact.status.value,
            "error": artifact.error_message,
            "wall_s": elapsed,
            "scf_iterations": measurements.get("scf_iterations"),
            "eigen_iterations": measurements.get("eigen_iterations"),
            "hamiltonian_applications": measurements.get("hamiltonian_applications"),
            "gpu_peak_bytes": artifact.provenance.gpu_peak_bytes,
            "grid_shape": (measurements.get("grid") or {}).get("shape"),
            "n_points": (measurements.get("grid") or {}).get("n_points"),
            "n_nodes": (measurements.get("cusp_quadrature") or {}).get("n_nodes"),
            "total_energy": None if artifact.result is None else artifact.result.energies.total,
            "deterministic_algorithms": artifact.provenance.deterministic_algorithms,
            "fused_tier": fused.kernel_record(device)["tier"],
            "fused_compile_s": _compile_s(device) - compile_before,
        }
        entry["wall_s_excluding_compile"] = elapsed - entry["fused_compile_s"]
        recorder.add("scenarios", entry)
        peak = entry["gpu_peak_bytes"]
        print(
            f"[scenario] {scenario_id}: {entry['status']} in {elapsed:.1f} s, scf iterations "
            f"{entry['scf_iterations']}, peak {'n/a' if peak is None else f'{peak / 2**30:.2f} GiB'}"
        )
        del artifact
        if device.type == "cuda":
            torch.cuda.empty_cache()


def profile_layer(recorder: Recorder, device, profiles: list[str]) -> None:
    """Time ``test_suite.py --profile <p> --device <d>`` end to end."""
    for profile in profiles:
        command = [
            sys.executable, str(REPO_ROOT / "test_suite.py"), "--profile", profile,
            "--device", device.type, "--no-json",
        ]
        print(f"[profile] {' '.join(command[1:])}")
        start = time.perf_counter()
        completed = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
        elapsed = time.perf_counter() - start
        tail = completed.stdout.strip().splitlines()[-3:]
        recorder.add(
            "profiles",
            {"profile": profile, "exit_code": completed.returncode, "wall_s": elapsed, "tail": tail},
        )
        print(f"[profile] {profile}: exit {completed.returncode} in {elapsed:.0f} s")


def markdown(data: dict) -> str:
    """The results as Markdown tables."""
    header = data["header"]
    lines = [
        f"### Benchmark -- {header.get('hostname')} / {header.get('device_name')} "
        f"(torch {header.get('torch')}, CUDA {header.get('torch_cuda')}, {header.get('device')})",
        "",
        f"fused kernels: {header.get('fused_kernels', {})}",
        "",
        "| kernel | size | dtype | median ms | GB/s | tier | compile s |",
        "|---|---|---|---:|---:|---|---:|",
    ]
    for k in data["kernels"]:
        rate = k.get("effective_GB_per_s")
        lines.append(
            f"| {k['kernel']} | {k['size']} | {k['dtype']} | {k['median_s'] * 1e3:.3f} | "
            f"{'' if rate is None else f'{rate:.3f}'} | {k.get('fused_tier', '')} | "
            f"{k.get('fused_compile_s') or 0.0:.2f} |"
        )
    if data["scenarios"]:
        lines += [
            "",
            "| scenario | status | wall s | compile s | SCF it. | peak GiB |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for s in data["scenarios"]:
            peak = s.get("gpu_peak_bytes")
            iterations = s.get("scf_iterations")
            lines.append(
                f"| {s['scenario']} | {s['status']} | {s['wall_s']:.1f} | {s.get('fused_compile_s') or 0.0:.1f} | "
                f"{'' if iterations is None else int(iterations)} | "
                f"{'' if peak is None else f'{peak / 2**30:.2f}'} |"
            )
    if data["profiles"]:
        lines += ["", "| profile | exit | wall s |", "|---|---:|---:|"]
        for p in data["profiles"]:
            lines.append(f"| {p['profile']} | {p['exit_code']} | {p['wall_s']:.0f} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the layers, write the JSON and print the tables."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--sizes", nargs="+", type=int, default=[64, 96, 128])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--scenarios", nargs="+", default=None, help="scenario ids (default: all registered)")
    parser.add_argument("--no-scenarios", action="store_true", help="skip the scenario layer")
    parser.add_argument("--no-kernels", action="store_true", help="skip the kernel layer")
    parser.add_argument(
        "--csr-scenarios", nargs="*", default=list(CSR_SCENARIOS),
        help="scenario grids for the L / L^T kernels (default: %(default)s)",
    )
    parser.add_argument("--profiles", nargs="*", default=[], choices=("quick", "full", "gpu"))
    parser.add_argument(
        "--quick", action="store_true",
        help="validation run: 64^3 only, 3 repeats after 1 warm-up, L / L^T on h_atom_lda only, no scenarios",
    )
    parser.add_argument(
        "--fused", choices=("auto", "compile", "gemm", "eager"), default=None,
        help="set CDFT_FUSED for this run (default: the environment as found)",
    )
    parser.add_argument(
        "--csc-transpose", action="store_true",
        help="hold L^T as the CSC view of L (quadrature.TRANSPOSE_AS_CSC, D-62) for the interp_L / lift_LT rows",
    )
    parser.add_argument("--output", default=None, help="JSON path (default reports/benchmarks/<host>_<device>.json)")
    args = parser.parse_args(argv)

    from cdft.operators.fused import ENV_FUSED, kernel_record, reset_registry
    from cdft.physics_config import REGISTRY
    from cdft.precision import resolve_device
    from contract import Device

    try:
        device = resolve_device(Device(args.device))
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.fused is not None:
        os.environ[ENV_FUSED] = args.fused
        reset_registry()
    if args.csc_transpose:
        import cdft.operators.quadrature as quadrature_module

        quadrature_module.TRANSPOSE_AS_CSC = True
    if args.quick:
        args.sizes, args.repeats, args.warmup = [64], 3, 1
        args.csr_scenarios = ["h_atom_lda"]
        if args.scenarios is None:
            args.no_scenarios = True
    output = pathlib.Path(args.output) if args.output else (
        REPO_ROOT / "reports" / "benchmarks" / f"{platform.node() or 'host'}_{device.type}.json"
    )
    recorder = Recorder(
        output,
        {
            **header_for(device), "argv": sys.argv[1:], "quick": args.quick,
            "fused_requested": args.fused or os.environ.get(ENV_FUSED) or "auto",
            "transpose_as_csc": bool(args.csc_transpose),
        },
    )
    print(f"benchmark on {device} -> {output}")
    if not args.no_kernels:
        kernel_layer(recorder, device, args.sizes, args.repeats, args.warmup, args.csr_scenarios)
    if not args.no_scenarios:
        ids = args.scenarios or list(REGISTRY.ids())
        scenario_layer(recorder, device, ids)
    if args.profiles:
        profile_layer(recorder, device, args.profiles)
    recorder.data["header"]["fused_kernels"] = kernel_record(device)
    recorder.data["header"]["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    recorder.write()
    print()
    print(markdown(recorder.data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
