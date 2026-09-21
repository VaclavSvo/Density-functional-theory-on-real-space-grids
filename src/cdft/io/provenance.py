"""Provenance capture -- gate G5.1.

Every record embeds its commit and dirty flag, the config hash, the versions of every package that
touched a number, the device block (device, name, compute capability, CUDA toolkit and driver
versions, deterministic-algorithm mode, cuBLAS workspace setting, peak GPU memory), the precision
policy in force and the wall time; without them a record is not a repeatable experiment (S4).
Rationale: ``docs/03_METHOD.md (Part E)``.
"""

from __future__ import annotations

import datetime as _dt
import importlib.metadata as _metadata
import pathlib
import platform
import subprocess

import torch

from contract import CONTRACT_VERSION, NumericsConfig, Provenance

from ..precision import device_record, policy_record, resolve_device

__all__ = ["capture", "git_state", "package_versions", "device_description"]

_TRACKED_PACKAGES = ("torch", "numpy", "scipy", "h5py", "pyyaml", "pylibxc2", "pyscf", "ase")


def git_state(repo: pathlib.Path | None = None) -> tuple[str, bool]:
    """Return ``(commit_sha, dirty)`` for the repository, or ``("", False)`` outside one.

    Both calls are short-timeout and non-fatal, so a host without git still produces a record
    and the empty SHA says so. ``git_dirty`` is recorded rather than forbidden: an uncommitted
    result is useful but must not be mistaken for a reproducible one.
    """
    root = repo or pathlib.Path(__file__).resolve().parents[3]
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - environment dependent
        return "", False
    return sha, bool(status)


def package_versions() -> dict[str, str]:
    """Return the installed versions of every package that can influence a number.

    An absent package is recorded as ``"absent"`` rather than omitted: not installed and not
    recorded are different states.
    """
    versions: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = _metadata.version(name)
        except _metadata.PackageNotFoundError:
            versions[name] = "absent"
    versions["python"] = platform.python_version()
    return versions


def device_description(device: torch.device) -> tuple[str, str]:
    """Return ``(device_name, driver_version)``; the driver is the NVIDIA one, empty off CUDA."""
    if device.type == "cuda":  # pragma: no cover - no CUDA in the verification environment
        record = device_record(device)
        return record["device_name"], record["cuda_driver_version"]
    return f"cpu:{platform.processor() or platform.machine()}", ""


def capture(
    numerics: NumericsConfig,
    wall_time_s: float = 0.0,
    *,
    device: torch.device | None = None,
    gpu_peak_bytes: int | None = None,
) -> Provenance:
    """Assemble the provenance block for one run.

    ``device`` is the device the run actually used; when omitted it is resolved from
    ``numerics.device`` as the solver does. ``gpu_peak_bytes`` is the solve's CUDA peak
    (:func:`cdft.precision.peak_memory_bytes`), ``None`` off CUDA.
    """
    device = device if device is not None else resolve_device(numerics.device)
    sha, dirty = git_state()
    block = device_record(device)
    # The same two values device_description returns, taken from the block already read.
    name = block["device_name"]
    driver = block["cuda_driver_version"] if device.type == "cuda" else ""
    return Provenance(
        contract_version=CONTRACT_VERSION,
        git_sha=sha,
        git_dirty=dirty,
        config_hash=numerics.fingerprint(),
        package_versions=package_versions(),
        device_name=name,
        driver_version=driver,
        precision_policy=policy_record(numerics.precision, device),
        hostname=platform.node(),
        timestamp_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        wall_time_s=wall_time_s,
        device=block["device"],
        device_capability=block["device_capability"],
        torch_cuda_version=block["torch_cuda_version"],
        cuda_driver_version=block["cuda_driver_version"],
        deterministic_algorithms=block["deterministic_algorithms"],
        cublas_workspace_config=block["cublas_workspace_config"],
        gpu_peak_bytes=gpu_peak_bytes,
    )
