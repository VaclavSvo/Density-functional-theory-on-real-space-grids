"""The HDF5 corpus: one file per batch, one group per run, append-only.

The schema is :data:`~contract.CORPUS_LAYOUT`, authoritative so a reader in another language can be
built from the contract alone. Two load-bearing properties, both gated: the default iterator yields
only trusted records (G5.4), and fields are stored rather than summarised (D-14). Rationale:
``docs/03_METHOD.md (Part E)``.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Iterator

import h5py
import numpy as np

from contract import (
    EnergyBreakdown,
    GateKind,
    GateReport,
    GateResult,
    GateVerdict,
    Provenance,
    RunArtifact,
    RunStatus,
    TRUSTED_STATUSES,
    SCFResult,
    SCFTrajectory,
    ScenarioSpec,
)

from ..config import numerics_from_dict, numerics_to_dict

__all__ = ["CorpusWriter"]

#: String attributes of the provenance device block, written and read by name.
_DEVICE_ATTRS = (
    "device",
    "device_capability",
    "torch_cuda_version",
    "cuda_driver_version",
    "deterministic_algorithms",
    "cublas_workspace_config",
)


def _peak_or_none(value: Any) -> int | None:
    """Map the stored ``gpu_peak_bytes`` attribute back to the provenance field (``-1`` is ``None``)."""
    number = int(value)
    return None if number < 0 else number


class CorpusWriter:
    """Append-only HDF5 corpus writer and reader, implementing :class:`~contract.CorpusWriterProtocol`."""

    def __init__(self, path: str | pathlib.Path, compression: str = "gzip", level: int = 4) -> None:
        """Open (or create) a corpus file. The file is opened per operation, never held."""
        self.path = pathlib.Path(path)
        self.compression = compression
        self.level = level
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _dataset_kwargs(self, array: np.ndarray) -> dict[str, Any]:
        """Return compression options, disabled for arrays too small to benefit."""
        if array.size < 256 or not self.compression:
            return {}
        return {"compression": self.compression, "compression_opts": self.level}

    def write(self, artifact: RunArtifact) -> str:
        """Append one record and return its group path."""
        group_path = f"/{artifact.run_id}"
        with h5py.File(self.path, "a") as handle:
            if group_path in handle:
                raise ValueError(f"{artifact.run_id} is already present; the corpus is append-only")
            group = handle.create_group(group_path)
            self._write_attrs(group, artifact)
            self._write_scenario(group, artifact.scenario)
            self._write_result(group, artifact)
            self._write_gates(group, artifact.gates)
            self._write_measurements(group, artifact.measurements)
        return group_path

    def _write_attrs(self, group: h5py.Group, artifact: RunArtifact) -> None:
        """Write the record attributes of :data:`~contract.CORPUS_LAYOUT`."""
        prov = artifact.provenance
        group.attrs["contract_version"] = prov.contract_version
        group.attrs["status"] = artifact.status.value
        group.attrs["scenario_id"] = artifact.scenario.scenario_id
        group.attrs["git_sha"] = prov.git_sha
        group.attrs["git_dirty"] = bool(prov.git_dirty)
        group.attrs["config_hash"] = prov.config_hash
        group.attrs["package_versions"] = json.dumps(dict(prov.package_versions), sort_keys=True)
        group.attrs["device_name"] = prov.device_name
        group.attrs["driver_version"] = prov.driver_version
        group.attrs["precision_policy"] = json.dumps(dict(prov.precision_policy), sort_keys=True)
        group.attrs["hostname"] = prov.hostname
        group.attrs["timestamp_utc"] = prov.timestamp_utc
        group.attrs["wall_time_s"] = float(prov.wall_time_s)
        for key in _DEVICE_ATTRS:
            group.attrs[key] = str(getattr(prov, key))
        # HDF5 has no null; -1 stands for "not measured" (a CPU run) and is read back as None.
        group.attrs["gpu_peak_bytes"] = int(prov.gpu_peak_bytes) if prov.gpu_peak_bytes is not None else -1
        group.attrs["numerics"] = json.dumps(numerics_to_dict(artifact.numerics), sort_keys=True, default=str)
        group.attrs["error_message"] = artifact.error_message

    def _write_scenario(self, group: h5py.Group, scenario: ScenarioSpec) -> None:
        """Write the structure, the scenario summary and the reference table."""
        structure = group.create_group("structure")
        numbers = np.asarray(scenario.structure.numbers, dtype=np.int32)
        structure.create_dataset("numbers", data=numbers)
        positions = (
            np.asarray(scenario.structure.positions, dtype=np.float64).reshape(-1, 3)
            if scenario.structure.n_atoms
            else np.zeros((0, 3), dtype=np.float64)
        )
        structure.create_dataset("positions", data=positions)
        structure.attrs["charge"] = float(scenario.structure.charge)
        structure.attrs["multiplicity"] = int(scenario.structure.multiplicity)
        structure.attrs["label"] = scenario.structure.label

        meta = group.create_group("scenario")
        meta.attrs["scenario_id"] = scenario.scenario_id
        meta.attrs["xc_name"] = scenario.xc.name
        meta.attrs["xc_rung"] = int(scenario.xc.rung)
        meta.attrs["external_kind"] = scenario.external.kind.value
        meta.attrs["external_params"] = json.dumps(
            {
                "charges": list(scenario.external.charges),
                "softening": scenario.external.softening,
                "omega": scenario.external.omega,
                "box_length": scenario.external.box_length,
                "depth": scenario.external.depth,
                "width": scenario.external.width,
            },
            sort_keys=True,
        )
        meta.attrs["n_electrons"] = (
            float("nan") if scenario.electrons.n_electrons is None else float(scenario.electrons.n_electrons)
        )
        meta.attrs["magnetisation"] = float(scenario.electrons.magnetisation)
        meta.attrs["n_spin"] = int(scenario.electrons.n_spin)
        meta.attrs["pseudo_table"] = "" if scenario.pseudo is None else scenario.pseudo.table
        meta.attrs["all_electron"] = bool(scenario.all_electron)
        meta.attrs["notes"] = scenario.notes

        if scenario.reference:
            reference = group.create_group("reference")
            for key, value in scenario.reference.items():
                entry = reference.create_group(key)
                entry.attrs["value"] = float(value.value)
                entry.attrs["units"] = value.units
                entry.attrs["kind"] = value.kind.value
                entry.attrs["method"] = value.method
                entry.attrs["citation"] = value.citation
                entry.attrs["tolerance"] = float(value.tolerance)

    def _write_result(self, group: h5py.Group, artifact: RunArtifact) -> None:
        """Write the density, potentials, orbitals, energies and SCF trajectory."""
        result = artifact.result
        if result is None:
            return
        output = artifact.numerics.output

        if output.store_density:
            density = group.create_group("density")
            self._dataset(density, "n", result.density)
        potentials = group.create_group("potentials")
        if result.v_hartree is not None:
            self._dataset(potentials, "v_hartree", result.v_hartree)
        if result.v_xc is not None:
            self._dataset(potentials, "v_xc", result.v_xc)

        orbitals = group.create_group("orbitals")
        self._dataset(orbitals, "eigenvalues", result.eigenvalues)
        self._dataset(orbitals, "occupations", result.occupations)
        if output.store_orbitals and result.orbitals is not None:
            self._dataset(orbitals, "psi", result.orbitals)

        energies = group.create_group("energies")
        breakdown = result.energies
        for name in (
            "total",
            "kinetic",
            "external",
            "hartree",
            "xc",
            "nonlocal_ps",
            "ion_ion",
            "dispersion",
            "entropy",
        ):
            energies.attrs[name] = float(getattr(breakdown, name))
        energies.attrs["harris_foulkes"] = (
            float("nan") if breakdown.harris_foulkes is None else float(breakdown.harris_foulkes)
        )

        if output.store_scf_trajectory:
            scf = group.create_group("scf")
            traj = result.trajectory
            scf.create_dataset("energies", data=np.asarray(traj.energies, dtype=np.float64))
            scf.create_dataset(
                "residual_norms", data=np.asarray(traj.residual_norms, dtype=np.float64)
            )
            scf.create_dataset(
                "density_changes", data=np.asarray(traj.density_changes, dtype=np.float64)
            )
            scf.attrs["mixing_events"] = json.dumps(traj.mixing_events)
            scf.attrs["fallback_events"] = json.dumps(traj.fallback_events)
            scf.attrs["converged"] = bool(result.converged)
            scf.attrs["n_iterations"] = int(result.n_iterations)

    def _write_gates(self, group: h5py.Group, report: GateReport) -> None:
        """Write the gate table with measured values, not only verdicts (gate G5.3)."""
        gates = group.create_group("gates")
        if not report.results:
            return
        string_type = h5py.string_dtype(encoding="utf-8")
        gates.create_dataset(
            "gate_id", data=np.array([r.gate_id for r in report.results], dtype=object), dtype=string_type
        )
        gates.create_dataset(
            "name", data=np.array([r.name for r in report.results], dtype=object), dtype=string_type
        )
        gates.create_dataset(
            "verdict",
            data=np.array([r.verdict.value for r in report.results], dtype=object),
            dtype=string_type,
        )
        gates.create_dataset(
            "measured", data=np.asarray([r.measured for r in report.results], dtype=np.float64)
        )
        gates.create_dataset(
            "threshold", data=np.asarray([r.threshold for r in report.results], dtype=np.float64)
        )
        gates.create_dataset(
            "kind", data=np.array([r.kind.value for r in report.results], dtype=object), dtype=string_type
        )
        gates.create_dataset(
            "citation",
            data=np.array([r.citation for r in report.results], dtype=object),
            dtype=string_type,
        )
        gates.create_dataset(
            "units", data=np.array([r.units for r in report.results], dtype=object), dtype=string_type
        )
        gates.create_dataset(
            "reason", data=np.array([r.reason for r in report.results], dtype=object), dtype=string_type
        )
        gates.attrs["detail"] = json.dumps(
            {r.gate_id: _jsonable(r.detail) for r in report.results}, default=str
        )

    def _write_measurements(self, group: h5py.Group, measurements: Any) -> None:
        """Write the solve-time instrument readings that gates consumed."""
        group.create_group("measurements").attrs["json"] = json.dumps(
            _jsonable(dict(measurements)), default=str
        )

    def _dataset(self, group: h5py.Group, name: str, tensor: Any) -> None:
        """Write a torch tensor as a float64 dataset with compression where it pays."""
        array = tensor.detach().to("cpu").to(dtype=_np_dtype(tensor)).numpy()
        group.create_dataset(name, data=array, **self._dataset_kwargs(array))

    # -- reading --

    def run_ids(self, include_invalid: bool = False) -> tuple[str, ...]:
        """Return the record identifiers in the file, filtered by status unless asked otherwise.

        The filter is :data:`~contract.TRUSTED_STATUSES`, not ``VALID`` alone, which would drop
        every ``MARGINAL`` record -- a pass. G5.4 checks both directions.
        """
        if not self.path.exists():
            return ()
        trusted = {status.value for status in TRUSTED_STATUSES}
        with h5py.File(self.path, "r") as handle:
            return tuple(
                key
                for key in handle
                if include_invalid or handle[key].attrs.get("status") in trusted
            )

    def read(self, run_id: str) -> RunArtifact:
        """Read one record back into a :class:`~contract.RunArtifact`.

        Tensors come back as numpy arrays; the contract types them ``Any`` so that inspecting a
        corpus needs no torch.
        """
        with h5py.File(self.path, "r") as handle:
            group = handle[run_id]
            numerics = numerics_from_dict(json.loads(group.attrs["numerics"]))
            provenance = Provenance(
                contract_version=str(group.attrs["contract_version"]),
                git_sha=str(group.attrs["git_sha"]),
                git_dirty=bool(group.attrs["git_dirty"]),
                config_hash=str(group.attrs["config_hash"]),
                package_versions=json.loads(group.attrs["package_versions"]),
                device_name=str(group.attrs["device_name"]),
                driver_version=str(group.attrs["driver_version"]),
                precision_policy=json.loads(group.attrs["precision_policy"]),
                hostname=str(group.attrs["hostname"]),
                timestamp_utc=str(group.attrs["timestamp_utc"]),
                wall_time_s=float(group.attrs["wall_time_s"]),
                # Older records carry no device block: read as empty.
                **{key: str(group.attrs.get(key, "")) for key in _DEVICE_ATTRS},
                gpu_peak_bytes=_peak_or_none(group.attrs.get("gpu_peak_bytes", -1)),
            )
            scenario = ScenarioSpec(scenario_id=str(group.attrs["scenario_id"]))
            result = self._read_result(group)
            report = self._read_gates(group)
            measurements = json.loads(group["measurements"].attrs["json"]) if "measurements" in group else {}
            return RunArtifact(
                run_id=run_id,
                scenario=scenario,
                numerics=numerics,
                provenance=provenance,
                result=result,
                gates=report,
                measurements=measurements,
                status=RunStatus(str(group.attrs["status"])),
                error_message=str(group.attrs.get("error_message", "")),
            )

    def _read_result(self, group: h5py.Group) -> SCFResult | None:
        """Reconstruct the solution fields of a record, or ``None`` when the run produced none."""
        if "energies" not in group:
            return None
        energies_group = group["energies"]
        harris = float(energies_group.attrs["harris_foulkes"])
        breakdown = EnergyBreakdown(
            total=float(energies_group.attrs["total"]),
            kinetic=float(energies_group.attrs["kinetic"]),
            external=float(energies_group.attrs["external"]),
            hartree=float(energies_group.attrs["hartree"]),
            xc=float(energies_group.attrs["xc"]),
            nonlocal_ps=float(energies_group.attrs["nonlocal_ps"]),
            ion_ion=float(energies_group.attrs["ion_ion"]),
            dispersion=float(energies_group.attrs["dispersion"]),
            entropy=float(energies_group.attrs["entropy"]),
            harris_foulkes=None if np.isnan(harris) else harris,
        )
        scf_group = group.get("scf")
        trajectory = SCFTrajectory(
            energies=list(np.asarray(scf_group["energies"])) if scf_group else [],
            residual_norms=list(np.asarray(scf_group["residual_norms"])) if scf_group else [],
            density_changes=list(np.asarray(scf_group["density_changes"])) if scf_group else [],
            mixing_events=json.loads(scf_group.attrs["mixing_events"]) if scf_group else [],
            fallback_events=json.loads(scf_group.attrs["fallback_events"]) if scf_group else [],
        )
        return SCFResult(
            converged=bool(scf_group.attrs["converged"]) if scf_group else False,
            n_iterations=int(scf_group.attrs["n_iterations"]) if scf_group else 0,
            density=np.asarray(group["density"]["n"]) if "density" in group else None,
            eigenvalues=np.asarray(group["orbitals"]["eigenvalues"]),
            occupations=np.asarray(group["orbitals"]["occupations"]),
            energies=breakdown,
            trajectory=trajectory,
        )

    def _read_gates(self, group: h5py.Group) -> GateReport:
        """Reconstruct the gate table of a record."""
        gates = group.get("gates")
        if gates is None or "gate_id" not in gates:
            return GateReport()
        detail = json.loads(gates.attrs.get("detail", "{}"))
        results = []
        count = gates["gate_id"].shape[0]
        for index in range(count):
            gate_id = _text(gates["gate_id"][index])
            results.append(
                GateResult(
                    gate_id=gate_id,
                    name=_text(gates["name"][index]),
                    verdict=GateVerdict(_text(gates["verdict"][index])),
                    measured=float(gates["measured"][index]),
                    threshold=float(gates["threshold"][index]),
                    kind=GateKind(_text(gates["kind"][index])),
                    citation=_text(gates["citation"][index]),
                    units=_text(gates["units"][index]),
                    reason=_text(gates["reason"][index]),
                    detail=detail.get(gate_id, {}),
                )
            )
        return GateReport(results=results)

    def iterate(self, include_invalid: bool = False) -> Iterator[RunArtifact]:
        """Iterate over records, yielding only trusted ones unless explicitly asked otherwise.

        The default is G5.4: a consumer opts in to records that failed their gates.
        """
        for run_id in self.run_ids(include_invalid=include_invalid):
            yield self.read(run_id)


def _np_dtype(tensor: Any) -> Any:
    """Return the dtype a tensor should be stored in: float64 for anything floating point."""
    import torch

    return torch.float64 if tensor.dtype.is_floating_point else tensor.dtype


def _text(value: Any) -> str:
    """Decode an HDF5 string scalar, which may arrive as bytes."""
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _jsonable(value: Any) -> Any:
    """Convert a nested structure to something :mod:`json` can serialise."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    return str(value)
