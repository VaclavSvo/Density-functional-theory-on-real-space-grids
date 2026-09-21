"""Publication figures for one solved scenario: ``python figures.py <scenario>`` at the repository root.

Solves a :mod:`physics_config` scenario (optionally at another bond length, functional or electron
count) through :func:`cdft.scf.solve.solve_scenario`, attaches the gate verdict, writes the record
and renders the figures -- or re-renders them from a stored record without solving. Exit codes
follow G5.6; ``--list`` and ``--help`` state the scenarios, figures and options.

Output (default ``reports/figures/<scenario_id>/<run_id>/``): per figure a ``<name>.pdf``/``.png``,
``<name>.caption.txt`` and ``<name>.csv``/``.npz``, plus ``energies.tex``, ``gates.tex``,
``manifest.json`` and ``record.h5``; a partial re-render keeps the entries it left in place.

Only :func:`solve_and_gate` solves, every renderer reads a :class:`FigureRecord`, and nothing is
drawn without a verdict (D-72). Survey tags resolve in ``docs/00_LITERATURE_SURVEY.md``;
design, house style and what is deliberately not drawn in ``docs/03_METHOD.md`` (Part E).
"""

from __future__ import annotations

# Path bootstrap, as in run.py (D-57): root modules and src/cdft importable without installing.
import pathlib as _pathlib
import sys as _sys

_REPO_ROOT = _pathlib.Path(__file__).resolve().parent
for _entry in (_REPO_ROOT, _REPO_ROOT / "src"):
    if _entry.is_dir() and str(_entry) not in _sys.path:
        _sys.path.insert(0, str(_entry))
del _entry

import argparse
import dataclasses
import datetime
import hashlib
import json
import logging
import math
import pathlib
import re
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

import physics_config
from config import LEVELS, PRESETS, load_numerics, numerics_for, numerics_for_spec, numerics_from_dict
from contract import (
    CONTRACT_VERSION,
    TRUSTED_STATUSES,
    AtomicStructure,
    BoundaryMode,
    Device,
    DomainMode,
    ElectronSpec,
    ExternalPotentialKind,
    ExternalPotentialSpec,
    GateResult,
    GateVerdict,
    NumericsConfig,
    ReferenceKind,
    ReferenceValue,
    RunArtifact,
    RunStatus,
    ScenarioSpec,
    XCRung,
    XCSpec,
)
from physics_config import (
    NONINTERACTING_XC,
    REGISTRY,
    interacting,
    lda_functional,
    load_scenarios,
    pbe_functional,
)

__all__ = [
    "FIGURES",
    "FIGURES_VERSION",
    "GATE_PROFILES",
    "HouseStyle",
    "FigureRecord",
    "FieldEvaluator",
    "IntegrationGrid",
    "Verdict",
    "classify_record",
    "compose_scenario_id",
    "parse_scenario_id",
    "select_scenario",
    "select_numerics",
    "figure_scope_problem",
    "solve_and_gate",
    "record_from_artifact",
    "record_from_corpus",
    "render_figures",
    "main",
]

#: Version of the figure layer, in every manifest and file; bump it when a figure's content changes.
FIGURES_VERSION = "1.0.1"

#: Degree of the interpolant of ``rho = n / f^2``: that of ``CuspQuadrature``, so a figure reads the
#: field as the solver's quadrature does. Error ``(k h)^8 / 8!`` on scale ``1/k``.
INTERPOLATION_DEGREE = 7

#: Points per chunk when evaluating fields: 20 000 gather a (20 000, 8, 8, 8) float64 block, 82 MB.
EVALUATION_CHUNK = 20_000

#: Decades of density shown below the maximum; six reaches the tail the boxes are sized for (D-53).
DENSITY_DECADES = 6.0

#: Radius about each nucleus excluded when scaling Laplacian errors: ``lap n ~ -4 Z n / r`` there.
LAPLACIAN_EXCLUSION_BOHR = 0.05

#: ``2 (3 pi^2)^(1/3)``: ``s = |grad n| / (this * n^(4/3))`` [A32], [A4].
_S_PREFACTOR = 2.0 * (3.0 * math.pi**2) ** (1.0 / 3.0)


# --- House style ------------------------------------------------------------------------------

#: Categorical series colours in fixed order (blue, orange, aqua, violet), colour-blind-safe on
#: white; aqua is only 2.8:1 against white, so every series also carries a legend entry.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7")

#: Ink and chrome for a white print surface.
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#ffffff"

#: Status colours, used only for the record-quality badge and always with a text label.
STATUS_WARNING = "#fab219"
STATUS_CRITICAL = "#d03b3b"

#: The one-hue sequential ramp (light -> dark) for magnitudes such as log10 n.
SEQUENTIAL_BLUE = (
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
)

#: The diverging pair for signed fields: blue arm, neutral grey midpoint, red arm (equal steps).
DIVERGING_BLUE_RED = (
    "#0d366b", "#256abf", "#6da7ec", "#b7d3f6",
    "#f0efec",
    "#f5c6c5", "#ec8a89", "#e34948", "#9c2423",
)


@dataclass(frozen=True, slots=True)
class HouseStyle:
    """Figure geometry and typography for a two-column journal.

    Widths are inches, sizes points at print size; the standard bounding box keeps files exact.
    """

    single_width: float = 3.37
    double_width: float = 7.0
    font_family: str = "sans"
    base_size: float = 8.0
    small_size: float = 7.0
    min_size: float = 6.0
    line_width: float = 1.0
    thin_width: float = 0.6
    hairline: float = 0.4
    marker_size: float = 3.2
    dpi: int = 600
    usetex: bool = False
    formats: tuple[str, ...] = ("pdf", "png")

    def rc(self) -> dict[str, Any]:
        """Return the matplotlib rcParams of this style."""
        from cycler import cycler

        serif = self.font_family == "serif"
        params: dict[str, Any] = {
            "font.family": "serif" if serif else "sans-serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman", "Times"],
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "mathtext.fontset": "stix" if serif else "dejavusans",
            "font.size": self.base_size,
            "axes.titlesize": self.base_size,
            "axes.labelsize": self.base_size,
            "legend.fontsize": self.small_size,
            "xtick.labelsize": self.small_size,
            "ytick.labelsize": self.small_size,
            "axes.linewidth": self.thin_width,
            "axes.edgecolor": INK_SECONDARY,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.facecolor": SURFACE,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.prop_cycle": cycler(color=list(SERIES)),
            "grid.color": GRIDLINE,
            "grid.linewidth": self.hairline,
            "grid.linestyle": "-",
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "xtick.major.width": self.thin_width,
            "ytick.major.width": self.thin_width,
            "xtick.minor.width": self.hairline,
            "ytick.minor.width": self.hairline,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "lines.linewidth": self.line_width,
            "lines.markersize": self.marker_size,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "legend.frameon": False,
            "legend.handlelength": 1.8,
            "legend.borderaxespad": 0.4,
            "figure.facecolor": SURFACE,
            "figure.dpi": 150,
            "savefig.dpi": self.dpi,
            "savefig.facecolor": SURFACE,
            "savefig.bbox": "standard",
            "figure.constrained_layout.use": True,
            "figure.constrained_layout.h_pad": 0.03,
            "figure.constrained_layout.w_pad": 0.03,
            "figure.constrained_layout.hspace": 0.02,
            "figure.constrained_layout.wspace": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "cdft-figures",  # deterministic element ids: same record, same bytes
            "text.usetex": self.usetex,
            "image.interpolation": "nearest",
        }
        return params


_PYPLOT: Any = None


def pyplot():
    """Import matplotlib (Agg backend) once and return ``pyplot``; lazy, so the extra stays optional."""
    global _PYPLOT
    if _PYPLOT is None:
        try:
            import matplotlib
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "figures.py needs matplotlib: pip install -e \".[figures]\" (matplotlib>=3.8)"
            ) from exc
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        # Silence fontTools' cosmetic warning about the zero 'created' timestamp of DejaVu on save.
        logging.getLogger("fontTools").setLevel(logging.ERROR)

        _PYPLOT = plt
    return _PYPLOT


def colormap(name: str):
    """Return the house sequential (``"sequential"``) or diverging (``"diverging"``) colormap."""
    from matplotlib.colors import LinearSegmentedColormap

    if name == "sequential":
        cmap = LinearSegmentedColormap.from_list("cdft_sequential_blue", SEQUENTIAL_BLUE, N=256)
        cmap.set_under(SEQUENTIAL_BLUE[0])
        cmap.set_over(SEQUENTIAL_BLUE[-1])
        return cmap
    if name == "diverging":
        cmap = LinearSegmentedColormap.from_list("cdft_diverging_blue_red", DIVERGING_BLUE_RED, N=256)
        cmap.set_under(DIVERGING_BLUE_RED[0])
        cmap.set_over(DIVERGING_BLUE_RED[-1])
        return cmap
    raise KeyError(f"unknown colormap {name!r}")


# --- Scenario selection: registered, optionally at another geometry / functional / N ------------

#: ``--xc`` values and the scenario-id suffix each one carries in the registry's naming.
XC_SUFFIXES: dict[str, str] = {
    "none": "",
    "lda_vwn": "_lda",
    "lda_pw92": "_lda_pw92",
    "lda_pz81": "_lda_pz81",
    "pbe": "_pbe",
}

#: Aliases accepted by ``--xc``.
XC_ALIASES: dict[str, str] = {"lda": "lda_vwn", "vwn": "lda_vwn", "pw92": "lda_pw92", "pz81": "lda_pz81"}

#: Structure builders by id stem: rebuilding through one attaches that bond length's references.
_DIATOMIC_BUILDERS: dict[str, Callable[..., ScenarioSpec]] = {
    "h2plus": physics_config.h2_plus,
    "h2": physics_config.hydrogen_molecule,
}

#: Gate list of a fractional-N scenario (the gates left out are stated for integer N only).
_FRACTIONAL_N_GATES = ("G1.5", "G2.1", "G2.3", "G2.4", "G2.9", "G3.3", "G0.3", "G0.4", "G0.8")

_ID_XC = re.compile(r"(_lda_pw92|_lda_pz81|_lda|_pbe)$")
_ID_N = re.compile(r"_N(\d+(?:\.\d+)?)")
_ID_R = re.compile(r"_R(\d+(?:\.\d+)?)")


def canonical_xc(name: str | None) -> str | None:
    """Normalise an ``--xc`` value (``None`` passes through); raise on an unknown name."""
    if name is None:
        return None
    key = XC_ALIASES.get(name.lower(), name.lower())
    if key not in XC_SUFFIXES:
        choices = ", ".join(sorted(set(XC_SUFFIXES) | set(XC_ALIASES)))
        raise ValueError(f"unknown functional {name!r}; choose from {choices}")
    return key


def parse_scenario_id(scenario_id: str) -> tuple[str, float | None, float | None, str]:
    """Split a registry-style id: ``h2plus_R2_lda`` -> ``("h2plus", 2.0, None, "lda_vwn")``."""
    rest = scenario_id
    xc = "none"
    match = _ID_XC.search(rest)
    if match:
        xc = {"_lda": "lda_vwn", "_lda_pw92": "lda_pw92", "_lda_pz81": "lda_pz81", "_pbe": "pbe"}[match.group(1)]
        rest = rest[: match.start()]
    n_electrons = None
    match = _ID_N.search(rest)
    if match:
        n_electrons = float(match.group(1))
        rest = rest[: match.start()] + rest[match.end():]
    bond = None
    match = _ID_R.search(rest)
    if match:
        bond = float(match.group(1))
        rest = rest[: match.start()] + rest[match.end():]
    return rest, bond, n_electrons, xc


def compose_scenario_id(stem: str, bond_length: float | None, n_electrons: float | None, xc: str) -> str:
    """Inverse of :func:`parse_scenario_id`; numbers carry 12 digits, so ids never collide."""
    out = stem
    if bond_length is not None:
        out += f"_R{bond_length:.12g}"
    if n_electrons is not None:
        out += f"_N{n_electrons:.12g}"
    return out + XC_SUFFIXES[xc]


def _functional_spec(xc: str) -> XCSpec:
    """Return the :class:`~contract.XCSpec` physics_config would attach for a canonical ``--xc`` value."""
    if xc == "pbe":
        return pbe_functional()
    if xc.startswith("lda_"):
        return lda_functional(xc.removeprefix("lda_"))
    raise ValueError(f"{xc!r} is not an interacting functional")


def _with_bond_length(spec: ScenarioSpec, bond_length: float) -> ScenarioSpec:
    """Move the two nuclei of ``spec`` to ``bond_length`` about their midpoint, dropping references."""
    positions = np.asarray(spec.structure.positions, dtype=np.float64).reshape(-1, 3)
    midpoint = positions.mean(axis=0)
    axis = positions[1] - positions[0]
    axis = axis / np.linalg.norm(axis)
    moved = (midpoint - 0.5 * bond_length * axis, midpoint + 0.5 * bond_length * axis)
    structure = dataclasses.replace(
        spec.structure, positions=tuple(tuple(float(x) for x in row) for row in moved)
    )
    return dataclasses.replace(spec, structure=structure, reference={})


def _sibling_gates(
    registry, stem: str, xc: str, bond_length: float | None, fractional: bool
) -> tuple[tuple[str, ...], str] | None:
    """Gate list and id of the registered scenario nearest a built one (stem, N class, R)."""
    best: tuple[tuple[int, float], str, tuple[str, ...]] | None = None
    for candidate in registry:
        if candidate.xc.name.lower() == "none":
            continue
        c_stem, c_bond, c_electrons, c_xc = parse_scenario_id(candidate.scenario_id)
        if c_stem != stem:
            continue
        n = candidate.electrons.n_electrons
        c_fractional = n is not None and abs(n - round(n)) > 1.0e-12
        if c_fractional != fractional:
            continue
        distance = abs((bond_length or 0.0) - (c_bond or 0.0))
        key = (0 if c_xc == xc else 1, distance)
        if best is None or key < best[0]:
            best = (key, candidate.scenario_id, tuple(candidate.gates))
    return None if best is None else (best[2], best[1])


def _electron_count(spec: ScenarioSpec) -> float:
    """Return the electron count a scenario solves with (explicit, or derived from its structure)."""
    from cdft.structure import electron_count

    return float(electron_count(spec.structure, spec.external, spec.electrons.n_electrons))


def _scenario_bond_length(spec: ScenarioSpec) -> float | None:
    """Return the internuclear distance of a two-nucleus scenario (``None`` otherwise)."""
    if spec.structure.n_atoms != 2:
        return None
    positions = np.asarray(spec.structure.positions, dtype=np.float64).reshape(-1, 3)
    return float(np.linalg.norm(positions[1] - positions[0]))


def select_scenario(
    scenario_id: str,
    registry=REGISTRY,
    *,
    bond_length: float | None = None,
    xc: str | None = None,
    n_electrons: float | None = None,
) -> tuple[ScenarioSpec, list[str]]:
    """Return the scenario to solve and the notes explaining how it was obtained.

    A composed id that is registered comes from the registry, with its curated gates and references;
    otherwise the scenario is built. A functional needs Coulomb nuclei (D-53).
    """
    base = registry[scenario_id]
    notes: list[str] = []
    xc0 = base.xc.name.lower()
    if xc0 not in XC_SUFFIXES:
        if bond_length is None and xc is None and n_electrons is None:
            return base, notes
        raise ValueError(
            f"{scenario_id!r} uses the functional {base.xc.name!r}, which figures.py cannot name in an id; "
            f"overrides need one of {', '.join(sorted(XC_SUFFIXES))}"
        )
    target_xc = canonical_xc(xc) or xc0
    if bond_length is not None:
        if base.structure.n_atoms != 2:
            raise ValueError(
                f"--bond-length needs a two-nucleus scenario; {scenario_id!r} has {base.structure.n_atoms}"
            )
        if bond_length <= 0.0:
            raise ValueError("--bond-length must be positive (bohr)")
    if n_electrons is not None and n_electrons <= 0.0:
        raise ValueError("--n-electrons must be positive")
    if target_xc != "none" and (
        base.structure.n_atoms == 0 or base.external.kind is not ExternalPotentialKind.NUCLEAR_COULOMB
    ):
        raise ValueError(
            f"a functional needs Coulomb nuclei: the self-consistent path is the all-electron one (D-53), and "
            f"{scenario_id!r} is a {base.external.kind.value} scenario"
        )

    # Overrides that restate the scenario's own values are not overrides.
    own_bond = _scenario_bond_length(base)
    if bond_length is not None and own_bond is not None and abs(bond_length - own_bond) <= 1.0e-9:
        notes.append(f"R = {bond_length:.12g} bohr is {scenario_id!r}'s own bond length")
        bond_length = None
    if n_electrons is not None and abs(n_electrons - _electron_count(base)) <= 1.0e-12:
        notes.append(f"N = {n_electrons:.12g} is {scenario_id!r}'s own electron count")
        n_electrons = None
    if bond_length is None and n_electrons is None and target_xc == xc0:
        return base, notes

    stem, bond0, electrons0, id_xc = parse_scenario_id(scenario_id)
    if id_xc != xc0:
        # The id does not name the scenario's functional (a YAML entry): keep it whole as the stem.
        stem, bond0, electrons0 = scenario_id, None, None
    target_bond = bond_length if bond_length is not None else bond0
    target_electrons = n_electrons if n_electrons is not None else electrons0
    if bond_length is not None and bond0 is None:
        notes.append(f"{scenario_id!r} carries no _R token; the bond length is appended to the id")
    # An electron count equal to the base's own is the registry's unmarked case: no _N in the id.
    base_id = compose_scenario_id(stem, target_bond, None, "none")
    id_electrons = target_electrons
    if (target_electrons is not None and base_id in registry
            and abs(target_electrons - _electron_count(registry[base_id])) <= 1.0e-12):
        id_electrons = None
    candidate = compose_scenario_id(stem, target_bond, id_electrons, target_xc)
    if candidate in registry:
        notes.append(f"{candidate!r} is registered: the curated entry is used as it stands")
        return registry[candidate], notes
    if candidate == scenario_id:
        raise ValueError(f"the overrides compose {candidate!r} again; state them differently")

    if base_id in registry:
        geometry = registry[base_id]
    elif stem in _DIATOMIC_BUILDERS and target_bond is not None:
        geometry = _DIATOMIC_BUILDERS[stem](target_bond, scenario_id=base_id)
        notes.append(f"geometry rebuilt with physics_config.{_DIATOMIC_BUILDERS[stem].__name__}({target_bond:.12g})")
    elif bond_length is not None:
        geometry = _with_bond_length(base, target_bond)
        notes.append("nuclei moved along their axis; the references of the original geometry are dropped")
    else:
        geometry = base
    geometry = dataclasses.replace(geometry, xc=NONINTERACTING_XC)
    if target_electrons is None and n_electrons is None:
        # keep the electron count of the scenario that was asked for (e.g. he_plus_lda has N = 1)
        target_electrons = base.electrons.n_electrons

    if target_xc == "none":
        electrons = geometry.electrons if target_electrons is None else dataclasses.replace(
            geometry.electrons, n_electrons=target_electrons
        )
        spec = dataclasses.replace(geometry, scenario_id=candidate, electrons=electrons)
        if n_electrons is not None:
            spec = dataclasses.replace(spec, reference={})
            notes.append("electron count changed on a bare-potential scenario: references dropped")
        notes.append(f"built {candidate!r} (non-interacting)")
        return spec, notes

    count = target_electrons if target_electrons is not None else _electron_count(geometry)
    fractional = abs(count - round(count)) > 1.0e-12
    sibling = _sibling_gates(registry, stem, target_xc, target_bond, fractional)
    if sibling is not None:
        gates, extra = sibling[0], ()
        notes.append(f"gate list of the nearest registered relative {sibling[1]!r}")
    elif fractional:
        gates, extra = _FRACTIONAL_N_GATES, ()
        notes.append("gate list of a fractional-N scenario (as h_atom_N0.5_lda names it)")
    else:
        gates, extra = None, ("G4.7",)
        notes.append("default interacting gate list plus G4.7")
    spec = interacting(
        geometry,
        _functional_spec(target_xc),
        candidate,
        n_electrons=target_electrons,
        gates=gates,
        extra_gates=extra,
        notes=f"{geometry.scenario_id} at {target_xc}, built by figures.py (select_scenario).",
    )
    notes.append(f"built {candidate!r} = interacting({geometry.scenario_id!r}, {target_xc})")
    if count > 2.0 + 1.0e-12:
        notes.append("N > 2 lies outside the Phase 1 validation set (Z <= 2, N <= 2)")
    return spec, notes


# --- Numerics, gate profiles, the solve ---------------------------------------------------------

@dataclass(frozen=True, slots=True)
class GateProfile:
    """Which gates run on the figure's record. Never a physics setting (``02_STATUS.md`` A §2)."""

    name: str
    max_solves: int | None
    cross_check: bool
    description: str


#: The two gate budgets of test_suite.py, by name.
GATE_PROFILES: dict[str, GateProfile] = {
    "quick": GateProfile(
        "quick", 0, False,
        "every gate that needs no extra solve (test_suite.py quick budget; ladders, Janak, initial "
        "guesses and the G5.2 pair are reported SKIPPED naming the budget)",
    ),
    "full": GateProfile(
        "full", None, True,
        "every gate, including the convergence ladders, Janak, initial guesses, the LOBPCG "
        "cross-check and the G5.2 strict pair (test_suite.py full budget)",
    ),
}


def select_numerics(
    scenario: ScenarioSpec, preset: str, profile: GateProfile, device: str | None = None
) -> NumericsConfig:
    """Resolve the numerics: a preset name (``auto`` by system) or a YAML file, plus the device."""
    if preset == "auto":
        numerics, _ = numerics_for_spec(scenario)
    elif preset in PRESETS:
        numerics = PRESETS[preset]
    else:
        numerics = load_numerics(preset)
    numerics = dataclasses.replace(
        numerics, eigen=dataclasses.replace(numerics.eigen, cross_check=profile.cross_check)
    )
    if device is not None:
        numerics = dataclasses.replace(numerics, device=Device(device))
    return numerics


def resolve_resolution(
    scenario: ScenarioSpec, level: str, profile: GateProfile, device: str | None = None
) -> tuple[NumericsConfig, dict[str, float]]:
    """Return ``(numerics, grid rule targets)`` of a registered scenario at one resolution level.

    Only a registered id has a setup (D-73). ``standard`` is the production rule; every other level
    is exploratory, and its targets must be active for the solve and for the gate pass alike.
    """
    numerics, targets = numerics_for(scenario.scenario_id, level, device=device)
    numerics = dataclasses.replace(
        numerics, eigen=dataclasses.replace(numerics.eigen, cross_check=profile.cross_check)
    )
    return numerics, targets


def solve_and_gate(
    scenario: ScenarioSpec,
    numerics: NumericsConfig,
    profile: GateProfile,
    *,
    min_states: int | None = None,
) -> RunArtifact:
    """Solve one scenario through ``solve_scenario`` and attach its gates; the only solve here."""
    from cdft.gates.runner import GateSuite, states_required
    from cdft.scf.solve import solve_scenario

    suite = GateSuite(max_solves=profile.max_solves)
    required = max(states_required(scenario, suite.artifact_gates), int(min_states or 0))
    artifact = solve_scenario(scenario, numerics, n_states=required or None)
    return suite.run_on_artifact(artifact)


@dataclass(frozen=True, slots=True)
class Verdict:
    """Whether a record may be drawn and how it is marked: deferred, known-open, invalid (D-72)."""

    status: str
    drawable: bool
    mark: str
    new_failures: int
    known_open: int
    failed_gates: tuple[str, ...]
    reason: str
    #: The failed gates that are registered known-open rows within their band (D-43).
    known_gates: tuple[str, ...] = ()

    @property
    def blocking(self) -> bool:
        """Whether this record sets a non-zero exit code (G5.6): a new failure or a failed run."""
        return self.new_failures > 0 or self.status in (RunStatus.ERROR.value, RunStatus.UNCONVERGED.value)

    def badge(self) -> str:
        """Return the text of the quality badge drawn on every figure of a marked record ("" if none)."""
        if self.mark == "invalid":
            if self.failed_gates:
                new = [gate for gate in self.failed_gates if gate not in self.known_gates]
                what = f"{', '.join(new or self.failed_gates)} failed" + (
                    f" (known-open: {', '.join(self.known_gates)})" if new and self.known_gates else ""
                ) + ("" if self.status == RunStatus.INVALID.value else f" (status {self.status})")
            else:
                what = f"status {self.status}" + (f": {self.reason}" if self.reason else "")
            return f"INVALID record: {what} -- not for publication"
        if self.mark == "known-open":
            return (f"record INVALID on known-open {', '.join(self.failed_gates)} only (D-43) -- "
                    f"outside TRUSTED_STATUSES, not for publication")
        if self.mark == "deferred":
            return "record DEFERRED (a named gate belongs to a later increment) -- not for publication"
        return ""


def classify_record(
    scenario_id: str,
    status: str,
    gate_results: Sequence[GateResult],
    *,
    has_fields: bool,
    error_message: str = "",
    allow_invalid: bool = False,
) -> Verdict:
    """Apply the drawing policy (D-72): ``TRUSTED_STATUSES`` only, unless ``allow_invalid``."""
    from cdft.gates.known_open import partition_failures

    failed = tuple(r.gate_id for r in gate_results if r.verdict is GateVerdict.FAIL)
    split = partition_failures(scenario_id, list(gate_results))
    if not has_fields:
        return Verdict(status, False, "invalid", split.new, split.known, failed,
                       error_message or "the record holds no solution fields")
    if status in {s.value for s in TRUSTED_STATUSES}:
        return Verdict(status, True, "", 0, split.known, failed, "")
    if status == RunStatus.DEFERRED.value:
        return Verdict(status, allow_invalid, "deferred", split.new, split.known, failed,
                       "status deferred: a gate the scenario names belongs to a later increment")
    if status == RunStatus.INVALID.value and split.new == 0 and split.unimplemented == 0 and split.known > 0:
        return Verdict(status, allow_invalid, "known-open", 0, split.known, failed,
                       f"INVALID on registered known-open rows only ({', '.join(failed)}; D-43)", failed)
    from cdft.gates.known_open import classify

    new = [r.gate_id for r in gate_results
           if r.verdict is GateVerdict.FAIL and classify(scenario_id, r.gate_id, r.measured)[0] == "new"]
    known = [gate for gate in failed if gate not in new]
    if error_message:
        reason = error_message
    elif failed:
        reason = (f"{len(new)} gate failure(s) not in the known-open registry for {scenario_id!r}: "
                  f"{', '.join(new) or 'none'}" + (f"; known-open: {', '.join(known)}" if known else ""))
    else:
        reason = f"status {status}"
    return Verdict(status, allow_invalid, "invalid", split.new, split.known, failed, reason, tuple(known))


def gate_summary(gate_results: Sequence[GateResult]) -> dict[str, Any]:
    """Count a gate report by verdict and name the gates a budget (not applicability) left out."""
    counts = {verdict.value: 0 for verdict in GateVerdict}
    budget: list[str] = []
    for result in gate_results:
        counts[result.verdict.value] = counts.get(result.verdict.value, 0) + 1
        if result.verdict is GateVerdict.SKIPPED and "profile allows" in (result.reason or ""):
            budget.append(result.gate_id)
    return {"counts": counts, "skipped_for_budget": budget}


def gate_summary_line(gate_results: Sequence[GateResult]) -> str:
    """One line for captions: the verdict counts and whether a budget left gates unevaluated."""
    summary = gate_summary(gate_results)
    counts = ", ".join(f"{count} {name}" for name, count in summary["counts"].items() if count)
    budget = summary["skipped_for_budget"]
    tail = (f"; {len(budget)} gate(s) not evaluated under this gate budget ({', '.join(budget)}) -- "
            f"re-solve with --gates full before publishing" if budget else "")
    return f"Gates: {counts or 'none recorded'}{tail}."


# --- The record the figures read: from a live artifact or from an HDF5 corpus record -------------


@dataclass(slots=True)
class FigureRecord:
    """Everything a renderer may read: plain arrays on the recorded box, ``[i,j,k]`` in C order."""

    run_id: str
    scenario: ScenarioSpec
    numerics: NumericsConfig
    status: str
    error_message: str
    gates: list[GateResult]
    measurements: dict[str, Any]
    provenance: dict[str, Any]
    shape: tuple[int, int, int]
    spacing: float
    origin: tuple[float, float, float]
    boundary: str
    density: np.ndarray | None
    v_hartree: np.ndarray | None
    v_xc: np.ndarray | None
    eigenvalues: np.ndarray
    occupations: np.ndarray
    energies: dict[str, float]
    trajectory: dict[str, list[float] | list[str]]
    converged: bool
    n_iterations: int
    source: str

    @property
    def interacting(self) -> bool:
        """Whether the scenario carries Hartree and XC (``xc.name != "none"``, D-53)."""
        return self.scenario.xc.name.lower() != "none"

    @property
    def charges(self) -> np.ndarray:
        """Nuclear charges ``Z_a`` (empty for a model system)."""
        from cdft.structure import nuclear_charges

        if self.scenario.structure.n_atoms == 0:
            return np.zeros(0)
        return np.asarray(nuclear_charges(self.scenario.structure, self.scenario.external), dtype=np.float64)

    @property
    def positions(self) -> np.ndarray:
        """Nuclear positions in bohr, ``(n_atoms, 3)``."""
        if self.scenario.structure.n_atoms == 0:
            return np.zeros((0, 3))
        return np.asarray(self.scenario.structure.positions, dtype=np.float64).reshape(-1, 3)

    @property
    def cusp_factorised(self) -> bool:
        """Whether the density was solved as ``f^2 |phi|^2`` (D-35); the recorded value wins."""
        ext = self.scenario.external
        recorded = self.measurements.get("cusp_factorisation")
        flag = bool(recorded) if recorded is not None else bool(ext.cusp_factorisation)
        return (
            ext.kind is ExternalPotentialKind.NUCLEAR_COULOMB
            and flag
            and self.scenario.structure.n_atoms > 0
        )

    @property
    def n_electrons(self) -> float:
        """The electron count the run used (recorded; else derived from the scenario)."""
        recorded = self.measurements.get("n_electrons")
        if recorded is not None:
            return float(recorded)
        from cdft.structure import electron_count

        scenario = self.scenario
        return float(electron_count(scenario.structure, scenario.external, scenario.electrons.n_electrons))

    @property
    def upper(self) -> np.ndarray:
        """Coordinates of the last grid point along each axis."""
        return np.asarray(self.origin) + self.spacing * (np.asarray(self.shape) - 1)

    @property
    def homo_index(self) -> int | None:
        """Index of the highest state with a non-negligible occupation."""
        occupied = np.nonzero(self.occupations > 1.0e-12)[0]
        return int(occupied.max()) if occupied.size else None

    @property
    def homo(self) -> float | None:
        """The HOMO eigenvalue, in Hartree."""
        index = self.homo_index
        return None if index is None else float(self.eigenvalues[index])

    def functional(self):
        """Return the native functional object of the scenario (``None`` for a bare potential)."""
        if not self.interacting:
            return None
        from cdft.xc.dispatch import functional_by_name

        return functional_by_name(self.scenario.xc.name)

    def provenance_line(self) -> str:
        """One line of provenance for captions and file metadata."""
        prov = self.provenance
        sha = str(prov.get("git_sha") or "no-git")[:12]
        dirty = "+dirty" if prov.get("git_dirty") else ""
        config = str(prov.get("config_hash") or "")[:12]
        line = (
            f"run {self.run_id}; scenario {self.scenario.scenario_id}; status {self.status}; "
            f"git {sha}{dirty}; config {config}; contract {prov.get('contract_version', '')}; "
            f"device {prov.get('device', '')}; figures.py {FIGURES_VERSION}"
        )
        overrides = self.measurements.get("grid_overrides")
        rule = overrides.get("grid_rule") if isinstance(overrides, Mapping) else None
        return line if not rule else f"{line}; exploratory resolution -- {rule}"


def _box_field(values: Any, shape: tuple[int, int, int], what: str) -> np.ndarray:
    """Reshape a flat field (spin summed) onto the box, refusing a masked domain."""
    array = _as_numpy(values)
    if array.ndim == 2:
        array = array.sum(axis=0)
    expected = int(np.prod(shape))
    if array.size != expected:
        raise ValueError(
            f"{what} has {array.size} points but the recorded box has {expected}: a masked domain "
            f"(DomainMode.MASKED_SPHERES) is not supported by the figure layer -- Phase 1 runs on BOX domains"
        )
    return array.reshape(shape)


def _provenance_dict(provenance: Any) -> dict[str, Any]:
    """Return a provenance object (or mapping) as plain JSON-able data."""
    if isinstance(provenance, Mapping):
        return {str(k): v for k, v in provenance.items()}
    out = {}
    for item in dataclasses.fields(provenance):
        value = getattr(provenance, item.name)
        out[item.name] = dict(value) if isinstance(value, Mapping) else value
    return out


def _grid_geometry(
    measurements: Mapping[str, Any], numerics: NumericsConfig
) -> tuple[tuple[int, int, int], float, tuple[float, float, float], str]:
    """Return the recorded grid (``UniformGrid.describe``), as gates.base.artifact_grid reads it."""
    described = measurements.get("grid")
    if not isinstance(described, Mapping) or not {"shape", "spacing_bohr", "origin_bohr"} <= set(described):
        raise ValueError("the record carries no grid description (measurements['grid']); cannot place its fields")
    if float(described.get("mask_fraction", 1.0)) < 1.0 - 1.0e-12:
        raise ValueError("the record's domain is masked; the figure layer supports full BOX domains only")
    shape = tuple(int(n) for n in described["shape"])
    origin = tuple(float(x) for x in described["origin_bohr"])
    boundary = str(described.get("boundary", numerics.grid.boundary.value))
    return shape, float(described["spacing_bohr"]), origin, boundary  # type: ignore[return-value]


#: Placeholder geometry of a record without fields; never used, since such a record is not drawn.
_NO_GEOMETRY: tuple[tuple[int, int, int], float, tuple[float, float, float], str] = (
    (1, 1, 1), 1.0, (0.0, 0.0, 0.0), "zero"
)

#: The energy components copied from :class:`contract.EnergyBreakdown` into a figure record.
_ENERGY_FIELDS = ("total", "kinetic", "external", "hartree", "xc", "nonlocal_ps", "ion_ion", "dispersion", "entropy")


def _as_numpy(value: Any) -> np.ndarray:
    """Return a tensor or array-like as a float64 numpy array on the host."""
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", torch.float64).numpy()
    return np.asarray(value, dtype=np.float64)


def _restricted_levels(eigenvalues: Any, occupations: Any) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues and total occupations of a spin-restricted record; two spin channels are refused."""
    values = _as_numpy(eigenvalues)
    weights = _as_numpy(occupations)
    if values.ndim == 2:
        if values.shape[0] != 1:
            raise ValueError(
                f"the record has {values.shape[0]} spin channels; figures.py draws spin-restricted records only"
            )
        values = values[0]
        weights = weights.sum(axis=0) if weights.ndim == 2 else weights
    return values, weights


def _energies(result: Any) -> dict[str, float]:
    """Return the energy components of a result, with ``harris_foulkes`` as NaN when it was not recorded."""
    energies = {name: float(getattr(result.energies, name)) for name in _ENERGY_FIELDS}
    harris = result.energies.harris_foulkes
    energies["harris_foulkes"] = float("nan") if harris is None else float(harris)
    return energies


def _figure_record(
    run_id: str,
    scenario: ScenarioSpec,
    numerics: NumericsConfig,
    artifact: RunArtifact,
    result: Any,
    measurements: dict[str, Any],
    provenance: dict[str, Any],
    geometry: tuple[tuple[int, int, int], float, tuple[float, float, float], str],
    v_hartree: Any,
    v_xc: Any,
    source: str,
) -> FigureRecord:
    """Assemble the record of a solved result; the two builders differ only in where they read it."""
    shape, spacing, origin, boundary = geometry
    eigenvalues, occupations = _restricted_levels(result.eigenvalues, result.occupations)
    traj = result.trajectory
    return FigureRecord(
        run_id=run_id,
        scenario=scenario,
        numerics=numerics,
        status=artifact.status.value,
        error_message=artifact.error_message,
        gates=list(artifact.gates.results),
        measurements=measurements,
        provenance=provenance,
        shape=shape,
        spacing=spacing,
        origin=origin,
        boundary=boundary,
        density=None if result.density is None else _box_field(result.density, shape, "density"),
        v_hartree=None if v_hartree is None else _box_field(v_hartree, shape, "v_hartree"),
        v_xc=None if v_xc is None else _box_field(v_xc, shape, "v_xc"),
        eigenvalues=eigenvalues,
        occupations=occupations,
        energies=_energies(result),
        trajectory={
            "energies": [float(x) for x in traj.energies],
            "residual_norms": [float(x) for x in traj.residual_norms],
            "density_changes": [float(x) for x in traj.density_changes],
            "fallback_events": list(traj.fallback_events),
            "mixing_events": list(traj.mixing_events),
        },
        converged=bool(result.converged),
        n_iterations=int(result.n_iterations),
        source=source,
    )


def record_from_artifact(artifact: RunArtifact) -> FigureRecord:
    """Build the figure record of a live, gated artifact."""
    measurements = dict(artifact.measurements or {})
    result = artifact.result
    if result is None:
        shape, spacing, origin, boundary = _NO_GEOMETRY
        if "grid" in measurements:
            shape, spacing, origin, boundary = _grid_geometry(measurements, artifact.numerics)
        return FigureRecord(
            artifact.run_id, artifact.scenario, artifact.numerics, artifact.status.value,
            artifact.error_message, list(artifact.gates.results), measurements,
            _provenance_dict(artifact.provenance), shape, spacing, origin, boundary,
            None, None, None, np.zeros(0), np.zeros(0), {}, {}, False, 0, "solve",
        )
    return _figure_record(
        artifact.run_id, artifact.scenario, artifact.numerics, artifact, result, measurements,
        _provenance_dict(artifact.provenance), _grid_geometry(measurements, artifact.numerics),
        result.v_hartree, result.v_xc, "solve",
    )


def _text(value: Any) -> str:
    """Return an HDF5 string attribute or dataset element as ``str``."""
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _scenario_from_group(group) -> ScenarioSpec:
    """Rebuild a record's :class:`~contract.ScenarioSpec`; functional from the registry."""
    structure_group = group["structure"]
    numbers = tuple(int(z) for z in np.asarray(structure_group["numbers"]))
    positions = np.asarray(structure_group["positions"], dtype=np.float64).reshape(-1, 3)
    structure = AtomicStructure(
        numbers=numbers,
        positions=tuple(tuple(float(x) for x in row) for row in positions) if numbers else None,
        charge=float(structure_group.attrs.get("charge", 0.0)),
        multiplicity=int(structure_group.attrs.get("multiplicity", 1)),
        label=_text(structure_group.attrs.get("label", "")),
    )
    meta = group["scenario"]
    xc_name = _text(meta.attrs["xc_name"]).lower()
    if xc_name == "none":
        xc = NONINTERACTING_XC
    elif xc_name == "pbe":
        xc = pbe_functional()
    elif xc_name in ("lda_vwn", "lda_pw92", "lda_pz81"):
        xc = lda_functional(xc_name.removeprefix("lda_"))
    else:
        xc = XCSpec(name=xc_name, rung=XCRung(int(meta.attrs["xc_rung"])), libxc_reference=())
    params = json.loads(_text(meta.attrs["external_params"]))
    external = ExternalPotentialSpec(
        kind=ExternalPotentialKind(_text(meta.attrs["external_kind"])),
        charges=tuple(float(z) for z in params.get("charges", ())),
        softening=float(params.get("softening", 0.0)),
        omega=float(params.get("omega", 1.0)),
        box_length=float(params.get("box_length", 0.0)),
        depth=float(params.get("depth", 0.0)),
        width=float(params.get("width", 1.0)),
    )
    n_electrons = float(meta.attrs["n_electrons"])
    electrons = ElectronSpec(
        n_electrons=None if math.isnan(n_electrons) else n_electrons,
        magnetisation=float(meta.attrs.get("magnetisation", 0.0)),
        spin_polarised=int(meta.attrs.get("n_spin", 1)) == 2,
    )
    reference: dict[str, ReferenceValue] = {}
    if "reference" in group:
        for key, entry in group["reference"].items():
            reference[key] = ReferenceValue(
                value=float(entry.attrs["value"]),
                units=_text(entry.attrs.get("units", "Ha")),
                kind=ReferenceKind(_text(entry.attrs.get("kind", "analytic"))),
                method=_text(entry.attrs.get("method", "")) or "as recorded",
                citation=_text(entry.attrs.get("citation", "")),
                tolerance=float(entry.attrs.get("tolerance", 0.0)),
            )
    return ScenarioSpec(
        scenario_id=_text(meta.attrs["scenario_id"]),
        structure=structure,
        xc=xc,
        external=external,
        electrons=electrons,
        reference=reference,
        notes=_text(meta.attrs.get("notes", "")),
    )


def record_from_corpus(path: str | pathlib.Path, run_id: str | None = None) -> FigureRecord:
    """Build the figure record of one stored HDF5 run, without solving (``run_id`` if not unique)."""
    import h5py

    from cdft.io.hdf5 import CorpusWriter

    path = pathlib.Path(path)
    if not path.is_file():
        # Checked first: CorpusWriter would create the parent directory of a mistyped path.
        raise FileNotFoundError(f"no corpus file at {path}")
    if not h5py.is_hdf5(path):
        raise ValueError(f"{path} is not an HDF5 file")
    reader = CorpusWriter(path)
    ids = reader.run_ids(include_invalid=True)
    if not ids:
        raise ValueError(f"{path} holds no records")
    if run_id is None:
        if len(ids) != 1:
            raise ValueError(f"{path} holds {len(ids)} records; name one with --run-id: {', '.join(ids)}")
        run_id = ids[0]
    if run_id not in ids:
        raise KeyError(f"{run_id!r} is not in {path}; records: {', '.join(ids)}")
    artifact = reader.read(run_id)
    with h5py.File(path, "r") as handle:
        group = handle[run_id]
        scenario = _scenario_from_group(group)
        numerics = numerics_from_dict(json.loads(_text(group.attrs["numerics"])))
        potentials = group.get("potentials")
        stored = {} if potentials is None else {name: np.asarray(potentials[name]) for name in potentials}
    v_hartree = stored.get("v_hartree")
    v_xc = stored.get("v_xc")
    measurements = dict(artifact.measurements or {})
    result = artifact.result
    provenance = _provenance_dict(artifact.provenance)
    if result is None:
        shape, spacing, origin, boundary = (
            _grid_geometry(measurements, numerics) if "grid" in measurements else _NO_GEOMETRY
        )
        return FigureRecord(run_id, scenario, numerics, artifact.status.value, artifact.error_message,
                            list(artifact.gates.results), measurements, provenance, shape, spacing, origin,
                            boundary, None, None, None, np.zeros(0), np.zeros(0), {}, {}, False, 0,
                            f"corpus:{path}")
    return _figure_record(
        run_id, scenario, numerics, artifact, result, measurements, provenance,
        _grid_geometry(measurements, numerics), v_hartree, v_xc, f"corpus:{path}",
    )


# --- Fields at arbitrary points: n = f^2 I[rho], analytic factor, degree-7 interpolant -----------


def _lagrange_coefficients(degree: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Lagrange basis on integer nodes: ``(offsets, C0, C1, C2)`` for ``ell_j(tau)`` and derivatives."""
    half = degree // 2
    offsets = np.arange(-half, -half + degree + 1)
    k = degree + 1
    c0 = np.zeros((k, k))
    for j in range(k):
        roots = [float(x) for m, x in enumerate(offsets) if m != j]
        denominator = float(np.prod([offsets[j] - x for m, x in enumerate(offsets) if m != j]))
        c0[j] = np.polynomial.polynomial.polyfromroots(roots) / denominator
    powers = np.arange(k)
    c1 = np.zeros_like(c0)
    c1[:, :-1] = c0[:, 1:] * powers[1:]
    c2 = np.zeros_like(c0)
    c2[:, :-2] = c0[:, 2:] * (powers[2:] * powers[1:-1])
    return offsets, c0, c1, c2


_OFFSETS, _C0, _C1, _C2 = _lagrange_coefficients(INTERPOLATION_DEGREE)
_PAD = int(-_OFFSETS[0]) + 1  # 4 cells of halo on each side of the box


def factor_terms(points: np.ndarray, charges: np.ndarray, positions: np.ndarray, order: int = 1):
    """``u = sum_a Z_a |r - R_a|`` and derivatives (D-35); ``hess_u`` diverges at a nucleus."""
    m = points.shape[0]
    u = np.zeros(m)
    grad = np.zeros((m, 3))
    hess = np.zeros((m, 3, 3)) if order >= 2 else None
    for charge, centre in zip(charges, positions, strict=True):
        delta = points - centre
        distance = np.sqrt((delta * delta).sum(axis=1))
        safe = np.maximum(distance, 1.0e-300)
        unit = delta / safe[:, None]
        u += charge * distance
        grad += charge * unit
        if hess is not None:
            hess += charge * (np.eye(3)[None, :, :] - unit[:, :, None] * unit[:, None, :]) / safe[:, None, None]
    return u, grad, hess


class FieldEvaluator:
    """Evaluate the recorded density and smooth fields at arbitrary points.

    ``n(r) = f^2(r) I[rho](r)``: ``f^2`` analytic, ``rho = n_grid / f^2_grid`` interpolated by the
    degree-7 tensor-product Lagrange polynomial of the solver's quadrature (D-54), so ``n``
    reproduces the record at a grid point. Derivatives combine it with those of ``u``:

        grad n = f^2 (grad rho - 2 rho grad u)
        H_n    = f^2 [H_rho - 2 (grad rho grad u^T + grad u grad rho^T) - 2 rho H_u + 4 rho grad u grad u^T]

    The halo matches the solver's continuation beyond the box: zeros, or the even image of
    ``n = psi^2`` about the wall. Points beyond the box return ``inside = False`` and zero fields.
    """

    def __init__(self, record: FigureRecord) -> None:
        """Precompute ``rho`` on the padded box."""
        if record.density is None:
            raise ValueError("the record holds no density")
        self.record = record
        self.h = float(record.spacing)
        self.origin = np.asarray(record.origin, dtype=np.float64)
        self.shape = np.asarray(record.shape, dtype=np.int64)
        self.charges = record.charges if record.cusp_factorised else np.zeros(0)
        self.positions = record.positions if record.cusp_factorised else np.zeros((0, 3))
        self.even_halo = record.boundary == BoundaryMode.ODD_REFLECTION.value
        weight = self.grid_weight()
        self.rho = record.density / weight
        self._rho_pad = self._padded(self.rho)
        self._smooth: dict[str, np.ndarray] = {}

    def axes(self) -> list[np.ndarray]:
        """Return the three coordinate axes of the recorded box."""
        return [self.origin[d] + self.h * np.arange(int(self.shape[d])) for d in range(3)]

    def grid_weight(self) -> np.ndarray | float:
        """``f^2`` at the grid points (1 on a path without the factor)."""
        if self.charges.size == 0:
            return 1.0
        axes = self.axes()
        u = np.zeros(tuple(int(n) for n in self.shape))
        for charge, centre in zip(self.charges, self.positions, strict=True):
            dx = (axes[0] - centre[0])[:, None, None]
            dy = (axes[1] - centre[1])[None, :, None]
            dz = (axes[2] - centre[2])[None, None, :]
            u += charge * np.sqrt(dx * dx + dy * dy + dz * dz)
        return np.exp(-2.0 * u)

    def _padded(self, field: np.ndarray) -> np.ndarray:
        """Return the field with a halo of :data:`_PAD` cells: zeros, or the even image about the wall."""
        pad = _PAD
        out = np.pad(field, pad, mode="constant")
        if self.even_halo:
            # ODD_REFLECTION: the wall is one spacing out, so halo pad-1-d mirrors interior d-1.
            for axis in range(3):
                n = int(self.shape[axis])
                for d in range(1, pad):
                    lo_dst = [slice(None)] * 3
                    lo_src = [slice(None)] * 3
                    lo_dst[axis] = pad - 1 - d
                    lo_src[axis] = pad + d - 1
                    out[tuple(lo_dst)] = out[tuple(lo_src)]
                    hi_dst = [slice(None)] * 3
                    hi_src = [slice(None)] * 3
                    hi_dst[axis] = pad + n + d
                    hi_src[axis] = pad + n - d
                    out[tuple(hi_dst)] = out[tuple(hi_src)]
        return out

    def add_smooth_field(self, name: str, field: np.ndarray) -> None:
        """Register a smooth box field (a Hartree remainder, a potential) for :meth:`smooth`."""
        self._smooth[name] = np.pad(np.asarray(field, dtype=np.float64), _PAD, mode="constant")

    def inside(self, points: np.ndarray) -> np.ndarray:
        """Whether each point lies in the recorded box (on or inside the outermost points)."""
        t = (points - self.origin) / self.h
        return np.all((t >= -1.0e-9) & (t <= self.shape - 1 + 1.0e-9), axis=1)

    def nudge(self, points: np.ndarray, tolerance: float = 1.0e-9) -> np.ndarray:
        """Move points closer than ``tolerance`` to a nucleus by ``tolerance`` along +x."""
        out = np.array(points, dtype=np.float64, copy=True)
        for centre in self.record.positions:
            near = np.sqrt(((out - centre) ** 2).sum(axis=1)) < tolerance
            out[near, 0] += tolerance
        return out

    def _stencil(self, points: np.ndarray, order: int, one_sided: bool = False):
        """Indices into the padded box and 1-D weights per axis; ``one_sided`` shifts in at a face."""
        t = (points - self.origin) / self.h
        t = np.clip(t, 0.0, self.shape - 1)
        base = np.floor(t)
        base = np.minimum(base, self.shape - 2).astype(np.int64)  # the last point uses the cell below
        if one_sided:
            first = int(_OFFSETS[0])
            span = _OFFSETS.size
            start = np.clip(base + first, 0, self.shape - span)
            base = start - first
        tau = t - base
        powers = tau[:, :, None] ** np.arange(_C0.shape[0])[None, None, :]  # (m, 3, k)
        w0 = powers @ _C0.T
        w1 = (powers @ _C1.T) / self.h if order >= 1 else None
        w2 = (powers @ _C2.T) / self.h**2 if order >= 2 else None
        index = base[:, :, None] + _OFFSETS[None, None, :] + _PAD  # (m, 3, k)
        return index, w0, w1, w2

    @staticmethod
    def _gather(padded: np.ndarray, index: np.ndarray) -> np.ndarray:
        """Return the (m, k, k, k) block of ``padded`` around each point."""
        ix = index[:, 0, :, None, None]
        iy = index[:, 1, None, :, None]
        iz = index[:, 2, None, None, :]
        return padded[ix, iy, iz]

    def _interpolate(self, padded: np.ndarray, points: np.ndarray, order: int, one_sided: bool = False):
        """Value, gradient ``(m, 3)`` and Hessian ``(m, 3, 3)`` of the interpolant at ``points``."""
        index, w0, w1, w2 = self._stencil(points, order, one_sided)
        block = self._gather(padded, index)
        value = np.einsum("mabc,ma,mb,mc->m", block, w0[:, 0], w0[:, 1], w0[:, 2], optimize=True)
        grad = hess = None
        if order >= 1:
            grad = np.stack(
                [
                    np.einsum("mabc,ma,mb,mc->m", block, w1[:, 0], w0[:, 1], w0[:, 2], optimize=True),
                    np.einsum("mabc,ma,mb,mc->m", block, w0[:, 0], w1[:, 1], w0[:, 2], optimize=True),
                    np.einsum("mabc,ma,mb,mc->m", block, w0[:, 0], w0[:, 1], w1[:, 2], optimize=True),
                ],
                axis=1,
            )
        if order >= 2:
            hess = np.empty((points.shape[0], 3, 3))
            weights = (w0, w1, w2)
            for i in range(3):
                for j in range(i, 3):
                    orders = [0, 0, 0]
                    orders[i] += 1
                    orders[j] += 1
                    value_ij = np.einsum(
                        "mabc,ma,mb,mc->m", block,
                        weights[orders[0]][:, 0], weights[orders[1]][:, 1], weights[orders[2]][:, 2],
                        optimize=True,
                    )
                    hess[:, i, j] = hess[:, j, i] = value_ij
        return value, grad, hess

    def density(self, points: np.ndarray, order: int = 0) -> dict[str, np.ndarray]:
        """``n`` (plus ``grad``, ``hess``, ``lap`` for ``order`` 1/2); zero outside the box."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        m = points.shape[0]
        out: dict[str, np.ndarray] = {"n": np.zeros(m), "inside": self.inside(points)}
        if order >= 1:
            out["grad"] = np.zeros((m, 3))
        if order >= 2:
            out["hess"] = np.zeros((m, 3, 3))
            out["lap"] = np.zeros(m)
        for start in range(0, m, EVALUATION_CHUNK):
            stop = min(m, start + EVALUATION_CHUNK)
            chunk = points[start:stop]
            rho, grad_rho, hess_rho = self._interpolate(self._rho_pad, chunk, order)
            if self.charges.size:
                u, grad_u, hess_u = factor_terms(chunk, self.charges, self.positions, order)
                f2 = np.exp(-2.0 * u)
            else:
                f2 = np.ones(stop - start)
                grad_u = np.zeros((stop - start, 3))
                hess_u = np.zeros((stop - start, 3, 3)) if order >= 2 else None
            keep = out["inside"][start:stop]
            out["n"][start:stop] = np.where(keep, f2 * rho, 0.0)
            if order >= 1:
                grad_n = f2[:, None] * (grad_rho - 2.0 * rho[:, None] * grad_u)
                out["grad"][start:stop] = np.where(keep[:, None], grad_n, 0.0)
            if order >= 2:
                cross = grad_rho[:, :, None] * grad_u[:, None, :]
                hess_n = f2[:, None, None] * (
                    hess_rho
                    - 2.0 * (cross + np.transpose(cross, (0, 2, 1)))
                    - 2.0 * rho[:, None, None] * hess_u
                    + 4.0 * rho[:, None, None] * grad_u[:, :, None] * grad_u[:, None, :]
                )
                hess_n = np.where(keep[:, None, None], hess_n, 0.0)
                out["hess"][start:stop] = hess_n
                out["lap"][start:stop] = np.trace(hess_n, axis1=1, axis2=2)
        return out

    def smooth(self, name: str, points: np.ndarray, order: int = 0) -> tuple[np.ndarray, np.ndarray | None]:
        """Return a registered smooth field (and its gradient for ``order >= 1``) at ``points``."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        padded = self._smooth[name]
        values = np.zeros(points.shape[0])
        grads = np.zeros((points.shape[0], 3)) if order >= 1 else None
        for start in range(0, points.shape[0], EVALUATION_CHUNK):
            stop = min(points.shape[0], start + EVALUATION_CHUNK)
            value, grad, _ = self._interpolate(padded, points[start:stop], min(order, 1), one_sided=True)
            values[start:stop] = value
            if grads is not None:
                grads[start:stop] = grad
        keep = self.inside(points)
        values = np.where(keep, values, 0.0)
        if grads is not None:
            grads = np.where(keep[:, None], grads, 0.0)
        return values, grads


# --- Integration grids: Becke-partitioned atom-centred product rules [F25] ----------------------


def angular_rule(n_theta: int, n_phi: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the unit vectors and weights (sum 4 pi) of a Gauss--Legendre(cos theta) x uniform(phi) rule."""
    x, w = np.polynomial.legendre.leggauss(n_theta)
    phi = (np.arange(n_phi) + 0.5) * (2.0 * math.pi / n_phi)
    sin_theta = np.sqrt(1.0 - x * x)
    directions = np.stack(
        [
            (sin_theta[:, None] * np.cos(phi)[None, :]).ravel(),
            (sin_theta[:, None] * np.sin(phi)[None, :]).ravel(),
            np.repeat(x, n_phi),
        ],
        axis=1,
    )
    weights = np.repeat(w, n_phi) * (2.0 * math.pi / n_phi)
    return directions, weights


def radial_rule(n_radial: int, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Radii and weights for ``int g r^2 dr`` under Becke's map ``r = scale (1+x)/(1-x)`` [F25]."""
    x, w = np.polynomial.legendre.leggauss(n_radial)
    r = scale * (1.0 + x) / (1.0 - x)
    return r, w * 2.0 * scale / (1.0 - x) ** 2 * r * r


def _becke_step(mu: np.ndarray) -> np.ndarray:
    """Becke's cell function ``s(mu) = (1 - p(p(p(mu)))) / 2``, ``p(mu) = 3mu/2 - mu^3/2`` [F25]."""
    for _ in range(3):
        mu = 1.5 * mu - 0.5 * mu**3
    return 0.5 * (1.0 - mu)


@dataclass(slots=True)
class IntegrationGrid:
    """Quadrature for ``int n g d^3r``: Becke-fused atom-centred rules [F25], or the mesh."""

    points: np.ndarray
    weights: np.ndarray
    kind: str
    #: ``(start, (n0, n1, n2))`` per block, C order; atom-centred axes are radius, polar, azimuth.
    blocks: tuple[tuple[int, tuple[int, int, int]], ...] = ()
    #: ``(block axis, sub-point cap)`` along which :func:`smeared_histograms` spreads a node's
    #: weight: radius and polar angle when atom-centred, all three on a mesh.
    smear: tuple[tuple[int, int], ...] = ()

    @classmethod
    def for_record(
        cls, record: FigureRecord, n_radial: int = 80, n_theta: int = 20, n_phi: int = 40
    ) -> IntegrationGrid:
        """Build the grid for a record; the defaults integrate a 1s density to 1e-10."""
        positions = record.positions
        if positions.shape[0] == 0 and record.scenario.external.kind is ExternalPotentialKind.HARMONIC:
            centre = model_centre(record)
            directions, angular_weights = angular_rule(n_theta, n_phi)
            radii, radial_weights = radial_rule(n_radial, 1.0 / math.sqrt(float(record.scenario.external.omega)))
            pts = (centre[None, None, :] + radii[:, None, None] * directions[None, :, :]).reshape(-1, 3)
            wts = (radial_weights[:, None] * angular_weights[None, :]).reshape(-1)
            return cls(pts, wts, "centred", ((0, (radii.size, n_theta, n_phi)),), ((0, 32), (1, 8)))
        if positions.shape[0] == 0:
            axes = [record.origin[d] + record.spacing * np.arange(record.shape[d]) for d in range(3)]
            mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
            shape = tuple(int(n) for n in record.shape)
            return cls(mesh, np.full(mesh.shape[0], record.spacing**3), "box", ((0, shape),),
                       ((0, 4), (1, 4), (2, 4)))
        charges = record.charges
        directions, angular_weights = angular_rule(n_theta, n_phi)
        all_points = []
        all_weights = []
        blocks = []
        start = 0
        for atom, (charge, centre) in enumerate(zip(charges, positions, strict=True)):
            radii, radial_weights = radial_rule(n_radial, 1.0 / max(float(charge), 1.0e-12))
            pts = centre[None, None, :] + radii[:, None, None] * directions[None, :, :]
            wts = radial_weights[:, None] * angular_weights[None, :]
            pts = pts.reshape(-1, 3)
            wts = wts.reshape(-1)
            if positions.shape[0] > 1:
                distances = np.sqrt(((pts[:, None, :] - positions[None, :, :]) ** 2).sum(axis=-1))
                cell = np.ones((pts.shape[0], positions.shape[0]))
                for a in range(positions.shape[0]):
                    for b in range(positions.shape[0]):
                        if a == b:
                            continue
                        r_ab = float(np.linalg.norm(positions[a] - positions[b]))
                        cell[:, a] *= _becke_step((distances[:, a] - distances[:, b]) / r_ab)
                wts = wts * cell[:, atom] / cell.sum(axis=1)
            all_points.append(pts)
            all_weights.append(wts)
            blocks.append((start, (radii.size, n_theta, n_phi)))
            start += pts.shape[0]
        return cls(np.concatenate(all_points), np.concatenate(all_weights), "becke", tuple(blocks),
                   ((0, 32), (1, 8)))

    def half_cell_values(self, values: np.ndarray, valid: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
        """``values`` at both half-cell faces of a node along a block axis (own value at an end)."""
        lower = values.copy()
        upper = values.copy()
        for start, shape in self.blocks:
            block = slice(start, start + int(np.prod(shape)))
            v = np.moveaxis(values[block].reshape(shape), axis, 0)
            ok = np.moveaxis(valid[block].reshape(shape), axis, 0)
            both = ok[1:] & ok[:-1]
            with np.errstate(invalid="ignore", over="ignore"):
                middle = 0.5 * (v[1:] + v[:-1])
            low = v.copy()
            high = v.copy()
            low[1:] = np.where(both, middle, v[1:])
            high[:-1] = np.where(both, middle, v[:-1])
            lower[block] = np.moveaxis(low, 0, axis).reshape(-1)
            upper[block] = np.moveaxis(high, 0, axis).reshape(-1)
        return lower, upper

    def description(self) -> str:
        """How this grid integrates, for a caption."""
        if self.kind == "becke":
            if len(self.blocks) == 1:
                return "an atom-centred radial x angular product grid"
            return (f"{len(self.blocks)} atom-centred radial x angular product grids fused by Becke's "
                    f"fuzzy cells [F25]")
        if self.kind == "centred":
            return "a radial x angular product grid about the centre of the well"
        return "the recorded mesh with h^3 weights"


def _substeps(value: np.ndarray, lower: np.ndarray, upper: np.ndarray, width: float, top: float, cap: int) -> int:
    """Sub-points per cell axis so that no sub-step spans more than half a bin (1 if nothing varies)."""
    shown = np.isfinite(value) & (value <= top)
    if not np.any(shown):
        return 1
    with np.errstate(invalid="ignore"):
        span = np.maximum(np.abs(value - lower), np.abs(upper - value))[shown]
    span = span[np.isfinite(span)]
    if not span.size or float(span.max()) < 0.25 * width:
        return 1
    m = int(math.ceil(4.0 * float(span.max()) / width))
    return min(cap, m + m % 2)


def _cell_offset(lower: np.ndarray, value: np.ndarray, upper: np.ndarray, a: float) -> np.ndarray:
    """Return the piecewise-linear change from the node at index offset ``a`` in ``(-1/2, 1/2)``."""
    with np.errstate(invalid="ignore", over="ignore"):
        return (-2.0 * a) * (lower - value) if a < 0.0 else (2.0 * a) * (upper - value)


def smeared_histograms(
    grid: IntegrationGrid,
    weights: np.ndarray,
    valid: np.ndarray,
    series: Mapping[str, tuple[np.ndarray, np.ndarray]],
    joint: tuple[str, str] | None = None,
) -> dict[str, np.ndarray]:
    """Weighted histograms of fields sampled on a quadrature grid, without the node combs.

    Every node of a radial shell of a spherical density carries the same value, so a plain histogram
    is a comb of spikes. Each node's weight is instead spread over its cell along the grid's smear
    axes: the value runs linearly to its half-cell faces (cross terms dropped) and the cell is
    sampled finely enough that no sub-step spans half a bin; the total weight is unchanged.
    ``series`` maps a name to ``(values, bin_edges)``, ``joint`` names two of them for a 2-D
    histogram. Densities are per unit abscissa; ``"substeps"`` holds the sub-point count per axis.
    """
    import itertools

    faces: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    values_valid: dict[str, np.ndarray] = {}
    counts = [1] * len(grid.smear)
    for key, (values, edges) in series.items():
        v = values[valid]
        values_valid[key] = v
        faces[key] = []
        width = float(np.min(np.diff(edges)))
        top = float(edges[-1])
        for slot, (axis, cap) in enumerate(grid.smear):
            lower, upper = grid.half_cell_values(values, valid, axis)
            faces[key].append((lower[valid], upper[valid]))
            counts[slot] = max(counts[slot], _substeps(v, lower[valid], upper[valid], width, top, cap))
    offsets = [[0.0] if m == 1 else [(p + 0.5) / m - 0.5 for p in range(m)] for m in counts]
    lattice = list(itertools.product(*offsets)) if offsets else [()]
    w = weights[valid] / len(lattice)
    out: dict[str, np.ndarray] = {key: np.zeros(len(edges) - 1) for key, (_, edges) in series.items()}
    if joint is not None:
        ex, ey = series[joint[0]][1], series[joint[1]][1]
        out["joint"] = np.zeros((len(ex) - 1, len(ey) - 1))
    for point in lattice:
        sample = {}
        for key in series:
            value = values_valid[key]
            total = value.copy()
            for (lower, upper), a in zip(faces[key], point, strict=True):
                if a != 0.0:
                    total += _cell_offset(lower, value, upper, a)
            sample[key] = total
            out[key] += np.histogram(total, bins=series[key][1], weights=w)[0]
        if joint is not None:
            out["joint"] += np.histogram2d(sample[joint[0]], sample[joint[1]],
                                           bins=[series[joint[0]][1], series[joint[1]][1]], weights=w)[0]
    for key, (_, edges) in series.items():
        out[key] = out[key] / np.diff(edges)
    if joint is not None:
        area = np.diff(series[joint[0]][1])[:, None] * np.diff(series[joint[1]][1])[None, :]
        out["joint"] = out["joint"] / area
    out["substeps"] = np.array(counts)
    return out


def _half_fraction_below(q: float, start: np.ndarray, stop: np.ndarray) -> np.ndarray:
    """Fraction of a linear run from ``start`` to ``stop`` whose value is ``<= q``."""
    rising = stop > start
    falling = stop < start
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.clip((q - start) / (stop - start), 0.0, 1.0)
    flat = (start <= q).astype(np.float64)
    return np.where(rising, t, np.where(falling, 1.0 - t, flat))


def radial_model_cdf(q: float, value: np.ndarray, lower: np.ndarray, upper: np.ndarray, weights: np.ndarray) -> float:
    """Weight with value ``<= q`` under the half-cell linear model, so quantiles are not quantised."""
    below = 0.5 * (_half_fraction_below(q, lower, value) + _half_fraction_below(q, value, upper))
    return float((weights * below).sum())


def radial_model_quantiles(
    grid: IntegrationGrid, values: np.ndarray, weights: np.ndarray, valid: np.ndarray, quantiles: Sequence[float]
) -> list[float]:
    """Quantiles of ``values`` under ``weights`` from :func:`radial_model_cdf`, by bisection to 1e-10."""
    lower, upper = grid.half_cell_values(values, valid, grid.smear[0][0] if grid.smear else 0)
    v, lo, hi, w = values[valid], lower[valid], upper[valid], weights[valid]
    total = float(w.sum())
    finite = np.isfinite(v) & np.isfinite(lo) & np.isfinite(hi)
    v, lo, hi, w = v[finite], lo[finite], hi[finite], w[finite]
    out = []
    for target in quantiles:
        a = float(min(v.min(), lo.min()))
        b = float(max(v.max(), hi.max()))
        for _ in range(200):
            middle = 0.5 * (a + b)
            if radial_model_cdf(middle, v, lo, hi, w) < target * total:
                a = middle
            else:
                b = middle
            if b - a <= 1.0e-10 * max(1.0, abs(middle)):
                break
        out.append(0.5 * (a + b))
    return out


def radial_model_fraction_above(
    grid: IntegrationGrid, values: np.ndarray, weights: np.ndarray, valid: np.ndarray, threshold: float
) -> float:
    """Fraction of the weight with value ``> threshold`` under the radial piecewise-linear model."""
    lower, upper = grid.half_cell_values(values, valid, grid.smear[0][0] if grid.smear else 0)
    v, lo, hi, w = values[valid], lower[valid], upper[valid], weights[valid]
    total = float(w.sum())
    finite = np.isfinite(v) & np.isfinite(lo) & np.isfinite(hi)
    below = radial_model_cdf(threshold, v[finite], lo[finite], hi[finite], w[finite])
    return max(0.0, 1.0 - below / total)


def spherical_average(
    evaluator: FieldEvaluator, centre: np.ndarray, radii: np.ndarray, n_theta: int = 24, n_phi: int = 48
) -> dict[str, np.ndarray]:
    """``<n>(r)`` and ``d<n>/dr`` about ``centre`` (angular product rule; exact for a spherical n)."""
    directions, weights = angular_rule(n_theta, n_phi)
    points = centre[None, None, :] + radii[:, None, None] * directions[None, :, :]
    fields = evaluator.density(evaluator.nudge(points.reshape(-1, 3)), order=1)
    n = fields["n"].reshape(radii.size, -1)
    radial = (fields["grad"].reshape(radii.size, -1, 3) * directions[None, :, :]).sum(axis=-1)
    inside = fields["inside"].reshape(radii.size, -1).all(axis=1)
    norm = weights.sum()
    return {
        "n": (n * weights[None, :]).sum(axis=1) / norm,
        "dn_dr": (radial * weights[None, :]).sum(axis=1) / norm,
        "inside": inside,
    }


# --- References: exact, literature and oracle values a figure compares against -------------------


@dataclass(slots=True)
class DensityReference:
    """A reference density: ``evaluate -> (n, grad n, lap n)``; ``radial`` for a spherical one."""

    label: str
    source: str
    evaluate: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]
    radial: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None
    centre: np.ndarray | None = None
    #: The reference's own cusp charge, ``-(1/2) d ln n/dr`` at the nucleus (``None``: exactly Z).
    z_eff: float | None = None
    #: A legend-sized name (``label`` goes into captions).
    short: str = "reference"


def _hydrogenic_reference(charge: float, electrons: float, centre: np.ndarray) -> DensityReference:
    """``n = N Z^3/pi exp(-2 Z r)``: the exact density of a bare nucleus with the 1s shell filled."""
    amplitude = electrons * charge**3 / math.pi

    def radial(r):
        n = amplitude * np.exp(-2.0 * charge * r)
        return n, -2.0 * charge * n

    def evaluate(points):
        delta = points - centre
        r = np.maximum(np.sqrt((delta * delta).sum(axis=1)), 1.0e-300)
        n, dn = radial(r)
        return n, dn[:, None] * delta / r[:, None], (4.0 * charge**2 - 4.0 * charge / r) * n

    return DensityReference(f"exact 1s, Z = {charge:g}", "closed form (hydrogenic)", evaluate, radial, centre,
                            short="exact")


def _harmonic_reference(omega: float, electrons: float) -> DensityReference:
    """``n = N (omega/pi)^(3/2) exp(-omega r^2)``: the well's ground state (G1.2), at the origin."""
    amplitude = electrons * (omega / math.pi) ** 1.5
    centre = np.zeros(3)

    def radial(r):
        n = amplitude * np.exp(-omega * r * r)
        return n, -2.0 * omega * r * n

    def evaluate(points):
        r2 = (points * points).sum(axis=1)
        n = amplitude * np.exp(-omega * r2)
        return n, -2.0 * omega * points * n[:, None], (4.0 * omega**2 * r2 - 6.0 * omega) * n

    return DensityReference(f"exact ground state, omega = {omega:g}", "closed form (harmonic well)",
                            evaluate, radial, centre, short="exact")


def _box_reference(length: float, electrons: float, centre: np.ndarray) -> DensityReference:
    """``n = N prod_i (2/L) cos^2(pi (x_i - c_i)/L)``: the cubic well's ground state (G1.3)."""
    k = math.pi / length

    def evaluate(points):
        points = points - centre
        g = (2.0 / length) * np.cos(k * points) ** 2
        dg = -(2.0 / length) * k * np.sin(2.0 * k * points)
        d2g = -(2.0 / length) * 2.0 * k * k * np.cos(2.0 * k * points)
        n = electrons * g.prod(axis=1)
        grad = electrons * np.stack(
            [dg[:, 0] * g[:, 1] * g[:, 2], g[:, 0] * dg[:, 1] * g[:, 2], g[:, 0] * g[:, 1] * dg[:, 2]], axis=1
        )
        lap = electrons * (
            d2g[:, 0] * g[:, 1] * g[:, 2] + g[:, 0] * d2g[:, 1] * g[:, 2] + g[:, 0] * g[:, 1] * d2g[:, 2]
        )
        return n, grad, lap

    return DensityReference(f"exact ground state, L = {length:g}", "closed form (particle in a box)",
                            evaluate, None, np.asarray(centre, dtype=np.float64), short="exact")


_RADIAL_KS_CACHE: dict[tuple[float, float, str], Any] = {}


def radial_ks_solution(charge: float, electrons: float, functional_name: str):
    """The radial Kohn--Sham oracle, cached per process; ``None`` when it did not converge."""
    key = (float(charge), float(electrons), functional_name)
    if key not in _RADIAL_KS_CACHE:
        from cdft.reference.radial_ks import solve_radial_ks
        from cdft.xc.dispatch import functional_by_name

        solution = solve_radial_ks(charge, electrons, functional_by_name(functional_name))
        _RADIAL_KS_CACHE[key] = solution if bool(solution.converged) else None
    return _RADIAL_KS_CACHE[key]


#: The radial oracle's density is trusted from this radius out (O-25): its Dirichlet node at 1e-12
#: bohr admixes the irregular solution at ~r_min/r (4e-6 at 1e-6 bohr, < 1e-8 from 1e-4), and a
#: spline in ``ln r`` magnifies the layer enormously.
RADIAL_ORACLE_TRUST_BOHR = 1.0e-4

#: Inside this radius ``ln n`` is the cubic fit in ``r`` over ``[TRUST, FIT]``, outside it the
#: spline in ``ln r``; the two agree to 3e-7 in ``d ln n/dr`` at the join.
RADIAL_ORACLE_JOIN_BOHR = 3.0e-3
RADIAL_ORACLE_FIT_BOHR = 2.0e-2


def _radial_ks_reference(charge: float, electrons: float, functional_name: str, centre: np.ndarray) -> DensityReference:
    """The oracle's density as a reference, with ``n``, ``n'``, ``n''`` at any radius.

    Spline of ``ln n`` in ``ln r`` outside :data:`RADIAL_ORACLE_JOIN_BOHR`, Kato form inside;
    ``z_eff`` is the functional's cusp charge, which a GGA shifts off ``Z`` (D-55).
    """
    from scipy.interpolate import CubicSpline

    solution = radial_ks_solution(charge, electrons, functional_name)
    if solution is None:
        raise ValueError(f"the radial KS oracle did not converge for Z = {charge:g}, N = {electrons:g} "
                         f"at {functional_name}")
    radii = np.asarray(solution.radii, dtype=np.float64)
    density = np.asarray(solution.density, dtype=np.float64)
    keep = (density > 1.0e-280) & (radii >= RADIAL_ORACLE_TRUST_BOHR)
    spline = CubicSpline(np.log(radii[keep]), np.log(density[keep]))
    first = spline.derivative(1)
    second = spline.derivative(2)
    r_max = radii[keep][-1]
    window = keep & (radii <= RADIAL_ORACLE_FIT_BOHR)
    poly = np.polynomial.Polynomial.fit(radii[window], np.log(density[window]), 3).convert()
    d_poly = poly.deriv(1)
    d2_poly = poly.deriv(2)
    join = RADIAL_ORACLE_JOIN_BOHR

    def radial_parts(r):
        r = np.asarray(r, dtype=np.float64)
        rc = np.clip(r, join, r_max)
        xr = np.log(rc)
        g1 = first(xr)  # d ln n / d ln r
        g2 = second(xr)
        inner = r < join
        rp = np.minimum(r, join)
        log_n = np.where(inner, poly(rp), spline(xr))
        d_log = np.where(inner, d_poly(rp), g1 / rc)
        d2_log = np.where(inner, d2_poly(rp), (g2 - g1) / (rc * rc))
        n = np.exp(log_n)
        dn = n * d_log
        d2n = n * (d2_log + d_log * d_log)
        beyond = r > r_max
        n = np.where(beyond, 0.0, n)
        dn = np.where(beyond, 0.0, dn)
        d2n = np.where(beyond, 0.0, d2n)
        return n, dn, d2n

    def radial(r):
        n, dn, _ = radial_parts(r)
        return n, dn

    def evaluate(points):
        delta = points - centre
        r = np.maximum(np.sqrt((delta * delta).sum(axis=1)), 1.0e-300)
        n, dn, d2n = radial_parts(r)
        return n, dn[:, None] * delta / r[:, None], d2n + 2.0 * dn / r

    label = f"radial KS oracle ({functional_name}, Z = {charge:g}, N = {electrons:g})"
    return DensityReference(label, "cdft.reference.radial_ks (independent 1-D solver)", evaluate, radial, centre,
                            z_eff=-0.5 * float(d_poly(0.0)), short="radial KS oracle")


def density_reference(record: FigureRecord, use_oracle: bool = True) -> DensityReference | None:
    """Best reference density for a record: a closed form or the radial KS oracle, else ``None``."""
    scenario = record.scenario
    kind = scenario.external.kind
    electrons = record.n_electrons
    if scenario.electrons.spin_polarised or electrons > 2.0 + 1.0e-12:
        return None
    if kind in (ExternalPotentialKind.HARMONIC, ExternalPotentialKind.PARTICLE_IN_BOX) and record.interacting:
        return None  # the closed forms are the bare wells' ground states
    if kind is ExternalPotentialKind.HARMONIC:
        return _harmonic_reference(scenario.external.omega, electrons)
    if kind is ExternalPotentialKind.PARTICLE_IN_BOX:
        return _box_reference(scenario.external.box_length, electrons, model_centre(record))
    if kind is not ExternalPotentialKind.NUCLEAR_COULOMB or scenario.structure.n_atoms != 1:
        return None
    charge = float(record.charges[0])
    centre = record.positions[0]
    if not record.interacting:
        return _hydrogenic_reference(charge, electrons, centre)
    if not use_oracle:
        return None
    functional = record.functional()
    if functional is None or functional.rung.value > XCRung.GGA.value:
        return None
    if radial_ks_solution(charge, electrons, record.scenario.xc.name.lower()) is None:
        return None
    return _radial_ks_reference(charge, electrons, record.scenario.xc.name.lower(), centre)


def model_centre(record: FigureRecord) -> np.ndarray:
    """Centre of an atom-free record: the origin for the harmonic well, else the box centre."""
    if record.scenario.external.kind is ExternalPotentialKind.HARMONIC:
        return np.zeros(3)
    return 0.5 * (np.asarray(record.origin, dtype=np.float64) + record.upper)


@dataclass(frozen=True, slots=True)
class Level:
    """One reference eigenvalue (or ionisation-potential line) for the spectrum figure."""

    energy: float
    degeneracy: int
    label: str
    source: str


def _nist_system(record: FigureRecord) -> str | None:
    """NIST SRD 141 system name for a spin-restricted VWN-LDA atom (mirrors gates.tier4.reference_key_for)."""
    scenario = record.scenario
    if scenario.structure.n_atoms != 1 or scenario.xc.name.lower() != "lda_vwn" or scenario.electrons.spin_polarised:
        return None
    charge = int(scenario.structure.numbers[0])
    return {(1, 1.0): "H", (2, 2.0): "He", (2, 1.0): "He+"}.get((charge, record.n_electrons))


_TWO_CENTRE_CACHE: dict[float, Any] = {}


def two_centre_energy(bond_length: float):
    """H2+ ground state at ``bond_length`` from the prolate-spheroidal oracle (D-41), cached."""
    key = round(float(bond_length), 12)
    if key not in _TWO_CENTRE_CACHE:
        from cdft.reference.two_centre import h2_plus_energy

        _TWO_CENTRE_CACHE[key] = h2_plus_energy(key)
    return _TWO_CENTRE_CACHE[key]


def _is_homonuclear_hydrogen_pair(record: FigureRecord) -> bool:
    """Two nuclei of charge 1 on the Coulomb path: the systems the two-centre oracle solves."""
    return (
        record.scenario.external.kind is ExternalPotentialKind.NUCLEAR_COULOMB
        and record.scenario.structure.n_atoms == 2
        and np.allclose(record.charges, 1.0)
    )


def _bond_length(record: FigureRecord) -> float:
    """Internuclear distance of a two-nucleus record."""
    return float(np.linalg.norm(record.positions[1] - record.positions[0]))


def spectrum_references(record: FigureRecord, use_oracle: bool = True) -> list[tuple[str, list[Level]]]:
    """Reference level sets for the spectrum figure: exact spectra, NIST SRD 141 [B10], oracle, PySCF."""
    scenario = record.scenario
    kind = scenario.external.kind
    n_levels = max(int(record.eigenvalues.size), 1)
    columns: list[tuple[str, list[Level]]] = []
    if not record.interacting:
        levels: list[Level] = []
        if kind is ExternalPotentialKind.HARMONIC:
            omega = scenario.external.omega
            shell = 0
            while sum(lv.degeneracy for lv in levels) < n_levels:
                levels.append(Level((shell + 1.5) * omega, (shell + 1) * (shell + 2) // 2, f"n = {shell}", "analytic"))
                shell += 1
        elif kind is ExternalPotentialKind.PARTICLE_IN_BOX:
            base = math.pi**2 / (2.0 * scenario.external.box_length**2)
            counts: dict[int, int] = {}
            limit = 6
            for nx in range(1, limit + 1):
                for ny in range(1, limit + 1):
                    for nz in range(1, limit + 1):
                        q = nx * nx + ny * ny + nz * nz
                        counts[q] = counts.get(q, 0) + 1
            for q in sorted(counts):
                if sum(lv.degeneracy for lv in levels) >= n_levels:
                    break
                levels.append(Level(q * base, counts[q], f"|n|^2 = {q}", "analytic"))
        elif kind is ExternalPotentialKind.NUCLEAR_COULOMB and scenario.structure.n_atoms == 1:
            charge = float(record.charges[0])
            shell = 1
            while sum(lv.degeneracy for lv in levels) < n_levels:
                levels.append(Level(-0.5 * charge**2 / shell**2, shell * shell, f"n = {shell}", "analytic"))
                shell += 1
        elif _is_homonuclear_hydrogen_pair(record) and use_oracle:
            result = two_centre_energy(_bond_length(record))
            levels.append(Level(result.electronic_energy, 1, "1s sigma_g", "two-centre oracle (D-41)"))
        if levels:
            columns.append(("exact", levels))
        return columns
    system = _nist_system(record)
    if system is not None:
        from cdft.reference.literature import NIST_SRD141

        entry = NIST_SRD141[f"{system}/lda/eigenvalue_1s"]
        columns.append(("NIST SRD 141", [Level(entry.value, 1, "1s", "NIST SRD 141 [B10]")]))
    if use_oracle and scenario.structure.n_atoms == 1 and kind is ExternalPotentialKind.NUCLEAR_COULOMB:
        functional = record.functional()
        if functional is not None and functional.rung.value <= XCRung.GGA.value and record.n_electrons <= 2.0 + 1.0e-12:
            solution = radial_ks_solution(float(record.charges[0]), record.n_electrons, scenario.xc.name.lower())
            if solution is not None:
                columns.append(("radial KS", [Level(float(solution.eigenvalues[0]), 1, "1s", "radial KS oracle")]))
    from cdft.reference.computed import lookup_computed

    computed = lookup_computed(scenario.scenario_id)
    if computed is not None and computed.get("homo") is not None:
        columns.append(("PySCF cc-pV5Z", [Level(float(computed["homo"]), 1, "HOMO", "cdft.reference.computed")]))
    return columns


def exact_ionisation_level(record: FigureRecord, use_oracle: bool = True) -> Level | None:
    """``-I`` of the exact functional when known: its HOMO eigenvalue [A30], fractional N [A21]."""
    if not record.interacting:
        return None
    scenario = record.scenario
    electrons = record.n_electrons
    if scenario.external.kind is not ExternalPotentialKind.NUCLEAR_COULOMB:
        return None
    if scenario.structure.n_atoms == 1:
        charge = float(record.charges[0])
        if 0.0 < electrons <= 1.0 + 1.0e-12:
            return Level(-0.5 * charge**2, 1, "-I (exact)", f"E({charge:g}+ bare) - E(1 e), closed form")
        if 1.0 < electrons <= 2.0 + 1.0e-12 and abs(charge - 2.0) < 1.0e-12:
            from cdft.reference.literature import EXACT_NONRELATIVISTIC

            e_he = EXACT_NONRELATIVISTIC["He/total_energy"].value
            e_he_plus = EXACT_NONRELATIVISTIC["He+/total_energy"].value
            return Level(e_he - e_he_plus, 1, "-I (exact)", "E(He) - E(He+), Nakashima & Nakatsuji (2007)")
        return None
    if _is_homonuclear_hydrogen_pair(record) and 0.0 < electrons <= 1.0 + 1.0e-12 and use_oracle:
        result = two_centre_energy(_bond_length(record))
        return Level(result.electronic_energy, 1, "-I (exact)", "two-centre oracle (D-41), vertical")
    return None


def exact_total_energy(record: FigureRecord, use_oracle: bool = True) -> tuple[float, str] | None:
    """Exact total energy of the system (clamped nuclei) when available; fractional N by [A21]."""
    scenario = record.scenario
    electrons = record.n_electrons
    ion_ion = float(record.energies.get("ion_ion", 0.0))
    if scenario.external.kind is not ExternalPotentialKind.NUCLEAR_COULOMB:
        return None
    if scenario.structure.n_atoms == 1:
        charge = float(record.charges[0])
        if not record.interacting and electrons <= 2.0 + 1.0e-12:
            return -0.5 * charge**2 * electrons, "N (-Z^2/2), bare nucleus"
        if abs(electrons - 1.0) < 1.0e-12:
            return -0.5 * charge**2, "-Z^2/2 (one electron)"
        if 0.0 < electrons < 1.0:
            return -0.5 * charge**2 * electrons, "N (-Z^2/2): piecewise linearity [A21] between N = 0 and 1"
        if abs(charge - 2.0) < 1.0e-12 and 1.0 < electrons <= 2.0 + 1.0e-12:
            from cdft.reference.literature import EXACT_NONRELATIVISTIC

            e_he = EXACT_NONRELATIVISTIC["He/total_energy"].value
            if abs(electrons - 2.0) < 1.0e-12:
                return e_he, "exact He, Nakashima & Nakatsuji (2007)"
            e_he_plus = EXACT_NONRELATIVISTIC["He+/total_energy"].value
            return (e_he_plus + (electrons - 1.0) * (e_he - e_he_plus),
                    "E(He+) + (N - 1)(E(He) - E(He+)): piecewise linearity [A21]")
        return None
    if _is_homonuclear_hydrogen_pair(record) and use_oracle:
        result = two_centre_energy(_bond_length(record))
        if not record.interacting and electrons <= 2.0 + 1.0e-12:
            return electrons * result.electronic_energy + ion_ion, "N E_el(R) + 1/R, two-centre oracle"
        if abs(electrons - 1.0) < 1.0e-12:
            return result.total_energy, "E_el(R) + 1/R, two-centre oracle (D-41)"
    return None


def energy_references(record: FigureRecord, use_oracle: bool = True) -> list[tuple[str, dict[str, float], str]]:
    """Reference energy decompositions ``(column, {term: value}, source)`` in the record's terms."""
    columns: list[tuple[str, dict[str, float], str]] = []
    system = _nist_system(record)
    if system is not None:
        from cdft.reference.literature import NIST_SRD141

        names = {"kinetic": "kinetic_energy", "external": "external_energy", "hartree": "hartree_energy",
                 "xc": "xc_energy", "total": "total_energy"}
        values = {term: NIST_SRD141[f"{system}/lda/{key}"].value for term, key in names.items()
                  if f"{system}/lda/{key}" in NIST_SRD141}
        columns.append(("NIST SRD 141", values, "NIST SRD 141 [B10], VWN LDA, radial"))
    scenario = record.scenario
    if (use_oracle and record.interacting and scenario.structure.n_atoms == 1
            and scenario.external.kind is ExternalPotentialKind.NUCLEAR_COULOMB
            and record.n_electrons <= 2.0 + 1.0e-12):
        functional = record.functional()
        if functional is not None and functional.rung.value <= XCRung.GGA.value:
            solution = radial_ks_solution(float(record.charges[0]), record.n_electrons, scenario.xc.name.lower())
            if solution is not None:
                columns.append(("radial KS", {"kinetic": solution.kinetic, "external": solution.external,
                                              "hartree": solution.hartree, "xc": solution.xc,
                                              "total": solution.total},
                                "cdft.reference.radial_ks"))
    from cdft.reference.computed import lookup_computed

    computed = lookup_computed(scenario.scenario_id)
    if computed is not None:
        columns.append(("PySCF", {"total": float(computed["value"])},
                        f"cdft.reference.computed (+-{float(computed['uncertainty']):.1e} Ha basis-set spread)"))
    exact = exact_total_energy(record, use_oracle)
    if exact is not None:
        columns.append(("exact", {"total": exact[0]}, exact[1]))
    return columns


# --- Physics at points: potentials, exchange-correlation, the self-interaction integrand ---------


def external_at(record: FigureRecord, points: np.ndarray) -> np.ndarray:
    """``v_ext`` at points as the solver defines it, with the bare Coulomb pole left unaveraged."""
    spec = record.scenario.external
    kind = spec.kind
    r_sq = (points * points).sum(axis=1)
    if kind in (ExternalPotentialKind.NONE, ExternalPotentialKind.PARTICLE_IN_BOX):
        return np.zeros(points.shape[0])
    if kind is ExternalPotentialKind.HARMONIC:
        return 0.5 * spec.omega**2 * r_sq
    if kind is ExternalPotentialKind.GAUSSIAN_WELL:
        return -spec.depth * np.exp(-r_sq / (2.0 * spec.width**2))
    if kind not in (ExternalPotentialKind.NUCLEAR_COULOMB, ExternalPotentialKind.SOFT_COULOMB):
        raise NotImplementedError(f"no pointwise v_ext for {kind.value}")
    v = np.zeros(points.shape[0])
    for charge, centre in zip(record.charges, record.positions, strict=True):
        d_sq = ((points - centre) ** 2).sum(axis=1)
        if kind is ExternalPotentialKind.SOFT_COULOMB:
            v -= charge / np.sqrt(d_sq + spec.softening**2)
        else:
            v -= charge / np.maximum(np.sqrt(d_sq), 1.0e-12)
    return v


def _hydrogenic_potential(charge: float, distance: np.ndarray) -> np.ndarray:
    """:func:`cdft.operators.quadrature.hydrogenic_1s_potential` on numpy input (the D-54 split)."""
    from cdft.operators.quadrature import hydrogenic_1s_potential

    return hydrogenic_1s_potential(charge, torch.as_tensor(distance, dtype=torch.float64)).numpy()


def prepare_hartree(evaluator: FieldEvaluator) -> bool:
    """Register the smooth remainder of ``v_H`` for point evaluation; ``False`` when impossible.

    ``v_H = sum_a c_a v_1s(|r - R_a|; Z_a) + v_smooth`` (D-54), the remainder ``C^3`` at a nucleus.
    """
    record = evaluator.record
    if record.v_hartree is None:
        return False
    if "hartree" in evaluator._smooth:
        return True
    coefficients = record.measurements.get("hartree_split_coefficients")
    remainder = np.array(record.v_hartree, dtype=np.float64, copy=True)
    if coefficients and record.cusp_factorised:
        if len(coefficients) != record.charges.size:
            raise ValueError(
                f"the record holds {len(coefficients)} Hartree split coefficients for "
                f"{record.charges.size} nuclei (D-54 stores one per nucleus)"
            )
        axes = evaluator.axes()
        for c, charge, centre in zip(coefficients, record.charges, record.positions, strict=True):
            dx = (axes[0] - centre[0])[:, None, None]
            dy = (axes[1] - centre[1])[None, :, None]
            dz = (axes[2] - centre[2])[None, None, :]
            distance = np.sqrt(dx * dx + dy * dy + dz * dz)
            remainder -= float(c) * _hydrogenic_potential(float(charge), distance.ravel()).reshape(distance.shape)
        evaluator._hartree_split = [float(c) for c in coefficients]
    else:
        evaluator._hartree_split = []
    evaluator.add_smooth_field("hartree", remainder)
    return True


def hartree_at(evaluator: FieldEvaluator, points: np.ndarray) -> np.ndarray:
    """``v_H`` at points from the analytic cusp part plus the interpolated remainder."""
    record = evaluator.record
    values, _ = evaluator.smooth("hartree", points)
    # Not strict: before prepare_hartree() there is no split and the loop adds nothing.
    for c, charge, centre in zip(getattr(evaluator, "_hartree_split", []), record.charges, record.positions,
                                 strict=False):
        distance = np.sqrt(((points - centre) ** 2).sum(axis=1))
        values = values + c * _hydrogenic_potential(float(charge), distance)
    return np.where(evaluator.inside(points), values, np.nan)


def xc_at(record: FigureRecord, n: np.ndarray, grad: np.ndarray | None, hess: np.ndarray | None,
          with_potential: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
    """``e_xc`` (per volume) and the local ``v_xc`` at points from ``n``, ``grad n`` and ``H_n``.

    LDA: ``v_xc = de/dn``. GGA: ``v_xc = de/dn - div(2 e_sigma grad n)`` expanded exactly as
    ``2 [e_sigma lap n + (e_sigma,n grad n + e_sigma,sigma grad sigma) . grad n]`` with
    ``grad sigma = 2 H_n grad n``, the second derivatives from autograd. This is the potential the
    weak form integrates (D-55), not the record's finite-difference field, which is wrong near a
    nucleus.
    """
    functional = record.functional()
    if functional is None:
        raise ValueError("a bare-potential record has no exchange-correlation functional")
    needs_sigma = functional.rung.value >= XCRung.GGA.value
    if functional.rung.value > XCRung.GGA.value:
        raise NotImplementedError("meta-GGA ingredients (tau) are not stored in the record")
    density = torch.as_tensor(np.maximum(n, 0.0), dtype=torch.float64)[None, :]
    if not needs_sigma:
        out = functional.evaluate(density)
        return out.e_xc.numpy(), (out.v_xc[0].numpy() if with_potential else None)
    grad_t = torch.as_tensor(grad, dtype=torch.float64)
    sigma = (grad_t * grad_t).sum(dim=1)[None, :]
    dens = density.clone().requires_grad_(True)
    sig = sigma.clone().requires_grad_(True)
    with torch.enable_grad():
        energy = functional.energy(dens, sig)
        e_n, e_s = torch.autograd.grad(energy.sum(), (dens, sig), create_graph=with_potential)
        if not with_potential:
            return energy.detach().numpy(), None
        e_sn, e_ss = torch.autograd.grad(e_s.sum(), (dens, sig), allow_unused=True)
    e_sn = torch.zeros_like(dens) if e_sn is None else e_sn
    e_ss = torch.zeros_like(sig) if e_ss is None else e_ss
    hess_t = torch.as_tensor(hess, dtype=torch.float64)
    lap = torch.diagonal(hess_t, dim1=1, dim2=2).sum(dim=1)
    grad_sigma = 2.0 * torch.einsum("mij,mj->mi", hess_t, grad_t)
    grad_es = e_sn[0][:, None] * grad_t + e_ss[0][:, None] * grad_sigma
    divergence = 2.0 * (e_s[0].detach() * lap + (grad_es * grad_t).sum(dim=1))
    v_xc = e_n[0].detach() - divergence
    return energy.detach().numpy(), v_xc.numpy()


# --- Figure context: the record, its evaluator, cached samples and the frame of the plots --------


@dataclass(slots=True)
class FigureOptions:
    """Rendering options that are not physics."""

    style: HouseStyle = field(default_factory=HouseStyle)
    use_oracle: bool = True
    map_points: int = 361
    profile_points: int = 1601
    radial_points: int = 600
    grid_radial: int = 80
    grid_theta: int = 20
    grid_phi: int = 40
    #: Distribution grid: radially dense (it sets the resolution of g(s), g(r_s)), angularly light.
    dist_radial: int = 400
    dist_theta: int = 16
    dist_phi: int = 32


@dataclass(slots=True)
class RenderedFigure:
    """A drawn figure with its caption, ``columns``/``tables`` CSV views, ``arrays`` and metrics."""

    figure: Any
    caption: str
    columns: dict[str, np.ndarray] = field(default_factory=dict)
    tables: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)


class FigureContext:
    """What every renderer receives: the record, the verdict, the evaluator and shared samples."""

    def __init__(self, record: FigureRecord, verdict: Verdict, options: FigureOptions) -> None:
        """Bind the record; heavy objects are built on first use and cached."""
        self.record = record
        self.verdict = verdict
        self.options = options
        self.style = options.style
        self.evaluator = FieldEvaluator(record) if record.density is not None else None
        self._cache: dict[str, Any] = {}

    def cached(self, key: str, factory: Callable[[], Any]) -> Any:
        """Build ``factory()`` once per context."""
        if key not in self._cache:
            self._cache[key] = factory()
        return self._cache[key]

    @property
    def reference(self) -> DensityReference | None:
        """The reference density (cached; the radial oracle costs tens of seconds on two cores)."""
        return self.cached("reference", lambda: density_reference(self.record, self.options.use_oracle))

    @property
    def grid(self) -> IntegrationGrid:
        """The integration grid of the record (cached)."""
        opts = self.options
        return self.cached(
            "grid", lambda: IntegrationGrid.for_record(self.record, opts.grid_radial, opts.grid_theta, opts.grid_phi)
        )

    def grid_fields(self, order: int = 1) -> dict[str, np.ndarray]:
        """Return the density (and gradient) at the integration grid points (cached per order)."""
        evaluator = self.evaluator
        return self.cached(f"grid_fields_{order}",
                           lambda: evaluator.density(evaluator.nudge(self.grid.points), order=order))

    @property
    def distribution_grid(self) -> IntegrationGrid:
        """The radially dense grid for density-weighted distributions (cached)."""
        opts = self.options
        return self.cached(
            "distribution_grid",
            lambda: IntegrationGrid.for_record(self.record, opts.dist_radial, opts.dist_theta, opts.dist_phi),
        )

    def distribution_fields(self) -> dict[str, np.ndarray]:
        """Density and gradient on :attr:`distribution_grid` (cached)."""
        return self.cached(
            "distribution_fields",
            lambda: self.evaluator.density(self.evaluator.nudge(self.distribution_grid.points), order=1),
        )

    @property
    def n_max(self) -> float:
        """The largest recorded density value (at a nucleus on every Phase 1 grid, D-64)."""
        return float(np.max(self.record.density))

    def frame(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(centre, u, v)``: plot origin, principal (bond) axis and an in-plane perpendicular."""
        def compute():
            record = self.record
            positions = record.positions
            if positions.shape[0] == 0:
                return model_centre(record), np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])
            if positions.shape[0] == 1:
                return positions[0].copy(), np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])
            charges = record.charges
            centre = (charges[:, None] * positions).sum(axis=0) / charges.sum()
            second = None
            if positions.shape[0] == 2:
                u = positions[1] - positions[0]
            else:
                values, vectors = np.linalg.eigh(np.cov((positions - centre).T))
                u = vectors[:, -1]
                if values[-2] > 1.0e-10 * max(values[-1], 1.0e-300):
                    second = vectors[:, -2]  # the nuclei span a plane: draw that plane
            u = u / np.linalg.norm(u)
            if u[np.argmax(np.abs(u))] < 0:
                u = -u
            if second is None:
                candidates = np.eye(3)
                second = candidates[np.argmin(np.abs(candidates @ u))]
            v = second - (second @ u) * u
            v = v / np.linalg.norm(v)
            if v[np.argmax(np.abs(v))] < 0:
                v = -v
            return centre, u, v

        return self.cached("frame", compute)

    def reach(self, centre: np.ndarray, direction: np.ndarray) -> float:
        """Largest ``t`` with ``centre + t direction`` inside the recorded box."""
        low = np.asarray(self.record.origin, dtype=np.float64)
        high = self.record.upper
        limits = []
        for d in range(3):
            if abs(direction[d]) < 1.0e-12:
                continue
            bound = high[d] if direction[d] > 0 else low[d]
            limits.append((bound - centre[d]) / direction[d])
        return float(max(0.0, min(limits)))

    def extent(self, direction: np.ndarray) -> float:
        """Half-width along ``direction`` (both signs) that holds :data:`DENSITY_DECADES` of density."""
        centre, _, _ = self.frame()

        def compute():
            spans = []
            for sign in (1.0, -1.0):
                d = sign * direction
                reach = self.reach(centre, d)
                t = np.linspace(0.0, reach, 400)
                n = self.evaluator.density(self.evaluator.nudge(centre + t[:, None] * d))["n"]
                above = np.nonzero(n >= self.n_max * 10.0 ** (-DENSITY_DECADES))[0]
                found = t[above.max()] if above.size else reach
                spans.append(min(reach, 1.1 * found + 0.5))
            return float(min(spans))

        return self.cached(f"extent_{tuple(np.round(direction, 9))}", compute)

    def plane(self, points_per_axis: int | None = None):
        """Return the map plane: ``(a, b, points)`` with ``a`` along ``v``, ``b`` along ``u`` (cached)."""
        count = points_per_axis or self.options.map_points
        count = count + (1 - count % 2)  # odd, so the centre line is sampled

        def compute():
            centre, u, v = self.frame()
            half_b = self.extent(u)
            half_a = self.extent(v)
            a = np.linspace(-half_a, half_a, count)
            b = np.linspace(-half_b, half_b, count)
            points = centre[None, None, :] + a[:, None, None] * v[None, None, :] + b[None, :, None] * u[None, None, :]
            return a, b, self.evaluator.nudge(points.reshape(-1, 3))

        return self.cached(f"plane_{count}", compute)

    def nuclei_in_frame(self) -> list[tuple[float, float, str]]:
        """Nuclei as ``(b, a, symbol)`` in the map frame."""
        from cdft.constants import ELEMENT_SYMBOLS

        centre, u, v = self.frame()
        out = []
        for number, position in zip(self.record.scenario.structure.numbers, self.record.positions, strict=True):
            symbol = ELEMENT_SYMBOLS[number] if 0 < number < len(ELEMENT_SYMBOLS) else f"Z{number}"
            out.append((float((position - centre) @ u), float((position - centre) @ v), symbol))
        return out

    def system_label(self) -> str:
        """``H2+ (R = 2 bohr), LDA-VWN, N = 1`` -- the system as a caption names it."""
        record = self.record
        scenario = record.scenario
        functional = {"none": "bare potential (no Hartree, no XC)", "lda_vwn": "LDA (VWN)",
                      "lda_pw92": "LDA (PW92)", "lda_pz81": "LDA (PZ81)", "pbe": "PBE"}.get(
            scenario.xc.name.lower(), scenario.xc.name)
        kind = scenario.external.kind
        if kind is ExternalPotentialKind.HARMONIC:
            system = f"harmonic well (omega = {scenario.external.omega:g})"
        elif kind is ExternalPotentialKind.PARTICLE_IN_BOX:
            system = f"cubic well (L = {scenario.external.box_length:g} bohr)"
        elif kind is ExternalPotentialKind.GAUSSIAN_WELL:
            system = f"Gaussian well (depth = {scenario.external.depth:g} Ha, width = {scenario.external.width:g} bohr)"
        elif scenario.structure.n_atoms == 0:
            system = f"{kind.value} potential"
        else:
            from cdft.constants import ELEMENT_SYMBOLS

            numbers = scenario.structure.numbers
            formula = "".join(ELEMENT_SYMBOLS[z] for z in numbers)
            counts = {s: formula.count(s) for s in set(ELEMENT_SYMBOLS[z] for z in numbers)}
            if len(counts) == 1 and len(numbers) > 1:
                formula = f"{next(iter(counts))}{len(numbers)}"
            total_z = float(record.charges.sum())
            charge = total_z - record.n_electrons
            if abs(charge) > 1.0e-12:
                formula += f" (q = {charge:+g})"
            system = formula
            if len(numbers) == 2:
                system += f", R = {_bond_length(record):g} bohr"
            if kind is ExternalPotentialKind.SOFT_COULOMB:
                system += f", soft Coulomb (a = {scenario.external.softening:g} bohr)"
        return f"{system}; {functional}; N = {record.n_electrons:g}"

    def axis_label(self, which: str) -> str:
        """Axis label: ``"u"`` is the principal axis (``b_bohr``), ``"v"`` the perpendicular (``a_bohr``)."""
        if self.record.positions.shape[0] > 1:
            return ("position along the bond axis, $b$ (bohr)" if which == "u"
                    else "perpendicular, $a$ (bohr)")
        centre, _, _ = self.frame()
        name, index = ("z", 2) if which == "u" else ("x", 0)
        shift = float(centre[index])
        if abs(shift) <= 1.0e-12:
            return f"${name}$ (bohr)"
        return f"${name} {'-' if shift > 0 else '+'} {abs(shift):g}$ (bohr)"

    def recorded_gate(self, gate_id: str) -> GateResult | None:
        """Return the record's result for one gate."""
        return next((g for g in self.record.gates if g.gate_id == gate_id), None)


def _usetex() -> bool:
    """Whether text is being typeset by LaTeX (``--usetex``), where a few characters need care."""
    return bool(pyplot().rcParams.get("text.usetex", False))


def _percent() -> str:
    """Return a percent sign that survives both mathtext and LaTeX."""
    return r"\%" if _usetex() else "%"


def _badge(fig, verdict: Verdict) -> None:
    """Draw the record-quality badge (status colour with a text label, never colour alone)."""
    text = verdict.badge()
    if not text:
        return
    critical = verdict.mark == "invalid"
    # A suptitle, not free text: the layout reserves room, so the badge never covers data.
    icon = "" if _usetex() else ("▲ " if critical else "● ")  # the glyphs are outside LaTeX's text fonts
    fig.suptitle(
        icon + text,
        x=0.01, ha="left", fontsize=6.0, color=INK, fontweight="normal",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#fde8e8" if critical else "#fff5df",
              "edgecolor": STATUS_CRITICAL if critical else STATUS_WARNING, "linewidth": 0.6},
    )


def _panel_label(ax, label: str) -> None:
    """Label a panel ``(a)``, ``(b)`` ... as the axes' left title, so the layout reserves room."""
    text = rf"\textbf{{({label})}}" if _usetex() else f"({label})"  # LaTeX ignores fontweight
    ax.set_title(text, loc="left", fontsize=8.0, fontweight="bold", color=INK, pad=2.0)


def _shrink_to_aspect(fig, ax, iterations: int = 4) -> None:
    """Reduce the figure height until an equal-aspect ``ax`` fills its slot, leaving no blank."""
    width, height = fig.get_size_inches()
    for _ in range(iterations):
        fig.canvas.draw()
        slot = ax.get_position(original=True)
        active = ax.get_position(original=False)
        blank = (slot.height - active.height) * height
        if blank < 0.01:
            return
        height -= blank
        fig.set_size_inches(width, height)


def _one_two_five(low: float, high: float, limit: int = 7) -> list[float]:
    """Return the 1-2-5 values inside ``[low, high]`` (decades only when there would be more than ``limit``)."""
    if not (low > 0.0 and high > low):
        return []
    steps = [m * 10.0**e for e in range(math.floor(math.log10(low)), math.ceil(math.log10(high)) + 1)
             for m in (1.0, 2.0, 5.0)]
    ticks = [t for t in steps if low * (1.0 - 1.0e-9) <= t <= high * (1.0 + 1.0e-9)]
    if len(ticks) > limit:
        ticks = [t for t in ticks if abs(math.log10(t) - round(math.log10(t))) < 1.0e-9]
    return ticks


def _plain_log_axis(ax, axis: str = "x") -> None:
    """Label a short logarithmic axis at 1-2-5 values in plain numbers (no crowded 2x10^0 labels)."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    target = ax.xaxis if axis == "x" else ax.yaxis
    low, high = sorted(ax.get_xlim() if axis == "x" else ax.get_ylim())
    ticks = _one_two_five(low, high)
    if len(ticks) < 2:
        return
    target.set_major_locator(FixedLocator(ticks))
    target.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    target.set_minor_formatter(NullFormatter())


def _mark_nuclei(ax, ctx: FigureContext, light: bool = False) -> None:
    """Nuclei as white-ringed dots with their element symbols, labelled outward of the centre."""
    for b, a, symbol in ctx.nuclei_in_frame():
        ax.plot([b], [a], marker="o", markersize=3.6, markerfacecolor=SURFACE if not light else INK,
                markeredgecolor=INK if not light else SURFACE, markeredgewidth=0.6, linestyle="none", zorder=5)
        left = b < -1.0e-9
        ax.annotate(symbol, (b, a), xytext=(-3 if left else 3, 3), textcoords="offset points",
                    ha="right" if left else "left", fontsize=6.5, color=INK, zorder=6)


def _fmt(value: float | None, digits: int = 3) -> str:
    """Compact scientific formatting for captions (``n/a`` for missing values)."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    if value != 0.0 and (abs(value) < 1.0e-3 or abs(value) >= 1.0e4):
        return f"{value:.{digits}e}"
    return f"{value:.{digits + 3}g}"


# --- The figures --------------------------------------------------------------------------------


def _needs_density(ctx: FigureContext) -> str | None:
    """Applicability: every density figure needs the density field."""
    return None if ctx.record.density is not None else "the record holds no density"


def fig_density_map(ctx: FigureContext) -> RenderedFigure:
    """log10 n on the plane through the nuclei, cusp resolved (D-35), decade contours."""
    plt = pyplot()
    style = ctx.style
    a, b, points = ctx.plane()
    n = ctx.evaluator.density(points)["n"].reshape(a.size, b.size)
    vmax = math.log10(ctx.n_max)
    positive = n[n > 0.0]
    # At most DENSITY_DECADES, fewer where the plane never gets that low: no colour spent unused.
    vmin = vmax - DENSITY_DECADES
    if positive.size:
        vmin = max(vmin, math.floor(math.log10(float(positive.min())) * 2.0) / 2.0)
    log_n = np.log10(np.maximum(n, 1.0e-300))
    width = style.single_width
    height = width * max(0.55, min(1.1, (a[-1] - a[0]) / (b[-1] - b[0]))) + 0.35
    fig, ax = plt.subplots(figsize=(width, height))
    image = ax.imshow(
        log_n, origin="lower", extent=(b[0], b[-1], a[0], a[-1]), cmap=colormap("sequential"),
        vmin=vmin, vmax=vmax, interpolation="antialiased", aspect="equal",
    )
    levels = np.arange(math.ceil(vmin), math.floor(vmax) + 1)
    if levels.size:
        ax.contour(b, a, log_n, levels=levels, colors=INK_SECONDARY, linewidths=style.hairline, alpha=0.55,
                   linestyles="solid")
    ax.grid(False)
    _mark_nuclei(ax, ctx)
    ax.set_xlabel(ctx.axis_label("u"))
    ax.set_ylabel(ctx.axis_label("v"))
    bar = fig.colorbar(image, ax=ax, extend="min", fraction=0.046, pad=0.03)
    bar.set_label(r"$\log_{10}\, n$ (bohr$^{-3}$)")
    bar.outline.set_linewidth(style.hairline)
    bar.ax.tick_params(width=style.hairline)
    _badge(fig, ctx.verdict)
    _shrink_to_aspect(fig, ax)
    record = ctx.record
    caption = (
        f"Electron density of {ctx.system_label()} on the plane through the nuclei, "
        f"log10 n over {vmax - vmin:g} decades below n_max = {_fmt(ctx.n_max)} bohr^-3; contours at "
        f"integer decades. Evaluated as n = f^2 I[rho] with the analytic cusp factor f^2 and the degree-7 "
        f"interpolant of rho = n/f^2 on a {a.size} x {b.size} mesh, from the recorded "
        f"{'x'.join(str(s) for s in record.shape)} grid (h = {record.spacing:.4g} bohr)."
    )
    return RenderedFigure(fig, caption, arrays={"a_bohr": a, "b_bohr": b, "density": n})


def _listing(items: Sequence[str]) -> str:
    """Join caption items as prose: ``a``, ``a and b``, ``a, b and c``."""
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def fig_density_profile(ctx: FigureContext) -> RenderedFigure:
    """Density along the principal axis (log) against the reference, and d ln n/db with the exact limits."""
    plt = pyplot()
    style = ctx.style
    record = ctx.record
    evaluator = ctx.evaluator
    centre, u, _ = ctx.frame()
    half = ctx.extent(u)
    b = np.linspace(-half, half, ctx.options.profile_points)
    points = evaluator.nudge(centre[None, :] + b[:, None] * u[None, :])
    fields = evaluator.density(points, order=1)
    n = fields["n"]
    floor = ctx.n_max * 10.0 ** (-DENSITY_DECADES)
    with np.errstate(divide="ignore", invalid="ignore"):
        # Below what panel (a) shows, d ln n is the ratio of two interpolation errors: not drawn.
        slope = np.where(n > floor, (fields["grad"] @ u) / n, np.nan)
    fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(style.single_width, 3.9),
                                      gridspec_kw={"height_ratios": [1.25, 1.0]})
    top.semilogy(b, n, color=SERIES[0], label="this work")
    columns = {"b_bohr": b, "density": n, "dlnn_db": slope}

    # Grid samples on the axis, when the axis is a grid line (nuclei are lattice points, D-64).
    drew_grid_values = False
    axis_index = int(np.argmax(np.abs(u)))
    if abs(abs(u[axis_index]) - 1.0) < 1.0e-12:
        t = (centre - np.asarray(record.origin)) / record.spacing
        others = [d for d in range(3) if d != axis_index]
        if all(abs(t[d] - round(t[d])) < 1.0e-6 for d in others):
            index = [int(round(t[d])) if d != axis_index else slice(None) for d in range(3)]
            line = record.density[tuple(index)]
            coordinate = (record.origin[axis_index] + record.spacing * np.arange(record.shape[axis_index])
                          - centre[axis_index]) * np.sign(u[axis_index])
            keep = (np.abs(coordinate) <= half) & (line > 0)
            stride = max(1, int(keep.sum() // 60))
            top.semilogy(coordinate[keep][::stride], line[keep][::stride], linestyle="none", marker="o",
                         markersize=2.4, markerfacecolor=SERIES[0], markeredgecolor=SURFACE,
                         markeredgewidth=0.4, label="grid values")
            drew_grid_values = True
    reference = ctx.reference
    if reference is not None:
        n_ref, grad_ref, _ = reference.evaluate(points)
        top.semilogy(b, n_ref, color=INK_SECONDARY, linestyle="--", linewidth=style.thin_width,
                     label=reference.short)
        with np.errstate(divide="ignore", invalid="ignore"):
            slope_ref = np.where(n_ref > floor, (grad_ref @ u) / n_ref, np.nan)
        bottom.plot(b, slope_ref, color=INK_SECONDARY, linestyle="--", linewidth=style.thin_width,
                    label=reference.short)
        columns["density_reference"] = n_ref
        columns["dlnn_db_reference"] = slope_ref
    top.set_ylim(floor, ctx.n_max * 3.0)
    top.set_ylabel(r"$n$ (bohr$^{-3}$)")
    # Every Phase 1 profile falls away from the centre, so the band under the peak is free.
    top.legend(loc="lower center", handlelength=1.4)
    _panel_label(top, "a")

    bottom.plot(b, slope, color=SERIES[0], label="this work")
    metrics: dict[str, Any] = {}
    charges = record.charges if record.cusp_factorised else np.zeros(0)
    if charges.size:
        z_max = float(charges.max())
        # Kato [A29]: d ln n/db jumps by 4 Z_a across nucleus a, carried by the analytic factor at
        # any spacing; the interpolant's own kink is measured beside it. -+2Z: single centre only.
        placed = 0
        for atom, (charge, position) in enumerate(zip(charges, record.positions, strict=True)):
            along = float((position - centre) @ u)
            if abs(along) > half or np.linalg.norm(position - (centre + along * u)) > 1.0e-9:
                continue  # off the drawn line
            probe = centre[None, :] + np.array([[along - 1.0e-7], [along + 1.0e-7]]) * u[None, :]
            rho, grad_rho, _ = evaluator._interpolate(evaluator._rho_pad, probe, 1)
            kink = float(((grad_rho @ u) / rho)[0] - ((grad_rho @ u) / rho)[1])
            metrics[f"kato_jump_factor_nucleus{atom}"] = 4.0 * float(charge)
            metrics[f"interpolant_slope_kink_nucleus{atom}"] = kink
            if record.positions.shape[0] > 1:
                # Outward of each nucleus, low left and high right, clear of the opposite tail.
                left = along < 0.0 or (abs(along) < 1.0e-9 and placed % 2 == 0)
                bottom.annotate(f"jump 4Z = {4.0 * float(charge):g}",
                                (along, 0.35 if left else 0.65), xycoords=("data", "axes fraction"),
                                xytext=(-4 if left else 4, 0), textcoords="offset points", fontsize=6.0,
                                color=INK_SECONDARY, ha="right" if left else "left", va="center")
                placed += 1
        if record.positions.shape[0] == 1:
            for sign in (1.0, -1.0):
                bottom.axhline(sign * 2.0 * z_max, color=INK_MUTED, linewidth=style.hairline,
                               label=r"$\pm 2Z$ (Kato)" if sign > 0 else None)
        homo = record.homo
        if homo is not None and homo < 0.0:
            kappa = math.sqrt(-2.0 * homo)
            metrics["kappa_homo"] = kappa
            for sign in (1.0, -1.0):
                bottom.axhline(sign * 2.0 * kappa, color=SERIES[1], linestyle="--", linewidth=style.thin_width,
                               label=r"$\pm 2\sqrt{-2\varepsilon_{\rm HOMO}}$" if sign > 0 else None)
        exact = exact_ionisation_level(record, ctx.options.use_oracle)
        if exact is not None:
            kappa_exact = math.sqrt(-2.0 * exact.energy)
            metrics["kappa_exact"] = kappa_exact
            for sign in (1.0, -1.0):
                bottom.axhline(sign * 2.0 * kappa_exact, color=SERIES[2], linestyle=":", linewidth=style.line_width,
                               label=r"$\pm 2\sqrt{2I}$ (exact functional)" if sign > 0 else None)
        bottom.set_ylim(-3.2 * z_max, 3.2 * z_max)
    else:
        finite = slope[np.isfinite(slope)]
        if finite.size:
            span = np.percentile(np.abs(finite), 98)
            bottom.set_ylim(-1.2 * span, 1.2 * span)
    bottom.set_xlabel(ctx.axis_label("u"))
    bottom.set_ylabel(r"$d\ln n/dz$ (bohr$^{-1}$)" if record.positions.shape[0] <= 1 else r"$d\ln n/db$ (bohr$^{-1}$)")
    # Below the figure: every in-panel position collides with some system's curves. This work first.
    handles, labels = bottom.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda k: labels[k] != "this work")
    fig.legend([handles[k] for k in order], [labels[k] for k in order],
               loc="outside lower center", ncol=2, fontsize=6.0)
    _panel_label(bottom, "b")
    _badge(fig, ctx.verdict)
    guide = ""
    if charges.size:
        kinks = ", ".join(
            f"{metrics[f'interpolant_slope_kink_nucleus{k}']:.2e}"
            for k in range(record.positions.shape[0]) if f"interpolant_slope_kink_nucleus{k}" in metrics
        )
        guide = (
            " In (b) the logarithmic derivative jumps by 4Z across each nucleus (Kato's cusp, [A29]): the "
            "analytic factor f^2 carries that jump exactly; the drawn jump also contains the slope "
            f"discontinuity of the interpolant of rho at the node ({kinks} bohr^-1), a property of the "
            "figure's interpolation, not a measurement of the solver's cusp (gate G1.11 is that). It "
            "approaches -+2 kappa in the tails, kappa = sqrt(-2 eps_HOMO) for the Kohn-Sham density [A30], "
            "[A31]; kappa = sqrt(2I) is what the exact functional would give (ionisation-potential theorem)."
        )
    evaluation = ("the cusp-factorised interpolant n = f^2 I[rho]" if record.cusp_factorised
                  else "the degree-7 interpolant of the recorded density")
    drawn = [f"{evaluation} (line)"]
    if drew_grid_values:
        drawn.append("the recorded grid values (dots)")
    if reference is not None:
        drawn.append(f"{reference.label} (dashed)")
    caption = (
        f"Density of {ctx.system_label()} along the principal axis through "
        f"{('the centre', 'the nucleus', 'the nuclei')[min(record.positions.shape[0], 2)]}. "
        f"(a) n from {_listing(drawn)}. (b) d ln n / d(position), drawn where n exceeds the lower limit "
        f"of (a).{guide}"
    )
    return RenderedFigure(fig, caption, columns=columns, metrics=metrics)


def _single_centre(ctx: FigureContext) -> str | None:
    """Applicability: one nucleus, or the (centred) harmonic well."""
    if ctx.record.density is None:
        return "the record holds no density"
    record = ctx.record
    if record.scenario.external.kind is ExternalPotentialKind.HARMONIC:
        return None
    if record.positions.shape[0] != 1:
        return "radial figures are drawn for single-centre systems (one nucleus or the harmonic well)"
    return None


def _radial_maximum(radii: np.ndarray, values: np.ndarray, function: Callable[[float], float]) -> tuple[float, float]:
    """Largest sampled maximum, refined to 1e-8 bohr by bounded Brent search on ``function``."""
    from scipy.optimize import minimize_scalar

    peak = int(np.nanargmax(values))
    if peak == 0 or peak == radii.size - 1:
        return float(radii[peak]), float(values[peak])
    found = minimize_scalar(lambda r: -function(float(r)), bounds=(float(radii[peak - 1]), float(radii[peak + 1])),
                            method="bounded", options={"xatol": 1.0e-8})
    if not found.success or -found.fun < values[peak]:
        return float(radii[peak]), float(values[peak])
    return float(found.x), float(-found.fun)


def _box_rule(record: FigureRecord) -> str:
    """Return the rule that sized the record's box, as a parenthetical for a caption ("" if none applies)."""
    from cdft.grid import (
        COULOMB_HALF_BOX_PER_Z,
        INTERACTING_HALF_BOX,
        INTERACTING_TAIL_FRACTION,
        VACUUM_PADDING,
    )

    coulomb = (record.scenario.external.kind is ExternalPotentialKind.NUCLEAR_COULOMB
               and record.positions.shape[0] > 0)
    if coulomb and record.interacting:
        return (f" (D-53 sizes the box from a Slater-screened hydrogenic tail estimate: "
                f"{INTERACTING_TAIL_FRACTION:g} of the charge beyond the vacuum margin, which is at least "
                f"{INTERACTING_HALF_BOX:g} bohr; the number above is the measured value)")
    if coulomb and record.positions.shape[0] == 1:
        z = float(record.charges.max())
        return (f" (D-42: a half-width of {COULOMB_HALF_BOX_PER_Z:g}/Z = {COULOMB_HALF_BOX_PER_Z / z:g} bohr, "
                f"where the weight exp(-2ZR) of the exact tail is exp(-{2.0 * COULOMB_HALF_BOX_PER_Z:g}))")
    if coulomb:
        return f" (D-42: the nuclear span plus {VACUUM_PADDING:g} bohr of vacuum)"
    if record.scenario.external.kind is ExternalPotentialKind.PARTICLE_IN_BOX:
        return " (the box is the well itself)"
    return " (the box of the numerics preset; no sizing rule applies to a model potential)"


def fig_radial_density(ctx: FigureContext) -> RenderedFigure:
    """4 pi r^2 <n>(r) and the charge between r and the inscribed sphere, against the reference."""
    from scipy.integrate import cumulative_simpson

    plt = pyplot()
    style = ctx.style
    record = ctx.record
    centre, _, _ = ctx.frame()
    low = np.asarray(record.origin)
    r_max = float(min(np.min(centre - low), np.min(record.upper - centre)))
    radii = np.linspace(0.0, r_max, ctx.options.radial_points)
    radii[0] = 1.0e-9
    average = spherical_average(ctx.evaluator, centre, radii)
    radial = 4.0 * math.pi * radii**2 * average["n"]
    inner = cumulative_simpson(radial, x=radii, initial=0.0)
    outer = inner[-1] - inner  # charge between r and the inscribed sphere
    electrons = record.n_electrons
    fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(style.single_width, 3.6),
                                      gridspec_kw={"height_ratios": [1.0, 1.0]})
    top.plot(radii, radial, color=SERIES[0], label="this work (spherical average)")
    bottom.semilogy(radii[:-1], outer[:-1] / electrons, color=SERIES[0], label="this work")
    columns = {"r_bohr": radii, "radial_distribution": radial, "charge_r_to_rmax_per_electron": outer / electrons}
    reference = ctx.reference
    metrics: dict[str, Any] = {"charge_inside_inscribed_sphere": float(inner[-1]),
                               "charge_beyond_inscribed_sphere": float(electrons - inner[-1]),
                               "inscribed_radius_bohr": r_max}
    if reference is not None and reference.radial is not None:
        n_ref, _ = reference.radial(radii)
        radial_ref = 4.0 * math.pi * radii**2 * n_ref
        inner_ref = cumulative_simpson(radial_ref, x=radii, initial=0.0)
        outer_ref = inner_ref[-1] - inner_ref
        top.plot(radii, radial_ref, color=INK_SECONDARY, linestyle="--", linewidth=style.thin_width,
                 label=reference.short)
        bottom.semilogy(radii[:-1], outer_ref[:-1] / electrons, color=INK_SECONDARY, linestyle="--",
                        linewidth=style.thin_width, label=reference.short)
        columns["radial_distribution_reference"] = radial_ref
        columns["charge_r_to_rmax_reference"] = outer_ref / electrons
    peak_r, peak_value = _radial_maximum(
        radii, radial, lambda r: 4.0 * math.pi * r * r * spherical_average(ctx.evaluator, centre, np.array([r]))["n"][0]
    )
    metrics["radial_peak_bohr"] = peak_r
    if reference is not None and reference.radial is not None:
        metrics["radial_peak_reference_bohr"], _ = _radial_maximum(
            radii, radial_ref, lambda r: 4.0 * math.pi * r * r * float(reference.radial(np.array([r]))[0][0])
        )
    top.plot([peak_r], [peak_value], marker="o", markersize=3.5, color=SERIES[0],
             markeredgecolor=SURFACE, markeredgewidth=0.5, linestyle="none")
    top.annotate(f"$r_{{\\max}}$ = {peak_r:.4f} bohr", (peak_r, peak_value), xytext=(6, -2),
                 textcoords="offset points", fontsize=6.5, color=INK_SECONDARY, va="top")
    top.set_ylabel(r"$4\pi r^2 \langle n\rangle(r)$ (bohr$^{-1}$)")
    top.legend(loc="center right")
    _panel_label(top, "a")
    floor = max(1.0e-14, float(np.min(outer[:-1][outer[:-1] > 0]) / electrons) if np.any(outer[:-1] > 0) else 1.0e-14)
    bottom.set_ylim(floor, 2.0)
    bottom.set_xlabel(r"$r$ (bohr)")
    bottom.set_ylabel(r"charge in $[r, r_{\rm box}]$ / $N$")
    bottom.legend(loc="upper right")
    _panel_label(bottom, "b")
    _badge(fig, ctx.verdict)
    caption = (
        f"Radial structure of {ctx.system_label()}. (a) Radial distribution 4 pi r^2 <n>(r) of the spherically "
        f"averaged density (24 x 48 angular rule about the "
        f"{'nucleus' if record.positions.shape[0] else 'centre'}), maximum at r = {peak_r:.5f} bohr"
        + (f" (reference {metrics['radial_peak_reference_bohr']:.5f} bohr)"
           if "radial_peak_reference_bohr" in metrics else "")
        + ". "
        f"(b) Charge between r and the sphere inscribed in the box (r_box = {r_max:.3f} bohr), per electron; "
        f"{_fmt(electrons - inner[-1])} electrons lie beyond r_box, in the box corners{_box_rule(record)}."
        + (f" Dashed: {reference.label}." if reference is not None and reference.radial is not None else "")
    )
    return RenderedFigure(fig, caption, columns=columns, metrics=metrics)


def _has_nuclei(ctx: FigureContext) -> str | None:
    """Applicability: a Coulomb-nucleus density."""
    if ctx.record.density is None:
        return "the record holds no density"
    record = ctx.record
    if record.scenario.external.kind is not ExternalPotentialKind.NUCLEAR_COULOMB or record.positions.shape[0] == 0:
        return "no Coulomb nuclei on this potential"
    if not record.cusp_factorised:
        return ("the record was solved without the analytic cusp factor (D-35), so no cusp is resolved "
                "inside a grid cell")
    return None


def fig_cusp_and_decay(ctx: FigureContext) -> RenderedFigure:
    """Kato ratio at each unique nucleus; local decay exponent against the HOMO and exact asymptotes."""
    plt = pyplot()
    style = ctx.style
    record = ctx.record
    positions = record.positions
    charges = record.charges
    fig, (left, right) = plt.subplots(1, 2, figsize=(style.double_width, 2.55),
                                      gridspec_kw={"wspace": 0.06})
    tables: dict[str, dict[str, np.ndarray]] = {}
    metrics: dict[str, Any] = {}
    seen: set[float] = set()
    for atom, (charge, position) in enumerate(zip(charges, positions, strict=True)):
        if float(charge) in seen:
            continue
        seen.add(float(charge))
        nearest = min((float(np.linalg.norm(position - other)) for k, other in enumerate(positions) if k != atom),
                      default=math.inf)
        r_top = min(1.0 / charge, 0.25 * nearest, 0.5 * ctx.reach(position, np.array([1.0, 0.0, 0.0])))
        radii = np.logspace(-4.0, math.log10(r_top), 120)
        average = spherical_average(ctx.evaluator, position, radii, n_theta=16, n_phi=32)
        ratio = -average["dn_dr"] / (2.0 * charge * average["n"])
        left.semilogx(radii, ratio, color=SERIES[len(seen) - 1],
                      label=f"nucleus {atom} (Z = {charge:g})")
        tables[f"kato_atom{atom}"] = {"r_bohr": radii, "n_spherical": average["n"],
                                      "dn_dr_spherical": average["dn_dr"], "kato_ratio": ratio}
        # Inside one spacing the limit is the factor's exact 1 plus the interpolant's kink.
        metrics[f"kato_ratio_interpolant_limit_atom{atom}"] = float(ratio[0])
        metrics[f"kato_ratio_at_one_spacing_atom{atom}"] = float(
            np.interp(record.spacing, radii, ratio)) if radii[-1] >= record.spacing else None
    left.axvspan(1.0e-4, record.spacing, color=GRIDLINE, alpha=0.45, linewidth=0.0, zorder=0,
                 label="within one grid spacing")
    left.axhline(1.0, color=INK_MUTED, linewidth=style.thin_width, label="Kato: 1")
    reference = ctx.reference if positions.shape[0] == 1 else None
    if reference is not None and reference.z_eff is not None and abs(reference.z_eff / charges[0] - 1.0) > 1.0e-4:
        # A GGA potential diverges at the nucleus, so its exact KS cusp is Z_eff != Z (D-55).
        left.axhline(reference.z_eff / charges[0], color=SERIES[1], linestyle="--", linewidth=style.thin_width,
                     label=f"radial oracle: $Z_{{\\rm eff}}/Z$ = {reference.z_eff / charges[0]:.4f}")
        metrics["oracle_z_eff"] = reference.z_eff
    # At least [0, 1.2] always: an exact 1s density gives 1, and the axis would magnify its noise.
    drawn = np.concatenate([np.asarray(t["kato_ratio"]) for t in tables.values()])
    drawn = drawn[np.isfinite(drawn)]
    low = min(0.0, float(drawn.min()) - 0.05) if drawn.size else 0.0
    high = max(1.2, float(drawn.max()) + 0.05) if drawn.size else 1.2
    left.set_ylim(low, high)
    left.ticklabel_format(axis="y", useOffset=False)
    left.set_xlabel("distance from the nucleus (bohr)")
    left.set_ylabel(r"$-\frac{1}{2Z}\,\frac{d\ln\langle n\rangle}{dr}$")
    left.legend(loc="lower left")
    _panel_label(left, "a")

    centre, _, _ = ctx.frame()
    low = np.asarray(record.origin)
    r_box = float(min(np.min(centre - low), np.min(record.upper - centre)))
    r_start = float(max(np.linalg.norm(positions - centre, axis=1).max(), 0.0)) + 1.0
    kappa_values = []
    if r_start < r_box - 1.0:
        radii = np.linspace(r_start, r_box, 240)
        average = spherical_average(ctx.evaluator, centre, radii, n_theta=16, n_phi=32)
        with np.errstate(divide="ignore", invalid="ignore"):
            kappa_local = np.where(average["n"] > 0.0, -0.5 * average["dn_dr"] / average["n"], np.nan)
        right.plot(radii, kappa_local, color=SERIES[0], label="this work")
        decay = {"r_bohr": radii, "n_spherical": average["n"], "kappa_local": kappa_local}
        tables["decay"] = decay
        homo = record.homo
        if homo is not None and homo < 0.0:
            kappa = math.sqrt(-2.0 * homo)
            screened = float(charges.sum()) - (record.n_electrons if record.interacting else 0.0)
            asymptote = kappa - (screened / kappa - 1.0) / radii
            right.plot(radii, asymptote, color=SERIES[1], linestyle="--", linewidth=style.thin_width,
                       label=r"$\kappa - (Q/\kappa - 1)/r$, $\kappa = \sqrt{-2\varepsilon_{\rm HOMO}}$")
            kappa_values.append(kappa)
            decay["kappa_homo_asymptote"] = asymptote
            metrics["kappa_homo"] = kappa
            metrics["tail_charge_Q"] = screened
            window = (radii > r_start + 0.25 * (r_box - r_start)) & (radii < r_box - 0.25 * (r_box - r_start))
            if np.any(window):
                metrics["kappa_local_minus_asymptote_mid_window"] = float(
                    np.nanmedian(kappa_local[window] - asymptote[window])
                )
        exact = exact_ionisation_level(record, ctx.options.use_oracle)
        if exact is not None:
            kappa_exact = math.sqrt(-2.0 * exact.energy)
            right.axhline(kappa_exact, color=SERIES[2], linestyle=":", linewidth=style.line_width,
                          label=r"$\sqrt{2I}$ (exact functional)")
            kappa_values.append(kappa_exact)
            metrics["kappa_exact"] = kappa_exact
        right.axvline(r_box, color=INK_MUTED, linewidth=style.thin_width)
        right.annotate("box face", (r_box, 0.02), xycoords=("data", "axes fraction"), xytext=(-3, 0),
                       textcoords="offset points", ha="right", fontsize=6.0, color=INK_MUTED)
        finite = kappa_local[np.isfinite(kappa_local)]
        top = (2.2 * max(kappa_values) if kappa_values
               else 1.5 * float(np.percentile(finite, 95)) if finite.size else 0.0)
        right.set_ylim(0.0, top if math.isfinite(top) and top > 0.0 else 2.0)
    else:
        right.text(0.5, 0.5, "box too small for a tail window", transform=right.transAxes, ha="center",
                   color=INK_MUTED)
    right.set_xlabel("distance from the centre of charge (bohr)")
    right.set_ylabel(r"$-\frac{1}{2}\,\frac{d\ln\langle n\rangle}{dr}$ (bohr$^{-1}$)")
    right.legend(loc="upper left", fontsize=6.0)
    right.ticklabel_format(axis="y", useOffset=False)
    _panel_label(right, "b")
    _badge(fig, ctx.verdict)
    gate = ctx.recorded_gate("G1.11")
    gate_text = (f" Gate G1.11 on this record: {gate.verdict.value} ({_fmt(gate.measured)}"
                 f"{'; ' + gate.reason if gate.reason else ''})." if gate is not None else "")
    gga = record.interacting and record.functional().rung.value >= XCRung.GGA.value
    caption = (
        f"Exact conditions on the density of {ctx.system_label()}. (a) Kato's cusp condition [A29]: "
        f"-(1/2Z) d ln<n>/dr of the density averaged over a sphere about each nucleus (one per distinct "
        f"charge) tends to 1 as r -> 0"
        + (" (for a GGA the exact Kohn-Sham orbital has Z_eff != Z because v_xc diverges at the nucleus, "
           "D-55; the dashed line is the radial oracle's own value)" if gga else "")
        + (". Within one grid spacing (shaded) the curve is the analytic factor's exact 1 plus the "
           "interpolant of rho, so its limit there describes the figure's interpolation; beyond it the curve "
           "follows the solved density" if record.cusp_factorised else "")
        + f".{gate_text} (b) Local decay exponent -(1/2) d ln<n>/dr about the centre of charge against the "
        f"asymptotic form of a Kohn-Sham density with a -Q/r potential tail, kappa - (Q/kappa - 1)/r with "
        f"kappa = sqrt(-2 eps_HOMO) and Q = {_fmt(metrics.get('tail_charge_Q'))} [A30], [A31]; the dotted line "
        f"is sqrt(2I) of the exact functional where I is known. The rise at the box face is the Dirichlet wall."
    )
    return RenderedFigure(fig, caption, tables=tables, metrics=metrics)


def _has_reference(ctx: FigureContext) -> str | None:
    """Applicability: a reference density exists for this record."""
    if ctx.record.density is None:
        return "the record holds no density"
    if ctx.reference is None:
        if ctx.record.positions.shape[0] > 1:
            return "no reference density for a two-centre system (the two-centre oracle returns energies; I10)"
        if ctx.record.interacting and not ctx.options.use_oracle:
            return "the radial Kohn-Sham oracle was switched off (--no-oracle)"
        return "no reference density is available for this scenario"
    return None


def fig_density_error(ctx: FigureContext) -> RenderedFigure:
    """Density, gradient-norm and Laplacian deviations from the reference along the axis [B11]."""
    plt = pyplot()
    style = ctx.style
    record = ctx.record
    evaluator = ctx.evaluator
    reference = ctx.reference
    centre, u, _ = ctx.frame()
    half = ctx.extent(u)
    b = np.linspace(-half, half, ctx.options.profile_points)
    points = evaluator.nudge(centre[None, :] + b[:, None] * u[None, :])
    fields = evaluator.density(points, order=2)
    n_ref, grad_ref, lap_ref = reference.evaluate(points)
    grad_norm = np.linalg.norm(fields["grad"], axis=1)
    grad_norm_ref = np.linalg.norm(grad_ref, axis=1)
    # lap n diverges like -4Z n/r at a nucleus, so its scale is taken outside a ball about each.
    if record.positions.shape[0]:
        along = np.array([float((p - centre) @ u) for p in record.positions])
        distance = np.min(np.abs(b[:, None] - along[None, :]), axis=1)
    else:
        distance = np.full(b.size, np.inf)
    regular = distance >= LAPLACIAN_EXCLUSION_BOHR
    rows = (
        ("density", fields["n"], n_ref, r"$n$ (bohr$^{-3}$)", np.ones(b.size, dtype=bool)),
        ("gradient", grad_norm, grad_norm_ref, r"$|\nabla n|$ (bohr$^{-4}$)", np.ones(b.size, dtype=bool)),
        ("laplacian", fields["lap"], lap_ref, r"$\nabla^2 n$ (bohr$^{-5}$)", regular),
    )
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(style.single_width, 4.6),
                             gridspec_kw={"hspace": 0.0})
    columns: dict[str, np.ndarray] = {"b_bohr": b}
    metrics: dict[str, Any] = {"laplacian_exclusion_radius_bohr": LAPLACIAN_EXCLUSION_BOHR}
    relevant = n_ref > ctx.n_max * 1.0e-6
    handles = []
    for ax, (key, value, ref_value, label, where), letter in zip(axes, rows, "abc", strict=True):
        error = np.abs(value - ref_value)
        mask = relevant & where
        reference_line, = ax.semilogy(b, np.abs(ref_value), color=INK_MUTED, linewidth=style.thin_width,
                                      label="|reference|")
        error_line, = ax.semilogy(b, np.maximum(error, 1.0e-300), color=SERIES[0],
                                  label="|this work - reference|")
        handles = [reference_line, error_line]
        scale = float(np.max(np.abs(ref_value[mask]))) if np.any(mask) else float(np.max(np.abs(ref_value)))
        metrics[f"max_abs_error_{key}_scaled"] = float(np.max(error[mask]) / scale) if np.any(mask) else None
        ax.set_ylabel(label)
        floor = max(1.0e-18, scale * 1.0e-16)
        ax.set_ylim(floor, scale * 5.0)
        _panel_label(ax, letter)
        columns[f"{key}"] = value
        columns[f"{key}_reference"] = ref_value
    fig.legend(handles=handles, loc="outside upper right", ncol=2, fontsize=style.small_size)
    axes[-1].set_xlabel(ctx.axis_label("u"))

    grid = ctx.grid
    grid_fields = ctx.grid_fields(order=1)
    n_grid_ref, _, _ = reference.evaluate(grid.points)
    inside = grid_fields["inside"]
    difference = np.abs(grid_fields["n"] - n_grid_ref)
    metrics["l1_density_error_electrons"] = float((grid.weights * difference)[inside].sum())
    metrics["l1_density_error_per_electron"] = metrics["l1_density_error_electrons"] / record.n_electrons
    metrics["figure_quadrature_charge"] = float((grid.weights * grid_fields["n"]).sum())
    metrics["reference_charge_inside_box"] = float((grid.weights * n_grid_ref)[inside].sum())
    _badge(fig, ctx.verdict)
    caption = (
        f"Deviation of the density of {ctx.system_label()} from {reference.label} ({reference.source}) along the "
        f"principal axis: (a) n, (b) |grad n|, (c) the Laplacian, the three quantities Medvedev et al. use to "
        f"judge densities [B11]; thin grey lines give the reference magnitude. The Laplacian diverges at a "
        f"nucleus, so the range of (c) and its scaled error are taken beyond {LAPLACIAN_EXCLUSION_BOHR:g} bohr "
        f"of the nuclei. "
        + ("The fields are the degree-7 interpolant of rho = n / f^2 times the analytic cusp factor f^2, "
           "with the factor's derivatives taken analytically (D-35). " if record.cusp_factorised
           else "The fields are the degree-7 interpolant of the recorded density. ")
        + f"Integrated over the box ({grid.description()}), int |n - n_ref| = "
        f"{_fmt(metrics['l1_density_error_electrons'])} electrons "
        f"({_fmt(metrics['l1_density_error_per_electron'])} per electron); the figure quadrature integrates the "
        f"record to {_fmt(metrics['figure_quadrature_charge'], 10)} electrons. These are numerical-error "
        f"measurements of the solver against a reference of the same functional (density-driven errors [B7] "
        f"need a density of the exact functional). A radial-oracle reference is read from "
        f"{RADIAL_ORACLE_TRUST_BOHR:g} bohr outward (O-25: its inner Dirichlet node depresses n e^(2Zr) by "
        f"4e-6 at 1e-6 bohr)."
    )
    return RenderedFigure(fig, caption, columns=columns, metrics=metrics)


def fig_reduced_gradient(ctx: FigureContext) -> RenderedFigure:
    """Density-weighted distributions of s and r_s [A32], the joint map, and PBE F_x over the s band."""
    from matplotlib.colors import LogNorm

    from cdft.xc.gga import PBEExchange

    plt = pyplot()
    style = ctx.style
    grid = ctx.distribution_grid
    fields = ctx.distribution_fields()
    n = fields["n"]
    electrons_weight = grid.weights * n
    keep = fields["inside"] & (n > ctx.n_max * 1.0e-12)
    n_k = n[keep]
    w_k = electrons_weight[keep]
    total = float(w_k.sum())
    s_all = np.full(n.size, np.nan)
    rs_all = np.full(n.size, np.nan)
    s_all[keep] = np.linalg.norm(fields["grad"][keep], axis=1) / (_S_PREFACTOR * n_k ** (4.0 / 3.0))
    rs_all[keep] = (3.0 / (4.0 * math.pi * n_k)) ** (1.0 / 3.0)
    s = s_all[keep]
    rs = rs_all[keep]
    s_edges = np.linspace(0.0, 3.0, 61)
    rs_edges = np.linspace(0.0, 8.0, 65)
    weight = electrons_weight / total
    shown = smeared_histograms(grid, weight, keep, {"s": (s_all, s_edges), "rs": (rs_all, rs_edges)},
                               joint=("rs", "s"))
    g_s, g_rs, joint = shown["s"], shown["rs"], shown["joint"]
    q10, q50, q90 = radial_model_quantiles(grid, s_all, weight, keep, (0.1, 0.5, 0.9))
    rq10, rq50, rq90 = radial_model_quantiles(grid, rs_all, weight, keep, (0.1, 0.5, 0.9))
    metrics = {
        "mean_s": float((w_k * s).sum() / total),
        "median_s": q50,
        "s_q10": q10,
        "s_q90": q90,
        "fraction_s_above_3": radial_model_fraction_above(grid, s_all, weight, keep, 3.0),
        "mean_rs_bohr": float((w_k * rs).sum() / total),
        "median_rs_bohr": rq50,
        "rs_q10_bohr": rq10,
        "rs_q90_bohr": rq90,
        "fraction_rs_above_8": radial_model_fraction_above(grid, rs_all, weight, keep, 8.0),
        "figure_quadrature_charge": float(electrons_weight[fields["inside"]].sum()),
        "histogram_subpoints_per_axis": [int(x) for x in shown["substeps"]],
    }
    fig, axes = plt.subplots(2, 2, figsize=(style.double_width, 4.4),
                             gridspec_kw={"hspace": 0.04, "wspace": 0.06})
    (ax_s, ax_rs), (ax_fx, ax_joint) = axes
    centres_s = 0.5 * (s_edges[1:] + s_edges[:-1])
    centres_rs = 0.5 * (rs_edges[1:] + rs_edges[:-1])
    ax_s.fill_between(centres_s, g_s, step="mid", color=SERIES[0], alpha=0.10, linewidth=0.0)
    ax_s.step(centres_s, g_s, where="mid", color=SERIES[0])
    ax_s.axvline(q50, color=INK_SECONDARY, linewidth=style.thin_width)
    ax_s.annotate(f"median {q50:.2f}", (q50, 0.95), xycoords=("data", "axes fraction"), xytext=(3, 0),
                  textcoords="offset points", fontsize=6.0, color=INK_SECONDARY, va="top")
    ax_s.set_xlabel(r"reduced gradient $s$")
    ax_s.set_ylabel(r"$g(s)/N$")
    ax_s.set_xlim(0.0, 3.0)
    _panel_label(ax_s, "a")
    ax_rs.fill_between(centres_rs, g_rs, step="mid", color=SERIES[0], alpha=0.10, linewidth=0.0)
    ax_rs.step(centres_rs, g_rs, where="mid", color=SERIES[0])
    ax_rs.axvline(rq50, color=INK_SECONDARY, linewidth=style.thin_width)
    ax_rs.annotate(f"median {rq50:.2f}", (rq50, 0.95), xycoords=("data", "axes fraction"), xytext=(3, 0),
                   textcoords="offset points", fontsize=6.0, color=INK_SECONDARY, va="top")
    ax_rs.set_xlabel(r"Wigner-Seitz radius $r_s$ (bohr)")
    ax_rs.set_ylabel(r"$g(r_s)/N$ (bohr$^{-1}$)")
    ax_rs.set_xlim(0.0, 8.0)
    _panel_label(ax_rs, "b")
    s_line = np.linspace(0.0, 3.0, 301)
    kappa, mu = PBEExchange.kappa, PBEExchange.mu
    ax_fx.axvspan(q10, min(q90, 3.0), color=SERIES[0], alpha=0.10, linewidth=0.0)
    f_pbe = 1.0 + kappa - kappa / (1.0 + mu * s_line**2 / kappa)
    f_median = 1.0 + kappa - kappa / (1.0 + mu * q50**2 / kappa)
    if q50 <= 3.0:
        # A drop line reads F_x off at the median and leaves the top of the band free for its label.
        ax_fx.plot([q50, q50], [0.9, f_median], color=INK_SECONDARY, linewidth=style.thin_width)
        ax_fx.plot([q50], [f_median], marker="o", markersize=2.8, color=INK, linestyle="none")
    ax_fx.plot(s_line, f_pbe, color=INK)
    ax_fx.axhline(1.0, color=INK_MUTED, linewidth=style.thin_width)
    ax_fx.axhline(1.0 + kappa, color=INK_MUTED, linewidth=style.thin_width, linestyle="--")
    ax_fx.set_xlim(0.0, 3.0)
    ax_fx.set_ylim(0.9, 1.0 + kappa + 0.12)
    # Direct labels: identity never by colour alone, and no legend over the data.
    label_style = {"fontsize": 6.0, "color": INK_SECONDARY}
    ax_fx.annotate(r"PBE $F_x(s)$", (2.95, f_pbe[-1]), xytext=(0, 3), textcoords="offset points",
                   ha="right", va="bottom", **label_style)
    ax_fx.annotate("LDA", (2.95, 1.0), xytext=(0, -2), textcoords="offset points", ha="right", va="top",
                   **label_style)
    ax_fx.annotate(r"$1+\kappa$ (Lieb-Oxford)", (2.95, 1.0 + kappa), xytext=(0, 2), textcoords="offset points",
                   ha="right", va="bottom", **label_style)
    ax_fx.annotate(f"10-90 {_percent()} of the electrons", (q10, 1.0 + kappa), xytext=(2, -3),
                   textcoords="offset points", ha="left", va="top", **label_style)
    if q50 <= 3.0:
        ax_fx.annotate(f"median: $F_x$ = {f_median:.3f}", (q50, 0.9), xytext=(2, 2), textcoords="offset points",
                       ha="left", va="bottom", **label_style)
    ax_fx.set_xlabel(r"reduced gradient $s$")
    ax_fx.set_ylabel(r"exchange enhancement $F_x$")
    _panel_label(ax_fx, "c")
    positive = joint[joint > 0]
    if positive.size:
        mesh = ax_joint.pcolormesh(rs_edges, s_edges, np.ma.masked_less_equal(joint.T, 0.0),
                                   cmap=colormap("sequential"),
                                   norm=LogNorm(vmin=max(positive.max() * 1.0e-4, positive.min()), vmax=positive.max()),
                                   shading="flat", rasterized=True)
        bar = fig.colorbar(mesh, ax=ax_joint, fraction=0.05, pad=0.03)
        bar.set_label(r"$g(r_s, s)/N$ (bohr$^{-1}$)")
        bar.outline.set_linewidth(style.hairline)
        # 1-2-5 steps inside the range, so a narrow range never shows a single labelled tick.
        ticks = _one_two_five(mesh.norm.vmin, mesh.norm.vmax)
        bar.set_ticks(ticks, labels=[f"{t:g}" for t in ticks])
        bar.ax.minorticks_off()
    ax_joint.grid(False)
    ax_joint.set_xlabel(r"$r_s$ (bohr)")
    ax_joint.set_ylabel(r"$s$")
    _panel_label(ax_joint, "d")
    _badge(fig, ctx.verdict)
    caption = (
        f"Semi-local ingredients sampled by the density of {ctx.system_label()}. (a, b) Density-weighted "
        f"distributions g(s) = int n delta(s - s(r)) d^3r and g(r_s), per electron (Zupan, Burke, Ernzerhof and "
        f"Perdew [A32]; the same analysis rationalises functional performance in [A33]), with "
        f"s = |grad n| / (2 (3 pi^2)^(1/3) n^(4/3)) and r_s = (3 / 4 pi n)^(1/3); medians "
        f"s = {q50:.3f}, r_s = {rq50:.3f} bohr; {100 * metrics['fraction_s_above_3']:.2f} % of the electrons "
        f"have s > 3 (outside the plot). (c) The PBE exchange enhancement factor [A4] over the band that holds "
        f"10-90 % of the electrons (s = {q10:.3f}-{q90:.3f}); the drop line marks the median. (d) The joint "
        f"distribution. "
        + (f"Sampled on {grid.description()} ({ctx.options.dist_radial} radial x "
           f"{ctx.options.dist_theta * ctx.options.dist_phi} angular nodes per nucleus) with the "
           f"{'cusp-factorised ' if ctx.record.cusp_factorised else ''}interpolant's gradient; "
           if grid.kind == "becke" else
           f"Sampled on a radial x angular product grid about the centre ({ctx.options.dist_radial} x "
           f"{ctx.options.dist_theta * ctx.options.dist_phi} nodes); " if grid.kind == "centred" else
           "Sampled on the recorded mesh with the interpolant's gradient (quantiles good to the variation of "
           "s across one cell); ")
        + f"points below 1e-12 n_max are omitted. Each node's weight is spread linearly over its cell "
        f"({' x '.join(str(int(m)) for m in shown['substeps'])} sub-points along "
        f"{'x, y and z' if grid.kind == 'box' else 'radius and polar angle'}) so the histograms show the "
        f"continuous distribution rather than a comb at the nodes; quantiles come from the same linear model "
        f"along the first of those axes, means from the nodes."
    )
    return RenderedFigure(
        fig, caption,
        tables={"s": {"s_centre": centres_s, "g_s_per_electron": g_s},
                "rs": {"rs_centre_bohr": centres_rs, "g_rs_per_electron": g_rs}},
        arrays={"s_edges": s_edges, "rs_edges": rs_edges, "g_s": g_s, "g_rs": g_rs, "joint": joint},
        metrics=metrics,
    )


def _interacting(ctx: FigureContext) -> str | None:
    """Applicability: a self-consistent record with its potentials."""
    if ctx.record.density is None:
        return "the record holds no density"
    if not ctx.record.interacting:
        return "a bare-potential run has no Hartree or exchange-correlation potential"
    if ctx.record.v_hartree is None:
        return "the record stores no Hartree potential"
    if ctx.record.functional().rung.value > XCRung.GGA.value:
        return "meta-GGA potentials need tau, which the record does not store"
    return None


def _tail_regression(r: np.ndarray, v: np.ndarray) -> tuple[float, float] | None:
    """Least-squares ``-v ~ alpha r^-beta`` over the given samples (D1.7); ``None`` if ill-posed."""
    good = (v < 0.0) & np.isfinite(v) & (r > 0.0)
    if good.sum() < 5:
        return None
    slope, intercept = np.polyfit(np.log(r[good]), np.log(-v[good]), 1)
    return float(math.exp(intercept)), float(-slope)


def fig_potentials(ctx: FigureContext) -> RenderedFigure:
    """Kohn-Sham potential components along the axis; the v_xc tail against -1/r (D1.7)."""
    plt = pyplot()
    style = ctx.style
    record = ctx.record
    evaluator = ctx.evaluator
    prepare_hartree(evaluator)
    gga = record.functional().rung.value >= XCRung.GGA.value
    centre, u, _ = ctx.frame()
    reach = min(ctx.reach(centre, u), ctx.reach(centre, -u))
    b = np.linspace(-reach, reach, ctx.options.profile_points)
    points = evaluator.nudge(centre[None, :] + b[:, None] * u[None, :])
    fields = evaluator.density(points, order=2 if gga else 0)
    n = fields["n"]
    alive = fields["inside"] & (n > ctx.n_max * 1.0e-13)
    e_xc = np.full(n.size, np.nan)
    v_xc = np.full(n.size, np.nan)
    e_part, v_part = xc_at(record, n[alive], fields.get("grad", np.zeros((n.size, 3)))[alive] if gga else None,
                           fields.get("hess")[alive] if gga else None)
    e_xc[alive] = e_part
    v_xc[alive] = v_part
    v_h = hartree_at(evaluator, points)
    v_ext = external_at(record, points)
    v_s = v_ext + v_h + v_xc
    nuclei = [float((p - centre) @ u) for p in record.positions]
    near = np.zeros(b.size, dtype=bool)
    for position in nuclei:
        near |= np.abs(b - position) < 0.35
    fig = plt.figure(figsize=(style.double_width, 2.7))
    grid_spec = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 1.0], wspace=0.06)
    ax_all = fig.add_subplot(grid_spec[0])
    ax_tail = fig.add_subplot(grid_spec[1])
    ax_power = fig.add_subplot(grid_spec[2])
    ax_all.plot(b, np.where(near, np.nan, v_ext), color=INK_MUTED, linestyle="--", linewidth=style.thin_width,
                label=r"$v_{\rm ext}$")
    ax_all.plot(b, v_h, color=SERIES[1], label=r"$v_{\rm H}$")
    ax_all.plot(b, v_xc, color=SERIES[0], label=r"$v_{\rm xc}$")
    ax_all.plot(b, np.where(near, np.nan, v_s), color=INK, label=r"$v_s = v_{\rm ext}+v_{\rm H}+v_{\rm xc}$")
    homo = record.homo
    metrics: dict[str, Any] = {}
    if homo is not None:
        ax_all.axhline(homo, color=SERIES[2], linestyle=":", linewidth=style.line_width,
                       label=r"$\varepsilon_{\rm HOMO}$")
        metrics["eps_homo"] = homo
    exact = exact_ionisation_level(record, ctx.options.use_oracle)
    if exact is not None:
        ax_all.axhline(exact.energy, color=SERIES[3], linestyle="-.", linewidth=style.thin_width, label=r"$-I$ (exact)")
        metrics["minus_I_exact"] = exact.energy
    far = ~near & np.isfinite(v_s)
    lower = float(np.nanmin(v_s[far])) if np.any(far) else -2.0
    upper = float(np.nanmax(v_h)) if np.any(np.isfinite(v_h)) else 1.0
    ax_all.set_ylim(1.15 * min(lower, homo or 0.0, exact.energy if exact else 0.0), max(0.1, 1.15 * upper))
    ax_all.set_xlim(b[0], b[-1])
    ax_all.set_xlabel(ctx.axis_label("u"))
    ax_all.set_ylabel("potential (Ha)")
    _panel_label(ax_all, "a")

    outer = max(nuclei) if nuclei else 0.0
    tail = (b > outer + 1.0) & alive
    one_electron = abs(record.n_electrons - 1.0) < 1.0e-12
    r_tail = b[tail]
    exact_handles = []
    if r_tail.size > 5:
        ax_tail.loglog(r_tail, -v_xc[tail], color=SERIES[0])
        asymptote, = ax_tail.loglog(r_tail, 1.0 / r_tail, color=INK, linestyle=":", linewidth=style.line_width,
                                    label=r"exact asymptote ($-v_{\rm xc} \to 1/r$, $\beta = 1$)")
        exact_handles.append(asymptote)
        if one_electron:
            mirror, = ax_tail.loglog(r_tail, v_h[tail], color=SERIES[1], linestyle="--", linewidth=style.thin_width,
                                     label=r"exact for one electron: $-v_{\rm xc}[n] = v_{\rm H}[n]$")
            exact_handles.append(mirror)
        log_r = np.log(r_tail)
        with np.errstate(divide="ignore", invalid="ignore"):
            power = -np.gradient(np.log(-v_xc[tail]), log_r)
        ax_power.plot(r_tail, power, color=SERIES[0])
        if one_electron:
            with np.errstate(divide="ignore", invalid="ignore"):
                power_h = -np.gradient(np.log(v_h[tail]), log_r)
            ax_power.plot(r_tail, power_h, color=SERIES[1], linestyle="--", linewidth=style.thin_width)
        ax_power.axhline(1.0, color=INK, linestyle=":", linewidth=style.line_width)
        # The wall goes into the shared legend; a label beside it is crossed by the tail.
        wall = ax_tail.axvline(reach, color=INK_MUTED, linewidth=style.thin_width,
                               label="box face (Dirichlet wall)")
        ax_power.axvline(reach, color=INK_MUTED, linewidth=style.thin_width)
        exact_handles.append(wall)
        window = (r_tail > outer + 2.0) & (r_tail < 0.8 * r_tail.max())
        fitted = _tail_regression(r_tail[window], v_xc[tail][window]) if np.any(window) else None
        if fitted is not None:
            metrics["D1.7_alpha"], metrics["D1.7_beta"] = fitted
            metrics["D1.7_window_bohr"] = [float(r_tail[window].min()), float(r_tail[window].max())]
        metrics["D1.7_local_beta_at_window_end"] = float(power[window][-1]) if np.any(window) else None
        finite = power[np.isfinite(power)]
        ax_power.set_ylim(0.0, max(4.0, float(np.nanpercentile(finite, 90)) * 1.2) if finite.size else 4.0)
    else:
        ax_tail.text(0.5, 0.5, "no tail inside the box", transform=ax_tail.transAxes, ha="center", color=INK_MUTED)
    ax_tail.set_xlabel(r"$r$ (bohr)")
    ax_tail.set_ylabel(r"$-v_{\rm xc}$ (Ha)")
    if r_tail.size > 5:
        _plain_log_axis(ax_tail, "x")
    _panel_label(ax_tail, "b")
    ax_power.set_xlabel(r"$r$ (bohr)")
    ax_power.set_ylabel(r"local power $\beta = -d\ln(-v)/d\ln r$")
    _panel_label(ax_power, "c")
    # One legend below the three panels: the colours mean the same thing in each.
    handles, labels = ax_all.get_legend_handles_labels()
    fig.legend(handles + exact_handles, labels + [h.get_label() for h in exact_handles],
               loc="outside lower center", ncol=4, fontsize=6.0)
    _badge(fig, ctx.verdict)
    d17 = (f" A power law -v_xc ~ alpha r^-beta over {metrics['D1.7_window_bohr'][0]:.2f}-"
           f"{metrics['D1.7_window_bohr'][1]:.2f} bohr gives alpha = {metrics['D1.7_alpha']:.3g}, "
           f"beta = {metrics['D1.7_beta']:.3g} (diagnostic D1.7, recorded, never thresholded; exact: 1, 1)."
           if "D1.7_beta" in metrics else "")
    hartree_text = ("v_H is the analytic cusp part of the D-54 split plus the interpolated remainder"
                    if getattr(evaluator, "_hartree_split", []) else "v_H is the interpolated record field")
    caption = (
        f"Kohn-Sham potential of {ctx.system_label()} along the principal axis. (a) Components"
        + ("; v_ext and v_s are not drawn within 0.35 bohr of a nucleus" if nuclei else "")
        + "; dotted: eps_HOMO"
        + (", dash-dotted: -I of the exact functional (the exact eps_HOMO)" if exact is not None else "")
        + f". {hartree_text}; v_xc is evaluated pointwise from the "
        f"{'cusp-factorised ' if record.cusp_factorised else ''}density"
        f"{' with the GGA divergence taken analytically' if gga else ''}. "
        f"(b) The tail of -v_xc on log-log axes against the exact -1/r asymptote [A31]"
        + ("; for one electron the exact functional gives v_xc[n] = -v_H[n] on any density [A9], drawn dashed"
           if one_electron else "")
        + f". (c) Local power -d ln(-v)/d ln r: 1 for the exact potential, growing without bound for a semi-local "
        f"one whose potential decays exponentially. In (b) and (c) r is the distance from the centre along the "
        f"axis, up to the box face, where the Dirichlet wall takes the density (and a semi-local v_xc) to "
        f"zero.{d17}"
    )
    columns = {"b_bohr": b, "v_ext": v_ext, "v_hartree": v_h, "v_xc": v_xc, "v_s": v_s, "density": n, "e_xc": e_xc}
    return RenderedFigure(fig, caption, columns=columns, metrics=metrics)


def _one_electron_interacting(ctx: FigureContext) -> str | None:
    """Applicability: an interacting record with exactly one electron (D1.1)."""
    reason = _interacting(ctx)
    if reason is not None:
        return reason
    if abs(ctx.record.n_electrons - 1.0) > 1.0e-12:
        return "D1.1 is defined for exactly one electron (E_H + E_xc = 0 for the exact functional)"
    return None


def fig_self_interaction(ctx: FigureContext) -> RenderedFigure:
    """Draw the one-electron self-interaction integrand n v_H / 2 + e_xc (D1.1) and its cumulative integral."""
    from matplotlib.colors import SymLogNorm

    plt = pyplot()
    style = ctx.style
    record = ctx.record
    evaluator = ctx.evaluator
    prepare_hartree(evaluator)
    gga = record.functional().rung.value >= XCRung.GGA.value

    def integrand(points: np.ndarray, fields: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = fields["n"]
        alive = fields["inside"] & (n > 0.0)
        e_xc = np.zeros(n.size)
        e_part, _ = xc_at(record, n[alive], fields["grad"][alive] if gga else None, None, with_potential=False)
        e_xc[alive] = e_part
        v_h = np.nan_to_num(hartree_at(evaluator, points), nan=0.0)
        hartree_density = np.where(alive, 0.5 * n * v_h, 0.0)
        return hartree_density + e_xc, hartree_density, e_xc

    a, b, plane_points = ctx.plane()
    plane_fields = evaluator.density(plane_points, order=1 if gga else 0)
    si, _, _ = integrand(plane_points, plane_fields)
    si = si.reshape(a.size, b.size)
    limit = float(np.percentile(np.abs(si), 99.9)) or 1.0
    grid = ctx.grid
    grid_fields = ctx.grid_fields(order=1)
    grid_si, grid_h, grid_xc = integrand(grid.points, grid_fields)
    centre, _, _ = ctx.frame()
    distance = np.linalg.norm(grid.points - centre, axis=1)
    order = np.argsort(distance)
    radius = distance[order]
    cumulative_si = np.cumsum((grid.weights * grid_si)[order])
    cumulative_h = np.cumsum((grid.weights * grid_h)[order])
    cumulative_xc = np.cumsum((grid.weights * grid_xc)[order])
    recorded = record.energies.get("hartree", float("nan")) + record.energies.get("xc", float("nan"))
    stride = max(1, radius.size // 3000)
    fig, (ax_map, ax_cum) = plt.subplots(1, 2, figsize=(style.double_width, 2.9),
                                         gridspec_kw={"width_ratios": [1.05, 1.0], "wspace": 0.06})
    image = ax_map.imshow(si, origin="lower", extent=(b[0], b[-1], a[0], a[-1]), cmap=colormap("diverging"),
                          norm=SymLogNorm(linthresh=limit * 1.0e-3, linscale=0.6, vmin=-limit, vmax=limit),
                          interpolation="antialiased", aspect="equal")
    ax_map.grid(False)
    _mark_nuclei(ax_map, ctx)
    ax_map.set_xlabel(ctx.axis_label("u"))
    ax_map.set_ylabel(ctx.axis_label("v"))
    bar = fig.colorbar(image, ax=ax_map, fraction=0.046, pad=0.03)
    bar.set_label(r"$\frac{1}{2} n v_{\rm H} + e_{\rm xc}$ (Ha bohr$^{-3}$)")
    bar.outline.set_linewidth(style.hairline)
    # Decades outside the linear band, plus zero: the +-linthresh ticks would print on top of zero.
    linthresh = limit * 1.0e-3
    decades = range(math.floor(math.log10(linthresh)) + 1, math.floor(math.log10(limit)) + 1)
    positive = [10.0**k for k in decades]
    if len(positive) > 3:
        positive = positive[::2]
    ticks = [-t for t in reversed(positive)] + [0.0] + positive
    labels = ([f"$-10^{{{round(math.log10(t))}}}$" for t in reversed(positive)] + ["0"]
              + [f"$10^{{{round(math.log10(t))}}}$" for t in positive])
    bar.set_ticks(ticks, labels=labels)
    bar.ax.minorticks_off()
    _panel_label(ax_map, "a")
    ax_cum.plot(radius[::stride], cumulative_h[::stride], color=SERIES[1], label=r"$E_{\rm H}(<r)$")
    ax_cum.plot(radius[::stride], cumulative_xc[::stride], color=SERIES[0], label=r"$E_{\rm xc}(<r)$")
    ax_cum.plot(radius[::stride], cumulative_si[::stride], color=INK, label=r"$E_{\rm H}+E_{\rm xc}$ ($<r$)")
    ax_cum.axhline(0.0, color=INK_MUTED, linewidth=style.thin_width)
    ax_cum.annotate("exact functional: 0", (0.98, 0.0), xycoords=("axes fraction", "data"), xytext=(0, 3),
                    textcoords="offset points", ha="right", fontsize=6.0, color=INK_MUTED)
    if math.isfinite(recorded):
        ax_cum.axhline(recorded, color=INK, linestyle="--", linewidth=style.thin_width,
                       label=f"solver quadrature: {recorded:.6f} Ha")
    ax_cum.set_xlim(0.0, float(np.percentile(radius[grid_fields['inside'][order]], 99.5)))
    ax_cum.set_xlabel("distance from the centre (bohr)")
    ax_cum.set_ylabel("cumulative energy (Ha)")
    # The legend sits in the empty band between the E_H plateau and the E_H + E_xc plateau.
    low, high = ax_cum.get_ylim()
    band = 0.5 * (float(cumulative_h[-1]) + float(cumulative_si[-1]))
    anchor = min(0.95, max(0.05, (band - low) / (high - low)))
    ax_cum.legend(loc="center right", bbox_to_anchor=(1.0, anchor), fontsize=6.0)
    _panel_label(ax_cum, "b")
    _badge(fig, ctx.verdict)
    _shrink_to_aspect(fig, ax_map)
    inside = grid_fields["inside"]
    metrics = {
        "D1.1_EH_plus_Exc_recorded": recorded,
        "D1.1_EH_plus_Exc_figure_quadrature": float((grid.weights * grid_si)[inside].sum()),
        "EH_figure_quadrature": float((grid.weights * grid_h)[inside].sum()),
        "Exc_figure_quadrature": float((grid.weights * grid_xc)[inside].sum()),
        "EH_recorded": record.energies.get("hartree"),
        "Exc_recorded": record.energies.get("xc"),
    }
    caption = (
        f"One-electron self-interaction of {ctx.system_label()} (diagnostic D1.1). (a) The local integrand "
        f"n v_H / 2 + e_xc on the plane through the nuclei (symmetric-log colour scale, linear within "
        f"{limit * 1.0e-3:.1e}); for the exact functional its integral vanishes for every one-electron density "
        f"(Perdew-Zunger [A9]). (b) Its integral over the ball of radius r about the centre, with the Hartree "
        f"and exchange-correlation parts: E_H + E_xc = {_fmt(recorded)} Ha on the solver's "
        f"{'cusp-aware quadrature (D-54)' if record.cusp_factorised else 'quadrature'} and "
        f"{_fmt(metrics['D1.1_EH_plus_Exc_figure_quadrature'])} Ha on the figure's own quadrature "
        f"({grid.description()}); the exact value is 0. Recorded, never thresholded (D-20)."
    )
    return RenderedFigure(fig, caption, columns={"r_bohr": radius[::stride], "cumulative_EH": cumulative_h[::stride],
                                                 "cumulative_Exc": cumulative_xc[::stride],
                                                 "cumulative_sum": cumulative_si[::stride]},
                          arrays={"a_bohr": a, "b_bohr": b, "self_interaction_density": si}, metrics=metrics)


def _has_spectrum(ctx: FigureContext) -> str | None:
    """Applicability: eigenvalues were recorded."""
    return None if ctx.record.eigenvalues.size else "the record holds no eigenvalues"


def _clusters(eigenvalues: np.ndarray, occupations: np.ndarray, tolerance: float = 1.0e-6):
    """Group eigenvalues into degenerate levels: ``[(energy, degeneracy, occupation)]``."""
    out: list[list[float]] = []
    for energy, occupation in sorted(zip(eigenvalues, occupations, strict=True)):
        if out and abs(energy - out[-1][0]) <= tolerance * max(1.0, abs(energy)):
            count = out[-1][1]
            out[-1][0] = (out[-1][0] * count + energy) / (count + 1)
            out[-1][1] += 1
            out[-1][2] += occupation
        else:
            out.append([float(energy), 1, float(occupation)])
    return [(e, int(g), o) for e, g, o in out]


def fig_spectrum(ctx: FigureContext) -> RenderedFigure:
    """Eigenvalues and occupations against reference levels and the exact -I (D1.6)."""
    plt = pyplot()
    style = ctx.style
    record = ctx.record
    levels = _clusters(record.eigenvalues, record.occupations)
    columns_ref = spectrum_references(record, ctx.options.use_oracle)
    exact = exact_ionisation_level(record, ctx.options.use_oracle)
    titles = ["this work"] + [title for title, _ in columns_ref] + (["exact functional"] if exact else [])
    fig, ax = plt.subplots(figsize=(style.single_width, 2.9))
    half = 0.32
    energies_all = [e for e, _, _ in levels]
    for energy, degeneracy, occupation in levels:
        occupied = occupation > 1.0e-12
        ax.hlines(energy, -half, half, color=SERIES[0], linewidth=1.8 if occupied else 0.8)
        if occupied:
            ax.plot([-half - 0.07], [energy], marker="o", markersize=2.8, color=SERIES[0], linestyle="none")
            if abs(occupation - round(occupation)) > 1.0e-9 or degeneracy > 1:
                ax.annotate(f"f = {occupation:g}", (-half - 0.1, energy), xytext=(-2, 0), textcoords="offset points",
                            ha="right", va="center", fontsize=6.0, color=INK_SECONDARY)
        if degeneracy > 1:
            ax.annotate(rf"$\times${degeneracy}", (half, energy), xytext=(2, 0), textcoords="offset points",
                        va="center", fontsize=6.0, color=INK_SECONDARY)
    homo = record.homo
    lines = []
    for column, (title, reference_levels) in enumerate(columns_ref, start=1):
        for level in reference_levels:
            ax.hlines(level.energy, column - half, column + half, color=INK_SECONDARY, linewidth=1.0)
            energies_all.append(level.energy)
            if level.degeneracy > 1:
                ax.annotate(rf"$\times${level.degeneracy}", (column + half, level.energy), xytext=(2, 0),
                            textcoords="offset points", va="center", fontsize=6.0, color=INK_SECONDARY)
        if homo is not None and reference_levels:
            closest = min(reference_levels, key=lambda lv: abs(lv.energy - homo))
            lines.append(f"{title}: {homo - closest.energy:+.2e} Ha")
    if exact is not None:
        column = len(titles) - 1
        ax.hlines(exact.energy, column - half, column + half, color=SERIES[1], linewidth=1.8)
        energies_all.append(exact.energy)
        if homo is not None:
            lines.append(f"-I (exact): {homo - exact.energy:+.3f} Ha")
    span = max(energies_all) - min(energies_all) if len(energies_all) > 1 else 1.0
    pad = max(0.08 * span, 0.02)
    ax.set_ylim(min(energies_all) - pad - (0.35 * span if lines else 0.0), max(energies_all) + pad)
    ax.set_xlim(-0.8, len(titles) - 0.4)
    ax.set_xticks(range(len(titles)))
    # With more than three columns the titles break at their first space to fit a single column.
    ax.set_xticklabels([title.replace(" ", "\n", 1) if len(titles) > 3 else title for title in titles],
                       fontsize=6.5)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("eigenvalue (Ha)")
    if lines:
        ax.text(0.02, 0.03, r"$\varepsilon_{\rm HOMO}$ $-$ reference" + "\n" + "\n".join(lines),
                transform=ax.transAxes, fontsize=6.0, color=INK, va="bottom", ha="left")
    _badge(fig, ctx.verdict)
    compared = [f"{title} ({', '.join(sorted({lv.source for lv in lvs}))})" for title, lvs in columns_ref]
    if exact is not None:
        compared.append(f"-I, the HOMO eigenvalue of the exact functional (ionisation-potential theorem [A30], "
                        f"[A31]; fractional N via piecewise linearity [A21]), {exact.source}")
    caption = (
        f"{'Kohn-Sham eigenvalues' if record.interacting else 'Eigenvalues of the one-body Hamiltonian'} of "
        f"{ctx.system_label()} (thick: occupied, with occupation f where fractional or degenerate; x g: degeneracy)"
        + (" against " + "; ".join(compared) if compared else "; no reference levels are available for this record")
        + (". The difference eps_HOMO + I is the Koopmans-type deviation of diagnostic D1.6."
           if exact is not None else ".")
    )
    metrics = {"eps_homo": homo, "levels": [[e, g, o] for e, g, o in levels]}
    if exact is not None and homo is not None:
        metrics["D1.6_eps_homo_plus_I"] = homo - exact.energy
    energies = np.array([e for e, _, _ in levels])
    return RenderedFigure(fig, caption, columns={"energy_ha": energies,
                                                 "degeneracy": np.array([g for _, g, _ in levels], dtype=float),
                                                 "occupation": np.array([o for _, _, o in levels])},
                          metrics=metrics)


def _has_scf(ctx: FigureContext) -> str | None:
    """Applicability: an SCF trajectory with more than one iteration."""
    if not ctx.record.interacting:
        return "a bare-potential run has no self-consistent field iteration"
    if len(ctx.record.trajectory.get("energies", [])) < 2:
        return "the record's SCF trajectory has fewer than two iterations"
    return None


def fig_scf_convergence(ctx: FigureContext) -> RenderedFigure:
    """Energy change, energy error, Harris-Foulkes gap and density residual per SCF iteration."""
    plt = pyplot()
    style = ctx.style
    record = ctx.record
    trajectory = record.trajectory
    energies = np.asarray(trajectory["energies"], dtype=np.float64)
    iterations = np.arange(1, energies.size + 1)
    final = record.energies.get("total", energies[-1])
    harris = np.asarray(record.measurements.get("harris_foulkes_history", []), dtype=np.float64)
    residual = np.asarray(trajectory.get("residual_norms", []), dtype=np.float64)
    step = np.asarray(trajectory.get("density_changes", []), dtype=np.float64)
    fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(style.single_width, 3.6),
                                      gridspec_kw={"hspace": 0.0})
    tiny = 1.0e-17
    change = np.abs(np.diff(energies))
    top.semilogy(iterations[1:], np.maximum(change, tiny), color=SERIES[0], marker="o", markersize=2.2,
                 label=r"$|E_k - E_{k-1}|$")
    top.semilogy(iterations, np.maximum(np.abs(energies - final), tiny), color=SERIES[1], linewidth=style.thin_width,
                 label=r"$|E_k - E_{\rm final}|$")
    if harris.size == energies.size:
        top.semilogy(iterations, np.maximum(np.abs(energies - harris), tiny), color=SERIES[2],
                     linewidth=style.thin_width, linestyle="--", label=r"$|E_k - E^{\rm HF}_k|$ (Harris-Foulkes)")
    scf = record.numerics.scf
    top.axhline(scf.energy_tol, color=INK_MUTED, linestyle="--", linewidth=style.thin_width)
    top.annotate(f"energy tolerance {scf.energy_tol:g}", (1.0, scf.energy_tol), xycoords=("axes fraction", "data"),
                 xytext=(-2, 2), textcoords="offset points", ha="right", fontsize=6.0, color=INK_MUTED)
    top.set_ylabel("energy (Ha)")
    top.legend(loc="upper right", fontsize=6.0)
    _panel_label(top, "a")
    if residual.size:
        bottom.semilogy(np.arange(1, residual.size + 1), np.maximum(residual, tiny), color=SERIES[0], marker="o",
                        markersize=2.2, label=r"residual $\|n_{\rm out} - n_{\rm in}\|_1$")
    if step.size:
        bottom.semilogy(np.arange(1, step.size + 1), np.maximum(step, tiny), color=SERIES[1],
                        linewidth=style.thin_width, label=r"mixed step $\|n_{k+1} - n_k\|_1$")
    bottom.axhline(scf.density_tol, color=INK_MUTED, linestyle="--", linewidth=style.thin_width)
    bottom.annotate(f"density tolerance {scf.density_tol:g} e", (1.0, scf.density_tol),
                    xycoords=("axes fraction", "data"), xytext=(-2, 2), textcoords="offset points", ha="right",
                    fontsize=6.0, color=INK_MUTED)
    events = []
    for event in trajectory.get("fallback_events", []):
        match = re.match(r"iteration (\d+)", str(event))
        if match:
            k = int(match.group(1))
            events.append(str(event))
            for ax in (top, bottom):
                ax.axvline(k, color=STATUS_WARNING, linewidth=style.thin_width)
            bottom.annotate(("" if _usetex() else "▲ ") + "fallback", (k, 0.02),
                            xycoords=("data", "axes fraction"), rotation=90,
                            fontsize=6.0, color=INK, ha="right", va="bottom")
    bottom.set_xlabel("SCF iteration")
    bottom.set_ylabel("density (electrons)")
    bottom.legend(loc="upper right", fontsize=6.0)
    _panel_label(bottom, "b")
    _badge(fig, ctx.verdict)
    stop = record.measurements.get("scf_stop_reason", "")
    caption = (
        f"Self-consistent field history of {ctx.system_label()}: {energies.size} iterations "
        f"({'converged' if record.converged else 'NOT converged'}; {stop}). (a) Energy change between iterations, "
        f"distance to the final energy (from the closing full diagonalisation) and the Harris-Foulkes gap, which is "
        f"second order in the density error [F22]. (b) The L1 density residual"
        f"{' in the cusp-aware measure (D-54)' if record.cusp_factorised else ''} and the "
        f"step the mixer took ({record.measurements.get('mixer', 'Pulay')} [F8], [F9]). Convergence requires both "
        f"tolerances (gate G3.3); fallback-ladder steps [F12] are marked."
        + (f" Fallbacks: {'; '.join(events)}." if events else "")
    )
    columns = {"iteration": iterations, "energy_ha": energies}
    if harris.size == energies.size:
        columns["harris_foulkes_ha"] = harris
    if residual.size == energies.size:
        columns["residual_l1"] = residual
    if step.size == energies.size:
        columns["mixed_step_l1"] = step
    return RenderedFigure(fig, caption, columns=columns,
                          metrics={"scf_iterations": int(energies.size), "scf_converged": bool(record.converged)})


# --- Registry, output, tables, manifest ----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FigureSpec:
    """One figure: when it applies, how it is drawn, what it cites."""

    name: str
    applies: Callable[[FigureContext], str | None]
    render: Callable[[FigureContext], RenderedFigure]
    citations: tuple[str, ...]
    summary: str


#: Every figure, in drawing order. Appending a spec is all it takes to add one.
FIGURES: tuple[FigureSpec, ...] = (
    FigureSpec("density_map", _needs_density, fig_density_map, ("A1", "D-35"),
               "log10 n on the plane through the nuclei"),
    FigureSpec("density_profile", _needs_density, fig_density_profile, ("A29", "A30", "A31"),
               "n and d ln n/dz along the principal axis, with the Kato and asymptotic limits"),
    FigureSpec("radial_density", _single_centre, fig_radial_density, ("D-42", "D-53"),
               "4 pi r^2 <n>(r) and the charge between r and the inscribed sphere"),
    FigureSpec("cusp_and_decay", _has_nuclei, fig_cusp_and_decay, ("A29", "A30", "A31"),
               "Kato ratio and local decay exponent"),
    FigureSpec("density_error", _has_reference, fig_density_error, ("B11", "B7", "F25"),
               "n, |grad n| and lap n against the reference; int |n - n_ref|"),
    FigureSpec("reduced_gradient", _needs_density, fig_reduced_gradient, ("A32", "A33", "A4", "F25"),
               "density-weighted s and r_s distributions, PBE F_x over the sampled band"),
    FigureSpec("potentials", _interacting, fig_potentials, ("A31", "A9", "A16"),
               "v_ext, v_H, v_xc, v_s; the v_xc tail (D1.7)"),
    FigureSpec("self_interaction", _one_electron_interacting, fig_self_interaction, ("A9", "F25"),
               "n v_H / 2 + e_xc and its integral (D1.1)"),
    FigureSpec("spectrum", _has_spectrum, fig_spectrum, ("A21", "A30", "A31", "B10"),
               "eigenvalues against reference levels and -I (D1.6)"),
    FigureSpec("scf_convergence", _has_scf, fig_scf_convergence, ("F8", "F9", "F12", "F22"),
               "SCF energy and density residual histories"),
)


def _file_metadata(record: FigureRecord, spec: FigureSpec, caption: str, fmt: str) -> dict[str, Any]:
    """Provenance for the file's own metadata (PDF info dictionary, PNG text chunks, SVG Dublin Core)."""
    import cdft

    title = f"{spec.name} -- {record.scenario.scenario_id} ({record.run_id})"
    creator = f"cdft {cdft.__version__} figures.py {FIGURES_VERSION} (contract {CONTRACT_VERSION})"
    line = record.provenance_line()
    if fmt == "pdf":
        return {"Title": title, "Author": "cdft figures.py", "Subject": caption[:1500], "Keywords": line,
                "Creator": creator, "CreationDate": None}
    if fmt == "png":
        return {"Title": title, "Author": "cdft figures.py", "Description": caption, "Software": creator,
                "Source": line, "cdft-run-id": record.run_id, "cdft-scenario": record.scenario.scenario_id,
                "cdft-status": record.status}
    if fmt == "svg":
        return {"Title": title, "Creator": creator, "Description": caption, "Identifier": record.run_id,
                "Source": line, "Date": None}
    return {}


#: The coordinate columns of the tables and arrays, as their headers explain them.
COLUMN_GLOSSARY: dict[str, str] = {
    "b_bohr": "b = signed position along the plotted axis, from the centre named in the caption",
    "a_bohr": "a = in-plane perpendicular coordinate, from the same centre",
    "r_bohr": "r = distance from the centre or nucleus named in the caption",
}


def _glossary(names: Sequence[str]) -> str:
    """Return the sentence explaining the coordinate columns among ``names`` (or "")."""
    notes = [COLUMN_GLOSSARY[name] for name in names if name in COLUMN_GLOSSARY]
    return f"Columns: {'; '.join(notes)}. " if notes else ""


def _write_table(path: pathlib.Path, columns: Mapping[str, np.ndarray], header: str) -> None:
    """Write equal-length columns as CSV with a commented provenance header."""
    names = list(columns)
    if not names:
        raise ValueError(f"table {path.name}: no columns")
    lengths = {name: len(np.atleast_1d(columns[name])) for name in names}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"table {path.name}: columns of unequal length {lengths}")
    data = np.column_stack([np.asarray(columns[name], dtype=np.float64) for name in names])
    # UTF-8 and LF everywhere, so a table's hash in the manifest does not depend on the OS.
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        np.savetxt(handle, data, delimiter=",", header=header + "\n" + ",".join(names), comments="# ",
                   fmt="%.12e")


def _sha256(path: pathlib.Path) -> str:
    """Hex digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_rendered(rendered: RenderedFigure, spec: FigureSpec, record: FigureRecord, outdir: pathlib.Path,
                  style: HouseStyle) -> dict[str, Any]:
    """Write a figure in every requested format, its caption, its table view; return its manifest entry."""
    plt = pyplot()
    files: dict[str, str] = {}
    for fmt in style.formats:
        path = outdir / f"{spec.name}.{fmt}"
        rendered.figure.savefig(path, format=fmt, metadata=_file_metadata(record, spec, rendered.caption, fmt))
        files[fmt] = path.name
    plt.close(rendered.figure)
    line = record.provenance_line()
    caption_path = outdir / f"{spec.name}.caption.txt"
    sources = ", ".join(c if c.startswith("D-") else f"[{c}]" for c in spec.citations)
    caption_path.write_text(
        f"{rendered.caption}\n\nSources: {sources} "
        f"(docs/00_LITERATURE_SURVEY.md; D-entries in docs/05_DECISION_LOG.md).\n"
        f"{gate_summary_line(record.gates)}\nProvenance: {line}\n",
        encoding="utf-8",
        newline="\n",
    )
    files["caption"] = caption_path.name
    if rendered.columns:
        table_path = outdir / f"{spec.name}.csv"
        header = f"{spec.name}: table view of the figure. {_glossary(list(rendered.columns))}"
        _write_table(table_path, rendered.columns, header + line)
        files["table"] = table_path.name
    for key, table in rendered.tables.items():
        if not re.fullmatch(r"[A-Za-z0-9_]+", key):
            raise ValueError(f"table key {key!r} of {spec.name} is not a plain identifier")
        table_path = outdir / f"{spec.name}_{key}.csv"
        header = f"{spec.name} ({key}): table view of one panel. {_glossary(list(table))}"
        _write_table(table_path, table, header + line)
        files[f"table_{key}"] = table_path.name
    if rendered.arrays:
        array_path = outdir / f"{spec.name}.npz"
        extras = {"provenance": np.array(line)}
        glossary = _glossary(list(rendered.arrays))
        if glossary:
            extras["glossary"] = np.array(glossary.strip())
        np.savez_compressed(array_path, **extras, **rendered.arrays)
        files["arrays"] = array_path.name
    return {
        "name": spec.name,
        "summary": spec.summary,
        "files": files,
        "sha256": {kind: _sha256(outdir / name) for kind, name in files.items()},
        "caption": rendered.caption,
        "citations": list(spec.citations),
        "metrics": _jsonable(rendered.metrics),
    }


def _jsonable(value: Any) -> Any:
    """Plain JSON types for the manifest (numpy scalars and arrays included)."""
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else str(number)
    if isinstance(value, (np.integer, int, bool, np.bool_)):
        return value.item() if hasattr(value, "item") else value
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, pathlib.Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if hasattr(value, "value"):
        return _jsonable(value.value)
    return str(value)


def _tex_number(value: float | None, digits: int = 8) -> str:
    """Return a number for a LaTeX table cell (``--`` when absent)."""
    if value is None or not math.isfinite(value):
        return "--"
    if value != 0.0 and (abs(value) < 1.0e-3 or abs(value) >= 1.0e5):
        mantissa, exponent = f"{value:.2e}".split("e")
        return f"${mantissa}\\times10^{{{int(exponent)}}}$"
    return f"${value:.{digits}f}$"


_TEX_SPECIALS = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
    "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}",
}
_TEX_PATTERN = re.compile("|".join(re.escape(char) for char in _TEX_SPECIALS))


def _tex_escape(text: str) -> str:
    """Escape the LaTeX specials in gate names and ids (one pass, so no double escaping)."""
    return _TEX_PATTERN.sub(lambda match: _TEX_SPECIALS[match.group(0)], text)


def energy_table(record: FigureRecord, use_oracle: bool) -> tuple[str, dict[str, Any]]:
    """Return the energy decomposition against every available reference, as a booktabs table."""
    terms = (("kinetic", r"$T_s$"), ("external", r"$E_{\rm ext}$"), ("hartree", r"$E_{\rm H}$"),
             ("xc", r"$E_{\rm xc}$"), ("ion_ion", r"$E_{\rm ii}$"), ("total", r"$E$"),
             ("harris_foulkes", r"$E^{\rm HF}$"))
    references = energy_references(record, use_oracle)
    header = " & ".join(["term", "this work"] + [f"{_tex_escape(title)} & $\\Delta$" for title, _, _ in references])
    lines = [
        r"% generated by figures.py " + FIGURES_VERSION + " -- " + _tex_escape(record.provenance_line()),
        r"\begin{tabular}{l" + "r" * (1 + 2 * len(references)) + "}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
    ]
    deltas: dict[str, Any] = {}
    for key, label in terms:
        value = record.energies.get(key)
        cells = [label, _tex_number(value)]
        for title, values, _ in references:
            ref = values.get(key)
            cells.append(_tex_number(ref))
            delta = None if ref is None or value is None or not math.isfinite(value) else value - ref
            cells.append(_tex_number(delta))
            if delta is not None:
                deltas[f"{title}:{key}"] = delta
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    notes = "; ".join(f"{title}: {source}" for title, _, source in references) or "no reference available"
    lines.append(f"% references: {_tex_escape(notes)}. E_ext is the band-energy complement "
                 f"({_tex_escape(str(record.measurements.get('external_method', 'band_energy_complement')))}).")
    return "\n".join(lines) + "\n", deltas


def gate_table(record: FigureRecord, include_skipped: bool = False) -> str:
    """Return the record's gate report as a booktabs table (skipped rows counted, not listed, by default)."""
    lines = [
        r"% generated by figures.py " + FIGURES_VERSION + " -- " + _tex_escape(record.provenance_line()),
        r"\begin{tabular}{llrrl}",
        r"\toprule",
        r"gate & verdict & measured & threshold & name \\",
        r"\midrule",
    ]
    skipped = 0
    for result in record.gates:
        if result.verdict is GateVerdict.SKIPPED and not include_skipped:
            skipped += 1
            continue
        measured = "--" if result.verdict is GateVerdict.SKIPPED else _tex_number(result.measured, 3)
        lines.append(
            f"{_tex_escape(result.gate_id)} & {_tex_escape(result.verdict.value)} & {measured} & "
            f"{_tex_number(result.threshold, 3)} & {_tex_escape(result.name)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    if skipped:
        lines.append(f"% {skipped} skipped gate(s) omitted; their reasons are in manifest.json and record.h5")
    return "\n".join(lines) + "\n"


def render_figures(
    record: FigureRecord,
    verdict: Verdict,
    outdir: pathlib.Path,
    options: FigureOptions | None = None,
    *,
    only: Sequence[str] | None = None,
    skip: Sequence[str] = (),
    log: Callable[[str], None] = print,
    invocation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Draw every applicable figure into ``outdir`` and write the manifest; failures are recorded."""
    options = options or FigureOptions()
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plt = pyplot()
    known = {spec.name for spec in FIGURES}
    unknown = [name for name in list(only or []) + list(skip) if name not in known]
    if unknown:
        raise ValueError(f"unknown figure(s) {', '.join(unknown)}; choose from {', '.join(sorted(known))}")
    manifest: dict[str, Any] = {
        "figures_py_version": FIGURES_VERSION,
        "contract_version": CONTRACT_VERSION,
        "generated_utc": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_id": record.run_id,
        "scenario_id": record.scenario.scenario_id,
        "system": None,
        "source": record.source,
        "status": record.status,
        "verdict": {"drawable": verdict.drawable, "mark": verdict.mark, "new_failures": verdict.new_failures,
                    "known_open": verdict.known_open, "failed_gates": list(verdict.failed_gates),
                    "known_open_gates": list(verdict.known_gates),
                    "reason": verdict.reason},
        "provenance": _jsonable(record.provenance),
        "grid": {"shape": list(record.shape), "spacing_bohr": record.spacing, "origin_bohr": list(record.origin),
                 "boundary": record.boundary},
        "energies": _jsonable(record.energies),
        "gates": _jsonable([{"gate_id": g.gate_id, "verdict": g.verdict.value, "measured": g.measured,
                             "threshold": g.threshold, "reason": g.reason} for g in record.gates]),
        "gate_summary": gate_summary(record.gates),
        "invocation": _jsonable(dict(invocation or {})),
        "style": _jsonable(options.style),
        "figures": [],
        "not_drawn": [],
        "errors": [],
    }
    if not verdict.drawable:
        requested = [spec for spec in FIGURES if not ((only and spec.name not in only) or spec.name in skip)]
        manifest["not_drawn"] = [
            {"name": spec.name, "reason": f"record not drawable: {verdict.reason}"} for spec in requested
        ]
        manifest["drawn_this_call"] = []
        _write_manifest(outdir, manifest, touched={spec.name for spec in requested}, refused=True)
        return manifest
    ctx = FigureContext(record, verdict, options)
    manifest["system"] = ctx.system_label()
    touched: set[str] = set()
    drawn_now: list[str] = []
    with plt.rc_context(options.style.rc()):
        for spec in FIGURES:
            if (only and spec.name not in only) or spec.name in skip:
                continue
            touched.add(spec.name)
            try:
                reason = spec.applies(ctx)
            except Exception as exc:  # noqa: BLE001 - an applicability check must not stop the set
                reason = f"applicability check failed: {type(exc).__name__}: {exc}"
            if reason is not None:
                manifest["not_drawn"].append({"name": spec.name, "reason": reason})
                log(f"  -  {spec.name:<18} not drawn: {reason}")
                continue
            started = time.perf_counter()
            try:
                rendered = spec.render(ctx)
                entry = save_rendered(rendered, spec, record, outdir, options.style)
            except Exception as exc:  # noqa: BLE001 - one broken figure must not lose the others
                import traceback

                plt.close("all")
                manifest["errors"].append({"name": spec.name, "error": f"{type(exc).__name__}: {exc}",
                                           "traceback": traceback.format_exc()})
                log(f"  !  {spec.name:<18} FAILED: {type(exc).__name__}: {exc}")
                continue
            entry["seconds"] = round(time.perf_counter() - started, 2)
            manifest["figures"].append(entry)
            drawn_now.append(spec.name)
            written = ", ".join(entry["files"][k] for k in options.style.formats)
            log(f"  +  {spec.name:<18} {written}  ({entry['seconds']} s)")
    if record.eigenvalues.size or record.energies:
        table, deltas = energy_table(record, options.use_oracle)
        (outdir / "energies.tex").write_text(table, encoding="utf-8", newline="\n")
        manifest["tables"] = {"energies": "energies.tex", "gates": "gates.tex"}
        manifest["energy_deltas"] = _jsonable(deltas)
    (outdir / "gates.tex").write_text(gate_table(record), encoding="utf-8", newline="\n")
    if touched != {spec.name for spec in FIGURES}:
        _merge_previous_manifest(manifest, outdir / "manifest.json", touched)
    manifest["diagnostics"] = {
        entry["name"]: entry["metrics"] for entry in manifest["figures"] if entry["metrics"]
    }
    manifest["drawn_this_call"] = drawn_now
    (outdir / "manifest.json").write_text(json.dumps(_jsonable(manifest), indent=2), encoding="utf-8",
                                          newline="\n")
    return manifest


def _write_manifest(outdir: pathlib.Path, manifest: dict[str, Any], touched: set[str], refused: bool) -> None:
    """Write the manifest of a refused record, keeping entries an earlier call drew for this run."""
    previous_figures: list[dict[str, Any]] = []
    try:
        previous = json.loads((outdir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        previous = None
    if (refused and previous is not None and previous.get("run_id") == manifest["run_id"]
            and previous.get("figures_py_version") == FIGURES_VERSION):
        previous_figures = list(previous.get("figures", []))
        kept_not_drawn = [e for e in previous.get("not_drawn", []) if e.get("name") not in touched
                          and e.get("name") not in {f.get("name") for f in previous_figures}]
        manifest["not_drawn"] = kept_not_drawn + manifest["not_drawn"]
        manifest["previously_drawn_note"] = (
            "figures listed under 'figures' were drawn by an earlier call for this run and are still on disk"
        )
    manifest["figures"] = previous_figures
    (outdir / "manifest.json").write_text(json.dumps(_jsonable(manifest), indent=2), encoding="utf-8",
                                          newline="\n")


def _merge_previous_manifest(manifest: dict[str, Any], path: pathlib.Path, touched: set[str]) -> None:
    """Carry over entries of ``path`` (same run, same figures.py version) for figures not in ``touched``."""
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if previous.get("run_id") != manifest["run_id"] or previous.get("figures_py_version") != FIGURES_VERSION:
        return
    order = {spec.name: k for k, spec in enumerate(FIGURES)}
    for key in ("figures", "not_drawn", "errors"):
        kept = [entry for entry in previous.get(key, []) if entry.get("name") not in touched]
        manifest[key] = sorted(kept + manifest[key], key=lambda entry: order.get(entry.get("name"), len(order)))


# --- Command line -------------------------------------------------------------------------------


def _print_listing(registry) -> None:
    """Scenarios (with their functional and electron count) and the figure catalogue."""
    print("scenarios (physics_config registry):")
    for scenario in registry:
        kind = scenario.external.kind.value
        n = scenario.electrons.n_electrons
        print(f"  {scenario.scenario_id:<18} {kind:<16} xc={scenario.xc.name:<9} "
              f"N={'derived' if n is None else f'{n:g}'}")
    print("\noverrides: --bond-length R (two nuclei), --xc {" + ",".join(sorted(set(XC_SUFFIXES) | set(XC_ALIASES)))
          + "}, --n-electrons N")
    print("\nfigures:")
    for spec in FIGURES:
        print(f"  {spec.name:<18} {spec.summary}")
    print("\ngate profiles:")
    for profile in GATE_PROFILES.values():
        print(f"  {profile.name:<6} {profile.description}")


def build_parser() -> argparse.ArgumentParser:
    """Build the ``figures.py`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="figures.py",
        description="Solve (or load) one scenario and render publication figures of its density and diagnostics.",
        epilog="Exit code (G5.6): 0 the record was drawn with no new gate failure and no failed solve (a "
               "known-open or deferred record drawn with --allow-invalid exits 0); 1 a new gate failure, a "
               "failed solve, a figure that raised, or a record refused by the drawing policy; 2 a usage "
               "error, checked before anything is solved.",
    )
    parser.add_argument("scenario", nargs="?", help="registered scenario id (see --list)")
    parser.add_argument("--list", action="store_true", help="list scenarios, figures and gate profiles, then exit")
    parser.add_argument("--scenario-file", default=None,
                        help="YAML file of scenario definitions (physics_config format)")
    parser.add_argument("--bond-length", type=float, default=None, help="bond length for a two-nucleus scenario")
    parser.add_argument("--angstrom", action="store_true", help="--bond-length is in Angstrom (default bohr)")
    parser.add_argument("--xc", default=None, help="functional: none, lda (= lda_vwn), lda_pw92, lda_pz81, pbe")
    parser.add_argument("--n-electrons", type=float, default=None, help="electron number (fractional allowed)")
    parser.add_argument("--states", type=int, default=None, help="converge at least this many eigenstates")
    parser.add_argument("--numerics", default="auto",
                        help="auto (model for atom-free, all-electron otherwise), model, all-electron, or a YAML file")
    parser.add_argument("--res", choices=LEVELS, default="standard",
                        help="resolution level from the scenario's setup (a registered scenario at "
                             "--numerics auto); standard is the production rule, the rest exploratory")
    parser.add_argument("--device", choices=[d.value for d in Device], default=None,
                        help="override the numerics device; cuda without CUDA is an error, never a fallback (G5.5)")
    parser.add_argument("--gates", choices=sorted(GATE_PROFILES), default="quick",
                        help="gate budget attached before drawing (default quick)")
    parser.add_argument("--allow-invalid", action="store_true",
                        help="also draw a record outside TRUSTED_STATUSES (deferred, known-open, invalid, or an "
                             "unconverged run), badged 'not for publication'; new failures and failed solves "
                             "still exit 1")
    parser.add_argument("--from-corpus", default=None, help="render a stored HDF5 record instead of solving")
    parser.add_argument("--run-id", default=None, help="record id inside --from-corpus (needed if it holds several)")
    parser.add_argument("--out", default=None, help="output directory (default reports/figures/<scenario>/<run_id>)")
    parser.add_argument("--corpus", default=None,
                        help="also append the solved record to this HDF5 corpus (must not be <out>/record.h5)")
    parser.add_argument("--no-record", action="store_true",
                        help="do not write record.h5 beside the figures (by default it is written, or appended "
                             "to when the file exists)")
    parser.add_argument("--only", nargs="+", default=None, metavar="FIGURE", help="draw only these figures")
    parser.add_argument("--skip", nargs="+", default=(), metavar="FIGURE", help="do not draw these figures")
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"], choices=["pdf", "png", "svg"],
                        help="file formats (default pdf png)")
    parser.add_argument("--font", choices=["sans", "serif"], default="sans", help="figure typeface family")
    parser.add_argument("--usetex", action="store_true",
                        help="typeset text with LaTeX (needs TeX with cm-super and dvipng)")
    parser.add_argument("--dpi", type=int, default=600, help="raster resolution of PNG output")
    parser.add_argument("--map-points", type=int, default=361, help="points per axis of the density maps")
    parser.add_argument("--no-oracle", action="store_true",
                        help="skip the radial Kohn-Sham and two-centre oracles (no reference for interacting atoms)")
    return parser


def figure_scope_problem(scenario: ScenarioSpec, numerics: NumericsConfig) -> str | None:
    """Why this scenario's record could not be drawn (``None`` if it can), checked before a solve."""
    if scenario.electrons.spin_polarised:
        return ("the scenario is spin-polarised; figures.py draws spin-restricted records only "
                "(spin-resolved figures belong with the spin-polarised increment)")
    from cdft.grid import grid_config_for_scenario
    from cdft.scf.solve import is_interacting

    resolved, _ = grid_config_for_scenario(scenario, numerics.grid, derive_grid=True,
                                           interacting=is_interacting(scenario))
    if resolved.domain is not DomainMode.BOX:
        return ("the numerics resolve to a masked domain (DomainMode.MASKED_SPHERES), which the figure "
                "layer does not support; choose a box-domain numerics preset")
    return None


def _same_file(first: str | pathlib.Path, second: str | pathlib.Path) -> bool:
    """Whether two paths name the same file, existing or not (case-insensitive where the OS is)."""
    import os

    def canonical(path: str | pathlib.Path) -> str:
        return os.path.normcase(str(pathlib.Path(path).expanduser().resolve()))

    return canonical(first) == canonical(second)


def _corpus_problem(path: str | pathlib.Path) -> str | None:
    """Why ``path`` cannot be appended to as an HDF5 corpus (``None`` if it can: absent, or HDF5)."""
    import h5py

    path = pathlib.Path(path).expanduser()
    if not path.exists():
        return None
    if not path.is_file():
        return f"{path} exists and is not a file"
    if not h5py.is_hdf5(path):
        return f"{path} exists and is not an HDF5 file"
    return None


def _usage_problems(args: argparse.Namespace) -> list[str]:
    """Every usage error the arguments contain on their own (exit 2, before anything is loaded or solved)."""
    problems = []
    known = {spec.name for spec in FIGURES}
    unknown = sorted({name for name in list(args.only or []) + list(args.skip) if name not in known})
    if unknown:
        names = ", ".join(spec.name for spec in FIGURES)
        problems.append(f"unknown figure(s) {', '.join(unknown)}; choose from {names}")
    if args.only and set(args.only) <= set(args.skip):
        problems.append("--skip removes every figure named by --only")
    if args.dpi <= 0:
        problems.append("--dpi must be positive")
    if args.map_points < 5:
        problems.append("--map-points must be at least 5")
    if args.states is not None and args.states < 1:
        problems.append("--states must be at least 1")
    if args.angstrom and args.bond_length is None:
        problems.append("--angstrom qualifies --bond-length, which was not given")
    if args.run_id is not None and not args.from_corpus:
        problems.append("--run-id selects a record inside --from-corpus, which was not given")
    if args.from_corpus:
        selection = [flag for flag, given in (
            ("a scenario id", bool(args.scenario)), ("--bond-length", args.bond_length is not None),
            ("--xc", args.xc is not None), ("--n-electrons", args.n_electrons is not None),
            ("--scenario-file", args.scenario_file is not None),
        ) if given]
        solve_only = [flag for flag, given in (
            ("--numerics", args.numerics != "auto"), ("--device", args.device is not None),
            ("--gates", args.gates != "quick"), ("--states", args.states is not None),
            ("--corpus", args.corpus is not None), ("--no-record", args.no_record),
        ) if given]
        if selection:
            problems.append(f"--from-corpus renders a stored record; the scenario selection "
                            f"({', '.join(selection)}) does not apply")
        if solve_only:
            problems.append(f"--from-corpus does not solve; {', '.join(solve_only)} do not apply")
    elif not args.scenario and not args.list:
        problems.append("name a scenario (see --list) or give --from-corpus")
    return problems


def main(argv: list[str] | None = None) -> int:
    """Run the command line; return the process exit code (G5.6: 0 success, 1 failure, 2 usage)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.list:
        problems = _usage_problems(args)
        if problems:
            parser.print_usage(sys.stderr)
            for problem in problems:
                print(f"ERROR: {problem}", file=sys.stderr)
            return 2
    try:
        registry = load_scenarios(args.scenario_file) if args.scenario_file else REGISTRY
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: cannot load {args.scenario_file}: {exc}", file=sys.stderr)
        return 2
    if args.list:
        _print_listing(registry)
        return 0
    style = HouseStyle(font_family=args.font, usetex=args.usetex, dpi=args.dpi,
                       formats=tuple(dict.fromkeys(args.formats)))
    options = FigureOptions(style=style, use_oracle=not args.no_oracle, map_points=args.map_points)
    started = time.perf_counter()
    print(f"figures.py {FIGURES_VERSION}   contract {CONTRACT_VERSION}", flush=True)
    write_failures: list[str] = []

    if args.from_corpus:
        try:
            record = record_from_corpus(args.from_corpus, args.run_id)
        except (OSError, ValueError, KeyError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        print(f"record {record.run_id} ({record.scenario.scenario_id}, status {record.status}) from {args.from_corpus}",
              flush=True)
        invocation: dict[str, Any] = {"mode": "from-corpus", "path": str(args.from_corpus), "run_id": record.run_id}
        outdir = pathlib.Path(args.out) if args.out else (
            _REPO_ROOT / "reports" / "figures" / record.scenario.scenario_id / record.run_id
        )
    else:
        bond = args.bond_length
        if bond is not None and args.angstrom:
            from cdft.constants import ANGSTROM_TO_BOHR

            bond = bond * ANGSTROM_TO_BOHR
        try:
            scenario, notes = select_scenario(
                args.scenario, registry, bond_length=bond, xc=args.xc, n_electrons=args.n_electrons
            )
        except (KeyError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        for note in notes:
            print(f"  note: {note}", flush=True)
        profile = GATE_PROFILES[args.gates]
        targets: dict[str, float] = {}
        composed = bond is not None or args.xc is not None or args.n_electrons is not None
        if args.res != "standard" and (composed or args.numerics != "auto" or args.scenario_file):
            print(
                f"ERROR: --res {args.res} needs a registered scenario at its own setup (D-73): drop "
                f"--numerics, --scenario-file and the composition options (--bond-length, --xc, "
                f"--n-electrons), which build a scenario that has no resolution levels.",
                file=sys.stderr,
            )
            return 2
        try:
            if args.res == "standard":
                numerics = select_numerics(scenario, args.numerics, profile, args.device)
            else:
                numerics, targets = resolve_resolution(scenario, args.res, profile, args.device)
        except (OSError, ValueError, KeyError) as exc:
            print(f"ERROR: numerics {args.numerics!r}: {exc}", file=sys.stderr)
            return 2
        # Everything that would make the solve useless is refused before it starts.
        problems = []
        scope = figure_scope_problem(scenario, numerics)
        if scope is not None:
            problems.append(scope)
        planned = pathlib.Path(args.out) if args.out else None
        record_target = None if (args.no_record or planned is None) else planned / "record.h5"
        if args.corpus is not None:
            corpus_issue = _corpus_problem(args.corpus)
            if corpus_issue is not None:
                problems.append(f"--corpus: {corpus_issue}")
            if record_target is not None and _same_file(args.corpus, record_target):
                problems.append("--corpus names the record.h5 written beside the figures; the record would be "
                                "appended twice (give another corpus path, or --no-record)")
        if record_target is not None:
            record_issue = _corpus_problem(record_target)
            if record_issue is not None:
                problems.append(f"record file: {record_issue}")
        if problems:
            for problem in problems:
                print(f"ERROR: {problem}", file=sys.stderr)
            return 2
        if record_target is not None and record_target.is_file():
            print(f"  note: {record_target} already exists; this record is appended to it, so a later "
                  f"--from-corpus redraw of that file needs --run-id", flush=True)
        invocation = {"mode": "solve", "requested": args.scenario, "solved": scenario.scenario_id,
                      "bond_length_bohr": bond, "xc": args.xc, "n_electrons": args.n_electrons,
                      "numerics": args.numerics, "device": args.device, "gate_profile": profile.name,
                      "states": args.states, "selection_notes": notes, "resolution": args.res}
        if args.res != "standard":
            detail = ", ".join(f"{name}={value:g}" for name, value in targets.items())
            note = (f"exploratory resolution {args.res!r} ({detail or f'spacing={numerics.grid.spacing:g}'}): "
                    f"not comparable with the golden values or the known-open registry")
            invocation["resolution_note"] = note
            print(note, flush=True)
        if args.device is not None:
            from cdft.precision import resolve_device

            try:
                resolved = resolve_device(numerics.device)
            except RuntimeError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 2
            from cdft.device_policy import apply_policy, describe

            apply_policy(resolved)
            print(f"device: {resolved}\n{describe(resolved)}", flush=True)
        from cdft.grid import grid_rule

        print(f"solving {scenario.scenario_id} (gates: {profile.name}) ...", flush=True)
        # The gate pass re-solves, so the rule must be active for it too, not only for the solve.
        with grid_rule(**targets):
            artifact = solve_and_gate(scenario, numerics, profile, min_states=args.states)
        from cdft.gates.runner import format_report

        title = f"{scenario.scenario_id}  [status={artifact.status.value}]"
        if artifact.error_message:
            title += f"  error={artifact.error_message}"
        print(format_report(artifact.gates.results, title), flush=True)
        outdir = planned or _REPO_ROOT / "reports" / "figures" / scenario.scenario_id / artifact.run_id
        outdir.mkdir(parents=True, exist_ok=True)
        from cdft.io.hdf5 import CorpusWriter

        record_path = None
        targets = []
        if not args.no_record:
            targets.append(("record", outdir / "record.h5"))
        if args.corpus is not None:
            if targets and _same_file(args.corpus, targets[0][1]):
                print("  note: --corpus is the record file beside the figures; the record is written once",
                      flush=True)
            else:
                targets.append(("corpus", pathlib.Path(args.corpus).expanduser()))
        for role, target in targets:
            try:
                CorpusWriter(target).write(artifact)
            except (OSError, ValueError) as exc:
                write_failures.append(f"{role} {target}: {type(exc).__name__}: {exc}")
                print(f"ERROR: could not write the {role} file {target}: {exc}", file=sys.stderr)
                continue
            if role == "record":
                record_path = target
        invocation["written"] = {role: str(target) for role, target in targets}
        invocation["write_failures"] = list(write_failures)
        try:
            record = record_from_artifact(artifact)
        except ValueError as exc:
            print(f"ERROR: the solved record cannot be drawn: {exc}", file=sys.stderr)
            return 1
        record.source = f"solve (record {record_path})" if record_path else "solve"
        del artifact

    verdict = classify_record(
        record.scenario.scenario_id, record.status, record.gates,
        has_fields=record.density is not None, error_message=record.error_message,
        allow_invalid=args.allow_invalid,
    )
    print(f"verdict: status {verdict.status}; "
          + ("drawn" if verdict.drawable else "NOT drawn")
          + (f" ({verdict.mark})" if verdict.mark else "")
          + (f"; {verdict.reason}" if verdict.reason else ""), flush=True)
    if not verdict.drawable and record.density is not None:
        print("  the record is outside TRUSTED_STATUSES; rerun with --allow-invalid to draw it with a "
              f"{verdict.mark.upper()} mark (not for publication)", flush=True)
    budget = gate_summary(record.gates)["skipped_for_budget"]
    if budget and verdict.drawable:
        print(f"  note: {len(budget)} gate(s) were not evaluated under this gate budget; "
              f"re-solve with --gates full before publishing these figures", flush=True)
    try:
        manifest = render_figures(record, verdict, outdir, options, only=args.only, skip=args.skip,
                                  invocation=invocation)
    except ValueError as exc:  # pragma: no cover - the figure names were validated above
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    drawn = len(manifest.get("drawn_this_call", manifest["figures"]))
    described = len(manifest["figures"])
    print(f"\n{drawn} figure(s) drawn into {outdir}"
          + (f" ({described} described by manifest.json)" if described != drawn else "")
          + f"  ({time.perf_counter() - started:.0f} s total)", flush=True)
    if manifest["errors"]:
        print(f"{len(manifest['errors'])} figure(s) failed; tracebacks in manifest.json", file=sys.stderr)
    if write_failures:
        print(f"{len(write_failures)} record file(s) were not written; see the errors above", file=sys.stderr)
    failed = verdict.blocking or bool(manifest["errors"]) or not verdict.drawable or bool(write_failures)
    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
