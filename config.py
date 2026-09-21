"""Loading and validation of :class:`~contract.NumericsConfig` -- "how hard to try".

Never names a chemical species; that is :mod:`cdft.physics_config` (Q1.5). Every loader rejects
unknown keys and unknown enum values rather than defaulting them. The per-system setups at the
bottom name the resolution levels a scenario may be run at (D-73).

Rationale for the numbers in each named configuration: ``docs/03_METHOD.md (Part E)``.
"""

from __future__ import annotations

# Path bootstrap: this file is canonical at the repository root next to ``contract.py`` (D-57), so
# the three "what and how" modules import without the package being installed.
import pathlib as _pathlib
import sys as _sys

_REPO_ROOT = _pathlib.Path(__file__).resolve().parent
for _entry in (_REPO_ROOT, _REPO_ROOT / "src"):
    if _entry.is_dir() and str(_entry) not in _sys.path:
        _sys.path.insert(0, str(_entry))
del _entry

import dataclasses
import math
import pathlib
from typing import Any, Mapping, TypeVar

import yaml

from contract import (
    BoundaryMode,
    Device,
    DomainMode,
    EigenConfig,
    EigensolverKind,
    GridConfig,
    MixingConfig,
    MixingScheme,
    NumericsConfig,
    OccupationScheme,
    OutputConfig,
    PoissonKind,
    Precision,
    PrecisionConfig,
    ScenarioSpec,
    SCFConfig,
)
from physics_config import REGISTRY

__all__ = [
    "load_numerics",
    "numerics_from_dict",
    "numerics_to_dict",
    "DEFAULT_NUMERICS",
    "MODEL_SYSTEM_NUMERICS",
    "ALL_ELECTRON_NUMERICS",
    "Resolution",
    "Setup",
    "LEVELS",
    "SETUPS",
    "PRESETS",
    "numerics_for",
    "numerics_for_spec",
    "describe_setups",
]

_T = TypeVar("_T")

_ENUMS: dict[str, type] = {
    "domain": DomainMode,
    "boundary": BoundaryMode,
    "scheme": MixingScheme,
    "kind": EigensolverKind,  # overridden per-section below where ambiguous
    "occupation": OccupationScheme,
    "device": Device,
    "poisson": PoissonKind,
    "hot_path": Precision,
    "reduction": Precision,
    "subspace": Precision,
    "orthonormalisation": Precision,
}

_SECTIONS: dict[str, type] = {
    "grid": GridConfig,
    "precision": PrecisionConfig,
    "eigen": EigenConfig,
    "mixing": MixingConfig,
    "scf": SCFConfig,
    "output": OutputConfig,
}


def _coerce(value: Any, field: dataclasses.Field[Any], section: str) -> Any:
    """Coerce one YAML scalar to the type the dataclass field declares.

    Enumerations resolve by *value*, not by name, so a config file reads as physics.
    """
    if isinstance(value, str) and field.name in _ENUMS:
        enum_type = EigensolverKind if (section == "eigen" and field.name == "kind") else _ENUMS[field.name]
        try:
            return enum_type(value)
        except ValueError as exc:
            allowed = ", ".join(sorted(m.value for m in enum_type))
            raise ValueError(
                f"{section}.{field.name}: {value!r} is not one of [{allowed}]"
            ) from exc
    if isinstance(value, list):
        return tuple(value)
    return value


def _build(cls: type[_T], data: Mapping[str, Any], section: str) -> _T:
    """Construct one configuration dataclass from a mapping, rejecting unknown keys."""
    fields = {f.name: f for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = set(data) - set(fields)
    if unknown:
        allowed = ", ".join(sorted(fields))
        raise ValueError(
            f"unknown key(s) in section {section!r}: {', '.join(sorted(unknown))}. "
            f"Allowed: {allowed}. Unknown keys are rejected rather than ignored because an ignored "
            f"key means the run used a setting nobody asked for while the log says otherwise."
        )
    kwargs = {name: _coerce(value, fields[name], section) for name, value in data.items()}
    return cls(**kwargs)  # type: ignore[call-arg]


def numerics_from_dict(data: Mapping[str, Any]) -> NumericsConfig:
    """Build a :class:`~contract.NumericsConfig` from a nested mapping.

    Parameters
    ----------
    data:
        Optional sections ``grid``, ``precision``, ``eigen``, ``mixing``, ``scf``, ``output``, plus
        the scalar top-level keys of :class:`~contract.NumericsConfig`.

    Raises
    ------
    ValueError
        On an unknown key, an unknown enumeration value, or a value the contract's own validators
        reject -- all before any tensor is allocated.
    """
    unknown = set(data) - set(_SECTIONS) - {"device", "seed", "poisson", "deterministic"}
    if unknown:
        raise ValueError(f"unknown top-level numerics key(s): {', '.join(sorted(unknown))}")

    sections: dict[str, Any] = {
        name: _build(cls, data.get(name, {}) or {}, name) for name, cls in _SECTIONS.items()
    }
    scalars: dict[str, Any] = {}
    if "device" in data:
        scalars["device"] = Device(data["device"])
    if "poisson" in data:
        scalars["poisson"] = PoissonKind(data["poisson"])
    if "seed" in data:
        scalars["seed"] = int(data["seed"])
    if "deterministic" in data:
        scalars["deterministic"] = bool(data["deterministic"])
    return NumericsConfig(**sections, **scalars)


def numerics_to_dict(config: NumericsConfig) -> dict[str, Any]:
    """Serialise a numerics configuration to plain types.

    Exactly what :meth:`~contract.NumericsConfig.fingerprint` hashes, so a record's stored
    configuration and its ``config_hash`` cannot disagree.
    """
    return dataclasses.asdict(config)


def load_numerics(path: str | pathlib.Path) -> NumericsConfig:
    """Load a numerics configuration from a YAML file; ``{}`` yields the contract defaults."""
    text = pathlib.Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"{path}: expected a mapping at the top level, got {type(data).__name__}")
    return numerics_from_dict(data)


# ---------------------------------------------------------------------------------------------
# Named configurations
# ---------------------------------------------------------------------------------------------
# Not presets that hide a choice: each exists because a specific gate needs a specific resolution.

#: Production defaults, unchanged from the contract. The pseudopotential path of Phase 2.
DEFAULT_NUMERICS = NumericsConfig()

#: Analytic model systems -- harmonic well (G1.2), particle in a box (G1.3), grid isotropy (G1.4).
#: A box domain, because these scenarios have no atoms to mask spheres around. ``spacing = 0.20``
#: and ``residual_tol = 1e-8`` are measured requirements of G1.2 and G2.4, not preferences.
MODEL_SYSTEM_NUMERICS = NumericsConfig(
    grid=GridConfig(
        spacing=0.20,
        fd_order=8,
        domain=DomainMode.BOX,
        boundary=BoundaryMode.ZERO,
        box_lengths=(16.0, 16.0, 16.0),
        use_double_grid=False,
        fourier_filter_projectors=False,
    ),
    eigen=EigenConfig(
        n_extra_states=6, chebyshev_degree=16, max_iterations=200, residual_tol=1.0e-8, cross_check=True
    ),
    device=Device.CPU,
)

#: The Phase 1 all-electron path: a bare -Z/r nucleus on the cusp-factorised operator.
#:
#: **The grid below is a placeholder, not a choice.** Both box and spacing are derived per scenario
#: by :func:`cdft.grid.grid_config_for_scenario` and every derivation is written into the record as
#: an override (G5.5, D-42). The values are the single-centre Z = 1 ones, so a caller bypassing the
#: derivation still gets a sane grid. The component gates are not scenario-bound and derive nothing:
#: ``run.py --numerics all-electron`` runs them at this ``spacing = 1.0``, where G0.2's fixture
#: (a Gaussian of ``sigma = 4h = 4`` bohr in the gate's fixed 16-bohr box) truncates its own test
#: charge at ``2 sigma`` and fails at 5.77e-3 Ha -- a fixture limit, not a solver error (the same
#: solve in a 64-bohr box passes at 1e-16), and expected (B3). ``--numerics auto``, the default,
#: runs the component gates at :data:`MODEL_SYSTEM_NUMERICS`.
ALL_ELECTRON_NUMERICS = NumericsConfig(
    grid=GridConfig(
        spacing=1.0,
        fd_order=8,
        domain=DomainMode.BOX,
        boundary=BoundaryMode.ZERO,
        box_lengths=(32.0, 32.0, 32.0),
        use_double_grid=False,
        fourier_filter_projectors=False,
    ),
    eigen=EigenConfig(
        n_extra_states=4, chebyshev_degree=20, max_iterations=300, residual_tol=1.0e-8, cross_check=True
    ),
    device=Device.CPU,
)

#: The base presets addressable by name, for the command line and for ``figures.py``.
PRESETS: dict[str, NumericsConfig] = {
    "model": MODEL_SYSTEM_NUMERICS,
    "all-electron": ALL_ELECTRON_NUMERICS,
}


# ---------------------------------------------------------------------------------------------
# Per-system setups (D-73)
# ---------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Resolution:
    """One resolution level. ``None`` = leave to the base preset / the production grid rule."""

    spacing: float | None = None        # bohr; a derivation target on a Coulomb system
    half_box: float | None = None       # bohr of vacuum beyond the outermost nucleus (Coulomb)
    box: float | None = None            # bohr, full edge (model systems only)
    points_per_edge: int | None = None  # single bare nucleus only (D-42: box-limited)
    fd_order: int | None = None
    eigen_tol: float | None = None      # EigenConfig.residual_tol
    chebyshev_degree: int | None = None
    scf_energy_tol: float | None = None
    scf_density_tol: float | None = None
    max_scf_iterations: int | None = None


@dataclasses.dataclass(frozen=True)
class Setup:
    """One registered scenario: what it is for, which preset it starts from, and its levels."""

    scenario: str                 # id in physics_config.REGISTRY
    base: str                     # "model" | "all-electron"
    about: str                    # <= 90 chars: what the system is for
    levels: Mapping[str, Resolution]


LEVELS: tuple[str, ...] = ("draft", "standard", "fine", "reference")

#: ``standard`` is always empty: it *is* the production rule, and the only level whose numbers are
#: comparable with the golden values, the fingerprints and the known-open registry.
_STANDARD = Resolution()

#: Model wells need resolution, not box: their eigenfunctions genuinely oscillate, and 0.20 is
#: G1.2's measured requirement, so the ladders bracket it. Every spacing divides its scenario's box
#: edge exactly -- :meth:`cdft.grid.UniformGrid.from_config` re-derives ``h`` from the edge, and a
#: spacing that does not divide it comes back snapped, which G5.5 reads as an unexplained change.
#: A Coulomb scenario is immune: D-64 rounds the derived edge up to a whole number of cells.
_HARMONIC_LEVELS: Mapping[str, Resolution] = {
    "draft": Resolution(spacing=0.32),
    "standard": _STANDARD,
    "fine": Resolution(spacing=0.16),
    "reference": Resolution(spacing=0.125),
}

#: The infinite well's edge is 10 bohr, so it takes its own divisors (D-73).
_WELL_LEVELS: Mapping[str, Resolution] = {
    "draft": Resolution(spacing=0.40),
    "standard": _STANDARD,
    "fine": Resolution(spacing=0.125),
    "reference": Resolution(spacing=0.10),
}

#: Bare molecules are spacing-limited and the box is flat from 12 to 24 bohr (A-1), so every level
#: moves the spacing alone. H2+ at R = 2 against the two-centre oracle: 1.8e-3 Ha at h = 0.32,
#: 1.0e-3 at 0.25, 1.6e-4 at 0.16. The draft target is 0.35 rather than 0.40 because the lattice
#: rule may round a target up, and above h ~ 0.5 convergence stops being monotone: a scan run there
#: acquires scatter that reads as structure (A-4).
_MOLECULE_LEVELS: Mapping[str, Resolution] = {
    "draft": Resolution(spacing=0.35),
    "standard": _STANDARD,
    "fine": Resolution(spacing=0.20),
    "reference": Resolution(spacing=0.16),
}

#: Interacting scenarios are interpolation-limited to ~O(h^3) (A-10). He at VWN LDA against NIST:
#: 2.0e-4 Ha at h = 0.35, 1.7e-5 at 0.25, 5.9e-6 at 0.20 on a 16-bohr box, and the 1e-6 target is
#: met at h ~ 0.16 -- which needs the box too, hence the raised floor at ``reference`` (box 14 vs
#: 16 differ by 9e-7). ``half_box`` is a floor: the derived tail margin wins where it is wider.
_INTERACTING_LEVELS: Mapping[str, Resolution] = {
    "draft": Resolution(spacing=0.35),
    "standard": _STANDARD,
    "fine": Resolution(spacing=0.20),
    "reference": Resolution(spacing=0.16, half_box=8.0),
}


def _nucleus_levels(z: float) -> dict[str, Resolution]:
    """Levels for a single bare nucleus: a point count and a half-box, never a spacing (D-42).

    The transformed problem has a constant solution, so refining ``h`` only accumulates round-off
    (4.7e-14 Ha at 21 points, 1.2e-12 at 129) and accuracy is bought with box instead: the
    Dirichlet error is the weight ``exp(-2 Z R)`` left at the edge, 1e-14 at the production
    ``16 / Z`` and ~1e-6 at half of it, which is what makes ``draft`` cheap and exploratory.
    """
    return {
        "draft": Resolution(points_per_edge=17, half_box=8.0 / z),
        "standard": _STANDARD,
        "fine": Resolution(points_per_edge=49),
        "reference": Resolution(points_per_edge=65),
    }


#: Every registered scenario, in registry order, with the levels ``--res`` selects.
SETUPS: dict[str, Setup] = {
    setup.scenario: setup
    for setup in (
        Setup("harmonic_w1", "model",
              "isotropic harmonic well, omega = 1: the analytic spectrum (n + 3/2) (G1.2)",
              _HARMONIC_LEVELS),
        Setup("box_L10", "model",
              "particle in a 10-bohr infinite well: the boundary treatment alone (G1.3)",
              _WELL_LEVELS),
        Setup("h_atom", "all-electron",
              "bare H nucleus, cusp-factorised: exact to round-off, box-limited (D-42)",
              _nucleus_levels(1.0)),
        Setup("he_plus", "all-electron",
              "bare He+ nucleus at Z = 2: exact to round-off; the box scales as 16/Z",
              _nucleus_levels(2.0)),
        Setup("h2plus_R2", "all-electron",
              "H2+ at R = 2 bohr against the two-centre oracle; spacing-limited (A-1)",
              _MOLECULE_LEVELS),
        Setup("he_atom", "all-electron",
              "bare helium nucleus, two electrons, no functional: exact to round-off",
              _nucleus_levels(2.0)),
        Setup("h2_R1.4", "all-electron",
              "H2 at R = 1.4 bohr, bare nuclei: the flat-plane geometry of [A23]",
              _MOLECULE_LEVELS),
        Setup("h_atom_lda", "all-electron",
              "H at VWN LDA vs NIST; box-limited: the LDA density decays as exp(-1.37 r)",
              _INTERACTING_LEVELS),
        Setup("he_plus_lda", "all-electron",
              "one electron at Z = 2, VWN LDA, against NIST SRD 141",
              _INTERACTING_LEVELS),
        Setup("he_atom_lda", "all-electron",
              "He at VWN LDA vs NIST: interpolation-limited to ~O(h^3) (A-10)",
              _INTERACTING_LEVELS),
        Setup("he_atom_pbe", "all-electron",
              "He at PBE against the PySCF cc-pV5Z/6Z oracle (O-12)",
              _INTERACTING_LEVELS),
        Setup("h2_R1.4_lda", "all-electron",
              "H2 at R = 1.4 bohr, VWN LDA, against PySCF",
              _INTERACTING_LEVELS),
        Setup("h2_R1.4_pbe", "all-electron",
              "H2 at R = 1.4 bohr, PBE: the flat-plane system [A23]",
              _INTERACTING_LEVELS),
        Setup("h2plus_R2_lda", "all-electron",
              "H2+ at R = 2, VWN LDA: the self-interaction reference system (D-12)",
              _INTERACTING_LEVELS),
        Setup("h2plus_R8_lda", "all-electron",
              "stretched H2+ at R = 8, VWN LDA: the dissociation failure (G4.5)",
              _INTERACTING_LEVELS),
        Setup("h_atom_N0.5_lda", "all-electron",
              "H with half an electron at VWN LDA: the first point of D1.2",
              _INTERACTING_LEVELS),
    )
}

_BASES: tuple[str, ...] = ("model", "all-electron")

#: Resolution fields that name a grid quantity, and which of them each system class accepts.
_GRID_FIELDS: tuple[str, ...] = ("spacing", "half_box", "box", "points_per_edge")
_ACCEPTS: dict[str, tuple[str, ...]] = {
    "model": ("spacing", "box"),
    "well": ("spacing",),
    "nucleus": ("points_per_edge", "half_box"),
    "molecule": ("spacing", "half_box"),
    "interacting": ("spacing", "half_box"),
}
_KIND_LABEL: dict[str, str] = {
    "model": "model system",
    "well": "particle in an infinite well, whose edge fixes the box",
    "nucleus": "single bare nucleus, which is box-limited (D-42)",
    "molecule": "bare molecule (D-42)",
    "interacting": "interacting scenario (D-53)",
}

#: Resolution field -> :func:`cdft.grid.grid_rule` target, per system class. ``half_box`` is stated
#: per side in bohr, so it scales onto the constant each rule actually holds.
_TARGETS: dict[str, dict[str, str]] = {
    "model": {},
    "well": {},
    "nucleus": {"points_per_edge": "points_per_edge", "half_box": "coulomb_half_box_per_z"},
    "molecule": {"spacing": "molecular_spacing", "half_box": "vacuum_padding"},
    "interacting": {"spacing": "interacting_spacing", "half_box": "interacting_half_box"},
}

#: Resolution field -> (numerics section, field). Everything else is a grid rule target.
_NUMERICS_FIELDS: dict[str, tuple[str, str]] = {
    "spacing": ("grid", "spacing"),
    "box": ("grid", "box_lengths"),
    "fd_order": ("grid", "fd_order"),
    "eigen_tol": ("eigen", "residual_tol"),
    "chebyshev_degree": ("eigen", "chebyshev_degree"),
    "scf_energy_tol": ("scf", "energy_tol"),
    "scf_density_tol": ("scf", "density_tol"),
    "max_scf_iterations": ("scf", "max_iterations"),
}

_INT_FIELDS: tuple[str, ...] = (
    "points_per_edge", "fd_order", "chebyshev_degree", "max_scf_iterations",
)


def _setup(scenario_id: str) -> Setup:
    """Return one setup, or raise listing the registered identifiers."""
    if scenario_id not in SETUPS:
        raise ValueError(f"unknown scenario {scenario_id!r}. Registered: {', '.join(SETUPS)}")
    return SETUPS[scenario_id]


def _base_numerics(base: str) -> NumericsConfig:
    """Return the preset ``base`` names, read at call time (the device leg rebinds the presets)."""
    if base not in _BASES:
        raise ValueError(f"unknown base {base!r}. Known: {', '.join(_BASES)}")
    return MODEL_SYSTEM_NUMERICS if base == "model" else ALL_ELECTRON_NUMERICS


def _validate(field: str, value: Any) -> float | int:
    """Reject a resolution value the derivation rules cannot use, before anything is solved."""
    if field in _INT_FIELDS:
        number = int(value)
        if number != value or number < 1:
            raise ValueError(f"{field} must be a positive integer, got {value!r}")
        if field == "points_per_edge" and (number < 9 or number % 2 == 0):
            raise ValueError(f"points_per_edge must be an odd integer >= 9, got {value!r} (D-42)")
        return number
    number_f = float(value)
    if not math.isfinite(number_f) or number_f <= 0.0:
        raise ValueError(f"{field} must be a positive finite number, got {value!r}")
    return number_f


def _system_kind(scenario: ScenarioSpec) -> str:
    """Which derivation rule this scenario's grid follows, as a key of :data:`_ACCEPTS`."""
    from cdft.scf.solve import is_interacting
    from contract import ExternalPotentialKind

    if scenario.external.kind is ExternalPotentialKind.PARTICLE_IN_BOX:
        return "well"
    coulomb = scenario.external.kind is ExternalPotentialKind.NUCLEAR_COULOMB
    if not coulomb or scenario.structure.n_atoms == 0:
        return "model"
    if is_interacting(scenario):
        return "interacting"
    return "nucleus" if scenario.structure.n_atoms == 1 else "molecule"


def _grid_targets(kind: str, scenario: ScenarioSpec, values: Mapping[str, Any]) -> dict[str, float]:
    """Map the resolved grid fields onto :func:`cdft.grid.grid_rule` targets."""
    targets: dict[str, float] = {}
    for field, target in _TARGETS[kind].items():
        if field not in values:
            continue
        value = float(values[field])
        if target == "coulomb_half_box_per_z":
            value *= max(int(z) for z in scenario.structure.numbers)  # the rule is 16 / Z bohr
        elif target == "vacuum_padding":
            value *= 2.0  # the rule adds the padding to the span, i.e. both sides at once
        targets[target] = int(value) if target == "points_per_edge" else value
    return targets


def _apply(numerics: NumericsConfig, kind: str, values: Mapping[str, Any]) -> NumericsConfig:
    """Apply the fields the grid rule does not carry to the base preset."""
    updates: dict[str, dict[str, Any]] = {}
    for field, value in values.items():
        if field in _TARGETS[kind] or field not in _NUMERICS_FIELDS:
            continue
        section, name = _NUMERICS_FIELDS[field]
        updates.setdefault(section, {})[name] = (
            (value, value, value) if name == "box_lengths" else value
        )
    for section, fields in updates.items():
        replaced = dataclasses.replace(getattr(numerics, section), **fields)
        numerics = dataclasses.replace(numerics, **{section: replaced})
    return numerics


def _resolve(
    scenario: ScenarioSpec,
    base: str,
    resolution: Resolution,
    device: Any,
    overrides: Mapping[str, Any],
) -> tuple[NumericsConfig, dict[str, float]]:
    """Merge a level with its overrides and split the result into numerics and grid rule targets."""
    allowed = tuple(f.name for f in dataclasses.fields(Resolution))
    unknown = set(overrides) - set(allowed)
    if unknown:
        raise ValueError(
            f"unknown resolution field(s): {', '.join(sorted(unknown))}. "
            f"Allowed: {', '.join(allowed)}"
        )
    values = {name: getattr(resolution, name) for name in allowed}
    values.update(overrides)
    values = {name: _validate(name, v) for name, v in values.items() if v is not None}

    kind = _system_kind(scenario)
    for field in _GRID_FIELDS:
        if field in values and field not in _ACCEPTS[kind]:
            raise ValueError(
                f"{scenario.scenario_id}: {field!r} has no meaning for a {_KIND_LABEL[kind]}; "
                f"it takes {', '.join(_ACCEPTS[kind])}"
            )

    numerics = _base_numerics(base)
    if device is not None:
        numerics = dataclasses.replace(numerics, device=Device(device))
    if not values:
        return numerics, {}
    return _apply(numerics, kind, values), _grid_targets(kind, scenario, values)


def numerics_for(
    scenario_id: str, level: str = "standard", *, device: Any = None, **overrides: Any
) -> tuple[NumericsConfig, dict[str, float]]:
    """Return ``(numerics, grid_rule_targets)`` for one registered scenario at one level.

    The targets are the non-production derivation constants the solve must run under
    (``with cdft.grid.grid_rule(**targets):``); at ``standard`` with no overrides they are empty
    and the numerics are the scenario's base preset unchanged. ``overrides`` takes
    :class:`Resolution` field names.
    """
    setup = _setup(scenario_id)
    if level not in setup.levels:
        raise ValueError(
            f"unknown level {level!r} for {scenario_id!r}. Available: {', '.join(setup.levels)}"
        )
    return _resolve(REGISTRY[scenario_id], setup.base, setup.levels[level], device, overrides)


def numerics_for_spec(
    scenario: ScenarioSpec, *, device: Any = None, **overrides: Any
) -> tuple[NumericsConfig, dict[str, float]]:
    """Resolve a scenario that has no setup (``--scenario-file``): base by rule, no levels."""
    base = "model" if scenario.structure.n_atoms == 0 else "all-electron"
    return _resolve(scenario, base, _STANDARD, device, overrides)


def _grid_line(config: GridConfig) -> str:
    """One line of resolved geometry: spacing, box edge, and the point count they imply."""
    if config.domain is not DomainMode.BOX or config.box_lengths is None:
        return f"h = {config.spacing:<6.4g} masked spheres, r = {config.mask_radius:g}"
    edge = config.box_lengths[0]
    cells = int(round(edge / config.spacing))
    # As :meth:`cdft.grid.UniformGrid.from_config` counts them: a reflected wall sits one spacing
    # outside the outermost point, so the interior holds one fewer point per axis.
    points = max(cells - 1 if config.boundary is BoundaryMode.ODD_REFLECTION else cells + 1, 1)
    return (
        f"h = {config.spacing:<6.4g} box = {edge:<6.4g} {points:>4}^3 = "
        f"{points ** 3 / 1.0e6:7.3f} M points"
    )


def describe_setups(registry: Any = REGISTRY) -> str:
    """Render the table ``run.py --setups`` prints: what each system is, and each level's grid."""
    from cdft.grid import grid_config_for_scenario, grid_rule
    from cdft.scf.solve import is_interacting

    lines = [
        f"{len(SETUPS)} scenarios x {len(LEVELS)} levels. --res picks a level; --spacing, "
        f"--half-box, --box, --points-per-edge and the solver tolerances override one field.",
        "'standard' is the production rule: the only level comparable with the golden values, the "
        "fingerprints and the known-open registry.",
        "",
    ]
    for scenario in registry:
        setup = _setup(scenario.scenario_id)
        lines.append(f"{setup.scenario}  [{setup.base}]  {setup.about}")
        for level in setup.levels:
            numerics, targets = numerics_for(setup.scenario, level)
            with grid_rule(**targets):
                config, _ = grid_config_for_scenario(
                    scenario, numerics.grid, interacting=is_interacting(scenario)
                )
            lines.append(f"    {level:<10} {_grid_line(config)}")
        lines.append("")
    return "\n".join(lines)
