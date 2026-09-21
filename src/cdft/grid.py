"""The real-space grid: masked domain, quadrature, and the finite-difference operators on it.

Implements :class:`~contract.GridProtocol`. Every solver field lives on the flat point index defined
here, and the mapping back to the three-dimensional box lives here and nowhere else. The domain is
the union of spheres around the atoms rather than a cube (D-08), and the stencil halo is filled per
:class:`~contract.BoundaryMode`, never assumed zero. The sizing rules below are D-42/D-53/D-64;
their measurements are in ``docs/03_METHOD.md (Part E)``.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterator
from dataclasses import dataclass
from dataclasses import fields as dc_fields
from types import MappingProxyType
from typing import Any

import torch

from contract import AtomicStructure, BoundaryMode, DomainMode, GridConfig

from .operators.laplacian import (
    component_from_padded,
    gradient_from_padded,
    laplacian_from_padded,
    stencil_half_width,
)
from .structure import bounding_box, positions_tensor

__all__ = [
    "UniformGrid",
    "GridGeometry",
    "grid_config_for_scenario",
    "coulomb_box_edge",
    "coulomb_spacing",
    "POINTS_PER_EDGE",
    "VACUUM_PADDING",
    "INTERACTING_SPACING",
    "INTERACTING_HALF_BOX",
    "INTERACTING_TAIL_FRACTION",
    "SLATER_1S_SCREENING",
    "interacting_half_box",
    "interacting_box_edge",
    "interacting_spacing",
    "lattice_spacing",
    "lattice_box_edge",
    "lattice_offsets",
    "lattice_anchor",
    "LATTICE_TOLERANCE",
    "RULE_TARGETS",
    "PRODUCTION_RULE",
    "grid_rule",
    "active_rule",
]

#: Half-box in bohr per unit nuclear charge, cusp-factorised path: a box-limited centre (D-42).
COULOMB_HALF_BOX_PER_Z = 16.0

#: Points along each box edge for a single centre -- a count, not a spacing (D-42).
POINTS_PER_EDGE = 33

#: Vacuum in bohr beyond the nuclear span for a multi-centre system (D-42).
VACUUM_PADDING = 12.0

#: Spacing in bohr for a multi-centre system (D-42); the residual 1.0e-03 Ha is open item O-13.
MOLECULAR_SPACING = 0.25


#: Spacing in bohr for an interacting (Hartree + XC) all-electron scenario, atom or molecule alike:
#: once ``v_H + v_xc`` enter, a single centre is spacing-limited too (D-53). A cost choice, not a
#: limit -- ``scripts/interacting_ladder.py`` measures the O(h^3) ladder behind it.
INTERACTING_SPACING = 0.25

#: Floor in bohr for the interacting vacuum margin, where the helium ladder converges (D-53).
INTERACTING_HALF_BOX = 7.0

#: Charge fraction the derived box may leave outside it, ``exp(-2 kappa R) = 1e-9``: the margin is
#: derived per scenario from an a-priori decay estimate, floored at :data:`INTERACTING_HALF_BOX`.
INTERACTING_TAIL_FRACTION = 1.0e-9

#: Slater's 1s screening, ``Z - 0.30 (n - 1)``: an estimate of the decay, for box sizing only.
SLATER_1S_SCREENING = 0.30


#: Derivation targets :func:`grid_rule` may swap; each is the module constant of that name (D-73).
RULE_TARGETS: tuple[str, ...] = (
    "interacting_spacing",
    "interacting_half_box",
    "interacting_tail_fraction",
    "molecular_spacing",
    "vacuum_padding",
    "points_per_edge",
    "coulomb_half_box_per_z",
)

#: The production value of every target, captured at import; :func:`active_rule` diffs against it.
PRODUCTION_RULE = MappingProxyType({name: globals()[name.upper()] for name in RULE_TARGETS})


def _validated_target(name: str, value: float) -> float:
    """Return one :data:`RULE_TARGETS` value, or raise before any grid is derived from it."""
    if name not in RULE_TARGETS:
        raise ValueError(f"unknown grid rule target {name!r}. Known: {', '.join(RULE_TARGETS)}")
    if name == "points_per_edge":
        points = int(value)
        if points != value or points < 9 or points % 2 == 0:
            raise ValueError(f"points_per_edge must be an odd integer >= 9, got {value!r} (D-42)")
        return points
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    return number


@contextlib.contextmanager
def grid_rule(**targets: float) -> Iterator[None]:
    """Swap the D-42/D-53/D-64 derivation targets for the duration of the block (D-73).

    Module-scoped, not an argument: the gates that re-solve call the solver again from the
    artifact's numerics and must see the same rule. Nestable; no arguments is a no-op.
    """
    values = {name: _validated_target(name, value) for name, value in targets.items()}
    previous = {name: globals()[name.upper()] for name in values}
    globals().update({name.upper(): value for name, value in values.items()})
    try:
        yield
    finally:
        globals().update({name.upper(): value for name, value in previous.items()})


def active_rule() -> dict[str, float]:
    """Return the derivation targets currently differing from :data:`PRODUCTION_RULE`."""
    current = globals()
    return {
        name: current[name.upper()]
        for name in RULE_TARGETS
        if current[name.upper()] != PRODUCTION_RULE[name]
    }


def _rule_sentence() -> str:
    """Describe a non-production rule for ``measurements["grid_overrides"]`` (G5.5)."""
    return "non-production derivation targets: " + ", ".join(
        f"{name}={value:g} (production {PRODUCTION_RULE[name]:g})"
        for name, value in active_rule().items()
    )


def interacting_half_box(structure: AtomicStructure, n_electrons: float) -> float:
    """Return the vacuum margin in bohr for an interacting scenario (D-53).

    ``max(INTERACTING_HALF_BOX, ln(1 / INTERACTING_TAIL_FRACTION) / (2 kappa))`` with
    ``kappa = min_a (Z_a - 0.30 max(0, n_a - 1))`` over the electrons per atom: the hydrogenic decay
    of the most weakly bound centre after Slater screening.
    """
    numbers = [float(z) for z in structure.numbers]
    if not numbers:
        raise ValueError("interacting_half_box requires at least one nucleus")
    per_atom = float(n_electrons) / len(numbers)
    kappa = min(z - SLATER_1S_SCREENING * max(0.0, per_atom - 1.0) for z in numbers)
    kappa = max(kappa, 0.25)
    derived = math.log(1.0 / INTERACTING_TAIL_FRACTION) / (2.0 * kappa)
    return max(INTERACTING_HALF_BOX, derived)


def interacting_box_edge(structure: AtomicStructure, n_electrons: float | None = None) -> float:
    """Return the box edge for an interacting scenario: nuclear span plus twice the derived margin.

    With ``n_electrons`` omitted the margin is the floor :data:`INTERACTING_HALF_BOX`.
    """
    positions = [tuple(float(x) for x in r) for r in structure.positions]
    if not positions:
        raise ValueError("interacting_box_edge requires at least one nucleus")
    span = max(
        max(abs(a[d] - b[d]) for d in range(3)) for a in positions for b in positions
    )
    half = INTERACTING_HALF_BOX if n_electrons is None else interacting_half_box(structure, n_electrons)
    return span + 2.0 * half


#: Relative tolerance for "this nucleus sits on the lattice": 1e-9 of a spacing is round-off (D-64).
LATTICE_TOLERANCE = 1.0e-9


def lattice_spacing(structure: AtomicStructure, target: float) -> tuple[float, dict[str, object]]:
    """Return the spacing nearest ``target`` on which every nucleus is a lattice point (D-64).

    ``h = d / max(1, round(d / target))`` with ``d`` the smallest non-zero coordinate difference
    between nuclei; every other difference must then be a multiple of ``h`` to
    :data:`LATTICE_TOLERANCE`. A single centre gets ``target`` unchanged. Returns ``(h, detail)``;
    ``detail["commensurate"]`` is ``False`` when the nuclei admit no common lattice near ``target``,
    in which case ``h = target`` is returned and the caller records the residual offsets (G5.5).
    """
    positions = [tuple(float(x) for x in r) for r in structure.positions]
    differences = sorted(
        {
            abs(a[d] - b[d])
            for i, a in enumerate(positions)
            for b in positions[i + 1 :]
            for d in range(3)
            if abs(a[d] - b[d]) > 0.0
        }
    )
    if not differences:
        return float(target), {"commensurate": True, "cells_between_nuclei": [], "target": target}
    smallest = differences[0]
    cells = max(1, int(round(smallest / target)))
    h = smallest / cells
    multiples = [delta / h for delta in differences]
    commensurate = all(abs(x - round(x)) <= LATTICE_TOLERANCE * max(1.0, x) for x in multiples)
    if not commensurate:
        return float(target), {
            "commensurate": False,
            "cells_between_nuclei": multiples,
            "target": target,
        }
    return h, {
        "commensurate": True,
        "cells_between_nuclei": [int(round(x)) for x in multiples],
        "target": target,
    }


def lattice_box_edge(edge: float, spacing: float) -> float:
    """Round ``edge`` up to a whole number of cells of ``spacing`` (D-64).

    Up, never to nearest: the edge carries a vacuum margin (D-42, D-53) that rounding down would
    shave. The exact multiple lets :meth:`UniformGrid.from_config` reproduce ``spacing`` to the bit.
    """
    cells = int(math.ceil(edge / spacing - LATTICE_TOLERANCE))
    return max(cells, 1) * spacing


def lattice_offsets(structure: AtomicStructure, origin: tuple[float, float, float], spacing: float) -> list[float]:
    """Distance of every nucleus to its nearest lattice point, in bohr (the O-23 measurement)."""
    out = []
    for r in structure.positions:
        total = 0.0
        for d in range(3):
            frac = (float(r[d]) - origin[d]) / spacing
            total += ((frac - round(frac)) * spacing) ** 2
        out.append(math.sqrt(total))
    return out


def interacting_spacing(structure: AtomicStructure, edge: float | None = None) -> float:
    """Return :func:`lattice_spacing` at :data:`INTERACTING_SPACING` for an interacting scenario.

    ``edge`` is accepted and ignored: since D-64 the edge is rounded to the spacing, not conversely.
    """
    h, _ = lattice_spacing(structure, INTERACTING_SPACING)
    return h


def coulomb_box_edge(structure: AtomicStructure) -> float:
    """Return the box edge in bohr that the physics of ``structure`` requires.

    Single centre ``2 * COULOMB_HALF_BOX_PER_Z / Z``; multi centre the nuclear span plus
    :data:`VACUUM_PADDING`, the box being irrelevant there beyond a few bohr of vacuum.
    """
    numbers = [int(z) for z in structure.numbers]
    if not numbers:
        raise ValueError("coulomb_box_edge requires at least one nucleus")
    z_max = max(max(numbers), 1)
    if len(numbers) == 1:
        return 2.0 * COULOMB_HALF_BOX_PER_Z / z_max
    positions = [tuple(float(x) for x in r) for r in structure.positions]
    span = max(
        max(abs(a[d] - b[d]) for d in range(3)) for a in positions for b in positions
    )
    return span + VACUUM_PADDING


def coulomb_spacing(structure: AtomicStructure, edge: float) -> float:
    """Return the spacing in bohr that pairs with ``edge`` for ``structure``.

    Single centre ``edge / (POINTS_PER_EDGE - 1)``, so the rule scales with ``Z``; multi centre the
    lattice-compatible spacing nearest :data:`MOLECULAR_SPACING` (D-64), independent of the box.
    """
    numbers = [int(z) for z in structure.numbers]
    if len(numbers) == 1:
        return edge / (POINTS_PER_EDGE - 1)
    h, _ = lattice_spacing(structure, MOLECULAR_SPACING)
    return h


def grid_config_for_scenario(
    scenario: "Any", config: GridConfig, *, derive_grid: bool = True, interacting: bool = False
) -> tuple[GridConfig, dict[str, str]]:
    """Reconcile a numerics grid configuration with what a scenario's physics dictates.

    A few numerics values are determined by the physics: an infinite well needs a box of its own
    edge and odd reflection, and on the cusp-factorised path box and spacing follow from the nuclei
    (D-42, D-53). ``derive_grid=False`` bypasses that rule, for the convergence gates G3.1/G3.2/G3.5
    whose job is to vary the very numbers it fixes; ``interacting`` selects the D-53 rule, D-42
    being exact only for the bare eigenproblem.

    Returns ``(resolved_config, overrides)``, ``overrides`` mapping each changed field to a sentence
    explaining why, for :attr:`~contract.RunArtifact.measurements` (gate G5.5: fallbacks are
    allowed, hidden fallbacks are not).
    """
    from contract import ExternalPotentialKind

    overrides: dict[str, str] = {}
    updates: dict[str, object] = {}
    if active_rule():
        overrides["grid_rule"] = _rule_sentence()

    if scenario.external.kind is ExternalPotentialKind.PARTICLE_IN_BOX:
        length = scenario.external.box_length
        if config.box_lengths != (length, length, length):
            updates["box_lengths"] = (length, length, length)
            overrides["box_lengths"] = (
                f"set to the well edge {length} bohr: an infinite well discretised on a box of a "
                f"different size is a different physical system, not a coarser one"
            )
        if config.domain is not DomainMode.BOX:
            updates["domain"] = DomainMode.BOX
            overrides["domain"] = "an infinite well has no atoms to mask spheres around"
        if config.boundary is not BoundaryMode.ODD_REFLECTION:
            updates["boundary"] = BoundaryMode.ODD_REFLECTION
            overrides["boundary"] = (
                "a hard wall requires odd reflection; with a zero-filled halo a wide stencil reads "
                "values the sine eigenfunctions do not have and loses its nominal order"
            )
    elif config.boundary is BoundaryMode.ODD_REFLECTION:
        updates["boundary"] = BoundaryMode.ZERO
        overrides["boundary"] = (
            "odd reflection is only correct at a hard wall; this scenario decays into vacuum"
        )

    # The Coulomb sizing rule (D-42): a single centre is box-limited, a molecule spacing-limited.
    if (
        derive_grid
        and interacting
        and scenario.external.kind is ExternalPotentialKind.NUCLEAR_COULOMB
        and scenario.structure.n_atoms > 0
    ):
        from .structure import electron_count

        n_electrons = electron_count(
            scenario.structure, scenario.external, scenario.electrons.n_electrons
        )
        half = interacting_half_box(scenario.structure, n_electrons)
        derived_h, lattice = lattice_spacing(scenario.structure, INTERACTING_SPACING)
        derived_edge = lattice_box_edge(interacting_box_edge(scenario.structure, n_electrons), derived_h)
        overrides["lattice"] = _lattice_sentence(derived_h, INTERACTING_SPACING, lattice)
        if config.box_lengths != (derived_edge, derived_edge, derived_edge):
            updates["box_lengths"] = (derived_edge, derived_edge, derived_edge)
            overrides["box_lengths"] = (
                f"derived {derived_edge:g} bohr for an interacting scenario (D-53): the nuclear span "
                f"plus {half:.3g} bohr of vacuum on each side, sized so that the Slater-screened "
                f"hydrogenic tail leaves {INTERACTING_TAIL_FRACTION:g} of the charge outside (floor "
                f"{INTERACTING_HALF_BOX:g} bohr), rounded up to a whole number of cells of "
                f"h = {derived_h:g} (D-64); the ladder script measures the residual truncation"
            )
        if config.domain is not DomainMode.BOX:
            updates["domain"] = DomainMode.BOX
            overrides["domain"] = "the derived edge is a box edge (D-53)"
        if abs(config.spacing - derived_h) > 1e-12:
            updates["spacing"] = derived_h
            overrides["spacing"] = (
                f"derived {derived_h:g} bohr for an interacting scenario (D-53): once v_H + v_xc "
                f"enter, the transformed amplitude is no longer constant and the atomic problem is "
                f"spacing-limited like a molecule; the D-42 atomic grid (h = 1/Z) integrates the "
                f"density to 15 % and is not used here"
            )
    elif (
        derive_grid
        and scenario.external.kind is ExternalPotentialKind.NUCLEAR_COULOMB
        and scenario.structure.n_atoms > 0
    ):
        derived_edge = coulomb_box_edge(scenario.structure)
        derived_h = coulomb_spacing(scenario.structure, derived_edge)
        single = scenario.structure.n_atoms == 1
        z_max = max(int(z) for z in scenario.structure.numbers)
        if not single:
            # The edge holds a whole number of cells (D-64), so `from_config` reproduces `derived_h`
            # exactly and every nucleus is a lattice point.
            _, lattice = lattice_spacing(scenario.structure, MOLECULAR_SPACING)
            derived_edge = lattice_box_edge(derived_edge, derived_h)
            overrides["lattice"] = _lattice_sentence(derived_h, MOLECULAR_SPACING, lattice)

        if config.box_lengths != (derived_edge, derived_edge, derived_edge):
            updates["box_lengths"] = (derived_edge, derived_edge, derived_edge)
            overrides["box_lengths"] = (
                f"derived {derived_edge:g} bohr: a single nucleus of charge Z = {z_max} needs "
                f"{COULOMB_HALF_BOX_PER_Z / z_max:g} bohr of half-box, because the transformed "
                f"amplitude is constant and the Dirichlet error is carried by the weight "
                f"exp(-2 Z R) at the edge"
                if single
                else
                f"derived {derived_edge:g} bohr: the nuclear span plus {VACUUM_PADDING:g} bohr of "
                f"vacuum, rounded up to a whole number of cells of h = {derived_h:g} (D-64). "
                f"Measured flat beyond this -- a multi-centre error is set by the spacing, "
                f"not by the box"
            )
        if config.domain is not DomainMode.BOX:
            updates["domain"] = DomainMode.BOX
            overrides["domain"] = (
                "the derived edge is a box edge; masked spheres would size the domain from "
                "mask_radius instead and silently discard the derivation"
            )
        if abs(config.spacing - derived_h) > 1e-12:
            updates["spacing"] = derived_h
            overrides["spacing"] = (
                f"derived {derived_h:g} bohr, which is {POINTS_PER_EDGE} points along each "
                f"edge. On the transformed single-centre problem the solution is constant, so "
                f"refining h only accumulates round-off: hydrogen measures 4.7e-14 Ha at 21 "
                f"points and 1.2e-12 Ha at 129, for 0.28 s against 262 s"
                if single
                else
                f"derived {derived_h:g} bohr: a multi-centre error is spacing-limited and "
                f"converges monotonically only below h = 0.5; the value is the lattice-compatible "
                f"spacing nearest {MOLECULAR_SPACING:g} (D-64)"
            )

    if not updates:
        return config, overrides
    fields = {f.name: getattr(config, f.name) for f in dc_fields(GridConfig)}
    fields.update(updates)
    return GridConfig(**fields), overrides  # type: ignore[arg-type]


def _lattice_sentence(spacing: float, target: float, detail: dict[str, object]) -> str:
    """The recorded explanation of the lattice rule (D-64) for ``measurements["grid_overrides"]``."""
    cells = detail.get("cells_between_nuclei", [])
    if not detail.get("commensurate", True):
        return (
            f"the nuclei admit no common lattice near h = {target:g} (coordinate differences of "
            f"{cells} cells); h = {spacing:g} is used and nucleus_offgrid_bohr records the residue (D-64)"
        )
    if not cells:
        return f"single centre: h = {spacing:g}, the nucleus anchors the lattice (D-64)"
    return (
        f"h = {spacing:g} chosen so that every inter-nuclear coordinate difference is a whole "
        f"number of cells ({cells}); the lattice is anchored on the first nucleus, so every "
        f"nucleus is a lattice point (D-64, closes O-23)"
    )


def lattice_anchor(scenario: "Any", *, derive_grid: bool) -> tuple[float, float, float] | None:
    """Return the lattice anchor for a solve: the first nucleus of a derived grid, else ``None``.

    ``None`` -- the coordinate origin is a lattice point, the pre-D-64 convention -- when the grid
    is not derived (the gates that vary the grid or the nuclei against a fixed lattice, whose sweeps
    an anchor following the nuclei would flatten) or when the scenario has no nuclei.
    """
    if not derive_grid or scenario.structure.n_atoms == 0:
        return None
    return tuple(float(x) for x in scenario.structure.positions[0])  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class GridGeometry:
    """The box a grid is built in: its origin, its shape in points, and its spacing.

    Separate from :class:`UniformGrid` so a geometry can be compared and its memory checked before
    the mask and the coordinate table are allocated (risk R2).
    """

    shape: tuple[int, int, int]
    spacing: float
    origin: tuple[float, float, float]

    @property
    def n_box_points(self) -> int:
        """Number of points in the enclosing box, before masking."""
        return self.shape[0] * self.shape[1] * self.shape[2]

    def lengths(self) -> tuple[float, float, float]:
        """Edge lengths of the box in bohr, measured between the outermost points."""
        return tuple((n - 1) * self.spacing for n in self.shape)  # type: ignore[return-value]


class UniformGrid:
    """A uniform Cartesian grid with an optional masked domain.

    Fields are stored flat, shape ``(..., n_points)``, in the order the mask enumerates them. The
    grid owns the padded buffer the operators use, the quadrature weights and the point coordinates.
    """

    def __init__(
        self,
        geometry: GridGeometry,
        mask: torch.Tensor,
        fd_order: int = 8,
        gradient_order: int = 8,
        boundary: BoundaryMode = BoundaryMode.ZERO,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        """Build a grid from a geometry and a boolean mask over the enclosing box.

        ``mask`` has shape ``geometry.shape``, ``True`` inside the domain; an empty domain raises.
        """
        if tuple(mask.shape) != geometry.shape:
            raise ValueError(f"mask shape {tuple(mask.shape)} != box shape {geometry.shape}")
        n_points = int(mask.sum().item())
        if n_points == 0:
            raise ValueError("the computational domain is empty; check mask_radius and box_lengths")

        self.geometry = geometry
        self.shape: tuple[int, int, int] = geometry.shape
        self.spacing: float = geometry.spacing
        self.origin: tuple[float, float, float] = geometry.origin
        self.boundary = boundary
        self.fd_order = fd_order
        self.gradient_order = gradient_order
        self.device = torch.device(device)
        self.dtype = dtype
        #: Grid settings :func:`grid_config_for_scenario` overrode, with a reason each (gate G5.5).
        self.resolution_overrides: dict[str, str] = {}

        self._mask = mask.to(self.device)
        self._n_points = n_points
        self._pad = max(stencil_half_width(2, fd_order), stencil_half_width(1, gradient_order))
        self._padded_shape = tuple(n + 2 * self._pad for n in self.shape)

        box_index = torch.nonzero(self._mask, as_tuple=False)  # (n_points, 3)
        self._box_index = box_index
        strides = (
            self._padded_shape[1] * self._padded_shape[2],
            self._padded_shape[2],
            1,
        )
        shifted = box_index + self._pad
        self._pad_flat_index = (
            shifted[:, 0] * strides[0] + shifted[:, 1] * strides[1] + shifted[:, 2] * strides[2]
        ).to(torch.long)
        self._box_flat_index = (
            box_index[:, 0] * (self.shape[1] * self.shape[2])
            + box_index[:, 1] * self.shape[2]
            + box_index[:, 2]
        ).to(torch.long)
        #: True when the mask keeps every box point: ``torch.nonzero`` lists points in C order, so
        #: the masked field is the flattened box and scatter/gather are plain copies -- no
        #: ``index_put_``, no ``index`` (D-60). Copies move bits unchanged, so the numbers stay
        #: those of the indexed path, which is verified against the index once here.
        self._full_box = n_points == geometry.n_box_points and bool(
            torch.equal(
                self._box_flat_index,
                torch.arange(n_points, device=self.device, dtype=torch.long),
            )
        )

    # --- Construction ---

    @classmethod
    def from_config(
        cls,
        config: GridConfig,
        structure: AtomicStructure | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
        anchor: tuple[float, float, float] | None = None,
    ) -> "UniformGrid":
        """Build the grid a configuration and a structure jointly imply.

        ``DomainMode.BOX`` uses the explicit ``box_lengths`` and keeps every point;
        ``MASKED_SPHERES`` sizes the box from the atoms plus ``mask_radius``. For ``ODD_REFLECTION``
        the walls sit one spacing outside the outermost points, so an edge ``L`` holds ``L/h - 1``
        interior points per axis -- which makes the analytic spectrum of the infinite well the
        spectrum of the discrete problem.

        ``anchor`` (D-64) is a point the lattice must pass through; the box is otherwise centred on
        the coordinate origin as before. ``None`` keeps the pre-D-64 convention, the origin itself
        on the lattice: the two agree whenever the anchor is a multiple of ``h`` from the origin
        (every single-centre scenario, H2+ at R = 2 and 8), so those grids do not move. The solvers
        pass the first nucleus for a derived grid (:func:`lattice_anchor`), ``None`` when a gate
        varies the grid.
        """
        if config.domain is DomainMode.BOX:
            if config.box_lengths is None:  # pragma: no cover - contract validates this
                raise ValueError("DomainMode.BOX requires box_lengths")
            lengths = config.box_lengths
            h = config.spacing
            if config.boundary is BoundaryMode.ODD_REFLECTION:
                counts = tuple(max(int(round(length / h)) - 1, 1) for length in lengths)
                h = lengths[0] / (counts[0] + 1)
                origin = tuple(-0.5 * length + h for length in lengths)
            else:
                counts = tuple(max(int(round(length / h)) + 1, 1) for length in lengths)
                h = lengths[0] / (counts[0] - 1) if counts[0] > 1 else h
                # Snap the box so the coordinate origin -- or the anchor -- lands exactly on a grid
                # point: the -Z/r regularisation applies only at a coincident point, and a nucleus
                # off the lattice changes the radial shells the Kato fit of G1.11 reads (O-23).
                if anchor is None:
                    origin = tuple(-h * round(0.5 * length / h) for length in lengths)
                else:
                    origin = tuple(
                        float(a) - h * round((float(a) + 0.5 * length) / h)
                        for a, length in zip(anchor, lengths)
                    )
            geometry = GridGeometry(shape=counts, spacing=h, origin=origin)  # type: ignore[arg-type]
            mask = torch.ones(counts, dtype=torch.bool, device=device)
            return cls(
                geometry,
                mask,
                fd_order=config.fd_order,
                gradient_order=config.fd_gradient_order,
                boundary=config.boundary,
                device=device,
                dtype=dtype,
            )

        if structure is None or structure.n_atoms == 0:
            raise ValueError(
                "DomainMode.MASKED_SPHERES needs atoms to draw spheres around; an atom-free "
                "scenario must use DomainMode.BOX with explicit box_lengths"
            )
        origin_xyz, lengths = bounding_box(structure, config.mask_radius)
        h = config.spacing
        counts = tuple(int(math.ceil(length / h)) + 1 for length in lengths)
        geometry = GridGeometry(shape=counts, spacing=h, origin=origin_xyz)  # type: ignore[arg-type]

        coords = _box_coordinates(geometry, device=device, dtype=dtype)
        positions = positions_tensor(structure, device=device, dtype=dtype)
        mask = torch.zeros(counts, dtype=torch.bool, device=device)
        radius_sq = config.mask_radius**2
        for atom in range(positions.shape[0]):
            delta = coords - positions[atom]
            mask |= (delta * delta).sum(dim=-1) <= radius_sq
        return cls(
            geometry,
            mask,
            fd_order=config.fd_order,
            gradient_order=config.fd_gradient_order,
            boundary=config.boundary,
            device=device,
            dtype=dtype,
        )

    # --- Contract surface ---

    @property
    def n_points(self) -> int:
        """Number of points inside the mask."""
        return self._n_points

    @property
    def mask(self) -> torch.Tensor:
        """The boolean domain mask over the enclosing box."""
        return self._mask

    @property
    def volume_element(self) -> float:
        """The quadrature weight of a single point, ``h**3``, in bohr cubed."""
        return self.spacing**3

    @property
    def weights(self) -> torch.Tensor:
        """Quadrature weights, shape ``(n_points,)``.

        Materialised though uniform: the contract declares a weight array and the corpus stores one.
        """
        return torch.full(
            (self._n_points,), self.volume_element, device=self.device, dtype=torch.float64
        )

    def points(self) -> torch.Tensor:
        """Cartesian coordinates of the domain points, shape ``(n_points, 3)``, in bohr."""
        origin = torch.tensor(self.origin, device=self.device, dtype=self.dtype)
        return origin + self._box_index.to(self.dtype) * self.spacing

    def integrate(self, field: torch.Tensor) -> torch.Tensor:
        """Integrate a field over the domain, accumulated in float64 whatever the operand dtype."""
        return (field.to(torch.float64) * self.volume_element).sum(dim=-1)

    def inner(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Return the L2 inner product ``<a|b>`` over the domain, accumulated in float64."""
        return self.integrate(a.to(torch.float64) * b.to(torch.float64))

    def norm(self, field: torch.Tensor) -> torch.Tensor:
        """Return the L2 norm of a field over the domain."""
        return self.inner(field, field).clamp_min(0.0).sqrt()

    def scatter_to_box(self, field: torch.Tensor) -> torch.Tensor:
        """Expand a masked field into the full box, zero-filled outside the domain."""
        batch = field.shape[:-1]
        if self._full_box:
            # A fresh contiguous copy, as the indexed path returns: callers may write into it.
            return field.clone(memory_format=torch.contiguous_format).view(*batch, *self.shape)
        flat = torch.zeros(
            (*batch, self.shape[0] * self.shape[1] * self.shape[2]),
            device=field.device,
            dtype=field.dtype,
        )
        flat[..., self._box_flat_index] = field
        return flat.view(*batch, *self.shape)

    def gather_from_box(self, box: torch.Tensor) -> torch.Tensor:
        """Restrict a full-box field to the masked domain."""
        batch = box.shape[: box.dim() - 3]
        if self._full_box:
            # A fresh tensor, as indexing returns (callers mutate the result in place): reshape
            # already copies a non-contiguous box; a view of the box is cloned.
            flat = box.reshape(*batch, -1)
            if flat.untyped_storage().data_ptr() == box.untyped_storage().data_ptr():
                flat = flat.clone(memory_format=torch.contiguous_format)
            return flat
        return box.reshape(*batch, -1)[..., self._box_flat_index]

    def scatter_to_padded(self, field: torch.Tensor) -> torch.Tensor:
        """Expand a masked field into the padded box and fill the halo per the boundary mode.

        The halo is the boundary condition: ``ZERO`` is Dirichlet for a decayed bound state,
        ``ODD_REFLECTION`` the closure the sine eigenfunctions of a hard-walled box satisfy and the
        only way a wide stencil keeps its nominal order there.
        """
        batch = field.shape[:-1]
        flat = torch.zeros(
            (*batch, self._padded_shape[0] * self._padded_shape[1] * self._padded_shape[2]),
            device=field.device,
            dtype=field.dtype,
        )
        padded = flat.view(*batch, *self._padded_shape)
        if self._full_box:
            p = self._pad
            interior = padded[..., p : p + self.shape[0], p : p + self.shape[1], p : p + self.shape[2]]
            interior.copy_(field.reshape(*batch, *self.shape))
        else:
            flat[..., self._pad_flat_index] = field
        if self.boundary is BoundaryMode.ODD_REFLECTION:
            _fill_odd_reflection(padded, self._pad, self.shape)
        return padded

    def laplacian(self, field: torch.Tensor) -> torch.Tensor:
        """Apply the order-``fd_order`` finite-difference Laplacian, shape-preserving."""
        padded = self.scatter_to_padded(field)
        out = laplacian_from_padded(padded, self.spacing, self.fd_order, self.shape, self._pad)
        return self.gather_from_box(out)

    def gradient(self, field: torch.Tensor) -> torch.Tensor:
        """Return the Cartesian gradient, shape ``(..., 3, n_points)``."""
        padded = self.scatter_to_padded(field)
        out = gradient_from_padded(
            padded, self.spacing, self.gradient_order, self.shape, self._pad
        )
        return self.gather_from_box(out)

    def partial_derivative(self, field: torch.Tensor, axis: int) -> torch.Tensor:
        """Return one Cartesian derivative, shape ``(..., n_points)``.

        Equal to ``gradient(field)[..., axis, :]`` bit for bit, for a third of the work: the three
        components are accumulated independently, so two of them need never be computed.
        """
        padded = self.scatter_to_padded(field)
        out = component_from_padded(
            padded, self.spacing, self.gradient_order, self.shape, self._pad, axis
        )
        return self.gather_from_box(out)

    # --- Reporting ---

    def memory_estimate(self, n_states: int, dtype: torch.dtype, blocks: int = 3) -> float:
        """Return the live orbital-block memory in bytes for ``n_states`` states.

        Three blocks is what the Chebyshev filter needs live at once (``01_PROJECT.md (Part B)`` §8).
        """
        itemsize = torch.empty((), dtype=dtype).element_size()
        return float(blocks * n_states * self._n_points * itemsize)

    def describe(self) -> dict[str, object]:
        """Return a summary for logs and for the record's ``grid`` group."""
        box_points = self.geometry.n_box_points
        return {
            "shape": self.shape,
            "spacing_bohr": self.spacing,
            "origin_bohr": self.origin,
            "n_points": self._n_points,
            "n_box_points": box_points,
            "mask_fraction": self._n_points / box_points,
            "boundary": self.boundary.value,
            "fd_order": self.fd_order,
            "gradient_order": self.gradient_order,
            "pad": self._pad,
        }

    def __repr__(self) -> str:
        """Return a one-line summary naming the shape, spacing and masked fraction."""
        frac = self._n_points / self.geometry.n_box_points
        return (
            f"UniformGrid(shape={self.shape}, h={self.spacing:.4f} bohr, "
            f"n_points={self._n_points} ({frac:.1%} of box), boundary={self.boundary.value})"
        )


def _box_coordinates(
    geometry: GridGeometry, device: torch.device | str, dtype: torch.dtype
) -> torch.Tensor:
    """Return the coordinates of every point of the enclosing box, shape ``(nx, ny, nz, 3)``."""
    axes = [
        torch.arange(geometry.shape[d], device=device, dtype=dtype) * geometry.spacing
        + geometry.origin[d]
        for d in range(3)
    ]
    mesh = torch.meshgrid(*axes, indexing="ij")
    return torch.stack(mesh, dim=-1)


def _fill_odd_reflection(padded: torch.Tensor, pad: int, shape: tuple[int, int, int]) -> None:
    """Fill the halo of a padded box by odd reflection about the walls, in place.

    The wall sits one spacing outside the outermost interior point, so the halo entry adjacent to
    the domain is the wall itself and stays zero; each entry beyond it is the negated mirror of an
    interior point. Written per axis in sequence, which fills the corners correctly.
    """
    for axis in range(3):
        n = shape[axis]
        for offset in range(1, pad):
            # Low side: index (pad - 1 - offset) mirrors interior index (offset - 1).
            lo_dst = _axis_slice(padded, axis, pad - 1 - offset)
            lo_src = _axis_slice(padded, axis, pad + offset - 1)
            torch.neg(lo_src, out=lo_dst)
            # High side: index (pad + n + offset) mirrors interior index (n - offset).
            hi_dst = _axis_slice(padded, axis, pad + n + offset)
            hi_src = _axis_slice(padded, axis, pad + n - offset)
            torch.neg(hi_src, out=hi_dst)


def _axis_slice(padded: torch.Tensor, axis: int, index: int) -> torch.Tensor:
    """Return the hyperplane of a padded box at ``index`` along one of the three spatial axes."""
    spatial = padded.dim() - 3
    slices: list[object] = [slice(None)] * padded.dim()
    slices[spatial + axis] = index
    return padded[tuple(slices)]  # type: ignore[index]
