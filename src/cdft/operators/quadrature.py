"""Cusp-aware quadrature: integrating ``f^2 * (smooth)`` exactly enough on a uniform grid (A-9, C7).

Integrals are ``integral f^2 g d^3r``, ``f^2 = exp(-2 sum_a Z_a s_a)`` closed form, ``g`` smooth.
The plain ``h^3`` sum is spectral on ``g`` but ``O((Zh)^4)`` on the cusp (G1.13). Do not patch that
with exact cell integrals near the nucleus: the mixed rule loses the Euler--Maclaurin cancellation
and is ``O(h^3)`` on a smooth integrand. The seam has to be smooth:

    Q[f^2 g] = sum_a Q_sphere,a[ P_a f^2 g ] + sum_i h^3 (1 - sum_a P_a)(r_i) f^2(r_i) g(r_i),

``p_a(r) = (1/2) erfc((s_a - r_c)/w)``, ``P_a = beta_a (1 - prod_b (1 - p_b))`` with the erf cells
``beta_a`` of D-58 (the weights sum to one exactly). In a sphere ``f^2``, ``grad u``, ``|grad u|^2``
are analytic at the nodes and ``g`` is interpolated; outside, ``1 - p_a ~ 1e-10`` at the nucleus.
The plain sum aliases the taper shell at ``exp(-pi^2 w^2/h^2) = 5.7e-8`` of its content (w = 1.3 h);
a single shell centred on a lattice nucleus cancels that by symmetry (4e-11 measured on H at
Z h = 1/4), a second centre modulates the shell and does not (2.7e-8 on H2+, the whole of A-11's
residue), so **overlapping spheres take their mass weights' far part from lumped weights**
``Q_far[f^2 ell_i]`` on a temporary twice-refined lattice (D-75); the far sums of grid fields keep
the plain ``h^3`` rule, a cusped ``v_xc`` being uninterpolable.
Being linear in the grid values of ``g``, the rule reduces to mass weights ``omega_i`` with
``sum_i omega_i g_i = Q[f^2 I[g]]`` -- the eigensolver's measure, and to lumped weights
``omega~_i = Q[f^2 v ell_i]`` for ``<a|v|b>`` (interpolating the *product* ``ab``, not the
factors), so a potential stays a multiplication by ``omega~_i / omega_i`` and needs no mass matrix.
The GGA term goes by parts onto the interpolant's gradient, so no divergence of a cusped field is
ever formed (D-52 item 3). Rationale: D-54, D-58, D-75, ``docs/03_METHOD.md`` (Parts B and E).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import torch

# Only a plain sparse matrix-vector product is used; torch's "beta" CSR warning is noise here.
warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta")
warnings.filterwarnings("ignore", message="Sparse invariant checks are implicitly disabled")

from ..grid import UniformGrid
from ..precision import csr_index_dtype
from .cusp import CuspFactor
from .gauss_legendre import gauss_legendre

__all__ = ["CuspQuadrature", "SphereRule", "hydrogenic_1s_potential", "lagrange_weights", "TRANSPOSE_AS_CSC"]

#: Hold ``L^T`` as the CSC view of ``L`` instead of a second CSR (D-69): halves the operator memory
#: and skips the SciPy build, bitwise equal on the CPU but ~15x slower there because torch converts
#: a CSC matrix per call; CUDA cost unmeasured. Part of the geometry-cache key.
TRANSPOSE_AS_CSC = False


def lagrange_weights(
    t: torch.Tensor, degree: int, with_derivative: bool = True
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return the Lagrange basis values and derivatives at fractional offset ``t`` in ``[0, 1)``.

    Stencil: the ``degree + 1`` points ``-half .. -half + degree`` around ``[0, 1]``. Shapes
    ``(n, degree + 1)``; the derivative is in the index coordinate, so the caller divides by the
    spacing. ``with_derivative=False`` returns ``None`` for it and skips ~1000 small kernels.
    """
    half = degree // 2
    offsets = torch.arange(-half, -half + degree + 1, dtype=t.dtype, device=t.device)
    x = t[:, None] - offsets[None, :]  # (n, k): distance from node to each stencil point
    weights = torch.ones_like(x)
    derivative = torch.zeros_like(x)
    # Node differences as host floats: offsets[j] - offsets[m] is exactly the integer j - m, so the
    # float64 value matches the tensor read without a device sync per pair when ``t`` is on a GPU.
    for j in range(degree + 1):
        for m in range(degree + 1):
            if m == j:
                continue
            denominator = float(j - m)
            weights[:, j] = weights[:, j] * x[:, m] / denominator
    if not with_derivative:
        return weights, None
    # d/dt ell_j = sum_{m != j} 1/(x_j - x_m) prod_{k != j, m} (t - x_k)/(x_j - x_k)
    for j in range(degree + 1):
        total = torch.zeros_like(t)
        for m in range(degree + 1):
            if m == j:
                continue
            term = torch.ones_like(t) / float(j - m)
            for k in range(degree + 1):
                if k in (j, m):
                    continue
                term = term * x[:, k] / float(j - k)
            total = total + term
        derivative[:, j] = total
    return weights, derivative


# --- The sphere rule around one nucleus ---


@dataclass(slots=True)
class SphereRule:
    """Nodes and weights of the product rule around one nucleus, in the nucleus's frame."""

    atom: int
    centre: torch.Tensor
    """Nucleus position, ``(3,)``."""
    radius_partition: float
    """``r_c``: the centre of the taper ``p_a = (1/2) erfc((s - r_c)/w)``."""
    radius_sphere: float
    """Where ``p_a`` is below the cutoff and the sphere rule stops."""
    points: torch.Tensor
    """Node coordinates, ``(m, 3)``."""
    weights: torch.Tensor
    """Bare volume weights ``r^2 dr dOmega``, ``(m,)``."""
    n_radial: int
    n_angular: int


def _gauss_legendre(order: int) -> tuple[np.ndarray, np.ndarray]:
    """The ``order``-point rule on ``[-1, 1]``; see :mod:`cdft.operators.gauss_legendre`."""
    return gauss_legendre(order)


def _sphere_rule(
    atom: int,
    centre: torch.Tensor,
    radius_partition: float,
    radius_sphere: float,
    spacing: float,
    radial_order: int,
    node_spacing: float,
    dtype: torch.dtype,
    device: torch.device,
) -> SphereRule:
    """Build the composite Gauss--Legendre (radius) x product (angles) rule for one sphere.

    The node set must resolve the grid, not only the physics: the rule also defines the mass weights
    ``omega_i = Q[f^2 ell_i]``, and a fixed-size angular rule aliases ``ell_i`` on the outer shells
    and yields negative lumped weights. Hence panels one spacing wide and
    ``n_theta = ceil(sqrt(2 pi) r/(node_spacing h))`` per radial node. Node count is ``(R/h)^3``, so
    the sphere radius is tied to ``h``, not ``1/Z``.
    """
    n_panels = max(2, int(math.ceil(radius_sphere / spacing)))
    edges = np.linspace(0.0, radius_sphere, n_panels + 1)
    gl_nodes, gl_weights = _gauss_legendre(radial_order)
    points: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    n_angular_max = 0
    n_radial = 0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mid, half = 0.5 * (hi + lo), 0.5 * (hi - lo)
        for node, weight in zip(gl_nodes, gl_weights):
            r = mid + half * node
            wr = half * weight * r * r
            n_theta = max(4, int(math.ceil(math.sqrt(2.0 * math.pi) * r / (node_spacing * spacing))))
            n_phi = 2 * n_theta
            cos_nodes, cos_weights = _gauss_legendre(n_theta)
            phis = 2.0 * math.pi * (np.arange(n_phi) + 0.5) / n_phi
            sin_theta = np.sqrt(np.clip(1.0 - cos_nodes**2, 0.0, 1.0))
            directions = np.stack(
                [
                    np.outer(sin_theta, np.cos(phis)).ravel(),
                    np.outer(sin_theta, np.sin(phis)).ravel(),
                    np.repeat(cos_nodes, n_phi),
                ],
                axis=1,
            )
            w_ang = np.repeat(cos_weights, n_phi) * (2.0 * math.pi / n_phi)
            points.append(r * directions)
            weights.append(wr * w_ang)
            n_angular_max = max(n_angular_max, int(directions.shape[0]))
            n_radial += 1
    pts = torch.as_tensor(np.concatenate(points), dtype=dtype, device=device) + centre
    return SphereRule(
        atom=atom,
        centre=centre,
        radius_partition=radius_partition,
        radius_sphere=radius_sphere,
        points=pts,
        weights=torch.as_tensor(np.concatenate(weights), dtype=dtype, device=device),
        n_radial=n_radial,
        n_angular=n_angular_max,
    )


# --- The quadrature ---


class CuspQuadrature:
    """The cusp-aware quadrature for one grid and one set of nuclei (module docstring).

    Everything geometric is built once per geometry and cached: sphere rules, analytic node fields,
    partition weights, interpolation stencils, mass weights.
    """

    #: Lagrange interpolation degree at the nodes; 7 matches the order-8 operator stencils (error
    #: ``(Zh)^8 / 8!``, below 1e-9 at ``Zh = 0.5``).
    degree: int = 7
    #: Taper width in grid spacings; the plain sum aliases the taper by ``exp(-2 pi^2 w^2/h^2)``
    #: (4e-15 at 1.3). An erf taper, not a smoothstep: suppression and resolvability both set by w.
    taper_width_in_spacings: float = 1.3
    #: The sphere ends this many widths past the taper centre, where ``p_a = erfc(4.5)/2 = 1e-10``.
    taper_margin: float = 4.5
    #: Smallest taper centre, in spacings, before ``w`` and ``r_c`` are scaled down together;
    #: below it the rule degrades towards the plain sum.
    min_centre_in_spacings: float = 3.5
    #: Largest sphere radius in spacings: D-42's half-box ``16 h`` minus the ``4.5 h`` margin.
    max_radius_in_spacings: float = 11.5
    #: Gauss points per radial panel of one spacing.
    radial_order: int = 6
    #: Largest arc between neighbouring angular nodes, in spacings (see :func:`_sphere_rule`).
    node_spacing: float = 1.0
    #: Angular node spacing for a sphere overlapping another's (D-58): it must then integrate the
    #: erf cell transition and the neighbour's suppressed cusp, both on the grid scale. 2x nodes.
    node_spacing_overlapping: float = 0.7
    #: Refinement per axis of the temporary lattice the lumped far weights are integrated on when
    #: spheres overlap (D-75). Measured on H2+: ``far(h/2) - far(h/3)`` is 5e-13 of ``integral f^2``.
    far_refine: int = 2
    #: Within this many spacings of a box face the lumped mass weights blend back to the plain ones
    #: (erf of width ``far_edge_width_in_spacings``): the one-sided stencils there make the lumped
    #: weight swing by 2x, which doubles the operator's local spectral radius and the Chebyshev
    #: iterations, while ``f^2`` at the faces is below 1e-12 of its mass.
    far_edge_margin_in_spacings: float = 6.0
    far_edge_width_in_spacings: float = 1.5

    def __init__(self, grid: UniformGrid, factor: CuspFactor) -> None:
        """Build the rule for ``factor``'s nuclei on ``grid``.

        Raises ``ValueError`` if a nucleus is too close to the box edge for a sphere plus an
        interpolation stencil: such a grid must be enlarged, not fall back to the plain rule.
        """
        self.grid = grid
        self.factor = factor
        self.dtype = torch.float64
        self.device = grid.device
        h = grid.spacing
        self.spheres: list[SphereRule] = []
        if factor.is_identity:
            self._build_identity()
            return

        positions = factor.positions.to(self.dtype)
        charges = factor.charges
        n_atoms = len(charges)
        origin = torch.tensor(grid.origin, dtype=self.dtype, device=self.device)
        upper = origin + torch.tensor([(n - 1) * h for n in grid.shape], dtype=self.dtype, device=self.device)
        half = self.degree // 2
        margin = (half + 1.5) * h  # stencil reach on the far side of the node, plus half a cell
        self.taper: list[tuple[float, float]] = []  # (centre, width) per sphere, in bohr
        self.overlapping: list[bool] = []

        for atom in range(n_atoms):
            centre = positions[atom]
            edge_distance = float(torch.minimum(centre - origin, upper - centre).min())
            radius_sphere = min(edge_distance - margin, self.max_radius_in_spacings * h)
            if radius_sphere <= 2.0 * h:
                raise ValueError(
                    f"nucleus {atom} is {edge_distance:.3g} bohr from the box edge; the cusp "
                    f"quadrature needs at least {2.0 * h + margin:.3g} bohr (a two-cell sphere plus "
                    f"a degree-{self.degree} stencil). Enlarge the box."
                )
            # Spheres may overlap and contain each other's nucleus (D-58): the Becke cells in
            # _partition_weights hand the region near b to sphere b smoothly. Do not cap the radius
            # at the internuclear distance -- that squeezes the taper and aliases (A-11, A-3).
            width = self.taper_width_in_spacings * h
            taper_centre = radius_sphere - self.taper_margin * width
            if taper_centre < self.min_centre_in_spacings * h:
                # Too little room: scale centre and width together, preserving the profile shape.
                scale = radius_sphere / (
                    (self.min_centre_in_spacings + self.taper_margin * self.taper_width_in_spacings) * h
                )
                width = self.taper_width_in_spacings * h * scale
                taper_centre = self.min_centre_in_spacings * h * scale
            self.taper.append((taper_centre, width))
            overlapping = False
            if n_atoms > 1:
                others = torch.cat([positions[:atom], positions[atom + 1:]])
                nearest = float((others - centre).norm(dim=-1).min())
                overlapping = nearest < 2.0 * radius_sphere
            self.overlapping.append(overlapping)
            self.spheres.append(
                _sphere_rule(
                    atom, centre, taper_centre, radius_sphere, h, self.radial_order,
                    self.node_spacing_overlapping if overlapping else self.node_spacing,
                    self.dtype, self.device,
                )
            )

        self.points = torch.cat([s.points for s in self.spheres])
        self.owner = torch.cat(
            [torch.full((s.points.shape[0],), s.atom, dtype=torch.long, device=self.device) for s in self.spheres]
        )
        base_weights = torch.cat([s.weights for s in self.spheres])
        self.n_nodes = int(self.points.shape[0])

        self.partition = self._partition_weights(self.points, self.owner)
        self.weights = base_weights * self.partition  # W_node P_node: the node's share
        u = torch.zeros(self.n_nodes, dtype=self.dtype, device=self.device)
        grad_u = torch.zeros((3, self.n_nodes), dtype=self.dtype, device=self.device)
        for atom, charge in enumerate(charges):
            delta = self.points - positions[atom]
            distance = delta.norm(dim=-1).clamp_min(1.0e-300)
            u = u + charge * distance
            grad_u = grad_u + charge * (delta / distance[:, None]).transpose(0, 1)
        self.f2 = torch.exp(-2.0 * u)
        self.grad_u = grad_u
        self.grad_u_squared = (grad_u * grad_u).sum(dim=0)
        self.transformed_potential = -0.5 * self.grad_u_squared

        self._build_stencils()

        grid_points = grid.points().to(self.dtype)
        self.far = self._far_factor(grid_points)
        # The far rule: the plain h^3 far_i for every integral of a grid field, and (D-75) the lumped
        # weight Q_far[f^2 ell_i] for the mass weights alone where a taper shell has a neighbour to
        # break its symmetry -- the measure the solver normalises in is then exact to 1e-9, while a
        # potential with its own cusp (v_xc ~ n^{1/3}) cannot be interpolated onto a finer lattice.
        self.far_weights = grid.volume_element * self.far
        self.plain_far = self.far_weights * factor.weight.to(self.dtype)
        self.far_lumped = any(self.overlapping)
        if self.far_lumped:
            self.lumped_far = self._lumped_far_weights()
            self.far_aliasing = float((self.plain_far.sum() - self.lumped_far.sum()) / self.lumped_far.sum())
        else:
            self.lumped_far = self.plain_far
            self.far_aliasing = 0.0
        self.mass_weights = self.lift(self.weights * self.f2) + self.lumped_far
        if bool((self.mass_weights <= 0.0).any()):
            worst = float(self.mass_weights.min())
            raise ValueError(
                f"the cusp-aware mass weights are not all positive (min {worst:.3e}). The lumped "
                f"weight Q[f^2 ell_i] goes negative where f^2 changes by more than e^-2 across a "
                f"cell, i.e. for Z h > 1 (here max Z h = {max(charges) * h:.2f}); the derived grids "
                f"(D-42: Z h = 1, D-53: Z h <= 0.5) never reach that. Refine the grid."
            )

    def _build_identity(self) -> None:
        """With no nuclei the rule is the plain sum: no spheres, unit far factor."""
        n = self.grid.n_points
        self.points = torch.zeros((0, 3), dtype=self.dtype, device=self.device)
        self.owner = torch.zeros((0,), dtype=torch.long, device=self.device)
        self.n_nodes = 0
        self.partition = self.weights = torch.zeros((0,), dtype=self.dtype, device=self.device)
        self.f2 = self.weights
        self.grad_u = torch.zeros((3, 0), dtype=self.dtype, device=self.device)
        self.grad_u_squared = self.transformed_potential = self.weights
        self.far = torch.ones(n, dtype=self.dtype, device=self.device)
        self.overlapping = []
        self.far_lumped = False
        self.far_aliasing = 0.0
        self.far_weights = torch.full((n,), self.grid.volume_element, dtype=self.dtype, device=self.device)
        self.plain_far = self.far_weights.clone()
        self.lumped_far = self.plain_far
        self.mass_weights = self.plain_far.clone()
        self._strides = (self.grid.shape[1] * self.grid.shape[2], self.grid.shape[2], 1)
        self._operator = self._operator_t = None
        self.csr_index_dtype = None
        self.transpose_layout = None

    def _bump(self, points: torch.Tensor, atom: int) -> torch.Tensor:
        """``p_a(r) = (1/2) erfc((s_a - r_c) / w)``, exactly zero beyond the sphere radius."""
        sphere = self.spheres[atom]
        centre, width = self.taper[atom]
        s = (points - sphere.centre).norm(dim=-1)
        value = 0.5 * torch.erfc((s - centre) / width)
        return torch.where(s < sphere.radius_sphere, value, torch.zeros_like(value))

    #: Width, in spacings, of the erf transition splitting overlapping spheres on their bisecting
    #: surface (D-58). Fixed in *space*, not in Becke's ``mu``: the rules resolve h, not the bond.
    cell_width_in_spacings: float = 1.0

    def _becke_cells(self, points: torch.Tensor) -> torch.Tensor:
        """Cell functions ``beta_a(r)``, shape ``(n_atoms, m)``, summing to one everywhere.

        ``beta_a = P_a / sum_c P_c``, ``P_a = prod_{b != a} (1/2) erfc((s_a - s_b)/(w_c sqrt 2))``:
        one at nucleus ``a``, ``1/2`` on the bisecting surface, ``erfc(R_ab/(w_c sqrt 2))/2`` at a
        neighbouring nucleus -- how far sphere ``a`` suppresses ``b``'s cusp. Width ``w_c`` in space
        whatever the bond length. For one nucleus ``beta = 1``.
        """
        n_atoms = len(self.spheres)
        if n_atoms == 1:
            return torch.ones((1, points.shape[0]), dtype=self.dtype, device=self.device)
        distances = torch.stack([(points - sphere.centre).norm(dim=-1) for sphere in self.spheres])
        width = self.cell_width_in_spacings * self.grid.spacing * math.sqrt(2.0)
        cells = torch.ones((n_atoms, points.shape[0]), dtype=self.dtype, device=self.device)
        for a in range(n_atoms):
            for b in range(n_atoms):
                if a == b:
                    continue
                cells[a] = cells[a] * 0.5 * torch.erfc((distances[a] - distances[b]) / width)
        return cells / cells.sum(dim=0, keepdim=True).clamp_min(1.0e-300)

    def _partition_weights(self, points: torch.Tensor, owner: torch.Tensor) -> torch.Tensor:
        """``P_a = beta_a Q`` at each node of sphere ``a``, ``Q = 1 - prod_b (1 - p_b)`` (D-58).

        ``Q`` is the share taken from the plain sum (complement: :meth:`_far_factor`); the Becke
        cells split it between spheres, so ``sum_a P_a + far = 1`` exactly, every factor is smooth,
        and each sphere's integrand vanishes to high order at any other nucleus it contains.

        The lune (D-75): where a node of sphere ``a`` lies outside sphere ``b``, ``beta_b Q`` has no
        rule of its own (sphere ``b``'s nodes stop at its radius), so its cell is handed to ``a``:
        ``P_a = Q (beta_a + sum_b beta_b [s_b >= R_b])``. The step is at sphere ``b``'s edge, where
        ``Q = p_b ~ 1e-10``; the mass it recovers is 9e-10 of ``integral f^2`` on H2+ and H2.
        """
        near = 1.0 - self._far_factor(points)
        cells = self._becke_cells(points)
        n_atoms = len(self.spheres)
        result = torch.zeros(points.shape[0], dtype=self.dtype, device=self.device)
        for atom in range(n_atoms):
            mask = owner == atom
            share = cells[atom]
            for other in range(n_atoms):
                if other == atom:
                    continue
                sphere = self.spheres[other]
                outside = (points - sphere.centre).norm(dim=-1) >= sphere.radius_sphere
                share = share + cells[other] * outside.to(self.dtype)
            result[mask] = (share * near)[mask]
        return result

    def _lumped_far_weights(self) -> torch.Tensor:
        """The lumped far mass weights ``Q_far[f^2 ell_i]`` of an overlapping geometry (D-75).

        The midpoint rule of ``far f^2 ell_i`` on the lattice refined ``far_refine`` times per axis
        over the whole box (``far`` and ``f^2`` analytic at the fine points, the cardinal function
        ``ell_i`` a tensor product of one-dimensional Lagrange weights, shifted one-sided at the box
        edge), applied as three one-dimensional transposed interpolations; ``sum_i omega_i g_i =
        Q_far[f^2 I[g]]`` for a grid field ``g``. It serves the mass weights only (``g = 1`` and
        the smooth ``|phi|^2``): a field with a cusp of its own, such as ``v_xc ~ n^{1/3}`` or
        ``e_xc``, interpolates badly where the taper's inner tail still carries weight and moved
        the H2 LDA total by 1e-6 Ha when the potentials used these weights, so every other far sum
        keeps the plain ``h^3 far_i``. Whole box, not a blend around the spheres: a blend ``chi``
        of grid-scale width is interpolated to 4e-3 by the degree-7 stencil, which costs 7e-9 of
        ``integral f^2`` on H2+ wherever ``f^2`` is not negligible. The only blend is back to the
        plain weights within ``far_edge_margin_in_spacings`` of a face.
        """
        grid = self.grid
        h = grid.spacing
        refine = int(self.far_refine)
        shape = grid.shape
        build = torch.device("cpu")
        positions = self.factor.positions.to(device=build, dtype=self.dtype)
        charges = self.factor.charges
        # One-dimensional transposed interpolation matrices, coarse x fine, per axis.
        k = self.degree + 1
        half = self.degree // 2
        matrices = []
        fine_axes = []
        for axis in range(3):
            n = shape[axis]
            n_fine = n * refine
            fine = grid.origin[axis] - 0.5 * h + h * (torch.arange(n_fine, dtype=self.dtype, device=build) + 0.5) / refine
            fine_axes.append(fine)
            fractional = (fine - grid.origin[axis]) / h
            first = (torch.floor(fractional).to(torch.long) - half).clamp_(0, n - k)
            t = fractional - first.to(self.dtype)
            x = t[:, None] - torch.arange(k, dtype=self.dtype, device=build)[None, :]
            weights = torch.ones_like(x)
            for j in range(k):
                for m in range(k):
                    if m != j:
                        weights[:, j] = weights[:, j] * x[:, m] / float(j - m)
            matrix = torch.zeros((n, n_fine), dtype=self.dtype, device=build)
            rows = (first[:, None] + torch.arange(k, device=build)[None, :]).reshape(-1)
            cols = torch.arange(n_fine, device=build).repeat_interleave(k)
            matrix.index_put_((rows, cols), weights.reshape(-1), accumulate=True)
            matrices.append(matrix)
        # far f^2 at the fine points, one z-plane at a time to bound memory, then the transposed
        # z-interpolation collapses each plane onto the coarse z index.
        n_x, n_y, n_z = (shape[0] * refine, shape[1] * refine, shape[2] * refine)
        xs, ys = torch.meshgrid(fine_axes[0], fine_axes[1], indexing="ij")
        plane_xy = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=1)
        collapsed = torch.zeros((n_x, n_y, shape[2]), dtype=self.dtype, device=build)
        spheres_cpu = [(s.centre.to(build), s.radius_sphere, self.taper[s.atom]) for s in self.spheres]
        volume = (h / refine) ** 3
        for iz in range(n_z):
            points = torch.cat([plane_xy, fine_axes[2][iz].expand(plane_xy.shape[0], 1)], dim=1)
            far = torch.ones(points.shape[0], dtype=self.dtype, device=build)
            u = torch.zeros(points.shape[0], dtype=self.dtype, device=build)
            for atom, (centre, radius, (taper_centre, width)) in enumerate(spheres_cpu):
                s = (points - centre).norm(dim=-1)
                bump = torch.where(s < radius, 0.5 * torch.erfc((s - taper_centre) / width), torch.zeros_like(s))
                far = far * (1.0 - bump)
                u = u + charges[atom] * s
            values = (volume * far * torch.exp(-2.0 * u)).reshape(n_x, n_y)
            collapsed = collapsed + values[:, :, None] * matrices[2][:, iz][None, None, :]
        lumped = torch.einsum("ax,by,xyc->abc", matrices[0], matrices[1], collapsed).reshape(-1)
        lumped = grid.gather_from_box(lumped.reshape(*shape).to(self.device))
        # The edge blend: plain weights where the stencils are one-sided (docstring of the class).
        points = grid.points().to(self.dtype)
        origin = torch.tensor(grid.origin, dtype=self.dtype, device=self.device)
        upper = origin + torch.tensor([(n - 1) * h for n in shape], dtype=self.dtype, device=self.device)
        face = torch.minimum((points - origin).min(dim=1).values, (upper - points).min(dim=1).values)
        margin = self.far_edge_margin_in_spacings * h
        width = self.far_edge_width_in_spacings * h
        interior = 0.5 * torch.erfc((margin - face) / width)
        return self.plain_far + interior * (lumped - self.plain_far)

    def _far_factor(self, points: torch.Tensor) -> torch.Tensor:
        """``prod_a (1 - p_a)``: the plain sum's share, exactly ``1 - sum_a P_a``."""
        far = torch.ones(points.shape[0], dtype=self.dtype, device=self.device)
        for atom in range(len(self.spheres)):
            far = far * (1.0 - self._bump(points, atom))
        return far

    def _build_stencils(self, index_dtype: torch.dtype | None = None) -> None:
        """The interpolation operator ``L`` (nodes x box points) as one sparse CSR matrix.

        Each node's row holds the ``(degree + 1)^3`` tensor-product Lagrange weights of its
        stencil; ``L^T``, built beside it by a counting sort, gives :meth:`lift`. Node gradients are
        the interpolated *grid* gradient (order-8 finite differences of the smooth field), so one
        matrix serves every pass and :meth:`lift_gradient` applies that composition's transpose.

        The build always runs on the CPU (one-off, SciPy counting sort, and the card would hold the
        int64 column and value arrays for nothing) with int64 indices; other devices get both
        operators with ``index_dtype`` indices (:func:`~cdft.precision.csr_index_dtype`: int32 on
        CUDA unless ``CDFT_CSR_INT64=1``, overridable here). The weights are elementwise products of
        the same numbers either way, so the device changes no value.
        """
        grid = self.grid
        target = self.device
        build = torch.device("cpu")
        if index_dtype is None:
            index_dtype = csr_index_dtype(target)
        h = grid.spacing
        origin = torch.tensor(grid.origin, dtype=self.dtype, device=build)
        fractional = (self.points.to(build) - origin) / h  # (m, 3)
        base = torch.floor(fractional)
        t = fractional - base
        half = self.degree // 2
        k = self.degree + 1
        shape = grid.shape
        weights_1d = []
        index_1d = []
        for axis in range(3):
            w, _ = lagrange_weights(t[:, axis], self.degree, with_derivative=False)
            index = base[:, axis].to(torch.long)[:, None] + torch.arange(-half, -half + k, device=build)[None, :]
            if bool((index < 0).any()) or bool((index >= shape[axis]).any()):
                raise ValueError("an interpolation stencil reaches outside the grid; the sphere radius bound is wrong")
            weights_1d.append(w)
            index_1d.append(index)
        self._strides = (shape[1] * shape[2], shape[2], 1)
        sx, sy, sz = self._strides
        wx, wy, wz = weights_1d
        ix, iy, iz = index_1d
        n_box = shape[0] * shape[1] * shape[2]
        columns = (ix[:, :, None, None] * sx + iy[:, None, :, None] * sy + iz[:, None, None, :] * sz).reshape(-1)
        values = (wx[:, :, None, None] * wy[:, None, :, None] * wz[:, None, None, :]).reshape(-1)
        del fractional, base, t, weights_1d, index_1d, wx, wy, wz, ix, iy, iz
        crow = torch.arange(0, self.n_nodes * k**3 + 1, k**3, dtype=torch.int64, device=build)
        if index_dtype == torch.int32 and max(int(values.numel()), n_box) >= 2**31 - 1:
            index_dtype = torch.int64  # int32 cannot address this operator; int64 is always valid
        self.csr_index_dtype = index_dtype
        native = target.type == "cpu" and index_dtype == torch.int64
        if native:
            self._operator = torch.sparse_csr_tensor(
                crow, columns, values, size=(self.n_nodes, n_box), device=target
            )
        else:
            self._operator = torch.sparse_csr_tensor(
                crow.to(device=target, dtype=index_dtype),
                columns.to(device=target, dtype=index_dtype),
                values.to(device=target),
                size=(self.n_nodes, n_box),
                device=target,
            )
        if TRANSPOSE_AS_CSC:
            # L^T as the CSC view of L: L's compressed rows are L^T's compressed columns.
            del values, columns, crow
            self.transpose_layout = "csc_view"
            self._operator_t = torch.sparse_csc_tensor(
                self._operator.crow_indices(),
                self._operator.col_indices(),
                self._operator.values(),
                size=(n_box, self.n_nodes),
                device=target,
            )
            return
        self.transpose_layout = "csr"
        # L^T via SciPy's CSR -> CSC counting sort (beats torch's argsort and its CSC -> CSR path).
        # The CSC arrays of L are the CSR arrays of L^T, handed to torch without a copy.
        import numpy as np
        import scipy.sparse as sparse

        csc = sparse.csr_matrix(
            (values.cpu().numpy(), columns.cpu().numpy(), np.arange(0, self.n_nodes * k**3 + 1, k**3)),
            shape=(self.n_nodes, n_box),
        ).tocsc()
        if not native:
            # Drop the CPU copies before L^T's arrays materialise: host peak stays at one operator.
            del values, columns, crow
        if native:
            self._operator_t = torch.sparse_csr_tensor(
                torch.from_numpy(csc.indptr.astype(np.int64)).to(self.device),
                torch.from_numpy(csc.indices.astype(np.int64)).to(self.device),
                torch.from_numpy(csc.data).to(self.device),
                size=(n_box, self.n_nodes),
                device=target,
            )
        else:
            numpy_index = np.int32 if index_dtype == torch.int32 else np.int64
            self._operator_t = torch.sparse_csr_tensor(
                torch.from_numpy(csc.indptr.astype(numpy_index)).to(target),
                torch.from_numpy(csc.indices.astype(numpy_index)).to(target),
                torch.from_numpy(csc.data).to(target),
                size=(n_box, self.n_nodes),
                device=target,
            )
        del csc

    # --- Interpolation and its transpose ---

    def interpolate(self, field: torch.Tensor) -> torch.Tensor:
        """Return the degree-``degree`` interpolant of a grid field at the nodes, ``(..., m)``."""
        batch = field.shape[:-1]
        if self.n_nodes == 0:
            return torch.zeros((*batch, 0), dtype=self.dtype, device=self.device)
        box = self.grid.scatter_to_box(field.to(self.dtype)).reshape(-1, self.grid.geometry.n_box_points)
        out = (self._operator @ box.transpose(0, 1)).transpose(0, 1)
        return out.reshape(*batch, self.n_nodes)

    def interpolate_gradient(self, field: torch.Tensor) -> torch.Tensor:
        """Return the gradient at the nodes, ``(..., 3, m)``: the interpolated order-8 grid gradient."""
        if self.n_nodes == 0:
            return torch.zeros((*field.shape[:-1], 3, 0), dtype=self.dtype, device=self.device)
        return self.interpolate(self.grid.gradient(field.to(self.dtype)))

    def lift(self, node_values: torch.Tensor) -> torch.Tensor:
        """Return ``L^T x``: node values scattered onto the grid by the stencil weights, ``(..., n_points)``.

        For ``x = W P f^2 g_node`` this is the near part of ``omega~_i = sum_nodes x ell_i(node)``.
        """
        batch = node_values.shape[:-1]
        if self.n_nodes == 0:
            return torch.zeros((*batch, self.grid.n_points), dtype=self.dtype, device=self.device)
        flat = node_values.to(self.dtype).reshape(-1, self.n_nodes)
        box = (self._operator_t @ flat.transpose(0, 1)).transpose(0, 1)
        return self.grid.gather_from_box(box.reshape(*batch, *self.grid.shape))

    def lift_gradient(self, node_vectors: torch.Tensor) -> torch.Tensor:
        """Return the grid weights of ``sum_nodes W P X . grad I[c]``: ``-div_FD (L^T X)``.

        The node gradient is the interpolated grid gradient, so its transpose is the lift followed
        by the exact transpose of the finite-difference gradient.
        """
        lifted = self.lift(node_vectors)  # (3, n_points)
        divergence = torch.zeros(self.grid.n_points, dtype=self.dtype, device=self.device)
        for d in range(3):
            divergence = divergence + self.grid.partial_derivative(lifted[d], d)
        return -divergence

    def interpolate_at(self, points: torch.Tensor, field: torch.Tensor) -> torch.Tensor:
        """Interpolate a grid field at arbitrary points ``(p, 3)`` with the same degree-7 stencil.

        Used for the density at the nuclei (the Hartree split coefficients). Points must lie at
        least ``degree//2 + 1`` spacings inside the box.
        """
        flat, w3 = self._point_stencil(points)
        box = self.grid.scatter_to_box(field.to(self.dtype)).reshape(-1)
        return (box[flat] * w3).sum(dim=(-3, -2, -1))

    def _point_stencil(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(flat_index, weights)`` of the degree-``degree`` stencil at ``points``, on the device.

        Computed on the CPU in the field-side operation order, so a CPU run is bit-identical, then
        moved to the device once. Cached against the points tensor object and its version counter
        (the cache holds a reference, so an ``id`` cannot be recycled while cached).
        """
        cache = self.__dict__.setdefault("_stencil_cache", [])
        for entry in cache:
            if entry[0] is points and entry[1] == points._version:
                return entry[2], entry[3]
        grid = self.grid
        h = grid.spacing
        host = torch.device("cpu")
        origin = torch.tensor(grid.origin, dtype=self.dtype, device=host)
        fractional = (points.detach().to(device=host, dtype=self.dtype) - origin) / h
        base = torch.floor(fractional)
        t = fractional - base
        half = self.degree // 2
        k = self.degree + 1
        weights = []
        indices = []
        for axis in range(3):
            w, _ = lagrange_weights(t[:, axis], self.degree, with_derivative=False)
            index = base[:, axis].to(torch.long)[:, None] + torch.arange(-half, -half + k, device=host)[None, :]
            if bool((index < 0).any()) or bool((index >= grid.shape[axis]).any()):
                raise ValueError("interpolation point too close to the box edge for the stencil")
            weights.append(w)
            indices.append(index)
        sx, sy, sz = self._strides
        flat = indices[0][:, :, None, None] * sx + indices[1][:, None, :, None] * sy + indices[2][:, None, None, :] * sz
        w3 = weights[0][:, :, None, None] * weights[1][:, None, :, None] * weights[2][:, None, None, :]
        flat, w3 = flat.to(self.device), w3.to(self.device)
        cache.insert(0, (points, points._version, flat, w3))
        del cache[4:]
        return flat, w3

    # --- Integrals and lumped weights ---

    def integrate(self, node_values: torch.Tensor, grid_values: torch.Tensor) -> torch.Tensor:
        """``sum_nodes W P g_node + sum_i h^3 far_i g_i``: the integral of a field given both ways.

        The caller supplies the integrand at the nodes and at the grid points; the partition
        weights are applied here.
        """
        near = (self.weights * node_values).sum(dim=-1) if self.n_nodes else 0.0
        far = (self.far_weights * grid_values).sum(dim=-1)
        return near + far

    def lumped_weights(self, node_values: torch.Tensor, grid_values: torch.Tensor) -> torch.Tensor:
        """Return ``omega~`` with ``sum_i c_i omega~_i = Q[I[c] * field]`` for a field given both ways.

        ``node_values`` includes ``f^2`` where the integrand carries it (e.g. ``f^2 v``).
        """
        return self.lift(self.weights * node_values) + self.far_weights * grid_values

    def lumped_gradient_weights(self, node_vectors: torch.Tensor, grid_vectors: torch.Tensor) -> torch.Tensor:
        """Return ``omega~`` with ``sum_i c_i omega~_i = Q[X . grad I[c]]`` for a vector field ``X``.

        Near: ``sum_nodes W P X . grad ell_i``. Far: the exact transpose of the grid's
        finite-difference gradient, ``-h^3 sum_d D_d[far X_d]``.
        """
        near = self.lift_gradient(self.weights * node_vectors)
        weighted = self.far[None, :] * grid_vectors
        divergence = torch.zeros(self.grid.n_points, dtype=self.dtype, device=self.device)
        for d in range(3):
            divergence = divergence + self.grid.gradient(weighted[d])[d]
        return near - self.grid.volume_element * divergence

    def bare_operator_terms(self) -> tuple[torch.Tensor, torch.Tensor, "Measure"]:
        """Return ``(W~ / omega, h^3 / omega, measure)`` of the bare transformed operator, built once.

        Geometry-only, but the SCF rebuilds the Hamiltonian every iteration, so caching them here
        saves a sparse ``L^T`` product and a host read per iteration. Nothing writes into them.
        """
        cached = self.__dict__.get("_bare_operator_terms")
        if cached is None:
            from ..eigen.measure import Measure

            potential = self.effective_potential(self.transformed_potential, self.factor.transformed_potential)
            inverse_weight = self.grid.volume_element / self.mass_weights
            measure = Measure(self.grid, self.mass_weights / self.grid.volume_element)
            cached = self.__dict__["_bare_operator_terms"] = (potential, inverse_weight, measure)
        return cached

    def effective_potential(self, node_values: torch.Tensor, grid_values: torch.Tensor) -> torch.Tensor:
        """Return the pointwise effective potential ``omega~_i / omega_i`` for ``v`` given both ways.

        The ``f^2`` is supplied here. Away from every sphere this is exactly ``v_i``.
        """
        tilde = self.lumped_weights(node_values * self.f2, grid_values * self.factor.weight.to(self.dtype))
        return tilde / self.mass_weights

    # --- Densities in both representations ---

    def density_gradient_at_nodes(self, rho: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(n, grad n)`` at the nodes: ``grad n = f^2 (grad rho - 2 rho grad u)``, exact in ``grad u``."""
        rho_node = self.interpolate(rho)
        grad_rho = self.interpolate_gradient(rho)
        n = self.f2 * rho_node
        grad_n = self.f2[None, :] * (grad_rho - 2.0 * rho_node[None, :] * self.grad_u)
        return n, grad_n

    def density_gradient_on_grid(self, rho: torch.Tensor) -> torch.Tensor:
        """``grad n = f^2 (grad rho - 2 rho grad u)`` on the grid, ``grad rho`` by finite differences.

        The difference acts on the smooth transformed density only; the cusp direction comes from
        ``grad u`` (cell-averaged at the nucleus, D-39). No stencil differences across the cusp.
        """
        grad_rho = self.grid.gradient(rho)
        f2 = self.factor.weight.to(self.dtype)
        return f2[None, :] * (grad_rho - 2.0 * rho[None, :] * self.factor.grad_u.to(self.dtype))

    def describe(self) -> dict[str, object]:
        """Summary for the record: sphere radii, node counts, the far fraction."""
        return {
            "degree": self.degree,
            "taper_width_in_spacings": self.taper_width_in_spacings,
            "far_rule": f"mass lumped-{self.far_refine}x, fields plain" if self.far_lumped else "plain",
            "far_aliasing": self.far_aliasing,
            "n_nodes": self.n_nodes,
            "csr_index_dtype": str(self.csr_index_dtype).removeprefix("torch.") if self.csr_index_dtype else None,
            "transpose_layout": self.transpose_layout,
            "spheres": [
                {
                    "atom": s.atom,
                    "taper_centre": s.radius_partition,
                    "taper_width": self.taper[s.atom][1],
                    "radius_sphere": s.radius_sphere,
                    "overlapping": bool(self.overlapping[s.atom]) if self.overlapping else False,
                    "n_radial": s.n_radial,
                    "n_angular": s.n_angular,
                }
                for s in self.spheres
            ],
            "mass_weight_sum": float(self.mass_weights.sum()),
            "plain_weight_sum": float(self.grid.integrate(self.factor.weight)) if not self.factor.is_identity else None,
        }


# --- Closed forms used by the Hartree split ---


def hydrogenic_1s_potential(charge: float, distance: torch.Tensor) -> torch.Tensor:
    """Electrostatic potential ``(pi/Z^3) [1 - e^{-2Zs}(1 + Zs)] / s`` of the density ``exp(-2 Z s)``.

    Finite limit ``pi / Z^2`` at ``s = 0``; total charge ``pi / Z^3``. Times ``c_a = n(R_a)`` this is
    the cusp part of the Hartree split ``n = sum_a c_a e^{-2 Z_a s_a} + n_smooth`` (D-54).
    """
    z = float(charge)
    x = z * distance
    small = x < 1.0e-3
    safe = distance.clamp_min(1.0e-300)
    full = (math.pi / z**3) * (1.0 - torch.exp(-2.0 * x) * (1.0 + x)) / safe
    # Series for small s: 1 - e^{-2x}(1+x) cancels catastrophically as x -> 0.
    series = (math.pi / z**2) * (
        1.0 - (2.0 / 3.0) * x**2 + (2.0 / 3.0) * x**3 - 0.4 * x**4 + (8.0 / 45.0) * x**5
    )
    return torch.where(small, series, full)
