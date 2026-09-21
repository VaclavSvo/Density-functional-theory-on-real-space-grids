"""The Kato cusp factor ``f = exp(-sum_a Z_a |r - R_a|)`` and the similarity transform it induces.

``A = f^-1 H f`` removes the ``-Z/r`` pole exactly, for any number of nuclei and any charges:

    A phi = -1/2 lap(phi) + (grad u) . grad(phi) - 1/2 |grad u|^2 phi,   u = sum_a Z_a |r - R_a|

``A`` and ``H`` share a spectrum (G1.12) and ``A`` is self-adjoint in the weighted measure
``w = f^2`` (G0.12). With no nuclei the transform is the identity. Derivation and why the cell
averaging in :meth:`CuspFactor._build_potential` is not optional: D-32, D-35,
``docs/03_METHOD.md`` (Parts B and E).
"""

from __future__ import annotations

import math

import torch

from ..grid import UniformGrid
from .gauss_legendre import gauss_legendre

__all__ = ["CuspFactor"]


def _gauss_legendre(order: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Gauss-Legendre nodes on ``[-1/2, 1/2]`` and weights, scaled so the caller multiplies by ``h``."""
    nodes, weights = gauss_legendre(order)
    return tuple(float(x) * 0.5 for x in nodes), tuple(float(w) * 0.5 for w in weights)


class CuspFactor:
    """The Kato factor evaluated on a grid.

    Holds ``grad u``, the weight ``f^2`` and the transformed potential ``-|grad u|^2 / 2``, all
    geometry-only, so an SCF pays for them once.
    """

    #: Cell averaging is applied within this many spacings of a nucleus.
    _average_radius_cells: float = 2.0

    #: Gauss-Legendre order per Cartesian direction for the cell average.
    _average_quadrature: int = 4

    def __init__(
        self,
        grid: UniformGrid,
        charges: tuple[float, ...],
        positions: torch.Tensor,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        """Build the factor for a set of point nuclei.

        Empty ``charges`` gives ``f = 1``, the identity transform every atom-free scenario needs.
        ``positions`` is in bohr, shape ``(n_atoms, 3)``, used at full precision and never snapped
        to the grid: the cusp is placed analytically.
        """
        self.grid = grid
        self._dtype = dtype
        self._n_cell_averaged = 0
        self._face_weight_cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.charges = tuple(float(z) for z in charges)
        self.positions = positions.to(device=grid.device, dtype=dtype)
        self.is_identity = len(self.charges) == 0

        n_points = grid.n_points
        points = grid.points().to(dtype)

        if self.is_identity:
            self._u = torch.zeros(n_points, device=grid.device, dtype=dtype)
            self._grad_u = torch.zeros((3, n_points), device=grid.device, dtype=dtype)
        else:
            u = torch.zeros(n_points, device=grid.device, dtype=dtype)
            grad_u = torch.zeros((3, n_points), device=grid.device, dtype=dtype)
            for index, charge in enumerate(self.charges):
                delta = points - self.positions[index]
                distance = delta.norm(dim=-1)
                u += charge * distance
                safe = distance.clamp_min(1.0e-30)
                on_nucleus = distance <= 1.0e-12
                unit = torch.where(
                    (~on_nucleus).unsqueeze(-1), delta / safe.unsqueeze(-1), torch.zeros_like(delta)
                )
                grad_u += charge * unit.transpose(0, 1)
            self._u = u
            # Zero at a nucleus is a placeholder: every point within two spacings is overwritten by
            # its cell average in _build_potential.
            self._grad_u = grad_u

        self._weight = torch.exp(-2.0 * self._u)
        self._transformed_potential = self._build_potential()

    def _build_potential(self) -> torch.Tensor:
        """Return ``W = -|grad u|^2 / 2``, cell-averaged wherever it is discontinuous.

        ``grad u`` holds one unit vector per nucleus, so ``|grad u|^2`` jumps at each nucleus. For
        one nucleus that is invisible (``|grad u|^2 = Z^2`` identically); for several, pointwise
        evaluation makes the energy depend on where the nuclei fall relative to the grid, so a finer
        spacing can give a worse H2+ error. The cell average removes that and handles the
        on-nucleus point by the same formula as every other. ``grad u`` is averaged in the same loop
        because the kinetic assembly needs the average of the vector, not of its square.
        """
        potential = -0.5 * (self._grad_u * self._grad_u).sum(dim=0)
        if self.is_identity:
            return potential

        points = self.grid.points().to(self._dtype)
        spacing = self.grid.spacing
        reach = self._average_radius_cells * spacing
        near = torch.zeros(self.grid.n_points, dtype=torch.bool, device=self.grid.device)
        for atom in range(len(self.charges)):
            near |= (points - self.positions[atom]).norm(dim=-1) <= reach
        count = int(near.sum())
        if count == 0:  # pragma: no cover - a grid with no point near a nucleus is degenerate
            return potential

        nodes, weights = _gauss_legendre(self._average_quadrature)
        offsets = [node * spacing for node in nodes]
        selected = points[near]
        mean_square = torch.zeros(count, dtype=self._dtype, device=self.grid.device)
        mean_vector = torch.zeros((count, 3), dtype=self._dtype, device=self.grid.device)
        total = 0.0
        for ix, wx in zip(offsets, weights):
            for iy, wy in zip(offsets, weights):
                for iz, wz in zip(offsets, weights):
                    weight = float(wx * wy * wz)
                    total += weight
                    shift = torch.tensor(
                        [ix, iy, iz], dtype=self._dtype, device=self.grid.device
                    )
                    gradient = torch.zeros_like(selected)
                    for atom, charge in enumerate(self.charges):
                        delta = (selected + shift) - self.positions[atom]
                        distance = delta.norm(dim=-1, keepdim=True).clamp_min(1.0e-300)
                        gradient = gradient + charge * delta / distance
                    mean_square = mean_square + weight * (gradient * gradient).sum(dim=-1)
                    mean_vector = mean_vector + weight * gradient

        potential = potential.clone()
        potential[near] = -0.5 * mean_square / total
        self._grad_u = self._grad_u.clone()
        self._grad_u[:, near] = (mean_vector / total).transpose(0, 1)
        self._n_cell_averaged = count
        return potential

    # --- Fields ---

    @property
    def grad_u(self) -> torch.Tensor:
        """``grad u``, shape ``(3, n_points)``. The coefficient of the drift term."""
        return self._grad_u

    @property
    def weight(self) -> torch.Tensor:
        """``f^2 = exp(-2u)``, shape ``(n_points,)``. The measure that makes ``A`` self-adjoint."""
        return self._weight

    @property
    def transformed_potential(self) -> torch.Tensor:
        """``-|grad u|^2 / 2``, shape ``(n_points,)``; exactly ``-Z^2/2`` for a single nucleus."""
        return self._transformed_potential

    def factor(self) -> torch.Tensor:
        """``f = exp(-u)``, shape ``(n_points,)``. Needed only to map ``phi`` back to ``psi``."""
        return torch.exp(-self._u)

    def face_weights(self, accuracy: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``f^2`` at the staggered faces of each axis, for the divergence form.

        Entry ``d`` has the box shape extended by ``accuracy`` along axis ``d``, indexed from face
        ``-p``, matching :func:`cdft.operators.divergence.staggered_derivative`. Evaluated
        analytically at the face coordinate, never interpolated. Cached per accuracy. Raises
        ``ValueError`` if ``exp(-2u)`` would underflow float64 (``u > 354``): the weight would be
        zero at the far corners, its reciprocal infinite, and the operator would return NaN.
        """
        cached = self._face_weight_cache.get(accuracy)
        if cached is not None:
            return cached

        shape = self.grid.shape
        half = accuracy // 2
        spacing = self.grid.spacing
        origin = torch.tensor(self.grid.origin, device=self.grid.device, dtype=self._dtype)
        weights: list[torch.Tensor] = []
        for axis in range(3):
            sizes = [shape[0], shape[1], shape[2]]
            sizes[axis] += accuracy
            coordinates = []
            for d in range(3):
                index = torch.arange(sizes[d], device=self.grid.device, dtype=self._dtype)
                if d == axis:
                    # Face n sits at grid index n + 1/2, with n running from -half.
                    index = index - half + 0.5
                coordinates.append(origin[d] + index * spacing)
            mesh = torch.meshgrid(*coordinates, indexing="ij")
            u = torch.zeros(sizes, device=self.grid.device, dtype=self._dtype)
            for atom, charge in enumerate(self.charges):
                squared = sum(
                    (mesh[d] - self.positions[atom, d]) ** 2 for d in range(3)
                )
                u = u + charge * squared.sqrt()
            largest = float(u.max())
            if largest > 354.0:
                raise ValueError(
                    f"the cusp weight exp(-2u) underflows float64 on this domain: max u = "
                    f"{largest:.1f} at the box corner, and the limit is 354. Reduce the box or the "
                    f"nuclear charge; a silent underflow would make the operator return NaN from a "
                    f"region where the orbital is already zero to 300 digits"
                )
            weights.append(torch.exp(-2.0 * u))
        result = (weights[0], weights[1], weights[2])
        self._face_weight_cache[accuracy] = result
        return result

    def weight_integral_exact(self) -> float | None:
        """Return ``integral f^2 d^3r`` in closed form, or ``None`` when no closed form is known.

        One centre: ``pi / Z^3``. Two centres of equal charge ``Z`` at separation ``R``, from the
        prolate spheroidal volume element, with ``a = 2 Z R``:

            (pi R^3 / 4) e^{-a} [ 2 (1/a + 2/a^2 + 2/a^3) - 2/(3 a) ].

        G1.13 uses it to measure how far the plain ``h^3`` rule is from integrating the cusp weight
        exactly -- the error every density integral on this path inherits (A-9).
        """
        if self.is_identity:
            return None
        if len(self.charges) == 1:
            z = self.charges[0]
            return math.pi / z**3
        if len(self.charges) == 2 and math.isclose(self.charges[0], self.charges[1]):
            z = self.charges[0]
            r = float((self.positions[0] - self.positions[1]).norm())
            a = 2.0 * z * r
            return (math.pi * r**3 / 4.0) * math.exp(-a) * (
                2.0 * (1.0 / a + 2.0 / a**2 + 2.0 / a**3) - 2.0 / (3.0 * a)
            )
        return None

    def weight_integral_grid(self) -> float:
        """Return the plain ``h^3`` quadrature of the stored weight, the rule the solver uses today."""
        return float(self.grid.integrate(self._weight))

    def weight_integral_outside_box(self) -> float:
        """Return ``integral f^2`` over the space *outside* the grid's box, numerically.

        :meth:`weight_integral_exact` is over all space and the grid integrates over its box
        (``[x_min - h/2, x_max + h/2]`` per axis), so they differ by this tail -- which G1.13 would
        otherwise read as a quadrature error.

        Spherical shells about the box centre from the inscribed radius outward; radial
        Gauss--Legendre panels of width ``1/(4 Z_min)``, order 8; angular GL(cos theta) x
        uniform(phi) of 160 x 320 while a shell crosses the box (the outside-the-box indicator is
        discontinuous, so only first order in the angular step, 0.6 % of the tail) and 16 x 32 once
        the shell is entirely outside. Shells stop where the weight is below 1e-40.

        Geometry- and record-only: computed once per factor and remembered.
        """
        cached = self.__dict__.get("_weight_integral_outside_box")
        if cached is None:
            cached = self.__dict__["_weight_integral_outside_box"] = self._weight_integral_outside_box()
        return cached

    def _weight_integral_outside_box(self) -> float:
        """The computation of :meth:`weight_integral_outside_box`, uncached."""
        if self.is_identity:
            return 0.0

        grid = self.grid
        points = grid.points()
        lo = points.min(dim=0).values - 0.5 * grid.spacing
        hi = points.max(dim=0).values + 0.5 * grid.spacing
        centre = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        r_inner = float(half.min())
        r_outside = float(half.norm())  # beyond this every point of a shell is outside the box
        z_min = min(self.charges)
        farthest_nucleus = float((self.positions - centre).norm(dim=-1).max())
        r_max = farthest_nucleus + 92.0 / (2.0 * z_min)
        if r_max <= r_inner:
            return 0.0

        def angular(n_theta: int, n_phi: int) -> tuple[torch.Tensor, torch.Tensor]:
            ct, wt = gauss_legendre(n_theta)
            ct_t = torch.tensor(ct, dtype=torch.float64, device=grid.device)
            wt_t = torch.tensor(wt, dtype=torch.float64, device=grid.device)
            phi = (torch.arange(n_phi, dtype=torch.float64, device=grid.device) + 0.5) * (2.0 * math.pi / n_phi)
            st = (1.0 - ct_t * ct_t).clamp_min(0.0).sqrt()
            direction = torch.stack(
                [
                    (st[:, None] * torch.cos(phi)[None, :]).reshape(-1),
                    (st[:, None] * torch.sin(phi)[None, :]).reshape(-1),
                    (ct_t[:, None] * torch.ones_like(phi)[None, :]).reshape(-1),
                ],
                dim=-1,
            )
            weights = (wt_t[:, None] * torch.full_like(phi, 2.0 * math.pi / n_phi)[None, :]).reshape(-1)
            return direction, weights

        radial_nodes, radial_weights = gauss_legendre(8)
        panel = 0.25 / z_min
        # Accumulated on the device and read once; do not reintroduce a per-node host read. A node
        # with no direction outside the box adds exactly +0.0, so skipping it would change nothing.
        total = torch.zeros((), dtype=torch.float64, device=grid.device)
        edge = r_inner
        fine = angular(160, 320)
        coarse = angular(16, 32)
        while edge < r_max:
            upper = min(edge + panel, r_max)
            mid, halfwidth = 0.5 * (edge + upper), 0.5 * (upper - edge)
            direction, ang_w = fine if edge < r_outside else coarse
            for node, weight in zip(radial_nodes, radial_weights):
                r = mid + halfwidth * float(node)
                xyz = centre + r * direction
                outside = ((xyz - centre).abs() > half).any(dim=-1)
                u = torch.zeros(xyz.shape[0], dtype=torch.float64, device=grid.device)
                for index, charge in enumerate(self.charges):
                    u = u + charge * (xyz - self.positions[index]).norm(dim=-1)
                f2 = torch.exp(-2.0 * u) * outside.to(torch.float64)
                total = total + float(weight) * halfwidth * r * r * (f2 * ang_w).sum()
            edge = upper
        return float(total)

    def density(self, phi: torch.Tensor, occupations: torch.Tensor) -> torch.Tensor:
        """Return ``n = sum_i f_i f^2 |phi_i|^2``, shape ``(n_spin, n_points)``.

        From the stored ``f^2``, not ``f`` applied twice: at ``Z = 2`` and 8 bohr the factor is
        ``e^-16`` and the difference is real.
        """
        return (occupations[..., None] * phi.pow(2)).sum(dim=-2) * self._weight

    def kinetic_energy(self, phi: torch.Tensor, occupations: torch.Tensor) -> float:
        """Return ``T_s = 1/2 sum_i f_i integral |grad psi_i|^2`` from the transformed orbitals.

        The gradient form ``grad psi = f (grad phi - phi grad u)``: first derivatives only, so it
        never differentiates across the kink in ``grad u``, and manifestly non-negative. Do not
        collapse it to ``|grad phi - phi grad u|^2``: the two appearances of ``grad u`` need
        different cell averages near a nucleus -- the cross term ``<grad u>`` (which vanishes
        there), the last term ``<|grad u|^2>`` (stored as ``-2 W``) -- and collapsing them costs
        1e-2 Ha on hydrogen at h = 0.4.
        """
        grad_phi = self.grid.gradient(phi)
        cross = (grad_phi * self._grad_u).sum(dim=-2)
        square = (grad_phi * grad_phi).sum(dim=-2)
        square = square - 2.0 * phi * cross - 2.0 * phi.pow(2) * self._transformed_potential
        per_state = 0.5 * self.grid.integrate(square * self._weight)
        return float((occupations.to(torch.float64) * per_state).sum())
