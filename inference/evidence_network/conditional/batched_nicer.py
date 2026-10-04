"""GPU-batched transformed-NICER likelihoods for conditional EN labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class SourceGrid:
    values: torch.Tensor
    mass_start: float
    mass_spacing: float
    radius_start: float
    radius_spacing: float
    mass_rows: int
    radius_rows: int
    fill: float
    centre_mass: float
    centre_radius: float


class BatchedShiftedNICER:
    """Evaluate many affine cloud transforms over many stellar curves.

    Rows are streamed in chunks.  This avoids materializing a
    ``scenario x EOS-row x quadrature`` tensor for the complete bank.
    """

    def __init__(
        self,
        interpolators,
        radius_curve: np.ndarray,
        mass_nodes: np.ndarray,
        maximum_mass: np.ndarray,
        centres: dict[str, tuple[float, float]],
        *,
        device: str,
        row_chunk: int = 50_000,
        quadrature_points: int = 40,
    ) -> None:
        self.device = torch.device(device)
        self.radius = np.asarray(radius_curve, dtype=np.float32)
        self.maximum_mass = np.asarray(maximum_mass, dtype=np.float32)
        self.mass_nodes = np.asarray(mass_nodes, dtype=np.float32)
        if self.radius.ndim != 2 or self.radius.shape[1] != len(self.mass_nodes):
            raise ValueError("radius curves and mass grid are inconsistent")
        if self.maximum_mass.shape != (len(self.radius),):
            raise ValueError("maximum-mass vector is inconsistent")
        self.row_chunk = int(row_chunk)
        if self.row_chunk < 1 or quadrature_points < 2:
            raise ValueError("invalid conditional-NICER batching settings")
        self.quadrature_points = int(quadrature_points)
        self.mass_start = float(self.mass_nodes[0])
        self.mass_spacing = float(self.mass_nodes[1] - self.mass_nodes[0])
        self.unit = torch.linspace(
            0.0, 1.0, self.quadrature_points, device=self.device
        )[None, :]
        self.mass_tensor = torch.as_tensor(self.mass_nodes, device=self.device)
        self.grids: dict[str, SourceGrid] = {}
        for name, interpolator in interpolators.items():
            mass_grid, radius_grid = interpolator.grid
            values = np.asarray(interpolator.values, dtype=np.float32)
            self.grids[name] = SourceGrid(
                values=torch.as_tensor(values, device=self.device),
                mass_start=float(mass_grid[0]),
                mass_spacing=float(mass_grid[1] - mass_grid[0]),
                radius_start=float(radius_grid[0]),
                radius_spacing=float(radius_grid[1] - radius_grid[0]),
                mass_rows=len(mass_grid),
                radius_rows=len(radius_grid),
                fill=float(interpolator.fill_value),
                centre_mass=float(centres[name][0]),
                centre_radius=float(centres[name][1]),
            )

    def chunks(self):
        for start in range(0, len(self.radius), self.row_chunk):
            stop = min(start + self.row_chunk, len(self.radius))
            yield start, stop

    def curve_quadrature(self, start: int, stop: int):
        radius = torch.as_tensor(self.radius[start:stop], device=self.device)
        maximum = torch.as_tensor(
            self.maximum_mass[start:stop], device=self.device
        )
        finite = torch.isfinite(radius) & (
            self.mass_tensor[None, :] <= maximum[:, None]
        )
        has_curve = finite.any(dim=1)
        first = torch.argmax(finite.float(), dim=1)
        last = (len(self.mass_nodes) - 1) - torch.argmax(
            finite.flip(1).float(), dim=1
        )
        lower = torch.where(
            first == 0,
            torch.tensor(1.0, device=self.device),
            self.mass_tensor[first],
        )
        upper = maximum
        valid = has_curve & (upper > lower)
        safe_lower = torch.where(valid, lower, torch.zeros_like(lower))
        safe_upper = torch.where(valid, upper, safe_lower + 1.0)
        mass = safe_lower[:, None] + (safe_upper - safe_lower)[:, None] * self.unit
        radius_mass = torch.maximum(mass, self.mass_tensor[first][:, None])
        index = torch.clamp(
            ((radius_mass - self.mass_start) / self.mass_spacing)
            .floor()
            .long(),
            0,
            len(self.mass_nodes) - 2,
        )
        last_segment = torch.maximum(last - 1, first)[:, None]
        index = torch.minimum(torch.maximum(index, first[:, None]), last_segment)
        fraction = torch.clamp(
            (radius_mass - (self.mass_start + index * self.mass_spacing))
            / self.mass_spacing,
            min=0.0,
        )
        radius0 = torch.gather(radius, 1, index)
        radius1 = torch.gather(
            radius, 1, torch.clamp(index + 1, max=len(self.mass_nodes) - 1)
        )
        radius1 = torch.where(torch.isfinite(radius1), radius1, radius0)
        curve_radius = radius0 * (1.0 - fraction) + radius1 * fraction
        return mass, curve_radius, valid, safe_lower, safe_upper

    @torch.inference_mode()
    def evaluate_chunk(
        self,
        name: str,
        transforms: torch.Tensor,
        quadrature,
    ) -> torch.Tensor:
        """Return ``(scenario,row)`` log likelihoods for one source."""

        if transforms.ndim != 2 or transforms.shape[1] != 4:
            raise ValueError("NICER transforms must have shape (scenario,4)")
        mass_shift, radius_shift, mass_scale, radius_scale = [
            transforms[:, column][:, None, None] for column in range(4)
        ]
        if torch.any(mass_scale <= 0) or torch.any(radius_scale <= 0):
            raise ValueError("NICER transformation scales must be positive")
        mass, radius, valid, safe_lower, safe_upper = quadrature
        grid = self.grids[name]
        query_mass = grid.centre_mass + (
            (mass[None, :, :] - mass_shift) - grid.centre_mass
        ) / mass_scale
        query_radius = grid.centre_radius + (
            (radius[None, :, :] - radius_shift) - grid.centre_radius
        ) / radius_scale
        mass_coordinate = (query_mass - grid.mass_start) / grid.mass_spacing
        radius_coordinate = (
            query_radius - grid.radius_start
        ) / grid.radius_spacing
        mass_index = torch.clamp(
            mass_coordinate.floor().long(), 0, grid.mass_rows - 2
        )
        radius_index = torch.clamp(
            radius_coordinate.floor().long(), 0, grid.radius_rows - 2
        )
        mass_fraction = mass_coordinate - mass_index
        radius_fraction = radius_coordinate - radius_index
        value00 = grid.values[mass_index, radius_index]
        value10 = grid.values[mass_index + 1, radius_index]
        value01 = grid.values[mass_index, radius_index + 1]
        value11 = grid.values[mass_index + 1, radius_index + 1]
        log_density = (
            value00 * (1.0 - mass_fraction) * (1.0 - radius_fraction)
            + value10 * mass_fraction * (1.0 - radius_fraction)
            + value01 * (1.0 - mass_fraction) * radius_fraction
            + value11 * mass_fraction * radius_fraction
        )
        outside = (
            (mass_coordinate < 0)
            | (mass_coordinate > grid.mass_rows - 1)
            | (radius_coordinate < 0)
            | (radius_coordinate > grid.radius_rows - 1)
        )
        log_density = torch.where(
            outside, torch.full_like(log_density, grid.fill), log_density
        )
        integral = torch.trapezoid(
            torch.exp(log_density), mass[None, :, :], dim=2
        ) / (safe_upper - safe_lower)[None, :]
        # The transformed cloud represents a normalized two-dimensional
        # density.  Unlike posterior-only training, absolute evidence cannot
        # discard its affine Jacobian: q'(M,R)=q(T^{-1}(M,R))/(s_M s_R).
        log_jacobian = (
            torch.log(mass_scale[:, 0, 0])
            + torch.log(radius_scale[:, 0, 0])
        )[:, None]
        return torch.where(
            valid[None, :],
            torch.log(integral + 1e-30) - log_jacobian,
            torch.full_like(integral, -1e30),
        )
