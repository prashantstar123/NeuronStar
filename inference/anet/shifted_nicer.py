"""Vectorized exact-quadrature shifted NICER likelihoods for A-NET scenarios."""

from __future__ import annotations

import numpy as np
import torch


def build_shifted_sources(
    interpolators,
    radius_curve,
    mass_nodes,
    maximum_mass,
    centres,
    *,
    device=None,
    chunk=200_000,
    quadrature_points=40,
):
    """Return a callable evaluating shifted/scaled per-source likelihoods.

    The quadrature reproduces ``likelihoods.nicer.log_likelihood_one`` on the
    identity transform while allowing the synthetic training cloud's mass and
    radius location/scale to vary.
    """

    torch_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    radius_numpy = np.asarray(radius_curve, dtype=np.float32)
    maximum_numpy = np.asarray(maximum_mass, dtype=np.float32)
    mass_numpy = np.asarray(mass_nodes, dtype=np.float64)
    rows = len(radius_numpy)
    ranges = [(start, min(start + chunk, rows)) for start in range(0, rows, chunk)]
    radius_chunks = [
        torch.as_tensor(radius_numpy[start:stop], device=torch_device)
        for start, stop in ranges
    ]
    maximum_chunks = [
        torch.as_tensor(maximum_numpy[start:stop], device=torch_device)
        for start, stop in ranges
    ]
    mass_tensor = torch.as_tensor(
        mass_numpy.astype(np.float32), device=torch_device
    )
    mass_start = float(mass_numpy[0])
    mass_spacing = float(mass_numpy[1] - mass_numpy[0])
    grids = {}
    for name, interpolator in interpolators.items():
        mass_grid, radius_grid = interpolator.grid
        values = np.asarray(interpolator.values, dtype=np.float32)
        grids[name] = {
            "values": torch.as_tensor(values, device=torch_device),
            "mass_start": float(mass_grid[0]),
            "mass_spacing": float(mass_grid[1] - mass_grid[0]),
            "radius_start": float(radius_grid[0]),
            "radius_spacing": float(radius_grid[1] - radius_grid[0]),
            "fill": float(interpolator.fill_value),
            "mass_rows": len(mass_grid),
            "radius_rows": len(radius_grid),
            "centre_mass": float(centres[name][0]),
            "centre_radius": float(centres[name][1]),
        }
    unit = torch.linspace(
        0.0, 1.0, quadrature_points, device=torch_device
    )[None, :]

    @torch.no_grad()
    def source_log_likelihood(
        name,
        mass_shift=0.0,
        radius_shift=0.0,
        mass_scale=1.0,
        radius_scale=1.0,
    ):
        if mass_scale <= 0 or radius_scale <= 0:
            raise ValueError("NICER transformation scales must be positive")
        grid = grids[name]
        values = grid["values"]
        output = np.empty(rows, dtype=np.float64)
        for (start, stop), radius, maximum in zip(
            ranges, radius_chunks, maximum_chunks
        ):
            finite = torch.isfinite(radius) & (
                mass_tensor[None, :] <= maximum[:, None]
            )
            has_curve = finite.any(dim=1)
            first = torch.argmax(finite.float(), dim=1)
            last = (len(mass_numpy) - 1) - torch.argmax(
                finite.flip(1).float(), dim=1
            )
            lower = torch.where(
                first == 0,
                torch.tensor(1.0, device=torch_device),
                mass_tensor[first],
            )
            upper = maximum
            valid = has_curve & (upper > lower)
            safe_lower = torch.where(
                valid, lower, torch.zeros((), device=torch_device)
            )
            safe_upper = torch.where(valid, upper, safe_lower + 1.0)
            quadrature_mass = safe_lower[:, None] + (
                safe_upper - safe_lower
            )[:, None] * unit

            radius_mass = torch.maximum(
                quadrature_mass, mass_tensor[first][:, None]
            )
            index = torch.clamp(
                ((radius_mass - mass_start) / mass_spacing)
                .floor()
                .to(torch.long),
                0,
                len(mass_numpy) - 2,
            )
            last_segment = torch.maximum(last - 1, first)[:, None]
            index = torch.minimum(
                torch.maximum(index, first[:, None]), last_segment
            )
            fraction = torch.clamp(
                (radius_mass - (mass_start + index * mass_spacing))
                / mass_spacing,
                min=0.0,
            )
            radius0 = torch.gather(radius, 1, index)
            radius1 = torch.gather(
                radius, 1, torch.clamp(index + 1, max=len(mass_numpy) - 1)
            )
            radius1 = torch.where(torch.isfinite(radius1), radius1, radius0)
            quadrature_radius = radius0 * (1.0 - fraction) + radius1 * fraction

            query_mass = grid["centre_mass"] + (
                (quadrature_mass - mass_shift) - grid["centre_mass"]
            ) / mass_scale
            query_radius = grid["centre_radius"] + (
                (quadrature_radius - radius_shift) - grid["centre_radius"]
            ) / radius_scale
            mass_coordinate = (
                query_mass - grid["mass_start"]
            ) / grid["mass_spacing"]
            radius_coordinate = (
                query_radius - grid["radius_start"]
            ) / grid["radius_spacing"]
            mass_index = torch.clamp(
                mass_coordinate.floor().to(torch.long),
                0,
                grid["mass_rows"] - 2,
            )
            radius_index = torch.clamp(
                radius_coordinate.floor().to(torch.long),
                0,
                grid["radius_rows"] - 2,
            )
            mass_fraction = mass_coordinate - mass_index
            radius_fraction = radius_coordinate - radius_index
            value00 = values[mass_index, radius_index]
            value10 = values[mass_index + 1, radius_index]
            value01 = values[mass_index, radius_index + 1]
            value11 = values[mass_index + 1, radius_index + 1]
            log_density = (
                value00 * (1.0 - mass_fraction) * (1.0 - radius_fraction)
                + value10 * mass_fraction * (1.0 - radius_fraction)
                + value01 * (1.0 - mass_fraction) * radius_fraction
                + value11 * mass_fraction * radius_fraction
            )
            outside = (
                (mass_coordinate < 0)
                | (mass_coordinate > grid["mass_rows"] - 1)
                | (radius_coordinate < 0)
                | (radius_coordinate > grid["radius_rows"] - 1)
            )
            log_density = torch.where(
                outside,
                torch.tensor(grid["fill"], device=torch_device),
                log_density,
            )
            density = torch.exp(log_density)
            integral = torch.trapezoid(
                density, quadrature_mass, dim=1
            ) / (safe_upper - safe_lower)
            result = torch.where(
                valid,
                torch.log(integral + 1e-300),
                torch.tensor(-1e30, device=torch_device),
            )
            output[start:stop] = result.double().cpu().numpy()
        return output

    return source_log_likelihood

