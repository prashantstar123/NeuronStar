"""Opt-in CUDA accelerator for the validated fixed-grid TOV/tidal kernel.

This module is deliberately not selected by the public solver automatically.
It mirrors :mod:`tov.solver` and :mod:`tov.tidal` and is accepted for
production only after its output passes the CPU row-replay and frozen-target
gates.  Each CUDA thread integrates one central-pressure star; EOS rows and
the 40-point central-pressure grid are parallelized together.
"""

from __future__ import annotations

import math

import numpy as np

from . import solver
from .api import StableBranch
from .crust import TOV_H, graft_bps_crust


def _cuda():
    """Import CUDA lazily so ordinary CPU workflows keep working unchanged."""

    from numba import cuda

    return cuda


def _build_kernel():
    cuda = _cuda()

    @cuda.jit(device=True, inline=True)
    def eps_of_p_lin(p, p_tab, eps_tab, eos_index, table_length):
        log_p = math.log(max(p, 1e-40))
        log_pt0 = math.log(p_tab[eos_index, 0])
        log_pt1 = math.log(p_tab[eos_index, table_length - 1])
        if log_p <= log_pt0:
            return eps_tab[eos_index, 0]
        if log_p >= log_pt1:
            return eps_tab[eos_index, table_length - 1]
        lo = 0
        hi = table_length - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if math.log(p_tab[eos_index, mid]) <= log_p:
                lo = mid
            else:
                hi = mid
        x0 = math.log(p_tab[eos_index, lo])
        x1 = math.log(p_tab[eos_index, hi])
        y0 = math.log(eps_tab[eos_index, lo])
        y1 = math.log(eps_tab[eos_index, hi])
        dx = x1 - x0
        if dx <= 0.0:
            return eps_tab[eos_index, lo]
        fraction = (log_p - x0) / dx
        return math.exp(y0 + fraction * (y1 - y0))

    @cuda.jit(device=True, inline=True)
    def dedp_of_p(p, p_tab, eps_tab, eos_index, table_length):
        h_local = max(p * 1e-4, 1e-30)
        e_p = eps_of_p_lin(
            p + h_local, p_tab, eps_tab, eos_index, table_length
        )
        e_m = eps_of_p_lin(
            max(p - h_local, p * 0.5),
            p_tab,
            eps_tab,
            eos_index,
            table_length,
        )
        return (e_p - e_m) / (2.0 * h_local)

    @cuda.jit(device=True, inline=True)
    def tov_rhs(r, p, m, y, p_tab, eps_tab, eos_index, table_length):
        pi = math.pi
        p_safe = max(p, 1e-40)
        eps = eps_of_p_lin(
            p_safe, p_tab, eps_tab, eos_index, table_length
        )
        dedp = dedp_of_p(
            p_safe, p_tab, eps_tab, eos_index, table_length
        )
        rr = max(r, 1e-12)
        safe_denom = max(rr - 2.0 * m, 1e-20)
        dp_dr = -(eps + p_safe) * (m + 4.0 * pi * rr**3 * p_safe) / (
            rr * safe_denom
        )
        dm_dr = 4.0 * pi * rr**2 * eps
        f_value = (rr - 4.0 * pi * rr**3 * (eps - p_safe)) / safe_denom
        q_value = (
            4.0
            * pi
            * rr
            * (
                5.0 * eps
                + 9.0 * p_safe
                + (eps + p_safe) * dedp
                - 6.0 / (4.0 * pi * rr**2)
            )
            / safe_denom
            - 4.0
            * (
                (m + 4.0 * pi * rr**3 * p_safe)
                / (rr**2 * (1.0 - 2.0 * m / rr))
            )
            ** 2
        )
        dy_dr = (-y * y - y * f_value - rr * rr * q_value) / rr
        return dp_dr, dm_dr, dy_dr

    @cuda.jit(fastmath=True)
    def mrl_kernel(
        central,
        p_tab,
        eps_tab,
        table_lengths,
        h,
        n_steps,
        mass,
        radius,
        tidal_lambda,
    ):
        flat_index = cuda.grid(1)
        n_stars = central.shape[0]
        eos_index = flat_index // n_stars
        star_index = flat_index - eos_index * n_stars
        if eos_index >= p_tab.shape[0]:
            return

        table_length = table_lengths[eos_index]
        p_c_geom = central[star_index]
        if table_length < 5 or p_c_geom >= 0.999 * p_tab[eos_index, table_length - 1]:
            mass[eos_index, star_index] = math.nan
            radius[eos_index, star_index] = math.nan
            tidal_lambda[eos_index, star_index] = math.nan
            return

        pi = math.pi
        r_init = 1e-4
        p_surface = max(1e-20, p_tab[eos_index, 0] * 1.001)
        eps_c = eps_of_p_lin(
            p_c_geom, p_tab, eps_tab, eos_index, table_length
        )
        p = p_c_geom - (2.0 * pi / 3.0) * (eps_c + p_c_geom) * (
            eps_c + 3.0 * p_c_geom
        ) * r_init**2
        m = (4.0 * pi / 3.0) * eps_c * r_init**3
        r = r_init
        y = 2.0
        r_surface = r_init
        m_surface = m
        y_surface = 2.0
        found = False

        for _ in range(n_steps):
            if found:
                break
            k1p, k1m, k1y = tov_rhs(
                r, p, m, y, p_tab, eps_tab, eos_index, table_length
            )
            k2p, k2m, k2y = tov_rhs(
                r + h / 2.0,
                p + h / 2.0 * k1p,
                m + h / 2.0 * k1m,
                y + h / 2.0 * k1y,
                p_tab,
                eps_tab,
                eos_index,
                table_length,
            )
            k3p, k3m, k3y = tov_rhs(
                r + h / 2.0,
                p + h / 2.0 * k2p,
                m + h / 2.0 * k2m,
                y + h / 2.0 * k2y,
                p_tab,
                eps_tab,
                eos_index,
                table_length,
            )
            k4p, k4m, k4y = tov_rhs(
                r + h,
                p + h * k3p,
                m + h * k3m,
                y + h * k3y,
                p_tab,
                eps_tab,
                eos_index,
                table_length,
            )
            p_new = p + h / 6.0 * (k1p + 2.0 * k2p + 2.0 * k3p + k4p)
            m_new = m + h / 6.0 * (k1m + 2.0 * k2m + 2.0 * k3m + k4m)
            y_new = y + h / 6.0 * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
            if p_new < p_surface:
                fraction = (p - p_surface) / max(p - p_new, 1e-30)
                fraction = min(max(fraction, 0.0), 1.0)
                r_surface = r + h * fraction
                m_surface = m + fraction * (m_new - m)
                y_surface = y + fraction * (y_new - y)
                found = True
            else:
                r += h
                p = p_new
                m = m_new
                y = y_new

        if not found:
            mass[eos_index, star_index] = math.nan
            radius[eos_index, star_index] = math.nan
            tidal_lambda[eos_index, star_index] = math.nan
            return

        compactness = 2.0 * m_surface / r_surface
        if compactness > 0.99:
            compactness = 0.99
        f1 = 2.0 - y_surface + (y_surface - 1.0) * compactness
        numerator = (
            (1.0 / 20.0)
            * compactness**5
            * (1.0 - compactness) ** 2
            * f1
        )
        denominator = (
            compactness
            * (
                6.0
                - 3.0 * y_surface
                + 1.5 * compactness * (5.0 * y_surface - 8.0)
            )
            + 0.25
            * compactness**3
            * (
                26.0
                - 22.0 * y_surface
                + compactness * (3.0 * y_surface - 2.0)
                + compactness**2 * (1.0 + y_surface)
            )
            + 3.0
            * (1.0 - compactness) ** 2
            * f1
            * math.log1p(-compactness)
        )
        value = (
            (2.0 / 3.0)
            * (numerator / denominator)
            * (r_surface / max(m_surface, 1e-10)) ** 5
        )
        mass[eos_index, star_index] = m_surface
        radius[eos_index, star_index] = r_surface * solver.KM_PER_MSOL
        tidal_lambda[eos_index, star_index] = value

    return mrl_kernel


_KERNEL = None


def prepare_crusted_tables(energy_batch, pressure_batch):
    """Apply the exact CPU crust convention and pack variable-length tables."""

    energy_batch = np.asarray(energy_batch, dtype=np.float64)
    pressure_batch = np.asarray(pressure_batch, dtype=np.float64)
    if energy_batch.shape != pressure_batch.shape or energy_batch.ndim != 2:
        raise ValueError("energy and pressure must have one matching 2-D shape")
    tables = []
    maximum_length = 0
    for energy, pressure in zip(energy_batch, pressure_batch, strict=True):
        valid = (
            np.isfinite(energy)
            & np.isfinite(pressure)
            & (energy > 0.0)
            & (pressure > 0.0)
        )
        local_energy = energy[valid]
        local_pressure = pressure[valid]
        if len(local_energy) < 5:
            tables.append(None)
            continue
        order = np.argsort(local_energy)
        full_energy, full_pressure = graft_bps_crust(
            local_energy[order], local_pressure[order]
        )
        full_energy = np.ascontiguousarray(
            full_energy * solver.MEV_FM3_TO_GEOM, dtype=np.float64
        )
        full_pressure = np.ascontiguousarray(
            full_pressure * solver.MEV_FM3_TO_GEOM, dtype=np.float64
        )
        tables.append((full_pressure, full_energy))
        maximum_length = max(maximum_length, len(full_energy))
    if maximum_length == 0:
        raise ValueError("no valid EOS tables were produced")

    p_tables = np.ones((len(tables), maximum_length), dtype=np.float64)
    eps_tables = np.ones_like(p_tables)
    lengths = np.zeros(len(tables), dtype=np.int32)
    for index, table in enumerate(tables):
        if table is None:
            continue
        pressure, energy = table
        length = len(pressure)
        p_tables[index, :length] = pressure
        eps_tables[index, :length] = energy
        if length < maximum_length:
            p_tables[index, length:] = pressure[-1]
            eps_tables[index, length:] = energy[-1]
        lengths[index] = length
    return p_tables, eps_tables, lengths


def mrl_curve_batch_cuda(
    p_tables,
    eps_tables,
    table_lengths,
    *,
    step=TOV_H,
    threads_per_block=128,
):
    """Run the fixed-grid M-R-Lambda integrations on CUDA."""

    global _KERNEL
    cuda = _cuda()
    if not cuda.is_available():
        raise RuntimeError("CUDA is not available")
    p_tables = np.ascontiguousarray(p_tables, dtype=np.float64)
    eps_tables = np.ascontiguousarray(eps_tables, dtype=np.float64)
    table_lengths = np.ascontiguousarray(table_lengths, dtype=np.int32)
    if p_tables.shape != eps_tables.shape or p_tables.ndim != 2:
        raise ValueError("packed pressure and energy tables must match")
    if table_lengths.shape != (len(p_tables),):
        raise ValueError("one packed table length is required per EOS")
    central = np.ascontiguousarray(solver.P_C_GEOM, dtype=np.float64)
    shape = (len(p_tables), len(central))
    d_mass = cuda.device_array(shape, dtype=np.float64)
    d_radius = cuda.device_array(shape, dtype=np.float64)
    d_lambda = cuda.device_array(shape, dtype=np.float64)
    if _KERNEL is None:
        _KERNEL = _build_kernel()
    count = shape[0] * shape[1]
    blocks = (count + threads_per_block - 1) // threads_per_block
    _KERNEL[blocks, threads_per_block](
        central,
        p_tables,
        eps_tables,
        table_lengths,
        float(step),
        int(solver.N_TOV_STEPS),
        d_mass,
        d_radius,
        d_lambda,
    )
    cuda.synchronize()
    return d_mass.copy_to_host(), d_radius.copy_to_host(), d_lambda.copy_to_host()


def stable_branches_from_curves(mass, radius, tidal_lambda):
    """Apply the public CPU solver's stable-branch selection exactly."""

    mass = np.asarray(mass, dtype=np.float64)
    radius = np.asarray(radius, dtype=np.float64)
    tidal_lambda = np.asarray(tidal_lambda, dtype=np.float64)
    if mass.shape != radius.shape or mass.shape != tidal_lambda.shape:
        raise ValueError("mass, radius, and tidal arrays must match")
    branches = []
    for local_mass, local_radius, local_lambda in zip(
        mass, radius, tidal_lambda, strict=True
    ):
        valid = (
            np.isfinite(local_mass)
            & np.isfinite(local_radius)
            & np.isfinite(local_lambda)
            & (local_mass > 0.0)
            & (local_radius > 0.0)
            & (local_lambda > 0.0)
        )
        local_mass = local_mass[valid]
        local_radius = local_radius[valid]
        local_lambda = local_lambda[valid]
        if len(local_mass) < 4:
            branches.append(None)
            continue
        maximum_index = int(np.argmax(local_mass))
        maximum_mass = float(local_mass[maximum_index])
        local_mass = local_mass[: maximum_index + 1]
        local_radius = local_radius[: maximum_index + 1]
        local_lambda = local_lambda[: maximum_index + 1]
        order = np.argsort(local_mass)
        local_mass = local_mass[order]
        local_radius = local_radius[order]
        local_lambda = local_lambda[order]
        unique_index = np.unique(local_mass, return_index=True)[1]
        local_mass = local_mass[unique_index]
        local_radius = local_radius[unique_index]
        local_lambda = local_lambda[unique_index]
        if len(local_mass) < 4:
            branches.append(None)
            continue
        branches.append(
            StableBranch(
                local_mass,
                local_radius,
                local_lambda,
                maximum_mass,
            )
        )
    return branches


def solve_stable_branches_cuda(energy_batch, pressure_batch, *, step=TOV_H):
    """Convenience wrapper for packed-table preparation, CUDA, and selection."""

    packed = prepare_crusted_tables(energy_batch, pressure_batch)
    curves = mrl_curve_batch_cuda(*packed, step=step)
    return stable_branches_from_curves(*curves)
