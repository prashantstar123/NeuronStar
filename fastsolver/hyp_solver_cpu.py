"""GENERATED (cpu) from hyp_solver_template.py sha256 0eaf23e0a7f66fc7; do not edit: fully compiled DDB Lambda/Xi-minus beta-equilibrium
sequence solver.  Generated into hyp_solver_cpu.py (numba njit, parallel rows)
and hyp_solver_cuda.py (numba CUDA, one thread per row) by gen_backends.py.

Port of /home/nucleartheory/ddb_hyperon_project/ddb_hyperon_eos.py (certified
accelerated bounded analytic Newton path, SolverOptions defaults with
use_analytic_newton=True).  Every formula, ordering, tolerance and gate is
copied from that file; the only intentional difference is that the SciPy
least_squares fallback is replaced by a per-row status flag so the driver
re-solves such rows with the certified CPU module.
"""
import math
import numpy as np
import numba
from numba import njit, prange

PI2 = math.pi ** 2
TWO_PI2 = 2.0 * math.pi ** 2
THREE_PI2 = 3.0 * math.pi ** 2
EIGHT_PI2 = 8.0 * math.pi ** 2
TWENTYFOUR_PI2 = 24.0 * math.pi ** 2
NMAX = 10

HBARC = 197.33
M_N = 4.7583690772
M_P = 4.7583690772
M_L = 1115.68 / 197.33
M_X = 1321.71 / 197.33
M_E = 2.5896e-3
M_MU = 0.53544
M_SIG = 550.0 / 197.33
M_OM = 783.0 / 197.33
M_RHO = 763.0 / 197.33
M_PHI = 1019.461 / 197.33
XW_L = 2.0 / 3.0
XW_X = 1.0 / 3.0
XR_L = 1.0
XR_X = 1.0
XP_L = -math.sqrt(2.0) / 3.0
XP_X = -2.0 * math.sqrt(2.0) / 3.0

NEWTON_TOL = 2.0e-11          # min(2e-11, 0.2 * relative_tolerance 1e-10)
NEWTON_MAXIT = 32
MAX_ASU = 8
THRESH_DENS_TOL = 1.0e-8
RESIDUAL_GATE = 1.0e-7        # max(1e-7, 100*1e-10, 100*1e-12)
PRESSURE_GATE_FACTOR = 1.0e-8 # max(1e-9, 100*1e-10)
CHEM_SCALE = 4.7583690772     # max(m_n, 1) * chemical_potential_residual_scale

ST_OK = 0
ST_FALLBACK = 1   # certified path would call SciPy least_squares here
ST_FAILED = 2     # certified path raises EquilibriumConvergenceError
ST_BADMASS = 3    # nonpositive effective mass (ValueError in certified path)


@njit(cache=True, error_model="numpy")
def cbrt(x):
    if x <= 0.0:
        return 0.0
    y = x ** (1.0 / 3.0)
    y = y - (y * y * y - x) / (3.0 * y * y)
    return y


@njit(cache=True, error_model="numpy")
def couplings(nB, a_s, a_w, a_r, g_s0, g_w0, g_r0, n0, xs_L, xs_X, cv, cd):
    x = nB / n0
    if a_s == 0.0:
        hs = 1.0
        dhs = 0.0
    else:
        hs = math.exp(1.0 - x ** a_s)
        dhs = -(a_s / n0) * (x ** (a_s - 1.0)) * hs
    if a_w == 0.0:
        hw = 1.0
        dhw = 0.0
    else:
        hw = math.exp(1.0 - x ** a_w)
        dhw = -(a_w / n0) * (x ** (a_w - 1.0)) * hw
    hr = math.exp(-a_r * (x - 1.0))
    dhr = -(a_r / n0) * hr
    sigma = g_s0 * hs
    omega = g_w0 * hw
    rho = g_r0 * hr
    d_sigma = g_s0 * dhs
    d_omega = g_w0 * dhw
    d_rho = g_r0 * dhr
    for code in range(2):
        cv[code, 0] = sigma
        cv[code, 1] = omega
        cv[code, 2] = rho
        cv[code, 3] = 0.0
        cd[code, 0] = d_sigma
        cd[code, 1] = d_omega
        cd[code, 2] = d_rho
        cd[code, 3] = 0.0
    cv[2, 0] = xs_L * sigma
    cv[2, 1] = XW_L * omega
    cv[2, 2] = XR_L * rho
    cv[2, 3] = XP_L * omega
    cd[2, 0] = xs_L * d_sigma
    cd[2, 1] = XW_L * d_omega
    cd[2, 2] = XR_L * d_rho
    cd[2, 3] = XP_L * d_omega
    cv[3, 0] = xs_X * sigma
    cv[3, 1] = XW_X * omega
    cv[3, 2] = XR_X * rho
    cv[3, 3] = XP_X * omega
    cd[3, 0] = xs_X * d_sigma
    cd[3, 1] = XW_X * d_omega
    cd[3, 2] = XR_X * d_rho
    cd[3, 3] = XP_X * d_omega
    return sigma, omega, rho


@njit(cache=True, error_model="numpy")
def scalar_density_k(momentum, mass):
    z = momentum / mass
    root = math.sqrt(1.0 + z * z)
    if z < 1.0e-3:
        z2 = z * z
        bracket = z ** 3 * (
            2.0 / 3.0
            + z2 * (-1.0 / 5.0 + z2 * (3.0 / 28.0 + z2 * (-5.0 / 72.0 + z2 * 35.0 / 704.0)))
        )
    else:
        bracket = z * root - math.asinh(z)
    return mass ** 3 * bracket / TWO_PI2


@njit(cache=True, error_model="numpy")
def scalar_density_derivatives_k(momentum, mass):
    z = momentum / mass
    root = math.sqrt(1.0 + z * z)
    energy = mass * root
    derivative_momentum = mass * momentum * momentum / (PI2 * energy)
    if z < 1.0e-3:
        z2 = z * z
        derivative_mass = mass * mass * z ** 5 * (
            2.0 / 5.0 + z2 * (-3.0 / 7.0 + z2 * (5.0 / 12.0 - z2 * 35.0 / 88.0))
        ) / TWO_PI2
    else:
        bracket = z * root - math.asinh(z)
        derivative_mass = mass * mass * (3.0 * bracket - 2.0 * z ** 3 / root) / TWO_PI2
    return derivative_momentum, derivative_mass


@njit(cache=True, error_model="numpy")
def kinetic_energy_density_k(momentum, mass):
    z = momentum / mass
    root = math.sqrt(1.0 + z * z)
    if z < 1.0e-3:
        z2 = z * z
        bracket = z ** 3 * (
            8.0 / 3.0 + z2 * (4.0 / 5.0 + z2 * (-1.0 / 7.0 + z2 * (1.0 / 18.0 - z2 * 5.0 / 176.0)))
        )
    else:
        bracket = z * root * (2.0 * z * z + 1.0) - math.asinh(z)
    return mass ** 4 * bracket / EIGHT_PI2


@njit(cache=True, error_model="numpy")
def kinetic_pressure_k(momentum, mass):
    z = momentum / mass
    root = math.sqrt(1.0 + z * z)
    if z < 1.0e-3:
        z2 = z * z
        bracket = z ** 5 * (8.0 / 5.0 + z2 * (-4.0 / 7.0 + z2 * (1.0 / 3.0 - z2 * 5.0 / 22.0)))
    else:
        bracket = z * root * (2.0 * z * z - 3.0) + 3.0 * math.asinh(z)
    return mass ** 4 * bracket / TWENTYFOUR_PI2


@njit(cache=True, error_model="numpy")
def layout(act, species_codes, active_columns):
    """Fermion layout n, p, [e], [mu], [L], [Xi]; returns species count."""
    for code in range(6):
        active_columns[code] = -1
    ns = 0
    species_codes[ns] = 0
    active_columns[0] = 4
    ns += 1
    species_codes[ns] = 1
    active_columns[1] = 5
    ns += 1
    if act[0] != 0:
        species_codes[ns] = 4
        active_columns[4] = 4 + ns
        ns += 1
    if act[1] != 0:
        species_codes[ns] = 5
        active_columns[5] = 4 + ns
        ns += 1
    if act[2] != 0:
        species_codes[ns] = 2
        active_columns[2] = 4 + ns
        ns += 1
    if act[3] != 0:
        species_codes[ns] = 3
        active_columns[3] = 4 + ns
        ns += 1
    return ns


@njit(cache=True, error_model="numpy")
def residual_k(x, nunk, species_codes, ns, active_columns, nB, cv, cd, pm, mm2, iso,
               field_scale, density_scale, chemical_scale,
               momenta, densities, effective_masses, scalar_densities, baryon_mu, residuals):
    for code in range(6):
        momenta[code] = 0.0
    for offset in range(ns):
        code = species_codes[offset]
        momenta[code] = x[4 + offset]
    for code in range(6):
        momentum = momenta[code]
        densities[code] = momentum ** 3 / THREE_PI2
    for code in range(4):
        effective_mass = pm[code] - cv[code, 0] * x[0]
        effective_masses[code] = effective_mass
        scalar_densities[code] = scalar_density_k(momenta[code], effective_mass)
    rearrangement = 0.0
    for code in range(4):
        density = densities[code]
        rearrangement += (
            -cd[code, 0] * x[0] * scalar_densities[code]
            + cd[code, 1] * x[1] * density
            + cd[code, 2] * iso[code] * x[2] * density
            + cd[code, 3] * x[3] * density
        )
    for code in range(4):
        baryon_mu[code] = (
            math.hypot(momenta[code], effective_masses[code])
            + cv[code, 1] * x[1]
            + cv[code, 2] * iso[code] * x[2]
            + cv[code, 3] * x[3]
            + rearrangement
        )
    electron_active = active_columns[4] >= 0
    if electron_active:
        electron_mu = math.hypot(momenta[4], pm[4])
    else:
        electron_mu = baryon_mu[0] - baryon_mu[1]
    muon_mu = math.hypot(momenta[5], pm[5])
    sigma_source = 0.0
    omega_source = 0.0
    rho_source = 0.0
    phi_source = 0.0
    baryon_sum = 0.0
    for code in range(4):
        sigma_source += cv[code, 0] * scalar_densities[code]
        omega_source += cv[code, 1] * densities[code]
        rho_source += cv[code, 2] * iso[code] * densities[code]
        phi_source += cv[code, 3] * densities[code]
        baryon_sum += densities[code]
    residuals[0] = (mm2[0] * x[0] - sigma_source) / field_scale
    residuals[1] = (mm2[1] * x[1] - omega_source) / field_scale
    residuals[2] = (mm2[2] * x[2] - rho_source) / field_scale
    residuals[3] = (mm2[3] * x[3] - phi_source) / field_scale
    residuals[4] = (baryon_sum - nB) / density_scale
    residuals[5] = (densities[1] - densities[3] - densities[4] - densities[5]) / density_scale
    row = 6
    if electron_active:
        residuals[row] = (baryon_mu[0] - baryon_mu[1] - electron_mu) / chemical_scale
        row += 1
    if active_columns[2] >= 0:
        residuals[row] = (baryon_mu[2] - baryon_mu[0]) / chemical_scale
        row += 1
    if active_columns[3] >= 0:
        residuals[row] = (baryon_mu[3] - baryon_mu[0] - electron_mu) / chemical_scale
        row += 1
    if active_columns[5] >= 0:
        residuals[row] = (muon_mu - electron_mu) / chemical_scale


@njit(cache=True, error_model="numpy")
def jacobian_k(x, nunk, species_codes, ns, active_columns, cv, pm, mm2, iso,
               field_scale, density_scale, chemical_scale,
               momenta, densities_derivative, effective_masses, fermi_energies,
               scalar_derivative_k, scalar_derivative_m, baryon_mu_derivative,
               electron_derivative, jacobian):
    for code in range(6):
        momenta[code] = 0.0
    for offset in range(ns):
        code = species_codes[offset]
        momenta[code] = x[4 + offset]
    for code in range(6):
        densities_derivative[code] = momenta[code] ** 2 / PI2
    for code in range(4):
        effective_mass = pm[code] - cv[code, 0] * x[0]
        effective_masses[code] = effective_mass
        fermi_energies[code] = math.hypot(momenta[code], effective_mass)
        dk, dm = scalar_density_derivatives_k(momenta[code], effective_mass)
        scalar_derivative_k[code] = dk
        scalar_derivative_m[code] = dm
    for r in range(nunk):
        for c in range(nunk):
            jacobian[r, c] = 0.0
    jacobian[0, 0] = mm2[0]
    jacobian[1, 1] = mm2[1]
    jacobian[2, 2] = mm2[2]
    jacobian[3, 3] = mm2[3]
    for code in range(4):
        jacobian[0, 0] += cv[code, 0] * cv[code, 0] * scalar_derivative_m[code]
        column = active_columns[code]
        if column >= 0:
            jacobian[0, column] -= cv[code, 0] * scalar_derivative_k[code]
            jacobian[1, column] -= cv[code, 1] * densities_derivative[code]
            jacobian[2, column] -= cv[code, 2] * iso[code] * densities_derivative[code]
            jacobian[3, column] -= cv[code, 3] * densities_derivative[code]
            jacobian[4, column] += densities_derivative[code]
    jacobian[5, active_columns[1]] += densities_derivative[1]
    if active_columns[3] >= 0:
        jacobian[5, active_columns[3]] -= densities_derivative[3]
    if active_columns[4] >= 0:
        jacobian[5, active_columns[4]] -= densities_derivative[4]
    if active_columns[5] >= 0:
        jacobian[5, active_columns[5]] -= densities_derivative[5]
    for column in range(nunk):
        jacobian[0, column] /= field_scale
        jacobian[1, column] /= field_scale
        jacobian[2, column] /= field_scale
        jacobian[3, column] /= field_scale
        jacobian[4, column] /= density_scale
        jacobian[5, column] /= density_scale
    for code in range(4):
        for column in range(nunk):
            baryon_mu_derivative[code, column] = 0.0
        energy = fermi_energies[code]
        baryon_mu_derivative[code, 0] = -cv[code, 0] * effective_masses[code] / energy
        baryon_mu_derivative[code, 1] = cv[code, 1]
        baryon_mu_derivative[code, 2] = cv[code, 2] * iso[code]
        baryon_mu_derivative[code, 3] = cv[code, 3]
        column = active_columns[code]
        if column >= 0:
            baryon_mu_derivative[code, column] = momenta[code] / energy
    for column in range(nunk):
        electron_derivative[column] = 0.0
    electron_active = active_columns[4] >= 0
    if electron_active:
        column = active_columns[4]
        electron_energy = math.hypot(momenta[4], pm[4])
        electron_derivative[column] = momenta[4] / electron_energy
    else:
        for column in range(nunk):
            electron_derivative[column] = baryon_mu_derivative[0, column] - baryon_mu_derivative[1, column]
    row = 6
    if electron_active:
        for column in range(nunk):
            jacobian[row, column] = (
                baryon_mu_derivative[0, column] - baryon_mu_derivative[1, column] - electron_derivative[column]
            ) / chemical_scale
        row += 1
    if active_columns[2] >= 0:
        for column in range(nunk):
            jacobian[row, column] = (
                baryon_mu_derivative[2, column] - baryon_mu_derivative[0, column]
            ) / chemical_scale
        row += 1
    if active_columns[3] >= 0:
        for column in range(nunk):
            jacobian[row, column] = (
                baryon_mu_derivative[3, column] - baryon_mu_derivative[0, column] - electron_derivative[column]
            ) / chemical_scale
        row += 1
    if active_columns[5] >= 0:
        muon_energy = math.hypot(momenta[5], pm[5])
        muon_column = active_columns[5]
        for column in range(nunk):
            value = -electron_derivative[column]
            if column == muon_column:
                value += momenta[5] / muon_energy
            jacobian[row, column] = value / chemical_scale


@njit(cache=True, error_model="numpy")
def lu_solve(a, b, n, out):
    """Solve a x = b in place (partial pivoting); returns False on an exact zero pivot."""
    for k in range(n):
        piv = k
        big = abs(a[k, k])
        for i in range(k + 1, n):
            v = abs(a[i, k])
            if v > big:
                big = v
                piv = i
        if big == 0.0:
            return False
        if piv != k:
            for j in range(n):
                t = a[k, j]
                a[k, j] = a[piv, j]
                a[piv, j] = t
            t = b[k]
            b[k] = b[piv]
            b[piv] = t
        for i in range(k + 1, n):
            f = a[i, k] / a[k, k]
            if f != 0.0:
                for j in range(k + 1, n):
                    a[i, j] -= f * a[k, j]
                b[i] -= f * b[k]
    for i in range(n - 1, -1, -1):
        s = b[i]
        for j in range(i + 1, n):
            s -= a[i, j] * out[j]
        out[i] = s / a[i, i]
    return True


@njit(cache=True, error_model="numpy")
def bounded_newton(initial, lower, upper, nunk, species_codes, ns, active_columns, nB, cv, cd, pm, mm2, iso,
                   field_scale, density_scale, chemical_scale,
                   safe_lower, safe_upper, x, best_x, residual, trial, trial_residual, step, rhs, jacobian,
                   w_mom, w_den, w_meff, w_sd, w_bmu, w_dd, w_fe, w_sdk, w_sdm, w_bmud, w_ed, xout):
    """Returns (status, evaluations): status 1 converged, 0 not converged, -1 singular."""
    for index in range(nunk):
        interior = 1.0e-12 * max(1.0, upper[index] - lower[index])
        safe_lower[index] = lower[index] + interior
        safe_upper[index] = upper[index] - interior
        x[index] = min(max(initial[index], safe_lower[index]), safe_upper[index])
    evaluations = 0
    for index in range(nunk):
        best_x[index] = x[index]
    best_norm = math.inf
    for _ in range(NEWTON_MAXIT):
        residual_k(x, nunk, species_codes, ns, active_columns, nB, cv, cd, pm, mm2, iso,
                   field_scale, density_scale, chemical_scale,
                   w_mom, w_den, w_meff, w_sd, w_bmu, residual)
        evaluations += 1
        norm = 0.0
        cost = 0.0
        finite_residual = True
        for index in range(nunk):
            value = residual[index]
            if not math.isfinite(value):
                finite_residual = False
            norm = max(norm, abs(value))
            cost += value * value
        if finite_residual and norm < best_norm:
            best_norm = norm
            for index in range(nunk):
                best_x[index] = x[index]
        if finite_residual and norm <= NEWTON_TOL:
            for index in range(nunk):
                xout[index] = x[index]
            return 1, evaluations
        if not finite_residual:
            break
        jacobian_k(x, nunk, species_codes, ns, active_columns, cv, pm, mm2, iso,
                   field_scale, density_scale, chemical_scale,
                   w_mom, w_dd, w_meff, w_fe, w_sdk, w_sdm, w_bmud, w_ed, jacobian)
        finite_jacobian = True
        for r in range(nunk):
            for c in range(nunk):
                if not math.isfinite(jacobian[r, c]):
                    finite_jacobian = False
        if not finite_jacobian:
            break
        for index in range(nunk):
            rhs[index] = -residual[index]
        if not lu_solve(jacobian, rhs, nunk, step):
            return -1, evaluations
        alpha = 1.0
        finite_step = True
        for index in range(nunk):
            value = step[index]
            if not math.isfinite(value):
                finite_step = False
                break
            if value > 0.0:
                alpha = min(alpha, 0.995 * (safe_upper[index] - x[index]) / value)
            elif value < 0.0:
                alpha = min(alpha, 0.995 * (safe_lower[index] - x[index]) / value)
        if (not finite_step) or (not math.isfinite(alpha)) or alpha <= 1.0e-12:
            break
        accepted = False
        trial_alpha = alpha
        for index in range(nunk):
            trial[index] = x[index]
        for _ in range(18):
            for index in range(nunk):
                trial[index] = min(max(x[index] + trial_alpha * step[index], safe_lower[index]), safe_upper[index])
            residual_k(trial, nunk, species_codes, ns, active_columns, nB, cv, cd, pm, mm2, iso,
                       field_scale, density_scale, chemical_scale,
                       w_mom, w_den, w_meff, w_sd, w_bmu, trial_residual)
            evaluations += 1
            trial_cost = 0.0
            finite_trial = True
            for index in range(nunk):
                value = trial_residual[index]
                if not math.isfinite(value):
                    finite_trial = False
                trial_cost += value * value
            if finite_trial and trial_cost < cost:
                for index in range(nunk):
                    x[index] = trial[index]
                accepted = True
                break
            trial_alpha *= 0.5
        if not accepted:
            break
        step_norm = 0.0
        state_norm = 0.0
        for index in range(nunk):
            step_norm = max(step_norm, abs(trial_alpha * step[index]))
            state_norm = max(state_norm, abs(x[index]))
        if step_norm / max(1.0, state_norm) <= 1.0e-13 and best_norm > NEWTON_TOL:
            break
    for index in range(nunk):
        xout[index] = best_x[index]
    return 0, evaluations


@njit(cache=True, error_model="numpy")
def initial_unknowns(nB, act, species_codes, ns, has_prev, p_nB, p_fields, p_dens, p_mu, p_meff,
                     a_s, a_w, a_r, g_s0, g_w0, g_r0, n0, trial_density, initial):
    k_max = cbrt(THREE_PI2 * nB)
    if has_prev == 0:
        hyperon_fraction = 0.02 * act[2] + 0.02 * act[3]
        proton_fraction = 0.10
        neutron_density = max((1.0 - proton_fraction - hyperon_fraction) * nB, 1.0e-8 * nB)
        proton_density = proton_fraction * nB
        trial_density[0] = neutron_density
        trial_density[1] = proton_density
        trial_density[2] = 0.02 * nB if act[2] != 0 else 0.0
        trial_density[3] = 0.02 * nB if act[3] != 0 else 0.0
        trial_density[5] = 0.20 * proton_density if act[1] != 0 else 0.0
        trial_density[4] = max(proton_density - trial_density[5] - trial_density[3], 1.0e-10 * nB)
        x = nB / n0
        if a_s == 0.0:
            hs = 1.0
        else:
            hs = math.exp(1.0 - x ** a_s)
        if a_w == 0.0:
            hw = 1.0
        else:
            hw = math.exp(1.0 - x ** a_w)
        hr = math.exp(-a_r * (x - 1.0))
        gamma_sigma = g_s0 * hs
        gamma_omega = g_w0 * hw
        gamma_rho = g_r0 * hr
        sigma = min(gamma_sigma * nB / M_SIG ** 2, 0.8 * min(M_N, M_P) / gamma_sigma)
        omega = gamma_omega * nB / M_OM ** 2
        rho = gamma_rho * (trial_density[1] - trial_density[0]) / (2.0 * M_RHO ** 2)
        initial[0] = sigma
        initial[1] = omega
        initial[2] = rho
        initial[3] = 0.0
    else:
        ratio = nB / p_nB
        initial[0] = p_fields[0] * ratio
        initial[1] = p_fields[1] * ratio
        initial[2] = p_fields[2] * ratio
        initial[3] = p_fields[3] * ratio
        for code in range(6):
            trial_density[code] = p_dens[code] * ratio
        if act[0] != 0 and trial_density[4] == 0.0:
            target = p_mu[4]
            k_seed = math.sqrt(max(target * target - M_E ** 2, 0.0))
            kk = max(k_seed, 0.02 * k_max)
            trial_density[4] = kk ** 3 / THREE_PI2
        if act[1] != 0 and trial_density[5] == 0.0:
            target = p_mu[4]
            k_seed = math.sqrt(max(target * target - M_MU ** 2, 0.0))
            kk = max(k_seed, 0.02 * k_max)
            trial_density[5] = kk ** 3 / THREE_PI2
        if act[2] != 0 and trial_density[2] == 0.0:
            target = p_mu[0]
            mu_zero = p_mu[2]
            effective_mass = p_meff[2]
            kinetic_target = effective_mass + max(target - mu_zero, 0.0)
            k_seed = math.sqrt(max(kinetic_target ** 2 - effective_mass ** 2, 0.0))
            kk = max(k_seed, 0.02 * k_max)
            trial_density[2] = kk ** 3 / THREE_PI2
        if act[3] != 0 and trial_density[3] == 0.0:
            target = p_mu[0] + p_mu[4]
            mu_zero = p_mu[3]
            effective_mass = p_meff[3]
            kinetic_target = effective_mass + max(target - mu_zero, 0.0)
            k_seed = math.sqrt(max(kinetic_target ** 2 - effective_mass ** 2, 0.0))
            kk = max(k_seed, 0.02 * k_max)
            trial_density[3] = kk ** 3 / THREE_PI2
    for offset in range(ns):
        code = species_codes[offset]
        initial[4 + offset] = cbrt(THREE_PI2 * max(trial_density[code], 0.0))


@njit(cache=True, error_model="numpy")
def unknown_bounds(nB, ns, cv, pm, lower, upper):
    sigma_upper = math.inf
    for code in range(4):
        if cv[code, 0] > 0.0:
            lim = pm[code] / cv[code, 0]
            if lim < sigma_upper:
                sigma_upper = lim
    sigma_upper = 0.999999 * sigma_upper
    k_upper = cbrt(THREE_PI2 * nB) * (1.0 + 1.0e-10)
    lower[0] = 0.0
    lower[1] = 0.0
    lower[2] = -5.0
    lower[3] = -5.0
    upper[0] = sigma_upper
    upper[1] = 5.0
    upper[2] = 5.0
    upper[3] = 5.0
    for offset in range(ns):
        lower[4 + offset] = 0.0
        upper[4 + offset] = k_upper


@njit(cache=True, error_model="numpy")
def audit_state(x, nunk, act, species_codes, ns, active_columns, nB, cv, cd, pm, mm2, iso,
                field_scale, density_scale, chemical_scale,
                momenta, s_fields, s_dens, s_mu, s_meff, s_sc, sdens, phys):
    """Reconstruct and audit one candidate solution (port of _state_from_optimizer_result).
    Returns 0 on success, 1 residual-audit failure, 2 pressure-audit failure, 3 negative density,
    4 nonpositive effective mass."""
    for i in range(4):
        s_fields[i] = x[i]
    if act[2] == 0 and act[3] == 0:
        s_fields[3] = 0.0
    for code in range(6):
        momenta[code] = 0.0
    for offset in range(ns):
        momenta[species_codes[offset]] = x[4 + offset]
    for code in range(6):
        s_dens[code] = momenta[code] ** 3 / THREE_PI2
    for code in range(4):
        s_meff[code] = pm[code] - cv[code, 0] * s_fields[0]
        if (not math.isfinite(s_meff[code])) or s_meff[code] <= 0.0:
            return 4
    for code in range(4):
        sdens[code] = scalar_density_k(momenta[code], s_meff[code])
    rearrangement = 0.0
    for code in range(4):
        density = s_dens[code]
        rearrangement += -cd[code, 0] * s_fields[0] * sdens[code]
        rearrangement += cd[code, 1] * s_fields[1] * density
        rearrangement += cd[code, 2] * iso[code] * s_fields[2] * density
        rearrangement += cd[code, 3] * s_fields[3] * density
    for code in range(4):
        fermi_energy = math.hypot(momenta[code], s_meff[code])
        s_mu[code] = (
            fermi_energy
            + cv[code, 1] * s_fields[1]
            + cv[code, 2] * iso[code] * s_fields[2]
            + cv[code, 3] * s_fields[3]
            + rearrangement
        )
    if act[0] != 0:
        s_mu[4] = math.hypot(momenta[4], M_E)
    else:
        s_mu[4] = s_mu[0] - s_mu[1]
    s_mu[5] = math.hypot(momenta[5], M_MU)
    sigma_source = 0.0
    omega_source = 0.0
    rho_source = 0.0
    phi_source = 0.0
    for code in range(4):
        sigma_source += cv[code, 0] * sdens[code]
    for code in range(4):
        omega_source += cv[code, 1] * s_dens[code]
    for code in range(4):
        rho_source += cv[code, 2] * iso[code] * s_dens[code]
    for code in range(4):
        phi_source += cv[code, 3] * s_dens[code]
    comp_baryon = s_dens[0] + s_dens[1] + s_dens[2] + s_dens[3]
    net_charge = s_dens[1] - s_dens[3] - s_dens[4] - s_dens[5]
    phys[0] = mm2[0] * s_fields[0] - sigma_source
    phys[1] = mm2[1] * s_fields[1] - omega_source
    phys[2] = mm2[2] * s_fields[2] - rho_source
    phys[3] = mm2[3] * s_fields[3] - phi_source
    phys[4] = comp_baryon - nB
    phys[5] = net_charge
    row = 6
    if act[0] != 0:
        phys[row] = s_mu[0] - s_mu[1] - s_mu[4]
        row += 1
    if act[2] != 0:
        phys[row] = s_mu[2] - s_mu[0]
        row += 1
    if act[3] != 0:
        phys[row] = s_mu[3] - s_mu[0] - s_mu[4]
        row += 1
    if act[1] != 0:
        phys[row] = s_mu[5] - s_mu[4]
        row += 1
    scaled_norm = 0.0
    for i in range(row):
        if i < 4:
            scale = field_scale
        elif i < 6:
            scale = density_scale
        else:
            scale = chemical_scale
        v = abs(phys[i] / scale)
        if not math.isfinite(v):
            scaled_norm = math.inf
        elif v > scaled_norm:
            scaled_norm = v
    baryon_energy = 0.0
    for code in range(4):
        baryon_energy += kinetic_energy_density_k(momenta[code], s_meff[code])
    lepton_energy = kinetic_energy_density_k(momenta[4], M_E) + kinetic_energy_density_k(momenta[5], M_MU)
    scalar_term = 0.5 * (M_SIG * s_fields[0]) ** 2
    omega_term = 0.5 * (M_OM * s_fields[1]) ** 2
    rho_term = 0.5 * (M_RHO * s_fields[2]) ** 2
    phi_term = 0.5 * (M_PHI * s_fields[3]) ** 2
    energy_density = baryon_energy + lepton_energy + scalar_term + omega_term + rho_term + phi_term
    euler_pressure = (
        s_mu[0] * s_dens[0]
        + s_mu[1] * s_dens[1]
        + s_mu[2] * s_dens[2]
        + s_mu[3] * s_dens[3]
        + s_mu[4] * s_dens[4]
        + s_mu[5] * s_dens[5]
        - energy_density
    )
    kinetic_pressure_sum = 0.0
    for code in range(4):
        kinetic_pressure_sum += kinetic_pressure_k(momenta[code], s_meff[code])
    kinetic_pressure_sum = kinetic_pressure_sum + (
        kinetic_pressure_k(momenta[4], M_E) + kinetic_pressure_k(momenta[5], M_MU)
    )
    explicit_pressure = (
        kinetic_pressure_sum - scalar_term + omega_term + rho_term + phi_term + nB * rearrangement
    )
    pressure_difference = euler_pressure - explicit_pressure
    pressure_scale = max(abs(energy_density), abs(euler_pressure), 1.0e-12)
    pressure_gate = PRESSURE_GATE_FACTOR * pressure_scale
    if (not math.isfinite(scaled_norm)) or scaled_norm > RESIDUAL_GATE:
        return 1
    if (not math.isfinite(pressure_difference)) or abs(pressure_difference) > pressure_gate:
        return 2
    for code in range(6):
        if s_dens[code] < 0.0:
            return 3
    s_sc[0] = nB
    s_sc[1] = energy_density
    s_sc[2] = euler_pressure
    s_sc[3] = scaled_norm
    s_sc[4] = rearrangement
    return 0


@njit(cache=True, error_model="numpy")
def boundary_exit(x, act, species_codes, ns, new_act):
    """Port of _active_set_after_boundary_exit; returns True if the set changed."""
    for i in range(4):
        new_act[i] = act[i]
    # candidate order: electron, muon, lambda, xi_minus  (act index 0,1,2,3; codes 4,5,2,3)
    for cand in range(4):
        if cand == 0:
            code = 4
        elif cand == 1:
            code = 5
        elif cand == 2:
            code = 2
        else:
            code = 3
        if act[cand] != 0:
            dens = 0.0
            for offset in range(ns):
                if species_codes[offset] == code:
                    dens = x[4 + offset] ** 3 / THREE_PI2
            if dens <= THRESH_DENS_TOL:
                new_act[cand] = 0
                if cand == 0:
                    new_act[1] = 0
                return True
    return False


@njit(cache=True, error_model="numpy")
def updated_active_set(act, s_dens, s_mu, s_meff, s_sc, new_act):
    """Port of _updated_active_set; returns True if the set changed."""
    tolerance = max(1.0e-9, 10.0 * s_sc[3] * max(s_mu[0], 1.0))
    muon_gap = s_mu[5] - s_mu[4]
    electron_gap = M_E - s_mu[4]
    # lambda gap
    density = s_dens[2]
    effective_mass = s_meff[2]
    momentum = cbrt(THREE_PI2 * density)
    kinetic_shift = math.hypot(momentum, effective_mass) - effective_mass
    lambda_gap = (s_mu[2] - kinetic_shift) - s_mu[0]
    density = s_dens[3]
    effective_mass = s_meff[3]
    momentum = cbrt(THREE_PI2 * density)
    kinetic_shift = math.hypot(momentum, effective_mass) - effective_mass
    xi_gap = (s_mu[3] - kinetic_shift) - (s_mu[0] + s_mu[4])
    new_act[0] = act[0]
    new_act[1] = act[1]
    new_act[2] = act[2]
    new_act[3] = act[3]
    if act[0] == 0 and electron_gap < -tolerance:
        new_act[0] = 1
        new_act[1] = 0
        return True
    if act[0] != 0 and act[1] == 0 and muon_gap < -tolerance:
        new_act[1] = 1
        return True
    if act[2] == 0 and lambda_gap < -tolerance:
        new_act[2] = 1
        return True
    if act[3] == 0 and xi_gap < -tolerance:
        new_act[3] = 1
        return True
    return False


@njit(cache=True, error_model="numpy")
def solve_point(nB, has_prev, p_fields, p_dens, p_mu, p_meff, p_act, p_sc,
                c_fields, c_dens, c_mu, c_meff, c_act, c_sc,
                a_s, a_w, a_r, g_s0, g_w0, g_r0, n0, xs_L, xs_X,
                cv, cd, pm, mm2, iso, species_codes, active_columns, act, new_act,
                initial, lower, upper, safe_lower, safe_upper, x, best_x, residual, trial, trial_residual,
                step, rhs, jacobian, w_mom, w_den, w_meff, w_sd, w_bmu, w_dd, w_fe, w_sdk, w_sdm, w_bmud, w_ed,
                xsol, trial_density, phys, sdens, nfev_out):
    """Port of solve_beta_equilibrium_point; writes the converged state into c_*.
    Returns ST_OK / ST_FALLBACK / ST_FAILED / ST_BADMASS."""
    if has_prev == 0:
        act[0] = 1
        act[1] = 0
        act[2] = 0
        act[3] = 0
    else:
        act[0] = p_act[0]
        act[1] = 1 if (p_act[1] != 0 and p_act[0] != 0) else 0
        act[2] = p_act[2]
        act[3] = p_act[3]
    couplings(nB, a_s, a_w, a_r, g_s0, g_w0, g_r0, n0, xs_L, xs_X, cv, cd)
    field_scale = max(nB, 1.0e-6)
    density_scale = max(nB, 1.0e-6)
    chemical_scale = CHEM_SCALE
    have_state = 0
    for update_index in range(MAX_ASU):
        ns = layout(act, species_codes, active_columns)
        nunk = 4 + ns
        if have_state == 0:
            initial_unknowns(nB, act, species_codes, ns, has_prev, p_sc[0], p_fields, p_dens, p_mu, p_meff,
                             a_s, a_w, a_r, g_s0, g_w0, g_r0, n0, trial_density, initial)
        else:
            initial_unknowns(nB, act, species_codes, ns, 1, c_sc[0], c_fields, c_dens, c_mu, c_meff,
                             a_s, a_w, a_r, g_s0, g_w0, g_r0, n0, trial_density, initial)
        unknown_bounds(nB, ns, cv, pm, lower, upper)
        for index in range(nunk):
            interior = 1.0e-12 * max(1.0, upper[index] - lower[index])
            initial[index] = min(max(initial[index], lower[index] + interior), upper[index] - interior)
        nst, evals = bounded_newton(initial, lower, upper, nunk, species_codes, ns, active_columns, nB, cv, cd, pm, mm2, iso,
                                    field_scale, density_scale, chemical_scale,
                                    safe_lower, safe_upper, x, best_x, residual, trial, trial_residual, step, rhs, jacobian,
                                    w_mom, w_den, w_meff, w_sd, w_bmu, w_dd, w_fe, w_sdk, w_sdm, w_bmud, w_ed, xsol)
        nfev_out[0] += evals
        if nst != 1:
            # Certified path: SciPy bounded least_squares is tried here.  Its only
            # effect in every observed event is to pin the infeasible threshold
            # species at zero density, fail the audit and hand that species to
            # _active_set_after_boundary_exit.  Newton's best iterate already sits
            # on that bound (density <= threshold_density_tolerance), so apply the
            # same exit rule to it; anything else is flagged for the certified
            # CPU re-solve (ST_FALLBACK) instead of being approximated.
            if boundary_exit(xsol, act, species_codes, ns, new_act):
                for i in range(4):
                    act[i] = new_act[i]
                nfev_out[1] += 1
                continue
            return ST_FALLBACK
        astat = audit_state(xsol, nunk, act, species_codes, ns, active_columns, nB, cv, cd, pm, mm2, iso,
                            field_scale, density_scale, chemical_scale,
                            w_mom, c_fields, c_dens, c_mu, c_meff, c_sc, sdens, phys)
        if astat == 4:
            return ST_BADMASS
        if astat != 0:
            if boundary_exit(xsol, act, species_codes, ns, new_act):
                for i in range(4):
                    act[i] = new_act[i]
                continue
            return ST_FALLBACK
        have_state = 1
        for i in range(4):
            c_act[i] = act[i]
        if not updated_active_set(act, c_dens, c_mu, c_meff, c_sc, new_act):
            return ST_OK
        for i in range(4):
            act[i] = new_act[i]
    return ST_FAILED


@njit(cache=True, error_model="numpy")
def solve_row(theta, grid, npts, E, P, aset, nfev_out):
    """Trace one parameter row over the increasing density grid.  Returns (status, failing_index)."""
    a_s = theta[0]
    a_w = theta[1]
    a_r = theta[2]
    g_s0 = theta[3]
    g_w0 = theta[4]
    g_r0 = theta[5]
    n0 = theta[6]
    xs_L = theta[7]
    xs_X = theta[8]
    cv = np.empty((4, 4), dtype=np.float64)
    cd = np.empty((4, 4), dtype=np.float64)
    pm = np.empty(6, dtype=np.float64)
    mm2 = np.empty(4, dtype=np.float64)
    iso = np.empty(4, dtype=np.float64)
    pm[0] = M_N
    pm[1] = M_P
    pm[2] = M_L
    pm[3] = M_X
    pm[4] = M_E
    pm[5] = M_MU
    mm2[0] = M_SIG ** 2
    mm2[1] = M_OM ** 2
    mm2[2] = M_RHO ** 2
    mm2[3] = M_PHI ** 2
    iso[0] = -0.5
    iso[1] = 0.5
    iso[2] = 0.0
    iso[3] = -0.5
    species_codes = np.empty(6, dtype=np.int64)
    active_columns = np.empty(6, dtype=np.int64)
    act = np.empty(4, dtype=np.int64)
    new_act = np.empty(4, dtype=np.int64)
    p_act = np.empty(4, dtype=np.int64)
    c_act = np.empty(4, dtype=np.int64)
    initial = np.empty(10, dtype=np.float64)
    lower = np.empty(10, dtype=np.float64)
    upper = np.empty(10, dtype=np.float64)
    safe_lower = np.empty(10, dtype=np.float64)
    safe_upper = np.empty(10, dtype=np.float64)
    x = np.empty(10, dtype=np.float64)
    best_x = np.empty(10, dtype=np.float64)
    residual = np.empty(10, dtype=np.float64)
    trial = np.empty(10, dtype=np.float64)
    trial_residual = np.empty(10, dtype=np.float64)
    step = np.empty(10, dtype=np.float64)
    rhs = np.empty(10, dtype=np.float64)
    xsol = np.empty(10, dtype=np.float64)
    phys = np.empty(10, dtype=np.float64)
    jacobian = np.empty((10, 10), dtype=np.float64)
    w_mom = np.empty(6, dtype=np.float64)
    w_den = np.empty(6, dtype=np.float64)
    w_dd = np.empty(6, dtype=np.float64)
    w_meff = np.empty(4, dtype=np.float64)
    w_sd = np.empty(4, dtype=np.float64)
    w_bmu = np.empty(4, dtype=np.float64)
    w_fe = np.empty(4, dtype=np.float64)
    w_sdk = np.empty(4, dtype=np.float64)
    w_sdm = np.empty(4, dtype=np.float64)
    w_ed = np.empty(10, dtype=np.float64)
    w_bmud = np.empty((4, 10), dtype=np.float64)
    trial_density = np.empty(6, dtype=np.float64)
    sdens = np.empty(4, dtype=np.float64)
    p_fields = np.empty(4, dtype=np.float64)
    p_dens = np.empty(6, dtype=np.float64)
    p_mu = np.empty(6, dtype=np.float64)
    p_meff = np.empty(4, dtype=np.float64)
    p_sc = np.empty(5, dtype=np.float64)
    c_fields = np.empty(4, dtype=np.float64)
    c_dens = np.empty(6, dtype=np.float64)
    c_mu = np.empty(6, dtype=np.float64)
    c_meff = np.empty(4, dtype=np.float64)
    c_sc = np.empty(5, dtype=np.float64)
    for i in range(4):
        p_fields[i] = 0.0
        p_meff[i] = 0.0
        p_act[i] = 0
    for i in range(6):
        p_dens[i] = 0.0
        p_mu[i] = 0.0
    for i in range(5):
        p_sc[i] = 0.0
    has_prev = 0
    for j in range(npts):
        nB = grid[j]
        st = solve_point(nB, has_prev, p_fields, p_dens, p_mu, p_meff, p_act, p_sc,
                         c_fields, c_dens, c_mu, c_meff, c_act, c_sc,
                         a_s, a_w, a_r, g_s0, g_w0, g_r0, n0, xs_L, xs_X,
                         cv, cd, pm, mm2, iso, species_codes, active_columns, act, new_act,
                         initial, lower, upper, safe_lower, safe_upper, x, best_x, residual, trial, trial_residual,
                         step, rhs, jacobian, w_mom, w_den, w_meff, w_sd, w_bmu, w_dd, w_fe, w_sdk, w_sdm, w_bmud, w_ed,
                         xsol, trial_density, phys, sdens, nfev_out)
        if st != ST_OK:
            return st, j
        E[j] = c_sc[1] * HBARC
        P[j] = c_sc[2] * HBARC
        aset[j] = c_act[0] + 2 * c_act[1] + 4 * c_act[2] + 8 * c_act[3]
        for i in range(4):
            p_fields[i] = c_fields[i]
            p_meff[i] = c_meff[i]
            p_act[i] = c_act[i]
        for i in range(6):
            p_dens[i] = c_dens[i]
            p_mu[i] = c_mu[i]
        for i in range(5):
            p_sc[i] = c_sc[i]
        has_prev = 1
    return ST_OK, npts


@njit(cache=True, parallel=True, error_model="numpy")
def solve_rows(thetas, grid, E, P, status, failidx, aset, nfev):
    n = thetas.shape[0]
    npts = grid.shape[0]
    for i in prange(n):
        st, j = solve_row(thetas[i], grid, npts, E[i], P[i], aset[i], nfev[i])
        status[i] = st
        failidx[i] = j


@njit(cache=True, error_model="numpy")
def solve_rows_serial(thetas, grid, E, P, status, failidx, aset, nfev):
    n = thetas.shape[0]
    npts = grid.shape[0]
    for i in range(n):
        st, j = solve_row(thetas[i], grid, npts, E[i], P[i], aset[i], nfev[i])
        status[i] = st
        failidx[i] = j
