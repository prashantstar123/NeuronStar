"""Validated nucleonic DDB forward model.

This module is a dependency-clean extraction of the EOS equations in
the hash-validated reference implementation.  Equations, constants, density
grid, nonlinear iterations, and parameter ordering are unchanged.  Sampling,
likelihood, crust, and TOV concerns deliberately live elsewhere.

The returned core energy density and pressure follow the validated unit convention:
the low-level JAX functions use fm^-1, while the public ``core_eos`` helpers
return MeV/fm^3.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["JAX_ENABLE_X64"] = "True"
os.environ.setdefault("CUDA_ROOT", "/tmp")

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, lax, vmap

OFM = 197.33
MN = 939.0
M_SIG = 550.0 / OFM
M_W = 783.0 / OFM
M_RHO = 763.0 / OFM
PI = jnp.pi
M_N = 4.7583690772
MB_PN = jnp.array([M_N, M_N])
H_FD = 1e-4

PARAMETER_NAMES = (
    "a_sigma",
    "a_omega",
    "a_rho",
    "Gamma_sigma",
    "Gamma_omega",
    "Gamma_rho",
    "rho0",
)
PRIOR_LOW = np.array([0.0, 0.0, 0.0, 6.5, 7.5, 5.0, 0.140])
PRIOR_HIGH = np.array(
    [0.3, 0.298446, 1.3, 13.4895, 14.5, 12.5471, 0.170]
)


# NMP forward: verbatim arithmetic from the validated reference implementation.
def couplings_malik22(rho, gs0, gv0, grho0, as_, av, ar, rho0):
    x = rho / rho0
    gs = gs0 * jnp.exp(-(x**as_ - 1.0))
    gw = gv0 * jnp.exp(-(x**av - 1.0))
    gr = grho0 * jnp.exp(-ar * (x - 1.0))
    dgs = -(gs * as_ / rho0) * x ** (as_ - 1.0)
    dgw = -(gw * av / rho0) * x ** (av - 1.0)
    dgr = -(gr * ar / rho0)
    return gs, gw, gr, dgs, dgw, dgr


def alpha_residual(sigma, rho, alpha, gs, m_sig):
    m_eff = MB_PN - gs * sigma
    rho_p = alpha * rho
    rho_n = (1.0 - alpha) * rho
    rho_B = jnp.stack([rho_p, rho_n])
    k_fb = jnp.cbrt(3.0 * PI**2 * rho_B)
    E_fb = jnp.sqrt(k_fb**2 + m_eff**2)
    log_arg = jnp.where(k_fb > 0, (E_fb + k_fb) / m_eff, 1.0)
    rho_SB = jnp.where(
        k_fb > 0,
        (m_eff / (2.0 * PI**2))
        * (E_fb * k_fb - m_eff**2 * jnp.log(log_arg)),
        0.0,
    )
    return sigma * m_sig**2 / gs - jnp.sum(rho_SB)


def alpha_energy_pressure(
    sigma, rho, alpha, gs, gw, gr, dgs, dgw, dgr, m_sig, m_w, m_rho
):
    m_eff = MB_PN - gs * sigma
    rho_p = alpha * rho
    rho_n = (1.0 - alpha) * rho
    rho_B = jnp.stack([rho_p, rho_n])
    k_fb = jnp.cbrt(3.0 * PI**2 * rho_B)
    E_fb = jnp.sqrt(k_fb**2 + m_eff**2)
    rho_S = sigma * m_sig**2 / gs
    omega = gw * rho / m_w**2
    rho03 = gr * (rho_p - rho_n) / (2.0 * m_rho**2)
    Sigma_0R = (
        dgw * omega * rho
        - dgs * sigma * rho_S
        + dgr * rho03**2 * m_rho**2 / gr
    )
    log_arg_e = jnp.where(k_fb > 0, (k_fb + E_fb) / m_eff, 1.0)
    eb_each = jnp.where(
        k_fb > 0,
        (1.0 / (8.0 * PI**2))
        * (
            k_fb * E_fb * (2.0 * k_fb**2 + m_eff**2)
            - jnp.log(log_arg_e) * m_eff**4
        ),
        0.0,
    )
    energy_b = jnp.sum(eb_each)
    aa = jnp.where(
        k_fb > 0,
        jnp.minimum(k_fb / jnp.maximum(E_fb, 1e-30), 1.0 - 1e-12),
        0.0,
    )
    ib = jnp.where(
        k_fb > 0,
        0.25
        * (
            1.5 * m_eff**4 * jnp.arctanh(aa)
            - 1.5 * k_fb * m_eff**2 * E_fb
            + k_fb**3 * E_fb
        ),
        0.0,
    )
    Pressure_b = (1.0 / 3.0) * (1.0 / PI**2) * jnp.sum(ib)
    st = 0.5 * (sigma * m_sig) ** 2
    ot = 0.5 * (omega * m_w) ** 2
    rt = 0.5 * (rho03 * m_rho) ** 2
    return energy_b + st + ot + rt, Pressure_b - st + ot + rt + Sigma_0R * rho


def alpha_solve_density(rho, alpha, tp):
    gs0, gv0, grho0, as_, av, ar, rho0_p, m_sig, m_w, m_rho = tp
    gs, gw, gr, dgs, dgw, dgr = couplings_malik22(
        rho, gs0, gv0, grho0, as_, av, ar, rho0_p
    )

    def r_fn(sigma):
        return alpha_residual(sigma, rho, alpha, gs, m_sig)

    def body(sigma, _):
        residual = r_fn(sigma)
        derivative = jax.grad(r_fn)(sigma)
        candidate = jnp.maximum(sigma - residual / derivative, 1e-12)
        return jnp.where(jnp.isfinite(candidate), candidate, sigma), None

    sigma_final, _ = lax.scan(
        body,
        jnp.asarray(gs * rho / m_sig**2, jnp.float64),
        None,
        length=30,
    )
    return alpha_energy_pressure(
        sigma_final,
        rho,
        alpha,
        gs,
        gw,
        gr,
        dgs,
        dgw,
        dgr,
        m_sig,
        m_w,
        m_rho,
    )


@jit
def derived_one(theta6, rho0):
    a_sig, a_w, a_r, Gsig0, Gw0, Gr0 = theta6
    tp = jnp.array(
        [Gsig0, Gw0, Gr0, a_sig, a_w, a_r, rho0, M_SIG, M_W, M_RHO]
    )
    rs = jnp.array([rho0 - H_FD, rho0, rho0 + H_FD])
    es, _ = vmap(lambda r: alpha_solve_density(r, 0.5, tp))(rs)
    EA = es * OFM / rs - MN
    eps0 = EA[1]
    K0 = 9.0 * rho0**2 * ((EA[2] - 2.0 * EA[1] + EA[0]) / H_FD**2)
    rp = jnp.array([rho0, 0.08, 0.12, 0.16])
    ep, pp = vmap(lambda r: alpha_solve_density(r, 0.0, tp))(rp)
    ep = ep * OFM
    pp = pp * OFM
    Jsym0 = ep[0] / rp[0] - MN - eps0
    return jnp.stack([eps0, K0, Jsym0, pp[1], pp[2], pp[3]])


@jit
def nmp_batch(tb, r0b):
    return vmap(derived_one)(tb, r0b)


# Beta-equilibrium EOS: verbatim arithmetic from the validated implementation.
M_E = 2.5896e-3
CHARGE_PN = jnp.array([1.0, 0.0])
ISO3_PN = jnp.array([0.5, -0.5])
B_BARYON = jnp.array([1.0, 1.0])
ML = jnp.array([M_E, 0.53544])
CHARGE_L = jnp.array([-1.0, -1.0])
B_LEPTON = jnp.array([0.0, 0.0])


def ssq(x):
    positive = x > 0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, x, 1.0)), 0.0)


def slg(x):
    positive = x > 0
    return jnp.where(positive, jnp.log(jnp.where(positive, x, 1.0)), 0.0)


def cpl(rho, gs0, gv0, grho0, as_, av, ar, rho0):
    x = rho / rho0
    gs = gs0 * jnp.exp(-(x**as_ - 1.0))
    gw = gv0 * jnp.exp(-(x**av - 1.0))
    gr = grho0 * jnp.exp(-ar * (x - 1.0))
    dgs = -(gs * as_ / rho0) * x ** (as_ - 1.0)
    dgw = -(gw * av / rho0) * x ** (av - 1.0)
    dgr = -(gr * ar / rho0)
    return gs, gw, gr, dgs, dgw, dgr


def ig(rho, gs, gw, gr, ms, mw, mr, rho0):
    sg = gs * rho / ms**2
    om = rho * (mw**2 / gw)
    r3 = -gr * rho / (2 * mr**2)
    me = MB_PN[1] - gs * sg
    mn = me + gw * om + gr * r3 * ISO3_PN[1]
    mu = 0.12 * M_E * (rho / rho0) ** (2 / 3)
    return jnp.array(
        [jnp.sqrt(sg), jnp.sqrt(om), r3, jnp.sqrt(jnp.abs(mn)), jnp.sqrt(jnp.abs(mu))]
    )


def rsd(x, rho, gs, gw, gr, dgs, dgw, dgr, ms, mw, mr):
    ss, os_, r3, mns, mes = x
    sg, om, mn, mu = ss**2, os_**2, mns**2, mes**2
    me = MB_PN - gs * sg
    S0 = dgw * om * rho - dgs * sg**2 * ms**2 / gs + dgr * r3**2 * mr**2 / gr
    mub = B_BARYON * mn - CHARGE_PN * mu
    Er = mub - gw * om - gr * r3 * ISO3_PN - S0
    k2 = Er**2 - me**2
    E = jnp.where(k2 <= 0, me, Er)
    k = ssq(k2)
    rB = k**3 / (3 * PI**2)
    rSB = (me / (2 * PI**2)) * (E * k - me**2 * slg((E + k) / me))
    mul = B_LEPTON * mn - CHARGE_L * mu
    k2l = mul**2 - ML**2
    kl = ssq(k2l)
    rL = kl**3 / (3 * PI**2)
    return jnp.array(
        [
            sg * ms**2 / gs - jnp.sum(rSB),
            om * mw**2 / gw - jnp.sum(rB),
            r3 * mr**2 / gr - jnp.sum(rB * ISO3_PN),
            rho - jnp.sum(rB),
            jnp.sum(CHARGE_PN * rB) + jnp.sum(CHARGE_L * rL),
        ]
    )


def epr(x, rho, gs, gw, gr, dgs, dgw, dgr, ms, mw, mr):
    ss, os_, r3, mns, mes = x
    sg, om, mn, mu = ss**2, os_**2, mns**2, mes**2
    me = MB_PN - gs * sg
    rS = sg * ms**2 / gs
    S0 = dgw * om * rho - dgs * sg * rS + dgr * r3**2 * mr**2 / gr
    mub = B_BARYON * mn - CHARGE_PN * mu
    Er = mub - gw * om - gr * r3 * ISO3_PN - S0
    k2 = Er**2 - me**2
    E = jnp.where(k2 <= 0, me, Er)
    k = ssq(k2)
    eb = jnp.sum(
        (1 / (8 * PI**2))
        * (k * E * (2 * k**2 + me**2) - slg((k + E) / me) * me**4)
    )
    ib = 0.25 * (
        1.5
        * me**4
        * jnp.arctanh(jnp.minimum(k / jnp.maximum(E, 1e-30), 1 - 1e-12))
        - 1.5 * k * me**2 * E
        + k**3 * E
    )
    Pb = (1 / 3) * (1 / PI**2) * jnp.sum(ib)
    muli = B_LEPTON * mn - CHARGE_L * mu
    k2l = muli**2 - ML**2
    mul = jnp.where(k2l < 0, ML, muli)
    kl = ssq(k2l)
    el = jnp.sum(
        (1 / (8 * PI**2))
        * (kl * mul * (2 * kl**2 + ML**2) - ML**4 * slg((kl + mul) / ML))
    )
    il = 0.25 * (
        1.5
        * ML**4
        * jnp.arctanh(jnp.minimum(kl / jnp.maximum(mul, 1e-30), 1 - 1e-12))
        - 1.5 * kl * ML**2 * mul
        + kl**3 * mul
    )
    Pl = (1 / 3) * (1 / PI**2) * jnp.sum(il)
    st = 0.5 * (sg * ms) ** 2
    ot = 0.5 * (om * mw) ** 2
    rt = 0.5 * (r3 * mr) ** 2
    return eb + el + st + ot + rt, Pb + Pl - st + ot + rt + S0 * rho


def curve(th6, rho0, N=200, rho_start=0.04):
    dt = (1.5 - rho_start) / (N - 1)
    a_, av, ar, G0, V0, R0 = th6
    tp = jnp.array([G0, V0, R0, a_, av, ar, rho0, M_SIG, M_W, M_RHO])
    dens = rho_start + jnp.arange(N) * dt
    gs, gw, gr, *_ = cpl(rho_start, G0, V0, R0, a_, av, ar, rho0)
    x0 = jnp.asarray(ig(rho_start, gs, gw, gr, M_SIG, M_W, M_RHO, rho0), jnp.float64)

    def st(xp, rho):
        g0, v0, r0, a2, av2, ar2, r02, ms2, mw2, mr2 = tp
        gs, gw, gr, dgs, dgw, dgr = cpl(rho, g0, v0, r0, a2, av2, ar2, r02)

        def b(carry, _):
            x, lam = carry
            residual = rsd(x, rho, gs, gw, gr, dgs, dgw, dgr, ms2, mw2, mr2)
            jacobian = jax.jacfwd(
                lambda y: rsd(y, rho, gs, gw, gr, dgs, dgw, dgr, ms2, mw2, mr2)
            )(x)
            delta = jnp.linalg.solve(
                jacobian.T @ jacobian + lam * jnp.eye(5),
                -jacobian.T @ residual,
            )
            candidate = x + delta
            next_residual = rsd(
                candidate, rho, gs, gw, gr, dgs, dgw, dgr, ms2, mw2, mr2
            )
            accepted = jnp.sum(next_residual * next_residual) < jnp.sum(
                residual * residual
            )
            return (
                jnp.where(accepted, candidate, x),
                jnp.where(accepted, lam * 0.5, lam * 2.0),
            ), None

        (xf, _), _ = lax.scan(b, (xp, 1e-3), None, length=40)
        energy, pressure = epr(
            xf, rho, gs, gw, gr, dgs, dgw, dgr, ms2, mw2, mr2
        )
        return xf, (energy, pressure)

    _, (energy, pressure) = lax.scan(st, x0, dens)
    return dens, energy, pressure


@jit
def eos_batch(tb, rb):
    return vmap(curve)(tb, rb)


def _as_theta(theta):
    value = np.asarray(theta, dtype=np.float64)
    if value.shape != (7,):
        raise ValueError(f"expected seven DDB parameters, got shape {value.shape}")
    return value


def nuclear_matter_observables(theta):
    """Return ``[E0, K0, Jsym, P08, P12, P16]`` for one parameter vector."""

    value = _as_theta(theta)
    return np.asarray(derived_one(jnp.asarray(value[:6]), value[6]), dtype=np.float64)


def nuclear_matter_observables_batch(theta):
    """Vectorized nuclear-matter observables for an ``(N, 7)`` array."""

    value = np.asarray(theta, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 7:
        raise ValueError(f"expected DDB parameter array with shape (N, 7), got {value.shape}")
    return np.asarray(
        nmp_batch(jnp.asarray(value[:, :6]), jnp.asarray(value[:, 6])),
        dtype=np.float64,
    )


def core_eos(theta):
    """Return ``(density, energy, pressure)`` for one DDB core EOS.

    Density is in fm^-3; energy density and pressure are in MeV/fm^3.
    """

    value = _as_theta(theta)
    density, energy, pressure = curve(jnp.asarray(value[:6]), value[6])
    return (
        np.asarray(density, dtype=np.float64),
        np.asarray(energy, dtype=np.float64) * OFM,
        np.asarray(pressure, dtype=np.float64) * OFM,
    )


def core_eos_batch(theta):
    """Vectorized core EOS generation for an ``(N, 7)`` parameter array."""

    value = np.asarray(theta, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 7:
        raise ValueError(f"expected DDB parameter array with shape (N, 7), got {value.shape}")
    density, energy, pressure = eos_batch(
        jnp.asarray(value[:, :6]), jnp.asarray(value[:, 6])
    )
    return (
        np.asarray(density, dtype=np.float64),
        np.asarray(energy, dtype=np.float64) * OFM,
        np.asarray(pressure, dtype=np.float64) * OFM,
    )
