#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Certified DDB forward model (density-dependent RMF, Malik22 form).

The validated JAX forward behind the final nuclear-matter table (derived_one, nmp_batch), the
beta-equilibrium EOS (eos_batch) and the maximum mass from the numba TOV solver with the BPS crust
(mmax_one, mmax_batch). The hyperonic pipeline and the astrophysical likelihood modules import it for
these physics functions only. Every physics function is unchanged from the certified module; only its
standalone command-line entry point and output path, which this package does not use, were removed.
"""
import os, sys, time, argparse, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("JAX_PLATFORMS","cpu"); os.environ["JAX_ENABLE_X64"]="True"; os.environ.setdefault("CUDA_ROOT","/tmp")
import numpy as np, jax, jax.numpy as jnp
from jax import jit, vmap, lax
from joblib import Parallel, delayed
_HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,_HERE)
import tov_numba as T, ddb_crust as L

OFM=197.33    # oneoverfm_MeV (hbar*c, MeV*fm) -- TOVsolver constant, hardcoded for standalone
MN=939.0; M_SIG=550.0/OFM; M_W=783.0/OFM; M_RHO=763.0/OFM    # DDB meson masses (MeV->fm^-1)
PI=jnp.pi; M_N=4.7583690772; MB_PN=jnp.array([M_N,M_N]); H_FD=1e-4; NW=20
H_TOV=L.TOV_H
NAMES=["a_sigma","a_omega","a_rho","Gamma_sigma","Gamma_omega","Gamma_rho","rho0"]
THETA_LOW =np.array([0.0,0.0,0.0,6.5,7.5,5.0,0.140]); THETA_HIGH=np.array([0.3,0.298446,1.3,13.4895,14.5,12.5471,0.170])
OBS_MU =np.array([0.153,-16.1,230.0,32.5,0.505714285714279,1.24142857142857,2.4857142857143])
OBS_SIG=np.array([0.005,0.2,40.0*2.0,1.8*2.0,2.0*0.097142857142855,2.0*0.304285714285714,2.0*0.691428571428572])  # K0,Jsym sigma x2
MMAX_MU,MMAX_W=2.0,0.05

# ============ NMP forward (VERBATIM forward_model_theta_to_x.py; rho0 INPUT; validated K0) ============
def couplings_malik22(rho,gs0,gv0,grho0,as_,av,ar,rho0):
    x=rho/rho0; gs=gs0*jnp.exp(-(x**as_-1.0)); gw=gv0*jnp.exp(-(x**av-1.0)); gr=grho0*jnp.exp(-ar*(x-1.0))
    dgs=-(gs*as_/rho0)*x**(as_-1.0); dgw=-(gw*av/rho0)*x**(av-1.0); dgr=-(gr*ar/rho0); return gs,gw,gr,dgs,dgw,dgr
def alpha_residual(sigma,rho,alpha,gs,m_sig):
    m_eff=MB_PN-gs*sigma; rho_p=alpha*rho; rho_n=(1.0-alpha)*rho; rho_B=jnp.stack([rho_p,rho_n])
    k_fb=jnp.cbrt(3.0*PI**2*rho_B); E_fb=jnp.sqrt(k_fb**2+m_eff**2); log_arg=jnp.where(k_fb>0,(E_fb+k_fb)/m_eff,1.0)
    rho_SB=jnp.where(k_fb>0,(m_eff/(2.0*PI**2))*(E_fb*k_fb-m_eff**2*jnp.log(log_arg)),0.0)
    return sigma*m_sig**2/gs-jnp.sum(rho_SB)
def alpha_energy_pressure(sigma,rho,alpha,gs,gw,gr,dgs,dgw,dgr,m_sig,m_w,m_rho):
    m_eff=MB_PN-gs*sigma; rho_p=alpha*rho; rho_n=(1.0-alpha)*rho; rho_B=jnp.stack([rho_p,rho_n])
    k_fb=jnp.cbrt(3.0*PI**2*rho_B); E_fb=jnp.sqrt(k_fb**2+m_eff**2); rho_S=sigma*m_sig**2/gs
    omega=gw*rho/m_w**2; rho03=gr*(rho_p-rho_n)/(2.0*m_rho**2)
    Sigma_0R=dgw*omega*rho-dgs*sigma*rho_S+dgr*rho03**2*m_rho**2/gr
    log_arg_e=jnp.where(k_fb>0,(k_fb+E_fb)/m_eff,1.0)
    eb_each=jnp.where(k_fb>0,(1.0/(8.0*PI**2))*(k_fb*E_fb*(2.0*k_fb**2+m_eff**2)-jnp.log(log_arg_e)*m_eff**4),0.0)
    energy_b=jnp.sum(eb_each); aa=jnp.where(k_fb>0,jnp.minimum(k_fb/jnp.maximum(E_fb,1e-30),1.0-1e-12),0.0)
    ib=jnp.where(k_fb>0,0.25*(1.5*m_eff**4*jnp.arctanh(aa)-1.5*k_fb*m_eff**2*E_fb+k_fb**3*E_fb),0.0)
    Pressure_b=(1.0/3.0)*(1.0/PI**2)*jnp.sum(ib); st=0.5*(sigma*m_sig)**2; ot=0.5*(omega*m_w)**2; rt=0.5*(rho03*m_rho)**2
    return energy_b+st+ot+rt, Pressure_b-st+ot+rt+Sigma_0R*rho
def alpha_solve_density(rho,alpha,tp):
    gs0,gv0,grho0,as_,av,ar,rho0_p,m_sig,m_w,m_rho=tp; gs,gw,gr,dgs,dgw,dgr=couplings_malik22(rho,gs0,gv0,grho0,as_,av,ar,rho0_p)
    def r_fn(sigma): return alpha_residual(sigma,rho,alpha,gs,m_sig)
    def body(sigma,_):
        r=r_fn(sigma); dr=jax.grad(r_fn)(sigma); s=jnp.maximum(sigma-r/dr,1e-12); return jnp.where(jnp.isfinite(s),s,sigma),None
    sf,_=lax.scan(body,jnp.asarray(gs*rho/m_sig**2,jnp.float64),None,length=30); return alpha_energy_pressure(sf,rho,alpha,gs,gw,gr,dgs,dgw,dgr,m_sig,m_w,m_rho)
@jit
def derived_one(theta6,rho0):
    a_sig,a_w,a_r,Gsig0,Gw0,Gr0=theta6; tp=jnp.array([Gsig0,Gw0,Gr0,a_sig,a_w,a_r,rho0,M_SIG,M_W,M_RHO])
    rs=jnp.array([rho0-H_FD,rho0,rho0+H_FD]); es,_=vmap(lambda r:alpha_solve_density(r,0.5,tp))(rs); EA=es*OFM/rs-MN
    eps0=EA[1]; K0=9.0*rho0**2*((EA[2]-2.0*EA[1]+EA[0])/H_FD**2)
    rp=jnp.array([rho0,0.08,0.12,0.16]); ep,pp=vmap(lambda r:alpha_solve_density(r,0.0,tp))(rp); ep=ep*OFM; pp=pp*OFM
    Jsym0=ep[0]/rp[0]-MN-eps0
    return jnp.stack([eps0,K0,Jsym0,pp[1],pp[2],pp[3]])   # [eps0,K0,Jsym0,P08,P12,P16]
@jit
def nmp_batch(tb,r0b): return vmap(derived_one)(tb,r0b)

# ============ beta-eq EOS (gen_B_eos) -> Mmax (numba TOV) ============
M_E=2.5896e-3; CHARGE_PN=jnp.array([1.0,0.0]); ISO3_PN=jnp.array([0.5,-0.5]); B_BARYON=jnp.array([1.0,1.0])
ML=jnp.array([M_E,0.53544]); CHARGE_L=jnp.array([-1.0,-1.0]); B_LEPTON=jnp.array([0.0,0.0])
def ssq(x): p=x>0; return jnp.where(p,jnp.sqrt(jnp.where(p,x,1.0)),0.0)
def slg(x): p=x>0; return jnp.where(p,jnp.log(jnp.where(p,x,1.0)),0.0)
def cpl(rho,gs0,gv0,grho0,as_,av,ar,rho0):
    x=rho/rho0; gs=gs0*jnp.exp(-(x**as_-1.0)); gw=gv0*jnp.exp(-(x**av-1.0)); gr=grho0*jnp.exp(-ar*(x-1.0))
    dgs=-(gs*as_/rho0)*x**(as_-1.0); dgw=-(gw*av/rho0)*x**(av-1.0); dgr=-(gr*ar/rho0); return gs,gw,gr,dgs,dgw,dgr
def ig(rho,gs,gw,gr,ms,mw,mr,rho0):
    sg=gs*rho/ms**2; om=rho*(mw**2/gw); r3=-gr*rho/(2*mr**2); me=MB_PN[1]-gs*sg; mn=me+gw*om+gr*r3*ISO3_PN[1]; mu=0.12*M_E*(rho/rho0)**(2/3)
    return jnp.array([jnp.sqrt(sg),jnp.sqrt(om),r3,jnp.sqrt(jnp.abs(mn)),jnp.sqrt(jnp.abs(mu))])
def rsd(x,rho,gs,gw,gr,dgs,dgw,dgr,ms,mw,mr):
    ss,os_,r3,mns,mes=x; sg,om,mn,mu=ss**2,os_**2,mns**2,mes**2; me=MB_PN-gs*sg
    S0=dgw*om*rho-dgs*sg**2*ms**2/gs+dgr*r3**2*mr**2/gr; mub=B_BARYON*mn-CHARGE_PN*mu
    Er=mub-gw*om-gr*r3*ISO3_PN-S0; k2=Er**2-me**2; E=jnp.where(k2<=0,me,Er); k=ssq(k2); rB=k**3/(3*PI**2)
    rSB=(me/(2*PI**2))*(E*k-me**2*slg((E+k)/me)); mul=B_LEPTON*mn-CHARGE_L*mu; k2l=mul**2-ML**2; kl=ssq(k2l); rL=kl**3/(3*PI**2)
    return jnp.array([sg*ms**2/gs-jnp.sum(rSB),om*mw**2/gw-jnp.sum(rB),r3*mr**2/gr-jnp.sum(rB*ISO3_PN),rho-jnp.sum(rB),jnp.sum(CHARGE_PN*rB)+jnp.sum(CHARGE_L*rL)])
def epr(x,rho,gs,gw,gr,dgs,dgw,dgr,ms,mw,mr):
    ss,os_,r3,mns,mes=x; sg,om,mn,mu=ss**2,os_**2,mns**2,mes**2; me=MB_PN-gs*sg; rS=sg*ms**2/gs
    S0=dgw*om*rho-dgs*sg*rS+dgr*r3**2*mr**2/gr; mub=B_BARYON*mn-CHARGE_PN*mu
    Er=mub-gw*om-gr*r3*ISO3_PN-S0; k2=Er**2-me**2; E=jnp.where(k2<=0,me,Er); k=ssq(k2)
    eb=jnp.sum((1/(8*PI**2))*(k*E*(2*k**2+me**2)-slg((k+E)/me)*me**4))
    ib=0.25*(1.5*me**4*jnp.arctanh(jnp.minimum(k/jnp.maximum(E,1e-30),1-1e-12))-1.5*k*me**2*E+k**3*E)
    Pb=(1/3)*(1/PI**2)*jnp.sum(ib); muli=B_LEPTON*mn-CHARGE_L*mu; k2l=muli**2-ML**2; mul=jnp.where(k2l<0,ML,muli); kl=ssq(k2l)
    el=jnp.sum((1/(8*PI**2))*(kl*mul*(2*kl**2+ML**2)-ML**4*slg((kl+mul)/ML)))
    il=0.25*(1.5*ML**4*jnp.arctanh(jnp.minimum(kl/jnp.maximum(mul,1e-30),1-1e-12))-1.5*kl*ML**2*mul+kl**3*mul)
    Pl=(1/3)*(1/PI**2)*jnp.sum(il); st=0.5*(sg*ms)**2; ot=0.5*(om*mw)**2; rt=0.5*(r3*mr)**2
    return eb+el+st+ot+rt, Pb+Pl-st+ot+rt+S0*rho
def curve(th6,rho0,N=200,rho_start=0.04):
    dt=(1.5-rho_start)/(N-1); a_,av,ar,G0,V0,R0=th6; tp=jnp.array([G0,V0,R0,a_,av,ar,rho0,M_SIG,M_W,M_RHO])
    dens=rho_start+jnp.arange(N)*dt; gs,gw,gr,*_=cpl(rho_start,G0,V0,R0,a_,av,ar,rho0); x0=jnp.asarray(ig(rho_start,gs,gw,gr,M_SIG,M_W,M_RHO,rho0),jnp.float64)
    def st(xp,rho):
        g0,v0,r0,a2,av2,ar2,r02,ms2,mw2,mr2=tp; gs,gw,gr,dgs,dgw,dgr=cpl(rho,g0,v0,r0,a2,av2,ar2,r02)
        def b(c,_):
            x,lam=c; r=rsd(x,rho,gs,gw,gr,dgs,dgw,dgr,ms2,mw2,mr2); J=jax.jacfwd(lambda y:rsd(y,rho,gs,gw,gr,dgs,dgw,dgr,ms2,mw2,mr2))(x)
            dd=jnp.linalg.solve(J.T@J+lam*jnp.eye(5),-J.T@r); xn=x+dd; rn=rsd(xn,rho,gs,gw,gr,dgs,dgw,dgr,ms2,mw2,mr2); acc=jnp.sum(rn*rn)<jnp.sum(r*r)
            return (jnp.where(acc,xn,x),jnp.where(acc,lam*0.5,lam*2.0)),None
        (xf,_),_=lax.scan(b,(xp,1e-3),None,length=40); e,p=epr(xf,rho,gs,gw,gr,dgs,dgw,dgr,ms2,mw2,mr2); return xf,(e,p)
    _,(e,p)=lax.scan(st,x0,dens); return dens,e,p
@jit
def eos_batch(tb,rb): return vmap(curve)(tb,rb)
def mmax_one(eps,P):
    ok=np.isfinite(eps)&np.isfinite(P)&(eps>0)&(P>0); eps,P=eps[ok],P[ok]
    if len(eps)<5: return np.nan
    o=np.argsort(eps); eps,P=eps[o],P[o]
    try:
        ef,pf=L.graft_bps_crust(eps,P); eg=np.ascontiguousarray(ef*T.MEV_FM3_TO_GEOM); pg=np.ascontiguousarray(pf*T.MEV_FM3_TO_GEOM)
        pc=T.P_C_GEOM[T.P_C_GEOM<0.999*pg[-1]]
        if len(pc)<4: return np.nan
        M,R=T._mr_curve(np.ascontiguousarray(pc),pg,eg,H_TOV); v=np.isfinite(M)&(M>0)
        return float(np.max(M[v])) if v.sum()>=4 else np.nan
    except Exception: return np.nan
def mmax_batch(th6,rho0):
    _,e,p=eos_batch(jnp.asarray(th6),jnp.asarray(rho0)); e=np.asarray(e)*OFM; p=np.asarray(p)*OFM
    return np.array(Parallel(n_jobs=NW,batch_size=8)(delayed(mmax_one)(e[k],p[k]) for k in range(len(th6))))

def loglike_vec(theta):
    th=np.atleast_2d(np.asarray(theta,float)); th6=th[:,:6]; rho0=th[:,6]
    nm=np.asarray(nmp_batch(jnp.asarray(th6),jnp.asarray(rho0)))   # [eps0,K0,Jsym,P08,P12,P16]
    x=np.column_stack([rho0,nm]); mm=mmax_batch(th6,rho0)
    ok=np.isfinite(x).all(1)&np.isfinite(mm)
    r=(OBS_MU[None,:]-x)/OBS_SIG[None,:]
    ll=-0.5*np.sum(r**2,1)-np.logaddexp(0.0,-(mm-MMAX_MU)/MMAX_W)
    ll[~ok]=-1e30
    return ll if np.asarray(theta).ndim>1 else ll[0]

