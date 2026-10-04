#!/usr/bin/env python3
"""SOLUTION 1: Green's-function Evidence Network (exact amortization over ARBITRARY source clouds).

Evidence is a LINEAR functional of each source density q_k:
        Z_k[q](d) = \int\int q(M,R) G_k(M,R; d) dM dR ,
where G_k is the (unnormalized) leave-source-k-out posterior-predictive M-R density of the bank.
By the Riesz representation theorem the whole source dependence IS the kernel G_k.  The network
learns log G_k(M,R; d) itself (cf. RieszNet, Chernozhukov+ 2022; linear-branch DeepONet, Lu+ 2021);
training integrates it against synthetic Gaussian test sources on a fixed lattice and regresses the
exact on-the-fly bank labels.  At query time the CERTIFIED density table (KDE x 1-99% rectangle, as
is) is integrated against the frozen kernel: no mixture fit, no encoding, any cloud shape.

Blindness: trained ONLY on single-Gaussian synthetic sources [0,7168); selected on dev [7168,7680);
tested on blind synthetic [7680,8192) AND on the three real baseline clouds, whose shapes the
network has never seen.  J0614/J1231/J1614 are never opened.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch
from torch import nn
HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("DDB_REPO_ROOT", str(HERE.parents[1])))
sys.path.insert(0, str(REPO_ROOT)); sys.path.insert(0, str(HERE.parent))
from train_onfly import FAMILIES, ExactLabels, draw_designs, summarize, sha256  # noqa: E402
from inference.evidence_network.conditional.build_dataset import SOURCE_NAMES  # noqa: E402
from inference.evidence_network.conditional.build_dataset_v3 import HELD_OUT_FILENAMES  # noqa: E402
from likelihoods.nicer import build_nicer_grid  # noqa: E402
from workflows.source_scenarios import BASE_NICER_SOURCES  # noqa: E402
from workflows.nuclear_scenarios import NUCLEAR_SCENARIO_NAMES, nuclear_observation  # noqa: E402
M_LO, M_HI, R_LO, R_HI = 1.0, 2.6, 8.2, 18.4


class GreenKernel(nn.Module):
    """log G_k(M,R; t) = head( phi_k(M,R) * psi(t) ) + u_k(M,R) + v(t): multiplicative (FiLM) coupling of a
    coordinate trunk (multi-scale Fourier features) with a nuclear-data trunk.  The lattice trunk is evaluated once
    per step, so a full-lattice overlap integral costs milliseconds."""
    def __init__(self, width=256, bands=6):
        super().__init__()
        self.register_buffer("freq", (2.0 ** torch.arange(bands)) * np.pi, persistent=True)
        self.trunk = nn.Sequential(nn.Linear(2 + 4 * bands, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.slot = nn.Parameter(torch.zeros(3, width)); nn.init.normal_(self.slot, std=0.5)
        self.tnet = nn.Sequential(nn.Linear(7, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.head = nn.Sequential(nn.SiLU(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))
        self.ux = nn.Linear(width, 1); self.vt = nn.Linear(width, 1)
        self.identity_head = nn.Sequential(nn.Linear(7, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 1))
    def forward(self, xy, t, slot):
        ph = xy[:, :, None] * self.freq
        f = self.trunk(torch.cat([xy, torch.sin(ph).flatten(1), torch.cos(ph).flatten(1)], 1)) + self.slot[slot]      # (P,W)
        g = self.tnet(t)                                                                                               # (D,W)
        return self.head(f[:, None, :] * (1.0 + g[None, :, :])).squeeze(-1) + self.ux(f) + self.vt(g).T
    def identity(self, t): return self.identity_head(t).squeeze(-1)


def unit(M, R, mass_low=M_LO, mass_high=M_HI, radius_low=R_LO, radius_high=R_HI):
    return torch.stack([(M - mass_low) / (mass_high - mass_low),
                        (R - radius_low) / (radius_high - radius_low)], -1)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--cache", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--baseline-root", type=Path, required=True)
    p.add_argument("--seed", type=int, default=9201); p.add_argument("--steps", type=int, default=12000); p.add_argument("--batch-scenarios", type=int, default=192)
    p.add_argument("--designs-per-step", type=int, default=9); p.add_argument("--lr", type=float, default=6e-4); p.add_argument("--nm", type=int, default=161); p.add_argument("--nr", type=int, default=205)
    p.add_argument("--mass-low", type=float, default=M_LO); p.add_argument("--mass-high", type=float, default=M_HI)
    p.add_argument("--radius-low", type=float, default=R_LO); p.add_argument("--radius-high", type=float, default=R_HI)
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    for f in HELD_OUT_FILENAMES:
        if (a.baseline_root / f).exists(): raise RuntimeError("held-out file present")
    t_start = time.perf_counter(); dev = torch.device("cuda"); torch.manual_seed(a.seed); torch.set_float32_matmul_precision("high")
    I = np.load(a.cache, allow_pickle=False); OBS, SIG = I["OBS"].astype(np.float64), I["SIG"].astype(np.float64)
    bank = np.load(a.bank, allow_pickle=False)
    domain = (a.mass_low, a.mass_high, a.radius_low, a.radius_high)
    Rb = bank["R"][I["kept_bank_rows"]][:, bank["MG"] >= a.mass_low]; print("label-bearing (in-box, finite) curve radius range: %.2f .. %.2f km; Mmax <= %.3f" % (np.nanmin(Rb), np.nanmax(Rb), bank["MM"][I["kept_bank_rows"]].max()), flush=True)
    assert np.nanmin(Rb) > a.radius_low and np.nanmax(Rb) < a.radius_high and bank["MM"][I["kept_bank_rows"]].max() < a.mass_high
    z = ((bank["X"][I["kept_bank_rows"]].astype(np.float64) - OBS) / SIG).astype(np.float32)
    labels = ExactLabels(torch.from_numpy(np.load(a.cache.with_suffix(".sourcelogw.npy"), mmap_mode="r")[:]).to(dev), torch.from_numpy(z).to(dev), float(I["denominator"]))
    mode, act = I["mode"], I["active_mask"]; mu = torch.as_tensor(I["mixture_mean"][:, :, 0], device=dev); cv = torch.as_tensor(I["mixture_covariance"][:, :, 0], device=dev)
    slot_of = torch.as_tensor(np.where(mode == 1, act.argmax(1), -1), device=dev)
    ntr, dev_stop, total = 7168, 7680, 8192
    single = {k: [torch.nonzero((slot_of[lo:hi] == k)).flatten() + lo for k in range(3)] for k, (lo, hi) in {"train": (0, ntr), "dev": (ntr, dev_stop), "test": (dev_stop, total)}.items()}
    ident_train = torch.nonzero(torch.as_tensor(mode[:ntr] == 0, device=dev)).flatten()
    Mg = torch.linspace(a.mass_low, a.mass_high, a.nm, device=dev); Rg = torch.linspace(a.radius_low, a.radius_high, a.nr, device=dev)
    MMg, RRg = torch.meshgrid(Mg, Rg, indexing="ij"); lat = torch.stack([MMg.flatten(), RRg.flatten()], 1); lat_u = unit(lat[:, 0], lat[:, 1], *domain)
    wM = torch.full((a.nm,), float(Mg[1] - Mg[0]), device=dev); wM[[0, -1]] *= 0.5; wR = torch.full((a.nr,), float(Rg[1] - Rg[0]), device=dev); wR[[0, -1]] *= 0.5
    logw = torch.log((wM[:, None] * wR[None, :]).flatten())
    print(f"lattice {a.nm}x{a.nr} = {len(lat)} nodes, dM={float(Mg[1]-Mg[0]):.4f} dR={float(Rg[1]-Rg[0]):.4f}", flush=True)

    def log_gauss_on_lattice(idx, k):
        m, C = mu[idx, k], cv[idx, k]; det = C[:, 0, 0] * C[:, 1, 1] - C[:, 0, 1] ** 2
        a_, b_ = lat[None, :, 0] - m[:, None, 0], lat[None, :, 1] - m[:, None, 1]
        maha = (C[:, 1, 1, None] * a_ * a_ - 2 * C[:, 0, 1, None] * a_ * b_ + C[:, 0, 0, None] * b_ * b_) / det[:, None]
        return -0.5 * maha - np.log(2 * np.pi) - 0.5 * torch.log(det)[:, None] + logw[None, :]           # (S,P) includes quadrature weight

    def log_overlap(logq, logG, log_floor=None):                                                           # (S,P),(P,D) -> (S,D)
        aS = logq.max(1, keepdim=True).values; bD = logG.max(0, keepdim=True).values
        main = torch.log(torch.exp(logq - aS) @ torch.exp(logG - bD) + 1e-37) + aS + bD
        if log_floor is None: return main
        loo = torch.logsumexp(logG + logw[:, None], dim=0)                                                 # log <1,G>(t): leave-one-out evidence
        return torch.logaddexp(main, log_floor[:, None] + loo[None, :])

    def gauss_log_floor(idx, k):                                                                           # the label evaluator floors the density at 1e-6 x peak
        C = cv[idx, k]; det = C[:, 0, 0] * C[:, 1, 1] - C[:, 0, 1] ** 2
        return np.log(1.0e-6) - np.log(2 * np.pi) - 0.5 * torch.log(det)

    model = GreenKernel().to(dev); opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps, eta_min=a.lr * 0.02); gen = torch.Generator(device=dev).manual_seed(a.seed + 7)
    fixed = {}
    for role, s0 in (("dev", 910_001), ("test", 920_001)):
        g = torch.Generator(device=dev).manual_seed(s0); fixed[role] = {f: draw_designs(f, 64, g, dev) for f in FAMILIES}
    paper = torch.as_tensor(np.stack([(np.asarray(nuclear_observation(n), float) - OBS) / SIG for n in NUCLEAR_SCENARIO_NAMES]), dtype=torch.float32, device=dev)
    with torch.no_grad(): probe = labels(torch.arange(0, ntr, 8, device=dev), draw_designs("gauss", 64, torch.Generator(device=dev).manual_seed(5), dev)); mean = float(probe.mean())

    def predict(idx, k, shift, chunk=8):
        out = []
        with torch.no_grad():
            lq = log_gauss_on_lattice(idx, k)
            for s in range(0, len(shift), chunk): out.append(log_overlap(lq, model(lat_u, shift[s:s + chunk], k) + mean, gauss_log_floor(idx, k)))
        return torch.cat(out, 1)

    def evaluate(role_sources, shifts):
        res = []
        for k in range(3):
            idx = single[role_sources][k]
            if len(idx): res.append((predict(idx, k, shifts, ) - labels(idx, shifts)).double().cpu().numpy())
        return res

    dev_shift = torch.cat([fixed["dev"][f][:16] for f in FAMILIES] + [paper]); best, best_state, hist = 1e9, None, []
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for step in range(1, a.steps + 1):
        per = a.designs_per_step // 3; shift = torch.cat([draw_designs(f, per, gen, dev) for f in FAMILIES]); loss = 0.0
        opt.zero_grad(set_to_none=True)
        for k in range(3):
            pool = single["train"][k]; idx = pool[torch.randint(len(pool), (a.batch_scenarios // 3,), generator=gen, device=dev)]
            pred = log_overlap(log_gauss_on_lattice(idx, k), model(lat_u, shift, k) + mean, gauss_log_floor(idx, k)); lk = torch.mean((pred - labels(idx, shift)) ** 2)
            (lk / 3).backward(); loss += float(lk) / 3
        ii = ident_train[torch.randint(len(ident_train), (8,), generator=gen, device=dev)]
        li = torch.mean((model.identity(shift) + mean - labels(ii[:1], shift)[0]) ** 2); li.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10.0); opt.step(); sched.step()
        if step == 1 or step % 250 == 0 or step == a.steps:
            model.eval(); r = np.concatenate([x.ravel() for x in evaluate("dev", dev_shift)]); dm = float(np.abs(r).mean()); model.train()
            hist.append({"step": step, "train_mse": loss, "dev_mae_one_source": dm, "dev_median_abs": float(np.median(np.abs(r))), "dev_p90_abs": float(np.quantile(np.abs(r), .9)), "elapsed": time.perf_counter() - t0}); print(json.dumps(hist[-1]), flush=True)
            if dm < best: best, best_state = dm, {k_: v.detach().cpu().clone() for k_, v in model.state_dict().items()}
    torch.cuda.synchronize(); train_s = time.perf_counter() - t0; model.load_state_dict(best_state); model.eval()

    report = {"blind_synthetic_one_source": {}, "best_dev_mae": best, "training_seconds": train_s, "steps": a.steps, "lattice": [a.nm, a.nr]}
    for fam, shift in list(fixed["test"].items()) + [("paper", paper)]:
        per_slot = evaluate("test", shift)
        report["blind_synthetic_one_source"][fam] = {"all": summarize(np.concatenate([x.ravel() for x in per_slot])), **{SOURCE_NAMES[k]: summarize(per_slot[k]) for k in range(3)}}
    with torch.no_grad():
        idr = (model.identity(paper) + mean - labels(ident_train[:1], paper)[0]).double().cpu().numpy(); report["identity_paper_residual"] = idr.tolist()
    # ---------- REAL CLOUDS: integrate the CERTIFIED table against the frozen kernel (no encoding) ----------
    grid = dict(nM=150, nR=150, max_rows=20_000, bw=0.08, seed=0); cert_id = labels(ident_train[:1], paper)[0].double().cpu().numpy(); real = {}
    for k, (name, src) in enumerate(zip(SOURCE_NAMES, BASE_NICER_SOURCES, strict=True)):
        tq = time.perf_counter(); it = build_nicer_grid(a.baseline_root / src.filename, mcol=src.mass_column, rcol=src.radius_column, wcol_or_None=src.weight_column, **grid); t_table = time.perf_counter() - tq
        tq = time.perf_counter(); gm, gr = (torch.as_tensor(x, dtype=torch.float32, device=dev) for x in it.grid); logq = torch.as_tensor(np.asarray(it.values), dtype=torch.float32, device=dev)
        GM, GR = torch.meshgrid(gm, gr, indexing="ij")
        keep = ((logq > it.fill_value + 1e-6) & (GM >= a.mass_low) & (GM <= a.mass_high)
                & (GR >= a.radius_low) & (GR <= a.radius_high))
        wm = torch.full_like(gm, float(gm[1] - gm[0])); wm[[0, -1]] *= 0.5; wr = torch.full_like(gr, float(gr[1] - gr[0])); wr[[0, -1]] *= 0.5
        lq = (logq + torch.log(wm[:, None] * wr[None, :]))[keep][None, :]
        with torch.no_grad():
            loo = torch.logsumexp(model(lat_u, paper, k) + mean + logw[:, None], dim=0)
            z_net = torch.logaddexp(log_overlap(lq, model(unit(GM[keep], GR[keep], *domain), paper, k) + mean)[0], float(it.fill_value) + loo).double().cpu().numpy()
        torch.cuda.synchronize(); t_net = time.perf_counter() - tq
        real[name] = {"network_logZ": z_net.tolist(), "certified_logZ": cert_id.tolist(), "TOTAL_error": (z_net - cert_id).tolist(), "mae": float(np.abs(z_net - cert_id).mean()),
                      "table_nodes_used": int(keep.sum()), "certified_table_build_s": t_table, "kernel_integration_s": t_net}
    report["real_baseline_clouds_no_encoding"] = real
    torch.save({"state_dict": best_state, "mean": mean, "lattice": [a.nm, a.nr], "domain": list(domain), "model_type": "green_function_en_v1", "query_ready": False}, a.output)
    report.update({"output_sha256": sha256(a.output), "bank_sha256": sha256(a.bank), "cache_sha256": sha256(a.cache),
                   "baseline_root": str(a.baseline_root), "held_out_source_files_used": [],
                   "total_wall_seconds": time.perf_counter() - t_start,
                   "peak_gpu_MB": int(torch.cuda.max_memory_allocated() >> 20), "history": hist})
    a.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print("\n=== blind synthetic one-source MAE ==="); [print(f"  {f:6s} all {v['all']['mae']:.4f}  " + "  ".join(f"{n} {v[n]['mae']:.4f}" for n in SOURCE_NAMES)) for f, v in report["blind_synthetic_one_source"].items()]
    print("identity residual at paper settings:", np.round(idr, 4))
    print("=== REAL baseline clouds, certified table integrated against the kernel (NO encoding) ===")
    for n, v in real.items(): print(f"  {n}: total error " + " ".join(f"{x:+.4f}" for x in v["TOTAL_error"]) + f"  | MAE {v['mae']:.4f} | nodes {v['table_nodes_used']} | integrate {v['kernel_integration_s']*1e3:.0f} ms")
    print(f"train {train_s:.0f}s  peak {report['peak_gpu_MB']} MB")

if __name__ == "__main__": main()
