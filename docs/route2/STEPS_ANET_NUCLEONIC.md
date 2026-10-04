# Nucleonic A-NET: how to rerun it from scratch

This page lists, step by step, how the nucleonic A-NET+IS results of the paper were made: the shared physics
bank, the 2,000 training scenarios, the trained network (called `fmpe9c`), the A1 query, the three held-out
NICER sources (J0614, J1231, J1614), the four nuclear shifts, and the A1 mass–tidal refinement. For each step it
gives the command for a rerun with the code in this repository, the settings, the seeds, the run time where it is
known, and what is missing.

The commands use the package code with the settings listed; they are rebuilt from the programs' options and the
settings saved with the paper's files. Where a setting of the paper's run is not known, this is said in the text.

**A full rerun is long.** The bank took about 9 hours on 22 CPU workers, and each 8,000-proposal query
needs 8,000 exact likelihood evaluations. To see that the code runs, use the smoke run (last section). To check the
published numbers, use the replay check (last section). Neither needs a full rerun.

## Before you start

1. Use the Route 2 Python environment (Python 3.11, numpy 2.4.6, scipy 1.17.1, torch 2.5.1, jax 0.10.2,
   joblib, scikit-learn). A GPU is optional (the paper's runs used one for the network steps).
2. Work from the repository root and set `PYTHONPATH=$PWD`.
3. Make sure the observation files are unpacked in `data/observations/` (`./reproduce.sh check` does it, and so
   does every Route 2 check or smoke run). The programs refuse to run if a file's SHA-256 differs from
   `workflows/data/observational-manifest.json`.
4. The programs are in `inference/anet/`, and the frozen network used for the paper is in
   `inference/anet/checkpoints/`; `inference/NUCLEONIC_CODE_MANIFEST.json` gives their SHA-256.
5. Machines of the paper's runs: a desktop workstation (Intel i7-13700F, 24 threads, NVIDIA RTX 3060 Ti) for the
   bank, the scenarios, the training and the A1, J0614 and J1231 queries; a 56-worker cluster job for the
   tidal-refinement curves. The files of the nuclear-shift and J1614 runs name a CUDA device but not the machine.
6. The large files of Steps 1, 2 and 7 (the shared physics bank, the exact astrophysical table, the training
   scenarios, and the tidal-refinement proposal and curves) are in the optional full-rerun download
   (`./reproduce.sh route2-fetch --group full_rerun`), so a rerun can start at any step. Their downloaded names
   start with `full_`; [FULL_RERUN_FILES.md](FULL_RERUN_FILES.md) gives both names.

## Step 1. Build the shared physics bank

**What it is.** Rows drawn from the uniform seven-parameter DDB prior. Each row carries its nuclear
observables, its mass–radius and mass–tidal curves on a 200-point mass grid (0.5–2.6 M☉), the pQCD and
shifted-GW grids, and the exact NICER and GW170817 likelihood terms. Extra rows near the A1 nuclear
observation and in two radius windows are added ("enrichment"); an exact counting correction (`logw_prior`)
brings every weight back to the uniform prior.

**Result.**

- `shared_prior_physics_bank.npz` (SHA-256 `06dda51a…`, 914,966 rows, 3.16 GB; downloaded as
  `full_nucleonic_shared_prior_physics_bank.npz`).
- `shared_exact_astrophysical_table.npz` (SHA-256 `e2dfb84d…`; downloaded as
  `full_nucleonic_shared_exact_astrophysical_table.npz`).

**Settings stored inside the bank.** 660,000 uniform rows (`N0`); enrichment seed 20260729;
160,000,000 screened proposals in stream B (nuclear support box of ±3 σ, `cbox = 3`), 2,640,000 in stream C and
1,320,000 in stream D; windows on the radius at 1.4 M☉ of C = 9.4–11.2 km and D = 14.8–18.0 km, both with
M_max ≥ 1.9 M☉. The note stored in the bank says that its first 660,000 rows are the uniform bank, byte for byte.

**Command.**

```bash
python inference/anet/build_bank.py --data-root data/observations \
    --output-dir runs/nucleonic_bank --phase all --workers 22
```

The defaults of `build_bank.py` are this recipe: 200,000 uniform rows with seed 1234 plus 460,000 with seed 2026,
enrichment seed 20260729, 160,000,000 stream-B proposals, `cbox 3`, and stream multipliers 5 (C) and 3 (D),
which give the 2,640,000 and 1,320,000 proposals above. The program is sharded and resumable. It stops unless
the conditional ESS at A1 is at least 2,000. Outputs: `final_bank.npz` and `final_exact.npz`.

**The paper's run.** 22 workers on the desktop workstation: 524.4 minutes; conditional ESS 4,384.

## Step 2. Build the 2,000 training scenarios

**What it is.** Each scenario is a synthetic "observation": three NICER mass–radius clouds (256 points each;
identity, one shifted/scaled source, or all three transformed), a GW tidal dial and a pQCD density dial, plus
the matching posterior rows drawn from the bank with exact weights.

**Result.** `anet_training_scenarios.npz` (SHA-256 `26958145…`, 1.08 GB; downloaded as
`full_nucleonic_anet_training_scenarios.npz`): 2,000 scenarios; `ntr = 1,900` for training, so 100 for
validation; `nreal = 200` identity scenarios; 201 to 6,000 posterior rows per scenario, 9,461,570 in total. The
frozen standardization file also stores 256 cloud points, the identity/one-slot fractions 0.25/0.5, and the
transformation ranges (mass shift −0.5 to 0.25 M☉, radius shift −1.6 to 1.6 km, scale 0.28 to 2.2).

**Command.**

```bash
python inference/anet/build_scenarios.py \
    --bank runs/nucleonic_bank/final_bank.npz --exact runs/nucleonic_bank/final_exact.npz \
    --data-root data/observations --output runs/nucleonic_anet/scenarios.npz \
    --scenarios 2000 --validation-scenarios 100 --identity-scenarios 200 \
    --maximum-posterior-rows 6000 --seed 9 --device cuda
```

The other defaults match the paper's archive: 256 cloud points, minimum bank ESS 250, identity fraction 0.25,
one-slot fraction 0.5. The random streams are `default_rng(9)` and `default_rng(786)` (that is, 777 + seed).

**Where the package default differs.** `--identity-scenarios` defaults to 15; the paper's archive has 200.

**The paper's run.** 729 s on the desktop workstation. 1,937 of the 3,937 drawn configurations were rejected
because their bank ESS was below 250 (smallest accepted ESS 252); the accepted tidal dial g_Λ therefore covers
[−0.58, 0.022] instead of the drawn [−0.58, 0.58].

## Step 3. Train the network (`fmpe9c`)

**What it is.** A DeepSets encoder (128 hidden, 64 outputs per NICER source) plus a flow-matching vector field
(512 wide, 4 layers). The context is 201 numbers: 3 × 64 cloud features, the 7 noisy nuclear observables and
the 2 dials. Sampling integrates the flow with 64 Heun steps.

**Result.** `inference/anet/checkpoints/fmpe9c_net.pt` (SHA-256 `ba18cc4b…`) and
`fmpe9c_std.npz` (SHA-256 `01adb633…`). The seven parameter means and standard deviations stored in
`fmpe9c_std.npz` equal, bit for bit, those recomputed from the scenario archive.

**Command.**

```bash
python inference/anet/train.py --scenarios runs/nucleonic_anet/scenarios.npz \
    --bank runs/nucleonic_bank/final_bank.npz --output-dir runs/nucleonic_anet/train \
    --epochs 600 --seed 21 --device cuda
```

Defaults: learning rate 1e-3, 4 scenarios per batch, hidden 512, 4 layers, 64 Heun steps. The validation loss is
computed every 10 epochs, and the checkpoint with the lowest one is kept.

**Where the package default differs.** `--seed` defaults to 2; the paper's network was trained with seed 21.

**The paper's run.** 4,216 s on the RTX 3060 Ti; best validation loss 0.4584.

## Step 4. Query A1, J0614 and J1231 (8,000 proposals each)

**What it is.** The frozen network is conditioned on the real data and draws 8,000 proposals. Each proposal gets
the exact full likelihood; the importance weights are L × prior / q.

**Settings.** 8,000 proposals and 64 Heun steps. Conditioning clouds from `default_rng(0)` (the check's log q
replay confirms this). Flow seed 0 on the GPU random generator (the check recovers each start point instead of
redrawing it). The posterior sample is 8,000 draws with replacement from `default_rng(1)` (replayed exactly by the
check). J0614 replaces J0437 (slot 2) and J1231 replaces J0030 (slot 0).

**Results.** One archive `anet_is_proposal_and_weights.npz` per query (arrays `theta`, `raw`, `w`, `logl`,
`logq`, `ess`) and the NICER single-weight correction terms `A1_J0614_J1231_singleweight_terms.npz`
(SHA-256 `bc1a84ff…`). These three runs use the target without the NICER single-weight correction; the saved
terms apply it:

| Query | Rows with finite weight | ESS as stored | ESS after the correction (paper) |
|---|---|---|---|
| A1 | 7,904 | 523.84 | 525.004 |
| J0614 | 7,894 | 445.86 | 445.84 |
| J1231 | 7,871 | 1,485.45 | 1,548.78 |

**Commands.**

```bash
python inference/anet/run_is.py --data-root data/observations --source-data-root data/observations \
    --source-scenario A1 --proposals 8000 --seed 0 --resample-seed 1 --workers 14 \
    --output runs/nucleonic_anet/A1.npz
# the same with --source-scenario J0614 and --source-scenario J1231
```

`run_is.py` first checks the shared target against its frozen certificate (1,870 rows, tolerance 1e-4; stored
maximum difference 2.05e-6), then samples and weights. The 1,870 test points of this certificate
(`workflows/data/a1_fixed_target_certification.npz`; the source certificates use the same points) were taken from
the UltraNest A1 posterior (870 rows) and a TSNPE+MIS A1 posterior sample (1,000 rows). They are only fixed points
at which every method's likelihood code must reproduce the stored log L; they do not enter any posterior, band or
evidence. (The 64 points of the hyperonic certificate are 32 rows of a hyperonic regression bank and 32 uniform
prior draws.)

**Where the package code differs.**

- It evaluates the corrected single-weight NICER target directly, so its weights correspond to the corrected
  ESS column above.
- It stores log q in physical parameter units. The paper's A1, J0614 and J1231 archives (and the nuclear-shift
  archives of Step 5) store log q in the network's standardized units, which differ by the constant
  Σ log(scale). The constant cancels in the weights and the ESS; it shifts log Z.

**The paper's runs.** 14 workers on the desktop workstation, 2026-07-30: about 100 s per query.

## Step 5. The four nuclear shifts (K0 = 200 and 260 MeV; Jsym = 29 and 36 MeV)

**What it is.** The same frozen network, conditioned on a shifted nuclear observation, with 8,000 proposals per
seed and the exact shifted target.

**Results.** Ten archives `anet_is_fmpe9c_<case>_n8000_seed<s>_singleweight.npz` and their `manifest.json`. They
use the single-weight NICER target.

| Case | Seeds pooled | Rows | Pooled ESS | Per-seed ESS |
|---|---|---|---|---|
| K0 = 200 MeV | 0 | 8,000 | 395.4 | 395.4 |
| K0 = 260 MeV | 0, 1, 2, 3 | 32,000 | 992.6 | 409.7, 112.2, 555.1, 471.7 |
| Jsym = 29 MeV | 0 | 8,000 | 1,278.5 | 1,278.5 |
| Jsym = 36 MeV | 0, 1, 2, 3 | 32,000 | 755.6 | 132.4, 428.3, 229.8, 165.3 |

**Settings** (stored in the archives and in the verification file; the check replays the resamples, the pooling
and the curve selections exactly).

- Flow seed = the run seed (0–3); conditioning clouds from `default_rng(0)`; 64 Heun steps; 8,000 proposals.
- Each run's resample: `default_rng(271828).choice(8000, 8000, p=w)`.
- Pooling: the log weights log L − log q of the seeds are joined and normalized together.
- Curves behind the published bands (`results/nucleonic/nuclear_amortization_verification.json`): 8,000 draws per
  method from the pooled weights. The A-NET draws use seed 2026082700 + 100 k, with k = 0, 1, 2, 3 for
  K0 = 200, K0 = 260, Jsym = 29, Jsym = 36 (the UltraNest draws use the same seed + 1). K0 = 260 is the
  exception: its A-NET curves use a deterministic systematic selection (positions (i + 0.5)/8000 in the
  cumulative weights), although the file lists seed 2026082800.
- Run times stored in the archives: 4–13 s of sampling and 135–361 s of likelihood evaluation per seed, on a
  CUDA device. Seed 0 of each case ran with 16 workers, and seeds 1–3 of K0 = 260 with 8.

**Command, one per case and seed.**

```bash
python inference/anet/run_is.py --data-root data/observations --scenario K0_260 \
    --proposals 8000 --seed 1 --resample-seed 271828 --workers 16 \
    --output runs/nucleonic_anet/K0_260_seed1.npz
```

There is no separate pooling program; the pooling above is two lines of numpy (the check shows them).

## Step 6. J1614 (four seeds pooled)

**What it is.** The A1 query with J1614 replacing J0740 (slot 1), run with four seeds and pooled.

**Result.** `anet_is_pooled.npz` and its `.json` report (SHA-256 `5f62ebac…`): seeds 0, 1, 2, 3 with 8,000 proposals
each (32,000 rows); per-seed ESS 1,055.4, 715.9, 1,320.7 and 579.3; pooled ESS 3,295.8; pooled
log Z = −26.383 ± 0.016 (the mean of the four seed evidences, with their delta-method errors combined; this uses the
64-step trajectory density, which sets log Z low, see Appendix C and Table XI; the 1,024-step value in Table XII is
−25.572 ± 0.030). These rows use the single-weight target and log q in physical units.

**Settings (replayed by the check).** For each seed, the flow seed and the conditioning-cloud seed both equal the
seed number: the check redraws each seed's clouds and matches the per-seed files from which the pooled file was
built.

**Command**, for s = 0, 1, 2, 3, then pool as in Step 5:

```bash
python inference/anet/run_is.py --data-root data/observations --source-data-root data/observations \
    --source-scenario J1614 --proposals 8000 --seed <s> --resample-seed 1 --workers 14 \
    --certificate workflows/data/source_target_certification_all3.npz \
    --output runs/nucleonic_anet/J1614_seed<s>.npz
```

**Where the package code differs.**

- `run_is.py` always draws the conditioning clouds with seed 0, so it does not repeat the per-seed clouds of the
  paper's J1614 run.
- The default target certificate, `workflows/data/source_target_certification.npz`, covers A1, J0614 and J1231
  only, so a J1614 run needs `--certificate workflows/data/source_target_certification_all3.npz`. That file is the
  certificate used for the paper, with a J1614 column (`51f84cf2…`); every numerical array is bit-identical, and
  only the name of one key of its stored code-hash record differs.

**The paper's run.** 184, 157, 247 and 163 s per seed, on a CUDA device.

## Step 7. A1 mass–tidal refinement ("A-NET-seeded direct-theta importance sampling")

**What it is.** A separate, larger importance-sampling run for the A1 mass–tidal band only. A-NET+IS rows are
used only to fit two Gaussian mixtures in parameter space (one for the whole posterior, one for the low-Λ tail);
fresh draws from those mixtures plus a uniform-prior component are then weighted with the exact target. The
adaptation rows are not part of the final estimate, and no flow density enters it.

**Result** (the tidal-refinement files):

- adaptation: 765,821 A-NET+IS rows (ESS 33,959); full mixture 6 components fitted on 100,000 rows; tail mixture
  4 components on 60,000 rows (tail mass 0.10; tail threshold `tail_threshold_Lambda1` = 2,026.7, a cut on Λ at
  1 M☉ as in `refine_tidal.py`);
- proposal: 320,000 fresh rows = 208,000 (weight 0.65, seeds 1000–1025) + 96,000 (0.30, seeds 1100–1111) +
  16,000 uniform-prior rows (0.05, seeds 1200–1201), 8,000 rows per seed; 301,507 inside the prior;
- exact curves: 298,846 valid rows, 1,064.7 s on 56 workers;
- files: the proposal `direct_theta_proposal_320k.npz` (SHA-256 `16096dda3499…`) and its exact curves
  `direct_theta_curves_320k.npz` (`1eec4b4b2c19…`), both in the optional full-rerun download;
- checks: total ESS 104,180 (at least 15,000), low-tail ESS about 20,000 (at least 2,000), split-half band change
  0.8% (at most 3%), and parameter and maximum-mass quantiles within 3% of the prior span and 0.04 M☉ of the
  8,000-row A1 run. The compact band is `band_anet_is_direct_gmm_320k`.

**Command.**

```bash
python inference/anet/refine_tidal.py --anet-is runs/nucleonic_anet/A1.npz \
    --data-root data/observations --output-dir runs/nucleonic_anet/tidal --workers 56
```

Defaults: 320,000 rows in parts of 8,000, seed 20260821, minimum ESS 15,000, minimum tail ESS 2,000.

**Where the package code differs.** It adapts only from the ordinary 8,000-row A1 archive given with
`--anet-is`. The paper's refinement adapted from the 765,821 A-NET+IS rows above, which are not part of this
package.

## Checks and smoke run

Replay check (no rerun; about 1–2 minutes on CPU):

```bash
PYTHONPATH=$PWD python -m route2.checks.nucleonic_anet
```

It recomputes every weight, correction, pooled ESS, resample and curve selection above from the saved archives,
and recomputes log L (exact target rebuilt from `data/observations`) and log q (frozen network) for 14 saved rows
per configuration.

Smoke run (tiny bank → scenarios → 2-epoch training → query and exact IS; about 2–3 minutes on 3 CPU cores):

```bash
PYTHONPATH=$PWD python -m route2.smoke.anet_nucleonic_smoke
```

## Gaps

- The command lines and logs of the paper's runs of Steps 1–6 are not available; the commands above are rebuilt
  from the programs' options and the settings saved with the paper's files.
- It has not been shown that the package programs reproduce the paper's bank, scenarios, network or archives
  byte for byte. The shared target is checked against its certificate (Step 4); the check replays the weights,
  log L and log q.
- `fmpe9c_std.npz` stores `nreal = 15`, while the scenario archive holds 200 identity scenarios. The network
  code does not use this number.
- The J1614 seed-0 command and the J1614 pooling script are not available (see Step 6).
- The tidal-refinement adaptation rows are not part of this package (see Step 7).
