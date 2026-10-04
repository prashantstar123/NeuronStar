# Route 2 steps: UltraNest (nucleonic and hyperonic)

This page explains how the UltraNest results of the paper were made, how to make them again from scratch with
the code in this repository, and how to check the saved results quickly.

- **UltraNest** is a nested sampler. It is the conventional reference of the paper: it gives the UltraNest column
  of Table VI (the Bayesian evidence log Z of each configuration) and the UltraNest posteriors that the figures
  compare with A-NET and TSNPE.
- UltraNest never builds or trains any other method. Its results are used only for comparison.

The commands use the package code with the settings listed. Where a setting of the published runs is not known,
this is said in the text.

Words used below:

| Word | Meaning |
|---|---|
| **run** | one complete UltraNest sampling of one configuration with one random seed |
| **configuration** | one of the 8 targets per sector: A1, K0 = 200 or 260 MeV, Jsym = 29 or 36 MeV, and PSR J0614, J1231 or J1614 in place of one baseline NICER source |

Files used on this page:

| File | What it holds | SHA-256 |
|---|---|---|
| `route2/fixtures/ultranest/records/evidence_aggregation.json` | the per-run values of the 23 runs other than the four hyperonic desktop runs (the same values as in the file below), their combinations and the combination rule of Step 4 | `87a27353…` |
| `route2/fixtures/ultranest/records/run_values.json` | the per-run values of all 27 runs and their combinations | `09035b3a…` |
| `route2/fixtures/ultranest/records/run_reports/` | the correction reports of the four hyperonic desktop runs (Step 3) | listed in `route2/fixtures/ultranest/MANIFEST.json` |
| `results/evidence/table_VI_independent_ultranest_rows.json` | the four hyperonic K0/Jsym rows of Table VI, with independent seeds only | `e9449e12…` |

## 1. The drivers

| Sector | Driver in this repository | What it samples |
|---|---|---|
| Nucleonic | `inference/ultranest/run.py` (the published driver, SHA-256 `57cc0342…`) | the paper's target, with the NICER single-weight convention (Section 4). No correction is needed afterwards. |
| Hyperonic, A1, K0, Jsym, J0614, J1231 | `hyperonic_pipeline/ddb_ultranest_hyp.py` | the double-weight NICER convention (certified modules in `legacy_stack/`); the single-weight correction is applied afterwards. `DDB_SWAP=J0614` or `J1231` replaces a NICER source. |
| Hyperonic, J1614 | `hyperonic_pipeline/ddb_ultranest_hyp_j1614.py` | the same, with J1614 in place of J0740. Here the two NICER conventions coincide, so no correction is needed. |

- The published nucleonic runs (except J1614) were made with other nucleonic drivers, which are not included in
  the package. They sampled the double-weight NICER convention and were corrected afterwards. Of these drivers,
  only the SHA-256 of the cluster driver is known (`fdb1889f…`). Every nucleonic likelihood used 20 workers.
- The nucleonic J1614 run sampled the paper's target directly.

## 2. Settings of every run

- **Active wall** is the measured sampling time of one completed run. The segments of a resumed run are added up;
  queue waiting and stopped gaps are not counted.
- **Evaluations** is the `ncall` value of UltraNest's `info/results.json`.

### Nucleonic (20 likelihood workers in every run)

| Configuration | Run | Seed | Live points | Where it ran | Active wall (min) | Evaluations |
|---|---|---|---:|---|---:|---:|
| A1 | `nuc_a1` | none (the driver does not seed numpy) | 1000 | desktop, interactive; two segments (22.1 + 88.1 min) | 110.2 | 411,372 |
| K0 = 200 | `nuc_K0_200` | none | 1000 | desktop; two segments | 170.5 | 356,354 |
| K0 = 260 | `nuc_k0260_s0` | none | 1000 | cluster; two segments on two nodes | 621.8 | 568,358 |
| K0 = 260 | `nuc_k0260_s1` | `np.random.seed(1001)` | 1000 | cluster | 600.0 | 476,477 |
| Jsym = 29 | `nuc_j29_s0` | none | 1000 | cluster; two segments on two nodes | about 376.6 (two node clocks) | 353,605 |
| Jsym = 29 | `nuc_j29_s1` | `np.random.seed(1001)` | 1000 | cluster | 414.1 | 375,178 |
| Jsym = 36 | `nuc_j36_s0` | none | 1000 | cluster; two segments on two nodes | 259.2 | 480,486 |
| Jsym = 36 | `nuc_j36_s1` | `np.random.seed(1001)` | 1000 | cluster | 559.6 | 516,161 |
| Jsym = 36 | `nuc_j36_s2` | `np.random.seed(1002)` | 1000 | cluster | 565.6 | 559,636 |
| PSR J0614 | `nuc_j0614` | none | 1000 | desktop; three segments (29.4 + 28.6 + 370.0 min) | 428.1 | 872,802 |
| PSR J1231 | `nuc_j1231` | none | 1000 | desktop | 179.8 | 407,458 |
| PSR J1614 | `nuc_J1614_20260917` | 20260917 | 2000 | desktop | 382.6 | 835,030 |

- The cluster jobs reserved 64 cores but used 20 likelihood workers: the imported TOV module fixes 20 workers.

### Second nucleonic A1 run (mass–tidal band only)

A second nucleonic A1 run is used for the A1 mass–tidal band (Fig. 3b and the tidal entries of Table III). It is
not used for the evidences, the parameter posterior or the mass–radius band.

| Run | Seed | Live points | Where it ran | Wall (min) | Evaluations | ESS | log Z |
|---|---|---:|---|---:|---:|---:|---|
| second A1 run | 20260822 | 2000 (`ndraw_min` 2048) | one desktop, 20 likelihood workers | 385.8 | 732,699 | 9,134 | −25.586 ± 0.150 |

- Target: the A1 likelihood with the single-weight PSR J0740+6620 term, checked against the 1,870-row target
  certificate before sampling (largest difference 2.1e-6).
- `results/nucleonic/a1_mass_tidal_bands.npz` holds the band of each run (`band_ultranest_old` for `nuc_a1`,
  `band_ultranest_new` for this run) and the pooled band (`band_ultranest_pooled`), which gives each run posterior
  mass 0.5. The paper's A1 tidal comparisons use the pooled band.
- The posterior of this run is not included in the package; the package ships its band.

### Hyperonic

| Configuration | Run | Seed | Live points | Workers | Where it ran | Active wall (min) | Evaluations | In Table VI |
|---|---|---|---:|---:|---|---:|---:|---|
| A1 | `hyp_a1_s1` | 771 | 1000 | 56 | cluster | 227.6 | 373,282 | yes |
| A1 | `hyp_a1_s2` | 772 | 1000 | 56 | cluster | 253.2 | 387,555 | yes |
| K0 = 200 | `hyp_k0200` | 200 | 1000 | 56 | cluster | 105.1 | 338,267 | yes |
| K0 = 200 | `hyp_k0200_desktopB` | not known | 1000 | 20 | desktop | 166.8 | 306,455 | no |
| K0 = 260 | `hyp_k0260` | 260 | 1000 | 56 | cluster | 245.6 | 398,546 | yes |
| K0 = 260 | `hyp_k0260_desktopB` | not known | 1000 | 20 | desktop | 213.0 | 444,551 | no |
| Jsym = 29 | `hyp_j29_s1` | 29 | 1000 | 56 | cluster | 210.6 | 339,395 | yes |
| Jsym = 29 | `hyp_j29_s2` | 129 | 1000 | 56 | cluster | 228.1 | 342,485 | yes |
| Jsym = 29 | `hyp_j29_desktopB` | not known | 1000 | 20 | desktop | 151.6 | 317,930 | no |
| Jsym = 36 | `hyp_j36` | 36 | 1000 | 56 | cluster | 255.9 | 423,889 | yes |
| Jsym = 36 | `hyp_j36_desktopB` | not known | 1000 | 20 | desktop | 224.5 | 461,929 | no |
| PSR J0614 | `hyp_j0614` | 614 | 1000 | 56 | cluster | 297.4 | 472,335 | yes |
| PSR J1231 | `hyp_j1231` | 1231 | 1000 | 56 | cluster | 377.3 | 635,820 | yes |
| PSR J1614 | `hyp_J1614_replica_1` | 20260921 | 2000 | 48 | cluster | 366.8 | 843,512 | yes |
| PSR J1614 | `hyp_J1614_replica_2` | 20260922 | 2000 | 44 | cluster | 457.2 | 872,943 | yes |

- The cluster jobs made in August (all hyperonic cluster runs except J1614) had 64-core allocations.
- The output summaries of the K0 and Jsym cluster runs give 56 workers, a minimum draw batch (`ndraw_min`) of 2048
  and HDF5 storage.
- The hyperonic runs made in August (all except the two J1614 replicas) used a slower solver path of the
  hyperonic EOS. Four timing runs with the faster production path and the same seeds (A1 771 and 772, J0614 614,
  J1231 1231) reproduced their trajectories to rounding. They took 140.8, 156.6, 185.8 and 243.3 min, but they
  are timing checks only, not extra evidence runs.

## 3. Before you start

1. Install the Route 2 environment. The from-scratch runs need `ultranest` 4.5.0. The hyperonic drivers also need
   `CompactObject-TOV` 2.1, which provides the pQCD module.
2. Work in the repository root, unpack the observation files once, and fetch the UltraNest files
   (29 files, 20.6 MB, each checked against its SHA-256):

   ```bash
   export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=""
   ./reproduce.sh check
   python -m route2.fetch --group ultranest
   ```

| Files in `route2/downloads/` | What they hold |
|---|---|
| `ultranest_nucleonic_<run>_posterior.npz` and `_terms.npz` (A1, J0614, J1231, Jsym_29_s0/s1, Jsym_36_s0/s1/s2) | corrected-posterior inputs and single-weight terms of these nucleonic runs |
| `ultranest_nucleonic_J1614_posterior.npz` | the corrected posterior of the nucleonic J1614 run |
| `ultranest_hyperonic_<config>_desktopB_input.npz` and `_terms.npz` | correction inputs and correction terms of the four hyperonic desktop runs |
| `ultranest_hyperonic_<config>_equal_weight.npz` (K0_200, K0_260, Jsym_29, Jsym_36) | equal-weight UltraNest samples of the hyperonic nuclear-shift configurations |

## 4. From-scratch rerun, step by step

The launch commands of the published runs are not available. The commands below use the drivers of Section 1
with the settings of Section 2. Nested sampling is random, so a rerun agrees with the published run within its
quoted error, not bit for bit. A seeded hyperonic cluster run repeated with the same seed and worker count is
expected to follow the same trajectory; the four timing runs above did.

**Hardware caution.** Each run takes 1.8 to 10.4 hours with 20 to 56 likelihood workers. The runs used for the
evidences add up to 4,668 min (78 h) with 20 workers for the nucleonic sector and 3,025 min (50 h) with 44 to 56
workers for the hyperonic Table VI runs. The full rerun is optional; the quick check of Section 6 needs 3 minutes.

### Step 1: nucleonic runs

```bash
mkdir -p runs/ultranest
python inference/ultranest/run.py --data-root data/observations --source-data-root data/observations \
    --output-dir runs/ultranest/nucleonic_A1 --workers 20 --min-live 1000 --seed 1001
```

- Nuclear shifts: add `--scenario K0_200` (or `K0_260`, `Jsym_29`, `Jsym_36`).
- Replaced NICER source: add `--source-scenario J0614` (or `J1231`).
- J1614: `--source-scenario J1614 --min-live 2000 --seed 20260917`.
- Seeds: the cluster replicas used `np.random.seed(1001)` and `1002`, which is what `--seed` does. The other
  published nucleonic runs were not seeded. `--workers 20` matches the published runs.
- Before sampling, the driver certifies its target on saved certificate rows (1,870 rows for A1; log L tolerance
  1e-4) and stops if this fails.
- It writes `ultranest_<configuration>_posterior.npz` (equal-weight samples, log Z, error, `ncall`) and
  `run_report.json`. This driver samples the paper's target, so its log Z needs no correction: skip Step 3.

### Step 2: hyperonic runs

```bash
export CERTIFIED_DDB_DIR=$PWD/legacy_stack/validated_code_DDB CERTIFIED_ASTRO_DIR=$PWD/legacy_stack/ddb_astro_mod
export HYPERON_PROJECT_DIR=$PWD/hyperonic_pipeline DDB_OBS_DATA_DIR=$PWD/data/observations
export DDB_J0437_FILE=$PWD/data/observations/J0437_post_equal_weights.dat
export DDB_GW_FILE=$PWD/data/observations/GW170817_GWTC-1.hdf5
mkdir -p runs/ultranest
python -c "import numpy as np; from workflows.nuclear_scenarios import nuclear_observation as o; \
from likelihoods.nuclear import A1_SIGMA; \
[np.savez(f'runs/ultranest/obs_{c}.npz', OBS=o(c), SIG=A1_SIGMA) for c in ('A1', 'K0_200', 'K0_260', 'Jsym_29', 'Jsym_36')]"
DDB_NUCLEAR_BANK=runs/ultranest/obs_A1.npz python hyperonic_pipeline/ddb_ultranest_hyp.py \
    --nw 56 --minlive 1000 --ndraw 2048 --seed 771 \
    --log-dir runs/ultranest/un_hyp_A1_s771 --output runs/ultranest/ultranest_hyp_A1_s771.npz
```

- The driver reads the nuclear observation and its widths from `DDB_NUCLEAR_BANK` (arrays `OBS` and `SIG`). The
  output summaries of the published K0 and Jsym runs hold exactly `nuclear_observation(<configuration>)` and the
  A1 widths.
- A1: seeds 771 and 772.
- Nuclear shifts: `DDB_NUCLEAR_BANK=runs/ultranest/obs_K0_200.npz` with seed 200. Likewise K0_260 with seed 260,
  Jsym_29 with seeds 29 and 129, and Jsym_36 with seed 36.
- J0614: `DDB_SWAP=J0614 DDB_J0614_FILE=$PWD/data/observations/J0614_mrsamples.dat`, seed 614.
- J1231: `DDB_SWAP=J1231 DDB_J1231_FILE=$PWD/data/observations/J1231_wmrsamples.txt`, seed 1231.
- J1614: run `hyperonic_pipeline/ddb_ultranest_hyp_j1614.py` with
  `DDB_J1614_FILE=$PWD/data/observations/J1614_STU_mrsamples_post_equal_weights.dat` and
  `--minlive 2000 --seed 20260921 --nw 48` (second replica: `--seed 20260922 --nw 44`).
- Set `HYPERON_PROJECT_DIR` as shown. The drivers put this folder and your home folder at the front of the Python import
  path, so keep no files named like the package's modules (for example `eos.py`) in your home folder.
- The weighted posterior rows are in `<log-dir>/chains/weighted_post.txt`. Step 3 needs them.

### Step 3: NICER single-weight correction

**Why it is needed.** In the double-weight NICER convention, the weighted J0740 samples (and the weighted J1231
samples) are weighted twice: once when the kernel-density rows are drawn in proportion to the weights, and again
inside the kernel density. The paper's single-weight convention uses the weights once. The runs that sampled the
double-weight convention are corrected afterwards.

For every posterior row i with UltraNest weight w_i > 0 (normalized to sum 1):

1. Compute its mass-radius curve and the per-row term Δ_i = log L_single − log L_double of J0740. In the J1231
   configuration, add the same difference for J1231. No term is needed for J1614, whose target is unchanged.
   - single weight (arrays `fixed_0740` and `fixed_1231` of the terms files):
     `likelihoods.nicer.single_weight.log_likelihood_one` with the target's own grid.
   - double weight (arrays `old_0740` and `old_1231`): `legacy_stack/ddb_astro_mod/nicer_like.py`, with
     `build_nicer_grid(..., nM=150, nR=150, max_rows=20000, bw=0.08)` and `logL_nicer_one`, as in the drivers.
2. Log Z shift = log Σ_i w_i exp(Δ_i).
3. Its Monte Carlo error = sqrt(CV² / ESS), with r_i = exp(Δ_i), CV² = Σ_i w_i (r_i / r̄ − 1)², r̄ = Σ_i w_i r_i
   and ESS = 1 / Σ_i w_i².
4. Corrected log Z = raw log Z + shift. Corrected error = sqrt(raw error² + Monte Carlo error²).
5. Corrected posterior weights ∝ w_i exp(Δ_i). For A1 this is written as log w = log(UltraNest weight)
   + fixed_0740 − old_0740, normalized.

`Targets.terms` and `correction_statistics` in `route2/checks/ultranest.py` implement exactly these steps.

Values of all 27 runs (from `route2/fixtures/ultranest/records/run_values.json`):

| Sector | Configuration | Run | Raw log Z | Shift | MC error of shift | Corrected log Z |
|---|---|---|---:|---:|---:|---:|
| nucleonic | A1 | `nuc_a1` | -25.658311 ± 0.242096 | +0.191676 | 0.000325 | -25.466635 ± 0.242096 |
| nucleonic | K0 = 200 | `nuc_K0_200` | -25.826779 ± 0.214648 | +0.196049 | 0.000302 | -25.630730 ± 0.214648 |
| nucleonic | K0 = 260 | `nuc_k0260_s0` | -26.122367 ± 0.102508 | +0.185420 | 0.000300 | -25.936947 ± 0.102508 |
| nucleonic | K0 = 260 | `nuc_k0260_s1` | -25.696159 ± 0.226689 | +0.185528 | 0.000317 | -25.510631 ± 0.226689 |
| nucleonic | Jsym = 29 | `nuc_j29_s0` | -25.850367 ± 0.112019 | +0.191234 | 0.000325 | -25.659133 ± 0.112019 |
| nucleonic | Jsym = 29 | `nuc_j29_s1` | -25.610224 ± 0.187926 | +0.191577 | 0.000312 | -25.418648 ± 0.187926 |
| nucleonic | Jsym = 36 | `nuc_j36_s0` | -26.066618 ± 0.124696 | +0.190316 | 0.000306 | -25.876303 ± 0.124697 |
| nucleonic | Jsym = 36 | `nuc_j36_s1` | -25.738844 ± 0.198555 | +0.190312 | 0.000297 | -25.548532 ± 0.198555 |
| nucleonic | Jsym = 36 | `nuc_j36_s2` | -25.848355 ± 0.259959 | +0.189892 | 0.000304 | -25.658464 ± 0.259959 |
| nucleonic | PSR J0614 | `nuc_j0614` | -27.169646 ± 0.160260 | +0.197302 | 0.000316 | -26.972345 ± 0.160260 |
| nucleonic | PSR J1231 | `nuc_j1231` | -26.676009 ± 0.213250 | +0.227899 | 0.001791 | -26.448110 ± 0.213257 |
| nucleonic | PSR J1614 | `nuc_J1614_20260917` | -25.565178 ± 0.285867 | 0 | 0 | -25.565178 ± 0.285867 |
| hyperonic | A1 | `hyp_a1_s1` | -28.859995 ± 0.160471 | +0.157825 | 0.000623 | -28.702170 ± 0.160472 |
| hyperonic | A1 | `hyp_a1_s2` | -29.010392 ± 0.204344 | +0.159261 | 0.000585 | -28.851130 ± 0.204345 |
| hyperonic | K0 = 200 | `hyp_k0200` | -29.557977 ± 0.357525 | +0.163479 | 0.000661 | -29.394498 ± 0.357525 |
| hyperonic | K0 = 200 | `hyp_k0200_desktopB` | -29.454975 ± 0.368700 | +0.164584 | 0.000651 | -29.290391 ± 0.368701 |
| hyperonic | K0 = 260 | `hyp_k0260` | -28.831234 ± 0.225369 | +0.150603 | 0.000598 | -28.680631 ± 0.225370 |
| hyperonic | K0 = 260 | `hyp_k0260_desktopB` | -28.871318 ± 0.167105 | +0.150117 | 0.000586 | -28.721201 ± 0.167106 |
| hyperonic | Jsym = 29 | `hyp_j29_s2` | -28.823731 ± 0.179540 | +0.155474 | 0.000624 | -28.668257 ± 0.179541 |
| hyperonic | Jsym = 29 | `hyp_j29_s1` | -28.616024 ± 0.161274 | +0.156616 | 0.000588 | -28.459407 ± 0.161275 |
| hyperonic | Jsym = 29 | `hyp_j29_desktopB` | -28.645931 ± 0.176242 | +0.155297 | 0.000620 | -28.490634 ± 0.176243 |
| hyperonic | Jsym = 36 | `hyp_j36` | -29.176823 ± 0.225979 | +0.161025 | 0.000600 | -29.015798 ± 0.225980 |
| hyperonic | Jsym = 36 | `hyp_j36_desktopB` | -28.992425 ± 0.223724 | +0.160316 | 0.000607 | -28.832109 ± 0.223725 |
| hyperonic | PSR J0614 | `hyp_j0614` | -31.708879 ± 0.258767 | +0.175571 | 0.000715 | -31.533308 ± 0.258768 |
| hyperonic | PSR J1231 | `hyp_j1231` | -29.413904 ± 0.195037 | +0.174269 | 0.001258 | -29.239635 ± 0.195041 |
| hyperonic | PSR J1614 | `hyp_J1614_replica_1` | -28.803829 ± 0.160382 | 0 | 0 | -28.803829 ± 0.160382 |
| hyperonic | PSR J1614 | `hyp_J1614_replica_2` | -28.775715 ± 0.149645 | 0 | 0 | -28.775715 ± 0.149645 |

Which figures use corrected posterior weights:

- Nucleonic A1, J0614 and J1231, and the nucleonic nuclear shifts: corrected weights.
- Nucleonic and hyperonic J1614: equal weights; no correction is needed.
- Hyperonic nuclear shifts (Appendix figure): the equal-weight UltraNest samples, corrected once for the NICER
  single-weight convention (`results/appendix/hyperonic_nuclear_amortization_report.json`).

### Step 4: combine the seeds of one configuration

For n completed independent runs with corrected values (z_k ± σ_k):

- centre = Σ_k z_k / σ_k² ÷ Σ_k 1 / σ_k²
- formal error = 1 / sqrt(Σ_k 1 / σ_k²)
- Q = Σ_k (z_k − centre)² / σ_k²
- final error = formal error × max(1, sqrt(Q / (n − 1))); a single run is unchanged.

The rule has three more points:

- Use `logz` and `logzerr` from each run's `info/results.json`.
- Never choose a seed because it agrees better with another method.
- Exclude only incomplete runs and timing-only repeats of a trajectory that is already counted.

This rule, applied to all 27 runs, gives the UltraNest columns of
`results/evidence/green_en_vs_ultranest_16_rows.json`.

**Why only independent seeds enter Table VI.** The rule assumes independent runs. For each of the hyperonic
K0 = 200, K0 = 260, Jsym = 29 and Jsym = 36 configurations, the desktop run (`*_desktopB`) and the cluster run
share bit-identical nine-dimensional posterior rows:

- the desktop posterior rows (the correction inputs of Section 3) and the cluster posterior share 71, 80, 107 and
  78 distinct rows, which occur 132, 148, 180 and 140 times in the cluster posterior;
- the equal-weight posteriors of the two runs share 52, 61, 82 and 60 distinct rows.

Independent Monte Carlo runs cannot share such rows, so these pairs share part of their initial random state.
The two independent control pairs share none: the desktop Jsym = 29 run against cluster seed 129, and A1 seed
771 against seed 772.

Table VI therefore uses the desktop runs as validation checks only. It combines only the strictly independent
cluster runs:

- one run each for K0 = 200, K0 = 260 and Jsym = 36;
- seeds 29 and 129 for Jsym = 29.

The rule is applied in the same way to all four configurations; no run is chosen per row. No EN value, posterior
or figure depends on this choice.

The UltraNest column of Table VI (separation = |EN − UltraNest| / sqrt(σ_EN² + σ_UN²), with the EN values of the
16-row file):

| Sector | Configuration | Runs combined in Table VI | log Z | n | Separation | All 27 runs combined, where different |
|---|---|---|---:|---:|---:|---|
| nucleonic | A1 | `nuc_a1` | -25.466635 ± 0.242096 | 1 | 0.59 | |
| nucleonic | K0 = 200 | `nuc_K0_200` | -25.630730 ± 0.214648 | 1 | 0.22 | |
| nucleonic | K0 = 260 | `nuc_k0260_s0`, `nuc_k0260_s1` | -25.864573 ± 0.160051 | 2 | 0.08 | |
| nucleonic | Jsym = 29 | `nuc_j29_s0`, `nuc_j29_s1` | -25.596086 ± 0.105768 | 2 | 0.33 | |
| nucleonic | Jsym = 36 | `nuc_j36_s0`, `nuc_j36_s1`, `nuc_j36_s2` | -25.765870 ± 0.101512 | 3 | 0.29 | |
| nucleonic | PSR J0614 | `nuc_j0614` | -26.972345 ± 0.160260 | 1 | 0.85 | |
| nucleonic | PSR J1231 | `nuc_j1231` | -26.448110 ± 0.213257 | 1 | 0.51 | |
| nucleonic | PSR J1614 | `nuc_J1614_20260917` | -25.565178 ± 0.285867 | 1 | 0.06 | |
| hyperonic | A1 | `hyp_a1_s1`, `hyp_a1_s2` | -28.758992 ± 0.126208 | 2 | 0.22 | |
| hyperonic | K0 = 200 | `hyp_k0200` | -29.394498 ± 0.357525 | 1 | 0.74 | -29.344046 ± 0.256668 (n = 2) |
| hyperonic | K0 = 260 | `hyp_k0260` | -28.680631 ± 0.225370 | 1 | 0.30 | -28.706809 ± 0.134232 (n = 2) |
| hyperonic | Jsym = 29 | `hyp_j29_s1`, `hyp_j29_s2` | -28.552671 ± 0.119978 | 2 | 0.99 | -28.533025 ± 0.099178 (n = 3) |
| hyperonic | Jsym = 36 | `hyp_j36` | -29.015798 ± 0.225980 | 1 | 0.37 | -28.923032 ± 0.158989 (n = 2) |
| hyperonic | PSR J0614 | `hyp_j0614` | -31.533308 ± 0.258768 | 1 | 0.29 | |
| hyperonic | PSR J1231 | `hyp_j1231` | -29.239635 ± 0.195041 | 1 | 0.53 | |
| hyperonic | PSR J1614 | `hyp_J1614_replica_1`, `hyp_J1614_replica_2` | -28.788799 ± 0.109414 | 2 | 0.28 | |

The column n gives the number of seeds combined in each row; Table VI of the paper does not mark it.

## 5. Where each number comes from

- `results/evidence/green_en_vs_ultranest_16_rows.json`: the UltraNest values of all 27 runs.
- `results/evidence/table_VI_independent_ultranest_rows.json`: the four hyperonic K0/Jsym rows with independent
  seeds only.
- `route2/fixtures/ultranest/records/`: `evidence_aggregation.json` and `run_values.json` (the per-run values and
  combinations of Steps 3 and 4) and the correction reports of the four hyperonic desktop runs (`run_reports/`).
- `results/nucleonic/a1_parameter_posteriors.npz` (`theta_ultranest`, `weight_ultranest`): the corrected A1
  UltraNest posterior.
- `results/nucleonic/nuclear_amortization_verification.json`: the corrected ESS values and the curve selections
  of the nucleonic nuclear-shift UltraNest bands.
- `results/nucleonic/j1614_ultranest_mass_radius_curves.npz` and `results/hyperonic/J1614/ultranest_mass_radius.npz`:
  the J1614 UltraNest samples and curves.

## 6. What the quick check replays

`route2/checks/ultranest.py` replays small, well-chosen pieces. It never samples again.

**A. Table VI arithmetic**

- All 27 corrected run values and all 16 combinations are rebuilt from the per-run values in
  `route2/fixtures/ultranest/records/`.
- They are compared with both evidence files and with the printed Table VI (values and separations; the seed
  counts are checked against the files in `route2/fixtures/ultranest/records/`, since the paper does not mark them).
- The raw log Z of 11 runs must also equal the value carried by another stored product: both J1614 posteriors,
  the hyperonic equal-weight posteriors and the four desktop-run reports in
  `route2/fixtures/ultranest/records/run_reports/`.

**B. Weights and corrections**

- The A1 corrected weights must equal `results/nucleonic/a1_parameter_posteriors.npz` bit for bit.
- The correction shift and its error are rebuilt from the downloaded rows for 12 runs: 8 nucleonic and the 4
  desktop runs.
- The ESS values and the 8,000-draw curve selections of the nucleonic Jsym bands are redrawn.
- The J1614 files must hold the UltraNest J1614 samples with equal weights.
- The shared-row counts behind the Table VI rows with independent seeds are recounted.

**C. Exact likelihood**

- 72 posterior rows are evaluated with the portable targets.
- Their stored correction terms, maximum masses and mass-radius curves are recomputed.

**D. Driver smoke**

- `inference/ultranest/run.py --smoke` certifies the nucleonic target.
- The two hyperonic drivers, in their `--cert` mode, must return the portable log L minus the NICER correction
  term (stored for the desktop rows, recomputed for the cluster rows), or the portable log L itself for J1614.

```bash
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 NUMBA_NUM_THREADS=3
python -m route2.fetch --group ultranest
python -m route2.checks.ultranest                      # about 3 minutes on 3 cores
python -m route2.checks.ultranest --skip-driver-smoke  # under a minute
```

The check writes `build/route2/ultranest_check.json` and prints PASS or FAIL.

Tolerances:

| Quantity | Tolerance | Why |
|---|---|---|
| log Z | 5e-13 | double-precision rounding only |
| Correction shift | 2e-12 | summation over about 15,000 rows |
| Monte Carlo error of the shift | 1e-13 | the stored value was derived from squared errors |
| log L of the correction terms | 1e-4 | the pipeline's certification tolerance |
| Masses and radii | 1e-6 Msun or km | changes log L by less than 2e-5 |

The printed Table VI strings must be identical.

In the reference run, every recomputed value agreed with its stored value:

- log Z within 1.8e-14 (most bit for bit);
- shifts within 2e-15;
- correction terms within 8e-12;
- masses and radii within 2e-12;
- the two hyperonic drivers within 7e-15; the nucleonic driver's own target certification at 2.1e-6 (its limit is 1e-4).

## 7. Gaps

1. **UltraNest output folders.** The `info/results.json` files, chains and logs of the runs are not included in
   the package. The raw log Z ± error of each run is taken from the per-run values in
   `route2/fixtures/ultranest/records/`.
2. **Correction inputs, 12 runs.** For 12 of the 27 runs the correction is taken from the per-run values and is
   not replayed, because the rows it needs are not included in the package. These are the nucleonic K0 = 200 run
   and the two K0 = 260 runs, and all nine hyperonic cluster runs. For the other 15 runs the correction is
   replayed (12) or is zero (3 J1614 runs).
3. **Scripts not included.** The scripts that computed the correction terms and combined the runs are not
   included in the package. Their formulas are written in the files of `route2/fixtures/ultranest/records/` and in
   Steps 3 and 4, and the check replays them.
4. **No per-row log L.** UltraNest's log L per posterior row is not stored. The likelihood replay therefore
   compares the correction terms, maximum masses and curves, and requires a finite exact log L.
5. **Drivers of the published runs.** The nucleonic drivers of the published runs (all except J1614) are not
   included in the package. The version of the hyperonic driver used for the runs made in August is not known.
   The launch commands are not available.
6. **Seeds.** The seeds of the hyperonic desktop runs are not known, and most nucleonic runs were not seeded.
   Those runs cannot be repeated bit for bit.
7. **UltraNest version.** The UltraNest version used on the desktop that ran `nuc_a1` and `nuc_J1614_20260917`
   is not known.
8. **Figure inputs not included.**
   - The nucleonic nuclear-shift, J0614 and J1231 UltraNest curve caches behind the figure bands are not
     included in the package. Only the curve selections of the Jsym = 29 and 36 bands are replayed.
   - The hyperonic A1, J0614 and J1231 UltraNest posterior rows are not included; only their bands are.
   - The cluster seed-129 Jsym = 29 posterior is not included, so its zero-overlap control cannot be recounted.
9. **Smoke run.** The smoke run does no nested sampling. The portable driver has no iteration cap, and a converged
   50-live-point run would need 10,000 to 100,000 likelihood calls.
