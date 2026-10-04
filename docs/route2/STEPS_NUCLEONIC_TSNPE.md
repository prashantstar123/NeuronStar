# Route 2 steps: nucleonic TSNPE+MIS (A1)

This page explains how the nucleonic TSNPE posterior and evidence for the A1 data were made, how to make them
again from scratch with the code in this repository, and how to check the saved results quickly.

- **TSNPE** (truncated sequential neural posterior estimation) trains a neural density (a "flow") for the
  posterior in several rounds.
- **MIS** (mixture importance sampling) then corrects the flow exactly: it draws parameter rows from a
  three-part mixture and weights every row by the exact likelihood. The weighted rows are the posterior; the
  mean weight gives the evidence Z.

The commands use the package code with the settings listed; where a setting of the published run is not known,
this is said in the text.

## 1. The chain in short

```text
uniform-prior nuclear-plus-maximum-mass support bank
    -> independent round-zero support flow
    -> TSNPE-owned restricted-prior rounds
    -> frozen TSNPE estimator
    -> fresh 60k flow + 15k broad + 30k mild proposals
    -> exact three-component balance-mixture correction (A1 target)
```

TSNPE uses no A-NET, Evidence-Network or conventional-sampler result. It shares only the prior, the EOS/TOV
code, the observations and the likelihood with the other methods.

The final stage (mixture importance sampling) is `inference/tsnpe/run_mis.py` with its default settings
(seed 0) and the frozen estimator of the TSNPE rounds, against the exact A1 target. It reads no other method's
result.

| Stage | Product | What is known |
|---|---|---|
| Round-zero support flow | `inference/tsnpe/checkpoints/flow_nucmm_tight.pt` (`5edc669c…`) and `flow_nucmm_tight_std.npz` (`0a9d955d…`) | The SHA-256 values, and that it was trained on prior-generated nuclear rows with a maximum-mass factor. Its only job is to define the first restricted prior; it is not a posterior. |
| TSNPE rounds | `density_estimator.pt` (`78dddcd5…`), `posterior_FINAL.npy`, `history.json`, `astro_meta.json` | 6 rounds: 30,000 simulations in round 0, then 20,000 per round. 132,000 simulations in total, so the pilot had 2,000 rows. Simulations kept per round (`history.json` key `valid`): 908, 994, 1,233, 1,442, 1,604, 1,695. Acceptance reference (`astro_meta.json` key `LOGL_REF`) −9.0363. Wall time 26.2 min. |
| MIS (final stage) | `tsnpe_nucleonic_mis_a1_20261003.npz` (`4f4c67bf…`) | 105,000 rows: 60,000 flow + 15,000 broad Gaussian + 30,000 mild Gaussian, with the three component densities at every row. Share of each Gaussian's mass inside the prior box: 0.3389215 (broad) and 0.729833 (mild). ESS 30,762.70, log Z = −25.54883 ± 0.00479. 104,846 rows have a positive weight; 3 rows have none (the target rejects them). |
| Compact posterior | `results/nucleonic/a1_parameter_posteriors.npz` (Route 1): `theta_tsnpe_mis`, `weight_tsnpe_mis` | The 104,846 positive-weight rows and their normalized weights. |

## 2. Before you start

1. Install the Route 2 environment (`requirements-route2.txt` or `environment-route2.yml`). The frozen
   estimator also needs `nflows` 0.14. It is installed automatically with `sbi` 0.25.0, through `pyknos`
   0.16.0.
2. Work in the repository root and unpack the observation files once:

   ```bash
   export PYTHONPATH=$PWD
   ./reproduce.sh check
   ```

3. Download the two large TSNPE files. Each is checked against its SHA-256.

   ```bash
   python -m route2.fetch --group tsnpe_nucleonic
   ```

| File in `route2/downloads/` | Size | What it is |
|---|---|---|
| `tsnpe_nucleonic_mis_a1_20261003.npz` | 10.1 MB | the output `tsnpe_mis_corrected.npz` of the final stage (Step 3) |
| `tsnpe_nucleonic_density_estimator.pt` | 12.0 MB | the estimator `density_estimator.pt` of the published TSNPE rounds (Step 2) |

## 3. From-scratch rerun, step by step

For the training rounds (Steps 1 and 2), the products of the published run are kept, but not its command lines,
seeds or logs. The commands below use the code in `inference/tsnpe/` with its default settings; in the published
run, the number of rounds, the simulations per round, the pilot size and the flow size all equal these defaults.
For the final stage (Step 3), the command and seeds of the published run are given below.

All three scripts use a CUDA GPU automatically if one is present. Add `--device cpu` to stay on the CPU.
`--workers N` (default 1) runs N likelihood processes in parallel. It changes only the wall time, not the
numbers.

### Step 1 (optional): train a round-zero support flow

Skip this step to use the frozen support flow shipped in `inference/tsnpe/checkpoints/`. `train.py` uses that
frozen flow by default.

```bash
python inference/tsnpe/train_seed.py \
    --bank uniform_prior_support_bank.npz \
    --output-dir runs/tsnpe_nucleonic/seed_flow
```

- **Input:** the uniform-prior support bank `uniform_prior_support_bank.npz` of the published run (75 MB,
  SHA-256 `4f50897b…`). It holds 660,000 rows drawn uniformly from the prior, with the nuclear predictions `X`
  and the maximum mass `MM`. It is not among the TSNPE check downloads, but it is in the optional full-rerun
  download (`./reproduce.sh route2-fetch --group full_rerun`) as `route2/downloads/full_nucleonic_uniform_prior_support_bank.npz`
  (see [FULL_RERUN_FILES.md](FULL_RERUN_FILES.md)).
- **What it does:** it trains a conditional flow (zuko NSF: 8 transforms, 3 hidden layers of 192, 10 bins) on
  the valid rows. Each row is weighted by 1 / (1 + exp(−(MM − 2.0)/0.05)), a soft 2 M☉ maximum-mass cut. The
  nuclear predictions get Gaussian noise with the A1 uncertainties.
- **Defaults:** `--steps 26000 --batch-size 4096 --learning-rate 1e-3 --seed 0`.
- **Output:** `flow_nuclear_mmax.pt` and `flow_nuclear_mmax_std.npz`. To use them in Step 2, add
  `--seed-checkpoint runs/tsnpe_nucleonic/seed_flow/flow_nuclear_mmax.pt` and
  `--seed-standardization runs/tsnpe_nucleonic/seed_flow/flow_nuclear_mmax_std.npz`.

### Step 2: TSNPE rounds

```bash
python inference/tsnpe/train.py \
    --data-root data/observations \
    --output-dir runs/tsnpe_nucleonic/train \
    --workers 8
```

- **Target gate.** The script first checks the exact A1 target against its immutable certificate
  (`workflows/data/a1_fixed_target_certification.npz`, 1,870 rows) and stops if it fails.
- **Pilot.** 2,000 rows from the first restricted prior fix the acceptance reference. It is the largest
  summed NICER + GW170817 + pQCD log-likelihood among these rows, plus 0.5.
- **Rounds.** Each round:
  1. draws parameters from the restricted prior. This is the prior box, cut to where the flow density is
     above its 1e-4 quantile (estimated from 200,000 draws). Round 0 uses the support flow; later rounds use
     the latest TSNPE flow;
  2. runs the simulator. A row is accepted with probability exp(log L_Mmax + min(log L_other − reference, 0)).
     Accepted rows get their nuclear predictions plus A1 noise;
  3. retrains the sbi NPE flow (NSF, 10 transforms, 256 hidden units).

  Round 0 uses 30,000 simulations; later rounds use 20,000. There are at most 6 rounds. From round 2 on,
  training stops early if the median and the 90% width of every parameter change by less than 5% of the
  width.
- **Defaults:** `--seed 0 --pilot 2000 --n0 30000 --n 20000 --max-rounds 6 --support-quantile 1e-4
  --support-samples 200000 --posterior-samples 4000 --final-samples 20000 --hidden-features 256
  --transforms 10`.
- **Output:** `density_estimator.pt`, `posterior_FINAL.npy` (20,000 draws), `history.json`,
  `run_report.json`.

### Step 3: three-part mixture importance sampling

```bash
python inference/tsnpe/run_mis.py \
    --data-root data/observations \
    --estimator runs/tsnpe_nucleonic/train/density_estimator.pt \
    --output-dir runs/tsnpe_nucleonic/mis \
    --workers 8
```

To redo only this stage with the frozen estimator, use
`--estimator route2/downloads/tsnpe_nucleonic_density_estimator.pt`. The published final stage is exactly this
command with `--workers 16 --device cpu` and all other settings at their defaults.

What the script does:

1. It loads the flow as an sbi `DirectPosterior` with a uniform (`BoxUniform`) prior on the 7-D prior box,
   conditioned on the A1 observation. It draws 60,000 rows and computes their exact log L. It weights them
   with the flow density. The flow density is normalized to the prior box: `log_prob(..., norm_posterior=True)`.
2. It fits the mean and covariance of these weighted rows, widens them by 2.5, truncates to the prior box and
   draws 15,000 **broad** rows. The 75,000 rows are then weighted with the two-part mixture.
3. It fits the 75,000 weighted rows the same way, widens by 1.5 and draws 30,000 **mild** rows.
4. It weights all 105,000 rows with the mixture density
   q = (60/105) q_flow + (15/105) q_broad + (30/105) q_mild.
   Here q_broad is the Gaussian that generated the broad rows, never a refit. Each Gaussian density is divided
   by its share of mass inside the box, estimated from 2,000,000 draws.

- **Defaults:** `--flow-draws 60000 --broad-draws 15000 --mild-draws 30000 --broad-scale 2.5 --mild-scale 1.5
  --normalization-draws 2000000 --seed 0 --resample-seed 1 --posterior-rows 20000 --minimum-ess 100`.
- **Seeds used by the script:** torch and numpy seed 0 (flow draws). Broad Gaussian: seed+101 (mass share)
  and seed+102 (draws). Mild Gaussian: seed+201 and seed+202. Resample: seed 1.
- **Output:** `stage_1_flow.npz`, `stage_2_flow_broad.npz`, `tsnpe_mis_corrected.npz` (theta, log L, log q
  and its three parts, log w, w, posterior resample) and `run_report.json`.

The formulas (N = 105,000; all rows count in N, including rows with zero weight):

| Quantity | Formula | Published value |
|---|---|---|
| log weight | log w_i = log L_i + log prior − log q_i, with log prior = −Σ log(high − low) = −0.25415 | — |
| normalized weight | w_i = exp(log w_i) / Σ_j exp(log w_j) | — |
| ESS | 1 / Σ w_i² | 30,762.69954869431 |
| log Z | log[(1/N) Σ exp(log w_i)] | −25.54882648807424 |
| error of log Z | sqrt(Σ w_i² − 1/N) | 0.004794068266360253 |

The error of log Z above is `final.log_evidence_standard_error` in `run_report.json`. The field
`logZ_standard_error` in `tsnpe_mis_corrected.npz` uses the ddof = 1 variance instead and is larger by a factor
of about sqrt(N/(N − 1)) (here 0.004794091095419824).

The exact A1 target of `run_mis.py` (`workflows/a1_problem.py`) is the complete exact likelihood, including the
PSR J0740+6620 NICER term, so the weights need no further correction. The 3 rows that the target rejects
(log L = −inf after the target marks a failed likelihood term) get no weight.

### Step 4: compact posterior

- `theta_tsnpe_mis` = theta of the 104,846 rows with a positive weight
- `weight_tsnpe_mis` = their weights, normalized to sum 1

They are stored in `results/nucleonic/a1_parameter_posteriors.npz`, which Route 1 uses for the figures. The
mass-radius band and the Table III numbers come from 4,000 rows resampled with numpy `default_rng(3)`; the
mass-tidal band uses every positive-weight row (`results/nucleonic/tsnpe_mis_a1_summary.json`).

## 4. Settings, sizes, times and hardware

| Item | Value |
|---|---|
| Support flow | frozen `flow_nucmm_tight.pt`. It loads exactly (strict loading) into the zuko NSF of `train.py`: 8 transforms, 3 × 192 hidden units, 10 bins |
| Support flow training command, seed, log | not known |
| TSNPE rounds and sizes | 6 rounds; 30,000 + 5 × 20,000 simulations; pilot 2,000; total 132,000 |
| TSNPE estimator | sbi NSF, 10 transforms, 256 hidden units, 10 bins (read from the file) |
| TSNPE training wall time | 26.2 min |
| TSNPE training hardware | a CUDA GPU (the estimator was saved from GPU memory); the machine is not known |
| TSNPE training seed | not known |
| Support quantile; draws to estimate it; posterior draws per round for the stopping rule | 1e-4; 200,000; 4,000 (code defaults; the values of the published run are not known) |
| MIS sizes | 60,000 + 15,000 + 30,000 = 105,000 rows |
| Share of Gaussian mass inside the box | broad 0.3389215, mild 0.729833 |
| Gaussian widening factors | broad 2.5, mild 1.5 |
| MIS seeds | torch and numpy 0; broad 101/102; mild 201/202; resample 1 |
| MIS wall time and hardware | 1,713 s with 16 CPU workers at low priority on a desktop with an Intel Core i7-13700F |
| Posterior resample | 20,000 rows with seed 1 |

**Hardware caution for a full rerun.** Steps 2 and 3 need about 241,000 exact likelihood evaluations:
132,000 simulations, 105,000 mixture rows and the 1,870-row target gate, run once in each step. On this
package's test machine (Intel Core i7-13700F), one worker needs about 0.04 s per row. That is about 2.6
CPU-hours in total, or about 20 minutes with 8 workers, plus the flow training (the published run used a
GPU). A rerun of the training rounds will not reproduce the published estimator bit for bit, because the
training seed is not known and the hardware differs. It reproduces the result only within its Monte Carlo
errors.

**Smoke runs (tested on the same machine, CPU only, 3 threads).** Each finishes in under 30 s:

```bash
python inference/tsnpe/train_seed.py --bank uniform_prior_support_bank.npz \
    --output-dir runs/tsnpe_smoke/seed --device cpu --smoke                      # 3 s
python inference/tsnpe/train.py --data-root data/observations \
    --output-dir runs/tsnpe_smoke/train --device cpu --pilot 32 --n0 64 --n 64 --max-rounds 1 \
    --posterior-samples 16 --final-samples 16 --support-samples 2000 --max-training-epochs 2 \
    --smoke --skip-target-gate                                                     # 23 s
python inference/tsnpe/run_mis.py --data-root data/observations \
    --estimator route2/downloads/tsnpe_nucleonic_density_estimator.pt \
    --output-dir runs/tsnpe_smoke/mis --device cpu --flow-draws 60 --broad-draws 15 --mild-draws 30 \
    --normalization-draws 20000 --posterior-rows 20 --minimum-ess 1 --smoke --skip-target-gate  # 25 s
```

`--skip-target-gate` is allowed only together with `--smoke`. A smoke result is marked `SMOKE_COMPLETED`; it
is a plumbing test, not a scientific result. The tiny MIS run above gave log Z = −25.68 ± 0.11 from 105
rows.

## 5. The check

```bash
python -m route2.checks.tsnpe_nucleonic          # add --device cuda to use a GPU for the flow
```

It runs in about 20 s on a CPU and writes `build/route2/tsnpe_nucleonic_check.json`. It prints one line per step
and a final `PASS` or `FAIL`. `./reproduce.sh route2-check` runs it together with the other Route 2 checks.

| Step | What is replayed | Tolerance and why | Result on the test machine |
|---|---|---|---|
| Code identity | the 7 files of `inference/tsnpe/` have the SHA-256 values listed in `route2/fixtures/tsnpe_nucleonic/expected.json` | exact | 7/7 identical |
| MIS arithmetic | the mixture density from the three stored component densities, then the weights, ESS, log Z and its error, with the production code; equal to the stored values and the run report; the 3 rows without a weight are exactly the rows with no finite log L | 1e-12 (float64 summation order only) | largest difference 0 |
| Proposal replay | the broad and mild Gaussians are rebuilt from the stored rows as `run_mis.py` builds them; their densities at all rows and the 45,000 Gaussian draws equal the stored ones; box shares 0.3389215 and 0.729833 | 1e-9 | largest difference 0 |
| Frozen estimator | (a) it loads as in `run_mis.py`; its density at the 14 replay rows equals the stored flow density; (b) 20,000 fresh draws (torch seed 0) against the published `posterior_FINAL.npy`; (c) the rounds, sizes and flow size of the published run equal the code defaults | (a) 0.01 in log: sbi normalizes the flow with a Monte Carlo estimate of its mass inside the box, made anew for each batch of 4,000 rows (standard error about 0.0014); (b) Kolmogorov–Smirnov D per parameter below the value for a 1e-3 false-alarm rate over the 7 parameters (0.0218) | (a) common offset −0.0021, row to row 0.0018; (b) largest D 0.0109 |
| Exact log L | the A1 target is rebuilt from `data/observations` and evaluated on 14 rows: the 6 largest weights and 8 random rows (seed 20261003); the 3 rows without a weight must be rejected (≤ −1e29) | 1e-4, the target-certificate tolerance | largest difference 4.1e-7; all 3 rejected |
| Route 1 identity | `theta_tsnpe_mis` and `weight_tsnpe_mis` of Route 1 are the 104,846 positive-weight rows and their normalized weights | exact (bit-identical arrays) | bit-identical |

Small inputs (the training files and the expected numbers) are in `route2/fixtures/tsnpe_nucleonic/`, with
`MANIFEST.json` giving the SHA-256 of each.

## 6. Gaps (what cannot be rerun exactly)

1. **No commands, seeds or logs for the TSNPE training rounds.** The products are kept (estimator,
   `posterior_FINAL.npy`, `history.json`, `astro_meta.json`), but not the command lines, the seeds or the logs
   of the training. The commands of Steps 1 and 2 are built from the code defaults. The sizes of the published
   run match them, but the training can be reproduced only statistically.
2. **The training files were not written by `train.py`.** `history.json` uses other key names
   (`nsim`, `med`, `w90`, `valid`) than `train.py` writes. Its sizes, the network size and the mixture sizes
   equal the code defaults.
3. **Support flow.** No training command, seed or log is known for `flow_nucmm_tight.pt`. Its standardization is
   not what `train_seed.py` computes from the support bank of Step 1 (largest relative difference 0.28). Step 1
   therefore trains a different support flow, not the frozen one. The support bank (75 MB) is not among the TSNPE
   check downloads; it is in the optional full-rerun download (Step 1).
