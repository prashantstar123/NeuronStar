# Route 2 steps: hyperonic TSNPE+MIS (A1)

This page explains how the hyperonic (DDBΛΞ⁻) TSNPE posterior and evidence for the A1 data were made, how to
make them again from scratch with the code in this repository, and how to check the saved result quickly.

- **TSNPE** (truncated sequential neural posterior estimation) learns where the posterior lies, round by
  round, and concentrates the expensive simulations there.
- **MIS** (deterministic-mixture importance sampling) then corrects everything exactly: it draws parameter rows
  from a mixture of frozen densities that covers the whole 9-D prior, computes the exact A1 likelihood of every
  row and weights each row by prior × likelihood / mixture density. The weighted rows are the posterior; the
  mean weight gives the evidence Z. The TSNPE steps only decide where the rows are drawn; they never change the
  target, which is always the original full prior times the exact likelihood.

The commands use the package code with the settings listed; where a setting of the published run is not known,
this is said in the text. A **frozen** file is a file whose SHA-256 is stored in a later file of the chain.

The code is in `hyperonic_pipeline/tsnpe/` (29 Python scripts; `CODE_MANIFEST.json` gives the SHA-256 and role
of each). The 7 scripts of Steps 6–10 are byte-identical to the scripts that produced the published result. The
other 22 are the scripts that these 7 import or that produced their inputs. The EOS, TOV and likelihood code is
the repository's portable target (`workflows/`, `likelihoods/`, `eos/`, `tov/`) and, for the curves,
`hyperonic_pipeline/fast_curve_full_shard.py`.

## 1. The chain in short

```text
original 9-D prior x exact A1 likelihood (the only target)
  |
  |- round 0: TSNPE-owned defensive bridge (Gaussian mixtures + 10% uniform prior), fitted to the
  |           prior-generated physics caches under the astrophysical likelihood only
  |           300,000 candidates -> exact simulation -> 947 accepted training rows
  |- round 1: learned truncated proposal, 20,000 candidates -> 72 accepted rows
  |- round 2: refined truncated proposal, 20,000 candidates -> 50 accepted rows
  |- two independently seeded frozen TSNPE flows (1,069 rows)
  |- defensive MIS (45,000 rows) and four adaptive stages: exact, but none reaches its ESS threshold;
  |   their rows are reused below (never as a posterior)
  |- held-out cross-validation: 60/20/20 split -> 4-component bounded-logit Gaussian mixture;
  |   both untouched folds forecast ESS > 3,000 (required)
  |- frozen final proposal: 90% cross-validated mixture + 10% parent TSNPE mixture, 150,000 rows,
  |   two uniform-prior defenses
  |- exact A1 likelihood of every row -> deterministic-MIS weights -> ESS 1,513.77 (minimum 1,500)
  |- 6,000-draw resample -> EOS/TOV curves -> 90% mass-radius band
```

| File of the published run (name written by the rerun) | SHA-256 | Numbers |
|---|---|---|
| posterior `tsnpe_hyperonic_clean_crossvalidated_9d_mis.npz` (Step 9; downloaded as `tsnpe_hyperonic_posterior_9d_mis.npz`) | `ae0bfba7…` | 150,000 rows, 128,707 valid; ESS 1,513.7694295 (minimum 1,500); log Z = −28.7091446 ± 0.0255723; largest weight 0.0134519; 20,000-row posterior resample, seed 2026095501 |
| final proposal `crossvalidated_hybrid_candidates_150k.npz` (Step 7; downloaded as `tsnpe_hyperonic_candidates_150k.npz`) | `82f46920…` | the frozen final proposal (seed 2026095401, RTX 4090); per row: θ, log q, `logq_repair`, `logq_parent_tsnpe`, component |
| cross-validated mixture `crossfit_gmm_bridge.npz` (Step 6; in `route2/fixtures/tsnpe_hyperonic/`) | `66948847…` | seed 2026095301; held-out forecasts 4,291.6 and 5,340.2 |
| partition `crossfit_training_stage.npz`, `crossfit_validation_stage_{1,2}.npz` (Step 6) | `581e9155…`, `60bed10b…`, `87e34547…` | training (147,000 rows) and two validation folds (24,000 each), split seed 2026095201 |
| exact shards `xfit150k_part0{0,1,2}_exact.npz` (Step 8) | `b448ab70…`, `97a56af4…`, `b53ee865…` | exact log L of the 150,000 rows in three shards |
| curve resample `tsnpe_curve_resample6000.npz` (Step 10) | `d3afca0e…` | 6,000 draws (seed 20260921), 4,015 unique rows; Route 1 ships it as `results/hyperonic/A1/tsnpe_resample6000.npz` |
| curves `tsnpe_curve_resample6000_curves.npz` (Step 10) | `9089d0cd…` | their EOS/TOV curves; Route 1 ships it as `results/hyperonic/A1/tsnpe_curves.npz` |

The final proposal also uses three frozen files, referenced by hash in the candidate file: the parent mixture
`optimized_hybrid_model.npz` (`a04f65dc…`) and the two flow proposals `57f39d74…` (3-member ensemble) and
`63d32d82…` (4-member ensemble). The hashes of their 7 network files are stored inside the proposal files.
The code calls the 3-member ensemble `old` and the 4-member ensemble `new` (options `--old-proposal` and
`--new-proposal`; download names `tsnpe_hyperonic_old_flow_*` and `tsnpe_hyperonic_new_flow_*`).

**Independence.** The TSNPE uses no A-NET, Evidence-Network or conventional-sampler product. It shares only
the prior, the observations, the EOS/TOV/likelihood code and the prior-generated physics caches (Step 1) with
the other methods. UltraNest and A-NET bands enter only the comparison figure (Step 10), after the posterior
and curves were frozen.

## 2. Before you start

1. Install the Route 2 environment. The flows need `torch` 2.5.1 and `zuko` 1.6.0; the rounds use `sbi`
   0.25.0; the Gaussian mixtures use `scikit-learn`.
2. Work in the repository root, set the paths and unpack the observation files once:

   ```bash
   export PYTHONPATH=$PWD JAX_PLATFORMS=cpu JAX_ENABLE_X64=True TOV_NW=1
   ./reproduce.sh check
   T=hyperonic_pipeline/tsnpe          # the TSNPE scripts
   R=runs/tsnpe_hyperonic              # where a rerun writes its files
   ```

   The exact likelihood steps start `--workers N` likelihood processes, each with one thread. Set this before
   them:

   ```bash
   export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
   ```

   The published run used 60 workers per cluster node (62 for the curves of Step 10) and 22 on the desktop
   workstation; `--workers` changes only the wall time. Each exact step reads the JSON report next to its input
   `.npz` (written by the step that made the input) and writes its own report next to its output. The scripts
   that write results refuse to overwrite an existing output; the three split scripts do not check, so give
   every run fresh folders.
3. Download the frozen files (84.6 MB). Each is checked against its SHA-256:

   ```bash
   python -m route2.fetch --group tsnpe_hyperonic
   ```

| File in `route2/downloads/` | Size | What it is |
|---|---|---|
| `tsnpe_hyperonic_posterior_9d_mis.npz` | 20.6 MB | the posterior (Step 9) |
| `tsnpe_hyperonic_candidates_150k.npz` | 13.6 MB | the final proposal (Step 7) |
| `tsnpe_hyperonic_old_flow_proposal.npz` + `.json` | 6.3 MB | the 3-member flow proposal of Step 5.2 (`57f39d74…`) and its report |
| `tsnpe_hyperonic_old_flow_0{0,1,2}.pt` | 3 × 5.1 MB | its three network files |
| `tsnpe_hyperonic_new_flow_proposal.npz` + `.json` | 8.4 MB | the 4-member flow proposal of Step 5.3 (`63d32d82…`) and its report |
| `tsnpe_hyperonic_new_flow_0{0,1,2,3}.pt` | 4 × 5.1 MB | its four network files |

The small frozen files (the cross-validated mixture and the parent mixture, each with its report) are in
`route2/fixtures/tsnpe_hyperonic/`.

## 3. From-scratch rerun, step by step

The settings and seeds below are those of the published run. Its command lines are known only for the exact
likelihood evaluations that ran as cluster jobs (Steps 2–5 and 8) and for the curves (Step 10); the other
commands are built from the code defaults and these settings. "GPU" marks a step that ran on one rented cloud
GPU (NVIDIA RTX 4090); add `--device cpu` to run it on a CPU (slower, and the float32 flow arithmetic differs
slightly). "Exact" marks a CPU step that evaluates the exact A1 likelihood: it first re-runs the frozen 64-row
target certificate (`workflows/data/hyperonic_target_certification.npz`) and stops if it fails. `--workers N`
changes only the wall time.

### Step 1: round-zero bridge (3 scripts; CPU)

```bash
python $T/build_tsnpe_astrophysical_bridge.py \
    --uniform-cache uniform_training_cache_part1.npz --uniform-cache uniform_training_cache_part2.npz \
    --uniform-cache uniform_training_cache_part3.npz --uniform-cache uniform_training_cache_part4.npz \
    --support-cache support_training_cache_part1.npz --support-cache support_training_cache_part2.npz \
    --bridge-cache $R/inputs/tsnpe_astrophysical_bridge_source.npz \
    --checkpoint $R/bridge_v2/tsnpe_owned_astro_gmm_ensemble.joblib \
    --output $R/round_00_bridge_v2_proposal.npz --seed 20260923
python $T/refine_tsnpe_astrophysical_bridge.py \
    --source-cache $R/inputs/tsnpe_astrophysical_bridge_source.npz \
    --base-checkpoint $R/bridge_v2/tsnpe_owned_astro_gmm_ensemble.joblib \
    --checkpoint $R/bridge_v3/tsnpe_owned_residual_astro_gmm_ensemble.joblib \
    --output $R/round_00_bridge_v3_proposal.npz --seed 20260924
python $T/extend_tsnpe_bridge_proposal.py --proposal $R/round_00_bridge_v3_proposal.npz \
    --checkpoint $R/bridge_v3/tsnpe_owned_residual_astro_gmm_ensemble.joblib \
    --extra-count 100000 --seed 20260925 --output $R/round_00_bridge_v3_320k_proposal.npz
```

- **Input:** the six shared prior-generated physics caches (uniform-prior draws with their EOS/TOV results; no
  posterior of any method). The same prior caches are also an input of A-NET (see the hyperonic A-NET steps):
  the four uniform training caches (`e51cf2f1…`, `e7ac3193…`, `e13a8566…`, `575ebf36…`) and the two support
  caches (`479206d0…`, `28e0894d…`). Together 499,445 rows (303,855 valid) from 600,000 uniform proposals and a
  160,000,000-proposal support screen (radius 3σ); their known sampling-count correction is applied. They are
  not among the TSNPE check downloads, but all six are in the optional full-rerun download
  (`./reproduce.sh route2-fetch --group full_rerun`) as `route2/downloads/full_hyperonic_uniform_training_cache_part1..4.npz`
  and `route2/downloads/full_hyperonic_support_training_cache_part1..2.npz` (see [FULL_RERUN_FILES.md](FULL_RERUN_FILES.md)).
- **Target of the bridge:** uniform prior × the astrophysical likelihood only (maximum mass, NICER, GW170817,
  pQCD). The A1 nuclear datum enters later, as simulated data, so it is not counted twice. The source cache
  has astrophysical ESS 12,745.5.
- **What it builds:** an ensemble of three 12-component Gaussian mixtures (120,000 training rows each) plus a
  0.10 uniform-prior part, then three 16-component residual-tail mixtures. The extension draws 100,000 more
  rows from the frozen density without refitting it.
- **Published run:** seeds 20260923, 20260924, 20260925 (all other options equal the code defaults); hashes:
  source cache `717938bd…`, base mixture `d81b8619…`, residual mixture `f29aaca7…`, bridge proposal `33a57313…`
  (20,000 pilot + 200,000 rows), extended proposal `728d7646…` (20,000 + 300,000 rows). Wall times 58.7 s
  and 96.5 s. The three shipped scripts have the same SHA-256 as the versions that ran.

### Step 2: round zero, exact simulation (Exact; the candidate parts as cluster jobs)

```bash
python $T/split_tsnpe_round0_proposal.py --proposal $R/round_00_bridge_v3_proposal.npz --output-dir $R/round0_bridge_v3_split
python $T/split_tsnpe_round0_proposal.py --proposal $R/round_00_bridge_v3_320k_proposal.npz --output-dir $R/round0_bridge_v3_320k_split
python $T/split_tsnpe_candidate_proposal.py --proposal $R/round0_bridge_v3_320k_split/round_00_bridge_v3_candidate_01.npz \
    --output-dir $R/round0_bridge_v3_320k_split/candidate_01_parts --prefix candidate_01
# pilot: defines the frozen acceptance reference (a desktop workstation, 22 workers, 526.8 s)
python $T/evaluate_tsnpe_round_9d.py --proposal $R/round0_bridge_v3_split/round_00_bridge_v3_pilot.npz \
    --data-root data/observations --output $R/round0_bridge_v3_split/round_00_bridge_v3_pilot_exact.npz \
    --workers 22 --accept-seed 2026092500
# the three candidate parts (150,000 + 75,000 + 75,000 rows), 60 workers each
S0=$R/round0_bridge_v3_320k_split
REF=$R/round0_bridge_v3_split/round_00_bridge_v3_pilot_exact.json
python $T/evaluate_tsnpe_round_9d.py --proposal $S0/round_00_bridge_v3_candidate_00.npz --data-root data/observations \
    --output $S0/round_00_bridge_v3_candidate_00_exact.npz --workers 60 --accept-seed 2026092501 \
    --reference-margin 1.0 --reference-report $REF
python $T/evaluate_tsnpe_round_9d.py --proposal $S0/candidate_01_parts/candidate_01_part_00.npz --data-root data/observations \
    --output $S0/candidate_01_parts/candidate_01_part_00_exact.npz --workers 60 --accept-seed 2026092502 \
    --reference-margin 1.0 --reference-report $REF
python $T/evaluate_tsnpe_round_9d.py --proposal $S0/candidate_01_parts/candidate_01_part_01.npz --data-root data/observations \
    --output $S0/candidate_01_parts/candidate_01_part_01_exact.npz --workers 60 --accept-seed 2026092503 \
    --reference-margin 1.0 --reference-report $REF
python $T/combine_tsnpe_round0_exact_shards.py --input $S0/round_00_bridge_v3_candidate_00_exact.npz \
    $S0/candidate_01_parts/candidate_01_part_00_exact.npz $S0/candidate_01_parts/candidate_01_part_01_exact.npz \
    --pilot-report $REF --output $R/round_00_bridge_v3_exact_combined.npz
```

- **Accept/reject:** each valid row is accepted with probability min(1, prior × L_astro / (q_bridge × e^ref)),
  where ref is the largest pilot value plus a margin of 1.0. The run stops if any row exceeds the reference
  (never clipped). Accepted rows get simulated nuclear data (the model's nuclear predictions plus Gaussian
  noise with the A1 uncertainties). Pilot rows only set the reference; they are never training rows.
- **Published run:** reference −9.16930 (bridge) and −8.45697 (restricted prior, used by later rounds); 0
  reference exceedances; 947 training rows (470 + 248 + 229) from 300,000 candidates; combined archive
  `9cf4ef59…`. Wall times 1,764.5 s, 1,930.6 s, 897.7 s. `--reference-margin 1.0` equals the code default.
  The machines that ran the three candidate parts are not known.

### Step 3: rounds 1 and 2 and the two frozen flows (GPU for fits and proposals; Exact for the rounds)

```bash
# round-zero estimator (947 rows)
python $T/train_tsnpe_round_9d.py --round-data $R/round_00_bridge_v3_exact_combined.npz --output-dir $R/round0_fit --seed <not known>
REF=$R/round0_bridge_v3_split/round_00_bridge_v3_pilot_exact.json
# round 1 (exact parts: 60 workers each)
python $T/generate_tsnpe_round_9d.py --round-index 1 --count 20000 --seed 2026092601 \
    --estimator $R/round0_fit/density_estimator.pt --estimator-report $R/round0_fit/run_report.json \
    --output $R/round1/round1_proposal.npz
python $T/split_tsnpe_candidate_proposal.py --proposal $R/round1/round1_proposal.npz --output-dir $R/round1/parts
python $T/evaluate_tsnpe_round_9d.py --proposal $R/round1/parts/round1_proposal_part_00.npz --data-root data/observations \
    --output $R/round1/parts/round1_part00_exact.npz --workers 60 --accept-seed 2026092611 \
    --reference-margin 1.0 --reference-report $REF
python $T/evaluate_tsnpe_round_9d.py --proposal $R/round1/parts/round1_proposal_part_01.npz --data-root data/observations \
    --output $R/round1/parts/round1_part01_exact.npz --workers 60 --accept-seed 2026092612 \
    --reference-margin 1.0 --reference-report $REF
python $T/combine_tsnpe_exact_round_shards.py --input $R/round1/parts/round1_part0{0,1}_exact.npz \
    --reference-report $REF --output $R/round1/round_01_exact_combined.npz
python $T/train_tsnpe_round_9d.py --round-data $R/round_00_bridge_v3_exact_combined.npz $R/round1/round_01_exact_combined.npz \
    --output-dir $R/round1_fit --seed 2026092602
# round 2 (exact parts: 60 workers each)
python $T/generate_tsnpe_round_9d.py --round-index 2 --count 20000 --seed 2026092603 --support-quantile 0.01 \
    --estimator $R/round1_fit/density_estimator.pt --estimator-report $R/round1_fit/run_report.json \
    --output $R/round2/round2_proposal.npz
python $T/split_tsnpe_candidate_proposal.py --proposal $R/round2/round2_proposal.npz --output-dir $R/round2/parts
python $T/evaluate_tsnpe_round_9d.py --proposal $R/round2/parts/round2_proposal_part_00.npz --data-root data/observations \
    --output $R/round2/parts/round2_part00_exact.npz --workers 60 --accept-seed 2026092621 \
    --reference-margin 1.0 --reference-report $REF
python $T/evaluate_tsnpe_round_9d.py --proposal $R/round2/parts/round2_proposal_part_01.npz --data-root data/observations \
    --output $R/round2/parts/round2_part01_exact.npz --workers 60 --accept-seed 2026092622 \
    --reference-margin 1.0 --reference-report $REF
python $T/combine_tsnpe_exact_round_shards.py --input $R/round2/parts/round2_part0{0,1}_exact.npz \
    --reference-report $REF --output $R/round2/round_02_exact_combined.npz
# the two frozen flows, fitted independently to the same 947 + 72 + 50 rows
ROUNDS="$R/round_00_bridge_v3_exact_combined.npz $R/round1/round_01_exact_combined.npz $R/round2/round_02_exact_combined.npz"
python $T/train_tsnpe_round_9d.py --round-data $ROUNDS --output-dir $R/round2_fit_seedA --seed 2026092604
python $T/train_tsnpe_round_9d.py --round-data $ROUNDS --output-dir $R/round2_fit_seedB --seed 2026093604
```

- **Rounds:** a round draws from the prior restricted to where the preceding estimator's density is above its
  quantile (1e-4 in round 1; 0.01 in round 2, because the round-1 support was broad), using sampling-importance
  resampling (200,000 support draws, 1,024× oversampling). The exact simulator accepts with probability
  L_Mmax × min(1, L_other / e^ref). The estimator is an sbi neural spline flow (10 transforms, 256 hidden
  units, at most 400 epochs), refitted on the full ordered prefix of rounds.
- **Published run:** round 1: proposal `762dfae8…` (118.8 s), 72 accepted of 20,000 (17,257 valid), archive
  `1ca31236…`, fit `85bf65ce…` on 1,019 rows (38.6 s). Round 2: proposal `407eb5ac…` (119.8 s), 50 accepted of
  20,000 (17,303 valid), archive `85e96b20…`. Frozen flows `e33e12c4…` (27.4 s) and `0bf2beb3…` (29.4 s). Exact
  shard wall times 269.4 s + 144.2 s (round 1) and 270.5 s + 144.4 s (round 2).
- The round-zero estimator (`283844cb…`) is not available and its seed is not known (see Gaps).

### Step 4: defensive MIS of the two flows (GPU for the candidates; Exact)

```bash
python $T/generate_clean_tsnpe_mis_candidates_9d.py --first-run-dir $R/round2_fit_seedA --second-run-dir $R/round2_fit_seedB \
    --uniform-draws 5000 --uniform-seed 2026092701 --output $R/mis/tsnpe_mis_candidates.npz
python $T/split_clean_tsnpe_mis_candidates.py --input $R/mis/tsnpe_mis_candidates.npz --output-dir $R/mis/parts
python $T/evaluate_clean_tsnpe_mis_shard_9d.py --candidate $R/mis/parts/tsnpe_mis_candidates_part_00.npz \
    --data-root data/observations --output $R/mis/parts/tsnpe_mis_part00_exact.npz --workers 60
python $T/evaluate_clean_tsnpe_mis_shard_9d.py --candidate $R/mis/parts/tsnpe_mis_candidates_part_01.npz \
    --data-root data/observations --output $R/mis/parts/tsnpe_mis_part01_exact.npz --workers 60
python $T/combine_clean_tsnpe_mis_exact_9d.py --input $R/mis/parts/tsnpe_mis_part0{0,1}_exact.npz \
    --density-calibration $R/mis/flow_density_calibration.json --output $R/mis/tsnpe_hyperonic_clean_full_prior_mis.npz \
    --resample-seed 2026092702
```

- **Published run:** 45,000 rows (20,000 from each flow + 5,000 uniform), `43727227…`. The flow densities were
  normalized to the prior box with 1,000,000 raw draws per flow (inside-prior shares 0.92750 and 0.93951; seeds
  2026092801 and 2026092802; file `flow_density_calibration.json`, `27a7bd4a…`); the script that wrote this
  file is not available (see Gaps). The two exact parts ran as cluster jobs, each on its own cluster node
  (60 workers; 62 cores allocated on each); shard wall times 550.9 s and 289.3 s. Result `b94f73a5…`: ESS 72.9,
  below the script's minimum ESS of 1,000, so it is not used as a posterior; its rows feed Step 5.

### Step 5: adaptive proposal stages (GPU for fits and candidates; Exact)

Each stage adds fresh exact rows. None of the four reaches its ESS threshold, so none is used as a posterior;
their rows and densities are the inputs of the cross-validated mixture (Step 6). The exact parts of Steps 5.1,
5.2 and 5.4 ran as cluster jobs with 60 workers each.

1. **Gaussian mixture + Student-t** (code defaults otherwise: 20,000 mild + 10,000 broad draws, 4 components,
   50,000 fit resamples). Exact: two shards of 15,000 rows (384.3 s, 201.7 s). Combined with Step 4: 75,000 rows,
   ESS 198.5, `ee1be027…`.

   ```bash
   python $T/generate_adaptive_tsnpe_mis_9d.py --stage-one $R/mis/tsnpe_hyperonic_clean_full_prior_mis.npz \
       --first-run-dir $R/round2_fit_seedA --second-run-dir $R/round2_fit_seedB \
       --density-calibration $R/mis/flow_density_calibration.json --seed 2026092901 \
       --output $R/mis/adaptive/adaptive_candidates.npz
   python $T/split_clean_tsnpe_mis_candidates.py --input $R/mis/adaptive/adaptive_candidates.npz \
       --output-dir $R/mis/adaptive/parts
   python $T/evaluate_clean_tsnpe_mis_shard_9d.py --candidate $R/mis/adaptive/parts/tsnpe_mis_candidates_part_00.npz \
       --data-root data/observations --output $R/mis/adaptive/parts/adaptive_mis_part00_exact.npz --workers 60
   python $T/evaluate_clean_tsnpe_mis_shard_9d.py --candidate $R/mis/adaptive/parts/tsnpe_mis_candidates_part_01.npz \
       --data-root data/observations --output $R/mis/adaptive/parts/adaptive_mis_part01_exact.npz --workers 60
   python $T/combine_adaptive_tsnpe_mis_9d.py --stage-one $R/mis/tsnpe_hyperonic_clean_full_prior_mis.npz \
       --adaptive-candidate $R/mis/adaptive/adaptive_candidates.npz \
       --adaptive-exact $R/mis/adaptive/parts/adaptive_mis_part0{0,1}_exact.npz \
       --output $R/mis/adaptive/tsnpe_adaptive_stage2.npz --resample-seed 2026092902
   ```

2. **3-member flow ensemble**: 3 unconditional zuko neural spline flows (8 transforms, 3 × 192 hidden units,
   10 bins, 120,000 training and 30,000 validation resamples, 100 epochs, patience 12; code defaults), plus a
   Student-t and the uniform prior: 60,000 + 10,000 + 5,000 rows, `57f39d74…`, 534.5 s. Member seeds 2026093001,
   2026094010, 2026095019. Exact: two shards of 26,000 and 49,000 rows (645.4 s, 605.0 s). Result: 75,000 rows,
   ESS 156.0, `71ea0979…` (the `--old-exact-stage` input of Steps 5.4 and 6).

   ```bash
   python $T/train_frozen_adaptive_flow_proposal_9d.py --adaptation-stage $R/mis/adaptive/tsnpe_adaptive_stage2.npz \
       --output-dir $R/mis/flow_ensemble3 --seed 2026093001
   python $T/split_clean_tsnpe_mis_candidates.py --input $R/mis/flow_ensemble3/final_adaptive_flow_candidates.npz \
       --output-dir $R/mis/flow_ensemble3/parts --parts 2 --part-counts 26000 49000
   python $T/evaluate_frozen_adaptive_flow_shard_9d.py --candidate $R/mis/flow_ensemble3/parts/tsnpe_mis_candidates_part_00.npz \
       --data-root data/observations --output $R/mis/flow_ensemble3/parts/final_flow_exact_part_00.npz --workers 60
   python $T/evaluate_frozen_adaptive_flow_shard_9d.py --candidate $R/mis/flow_ensemble3/parts/tsnpe_mis_candidates_part_01.npz \
       --data-root data/observations --output $R/mis/flow_ensemble3/parts/final_flow_exact_part_01.npz --workers 60
   python $T/combine_frozen_adaptive_flow_mis_9d.py --candidate $R/mis/flow_ensemble3/final_adaptive_flow_candidates.npz \
       --exact $R/mis/flow_ensemble3/parts/final_flow_exact_part_0{0,1}.npz \
       --output $R/mis/flow_ensemble3/tsnpe_hyperonic_clean_frozen_adaptive_flow_mis.npz --resample-seed 2026093101
   ```

3. **4-member flow ensemble**: 80,000 + 15,000 + 5,000 rows, `63d32d82…`, 827.7 s. Member seeds 2026094001,
   2026095010, 2026096019, 2026097028. No exact stage of its own.

   ```bash
   python $T/train_frozen_adaptive_flow_proposal_9d.py \
       --adaptation-stage $R/mis/flow_ensemble3/tsnpe_hyperonic_clean_frozen_adaptive_flow_mis.npz \
       --bridge-validation-stage $R/mis/adaptive/tsnpe_adaptive_stage2.npz --output-dir $R/mis/flow_ensemble4 \
       --ensemble 4 --training-resamples 150000 --validation-resamples 40000 --jitter 0.025 --broad-draws 15000 \
       --seed 2026094001
   ```

4. **Optimized parent mixture**: weights 0.04 and 0.10 for the flow and Student-t parts of the 3-member
   proposal, 0.04 and 0.10 for those of the 4-member proposal, and 0.72 for the uniform prior (`a04f65dc…`);
   forecast ESS 1,390.9 (minimum 1,200). The default of 100,000 rows gives a forecast of 1,184.8, below this
   minimum, so the command sets 120,000 rows. Its 120,000 candidates (`1b7cb69d…`) were evaluated exactly in two
   shards of 42,000 and 78,000 rows on two cluster nodes, 60 workers each, 997.2 s and 933.7 s.

   ```bash
   python $T/optimize_hybrid_flow_mis_proposal_9d.py \
       --old-proposal $R/mis/flow_ensemble3/final_adaptive_flow_candidates.npz \
       --old-checkpoints $R/mis/flow_ensemble3/adaptive_flow_0{0,1,2}.pt \
       --new-proposal $R/mis/flow_ensemble4/final_adaptive_flow_candidates.npz \
       --new-checkpoints $R/mis/flow_ensemble4/adaptive_flow_0{0,1,2,3}.pt \
       --old-exact-stage $R/mis/flow_ensemble3/tsnpe_hyperonic_clean_frozen_adaptive_flow_mis.npz \
       --validation-stage $R/mis/adaptive/tsnpe_adaptive_stage2.npz \
       --output-dir $R/mis/optimized_hybrid_120k --candidate-rows 120000 --seed <not known>
   python $T/split_clean_tsnpe_mis_candidates.py --input $R/mis/optimized_hybrid_120k/dual_stage_new_candidates.npz \
       --output-dir $R/mis/optimized_hybrid_120k/parts --parts 2 --part-counts 42000 78000
   python $T/evaluate_dual_stage_flow_shard_9d.py --candidate $R/mis/optimized_hybrid_120k/parts/tsnpe_mis_candidates_part_00.npz \
       --data-root data/observations --output $R/mis/optimized_hybrid_120k/parts/opt_hybrid120k_part00_exact.npz --workers 60
   python $T/evaluate_dual_stage_flow_shard_9d.py --candidate $R/mis/optimized_hybrid_120k/parts/tsnpe_mis_candidates_part_01.npz \
       --data-root data/observations --output $R/mis/optimized_hybrid_120k/parts/opt_hybrid120k_part01_exact.npz --workers 60
   ```

   Combined with the rows of Step 5.2, these rows give 195,000 rows with ESS 237.6; this combination is not an
   input of any later step, so its script is not included.

### Step 6: held-out cross-validated mixture (CPU)

```bash
python $T/prepare_crossvalidated_tsnpe_stages_9d.py \
    --old-exact-stage $R/mis/flow_ensemble3/tsnpe_hyperonic_clean_frozen_adaptive_flow_mis.npz \
    --old-cross-density $R/mis/optimized_hybrid_120k/old_exact_cross_density.npz \
    --new-candidate $R/mis/optimized_hybrid_120k/dual_stage_new_candidates.npz \
    --new-exact $R/mis/optimized_hybrid_120k/parts/opt_hybrid120k_part0{0,1}_exact.npz \
    --output-dir $R/mis/crossfit_partition --seed 2026095201
python $T/freeze_crossvalidated_gmm_bridge_9d.py --train $R/mis/crossfit_partition/crossfit_training_stage.npz \
    --validation $R/mis/crossfit_partition/crossfit_validation_stage_{1,2}.npz \
    --output $R/mis/crossfit_gmm_bridge.npz --seed 2026095301
```

- The 120,000 rows of Step 5.4 are split per mixture component into disjoint 60/20/20 parts. Training = the
  75,000 rows of Step 5.2 + 72,000 rows of Step 5.4 (ESS 174.8); folds of 24,000 rows (ESS 36.9 and 85.2). The
  folds never enter the fit.
- The cross-validated mixture (code defaults, listed in `route2/fixtures/tsnpe_hyperonic/crossfit_gmm_bridge.json`):
  a 4-component full-covariance Gaussian mixture in bounded-logit coordinates, fitted to 50,000 weighted
  resamples (jitter 0.05, regularization 0.01), plus a broad Student-t (8 degrees of freedom, scale 1.3) and the
  uniform prior, 0.80/0.15/0.05. Both untouched folds forecast the ESS of the 150,000-row final proposal (with a
  10% parent part) as 4,291.6 and 5,340.2, above the required 3,000. Hashes: partition `581e9155…`,
  `60bed10b…`, `87e34547…`; mixture `66948847…`.

### Step 7: final proposal (GPU)

```bash
python $T/build_crossvalidated_hybrid_proposal_9d.py --bridge $R/mis/crossfit_gmm_bridge.npz \
    --parent-hybrid-model $R/mis/optimized_hybrid_120k/optimized_hybrid_model.npz \
    --old-proposal $R/mis/flow_ensemble3/final_adaptive_flow_candidates.npz \
    --old-checkpoints $R/mis/flow_ensemble3/adaptive_flow_0{0,1,2}.pt \
    --new-proposal $R/mis/flow_ensemble4/final_adaptive_flow_candidates.npz \
    --new-checkpoints $R/mis/flow_ensemble4/adaptive_flow_0{0,1,2,3}.pt \
    --output $R/mis/crossvalidated_hybrid_candidates_150k.npz --seed 2026095401
```

- 150,000 rows = 0.9 × (the cross-validated mixture of Step 6: 0.80 Gaussian mixture + 0.15 Student-t + 0.05
  uniform) + 0.1 × (the parent mixture of Step 5.4). Component counts 108,000 / 20,250 / 6,750 / 600 / 1,500 /
  600 / 1,500 / 10,800, including two uniform full-prior parts. Every row stores log q = logsumexp(log 0.9 +
  `logq_repair`, log 0.1 + `logq_parent_tsnpe`), where `logq_repair` is the density of the cross-validated
  mixture. Published run: `82f46920…`, seed 2026095401, RTX 4090, about 14 s after staging.
- **Rebuild from the frozen inputs (tested on a CPU):**

  ```bash
  F=route2/fixtures/tsnpe_hyperonic; D=route2/downloads
  python $T/build_crossvalidated_hybrid_proposal_9d.py --bridge $F/crossfit_gmm_bridge.npz \
      --parent-hybrid-model $F/optimized_hybrid_model.npz \
      --old-proposal $D/tsnpe_hyperonic_old_flow_proposal.npz --old-checkpoints $D/tsnpe_hyperonic_old_flow_0{0,1,2}.pt \
      --new-proposal $D/tsnpe_hyperonic_new_flow_proposal.npz --new-checkpoints $D/tsnpe_hyperonic_new_flow_0{0,1,2,3}.pt \
      --output runs/tsnpe_hyperonic_rebuild/candidates_150k.npz --seed 2026095401 --device cpu
  ```

  On the test machine (CPU, 3 threads, 18 s) the component labels, counts and row order equal the published
  file; the 17,550 uniform rows are bit-identical, the 131,250 Gaussian-mixture and Student-t rows agree to
  2e-14 of the prior width (last-digit linear-algebra differences), and `logq_repair` agrees to 4e-12. The
  1,200 flow-drawn rows differ, because the torch random numbers of a CPU differ from those of the GPU. This is
  a smoke test of the code, not a replacement of the frozen file.

### Step 8: exact evaluation of the final proposal (Exact)

```bash
python $T/split_clean_tsnpe_mis_candidates.py --input $R/mis/crossvalidated_hybrid_candidates_150k.npz \
    --output-dir $R/mis/crossfit_final_parts --parts 3 --part-counts 15000 45000 90000
python $T/evaluate_crossvalidated_hybrid_shard_9d.py --candidate $R/mis/crossfit_final_parts/tsnpe_mis_candidates_part_00.npz \
    --data-root data/observations --output $R/mis/crossfit_final_exact/xfit150k_part00_exact.npz --workers 22
python $T/evaluate_crossvalidated_hybrid_shard_9d.py --candidate $R/mis/crossfit_final_parts/tsnpe_mis_candidates_part_01.npz \
    --data-root data/observations --output $R/mis/crossfit_final_exact/xfit150k_part01_exact.npz --workers 60
python $T/evaluate_crossvalidated_hybrid_shard_9d.py --candidate $R/mis/crossfit_final_parts/tsnpe_mis_candidates_part_02.npz \
    --data-root data/observations --output $R/mis/crossfit_final_exact/xfit150k_part02_exact.npz --workers 60
```

- Part 00 ran on a desktop workstation (22 workers, 406.4 s, 12,878 valid rows), part 01 as a cluster job on a
  cluster node (60 workers, 1,071.7 s, 38,593 valid), part 02 on a second cluster node (60 workers, 1,081.3 s,
  77,236 valid), all with one thread per worker. Target certificate: PASS, 51/51 valid rows, largest error 3.0e-9.

### Step 9: final MIS (CPU, seconds)

```bash
python $T/combine_crossvalidated_hybrid_is_9d.py --candidate $R/mis/crossvalidated_hybrid_candidates_150k.npz \
    --exact $R/mis/crossfit_final_exact/xfit150k_part0{0,1,2}_exact.npz \
    --output $R/mis/tsnpe_hyperonic_clean_crossvalidated_9d_mis.npz \
    --minimum-ess 1500 --posterior-rows 20000 --resample-seed 2026095501
```

With N = 150,000 (every row counts, also rows with zero weight) and log prior = −Σ log(high − low):

| Quantity | Formula | Published value |
|---|---|---|
| log weight | log w_i = log L_i + log prior − log q_i (−∞ for rejected rows) | — |
| normalized weight | w_i = exp(log w_i − max) / Σ_j exp(log w_j − max) | largest 0.013451868370136335 |
| ESS | 1 / Σ w_i² | 1,513.7694295249498 |
| log Z | max + log[(1/N) Σ exp(log w_i − max)] | −28.709144581496847 |
| error of log Z | std(exp(log w − max), ddof 1) / √N / mean(exp(log w − max)) | 0.025572256042419722 |

The command requires ESS ≥ 1,500 (`--minimum-ess 1500`); the published ESS meets it. The 20,000-row posterior
is a weighted resample with seed 2026095501.

### Step 10: curves and comparison figure

> **The curve command below cannot run from this repository.** `fast_curve_full_shard.py` imports the helper
> module `fast_newton_hyperon_audit`, which is not part of this package (the same gap as hyperonic A-NET
> Steps 12–13; see Gaps, item 9). The resample command before it runs.

```bash
python $T/prepare_frozen_tsnpe_curve_resample.py --posterior $R/mis/tsnpe_hyperonic_clean_crossvalidated_9d_mis.npz \
    --output $R/mis/crossfit_curve_replay/tsnpe_curve_resample6000.npz              # defaults: --draws 6000 --seed 20260921
CERTIFIED_DDB_DIR=legacy_stack/validated_code_DDB CERTIFIED_ASTRO_DIR=legacy_stack/ddb_astro_mod \
python hyperonic_pipeline/fast_curve_full_shard.py --input $R/mis/crossfit_curve_replay/tsnpe_curve_resample6000.npz \
    --output $R/mis/crossfit_curve_replay/tsnpe_curve_resample6000_curves.npz --workers 62
```

- The resample draws 6,000 rows from the weights (4,015 unique) and stores each unique row with its count.
  The published curves came from the curve command above, run as a cluster job (62 workers, one thread per
  worker): 49.1 s for 4,015 rows.
- `render_frozen_tsnpe_mass_radius_review.py` draws the comparison figure. It adds the published UltraNest and
  A-NET bands (`hyperonic_mass_radius_tail_support.npz`, `11b6778f…`) for comparison only, after the TSNPE
  posterior and curves were frozen. It reads the plotting package of the paper repository from a fixed path.
  Result: over 0.5–2.1146 M☉ the 90% band edges differ from UltraNest by 0.0192 km (median), 0.0427 km (95th
  percentile) and 0.0848 km (largest). Route 1 ships its data as `results/hyperonic/A1/tsnpe_comparison_bands.npz`.

## 4. Settings, sizes, times, hardware and cost

| Item | Value |
|---|---|
| Prior | canonical 9-D box of the hyperonic model (the full prior; every proposal covers it through uniform parts) |
| Target | exact A1 likelihood (nuclear, maximum mass, NICER, GW170817, pQCD), portable target code |
| Round sizes | round 0: 20,000 pilot + 300,000; rounds 1 and 2: 20,000 each; training rows 947 + 72 + 50 |
| Round estimators | sbi NSF, 10 transforms, 256 hidden units, ≤ 400 epochs |
| Flow ensembles | zuko NSF, 8 transforms, 3 × 192 hidden units, 10 bins; 3 members (Step 5.2) and 4 members (Step 5.3) |
| MIS stage sizes | 45,000; +30,000; 75,000; 120,000; final 150,000 |
| Seeds | bridge 20260923, 20260924, extension 20260925; accept 2026092500–03, 2026092611–12, 2026092621–22; round proposals 2026092601, 2026092603; fits 2026092602, 2026092604, 2026093604; MIS uniform 2026092701, resample 2026092702; calibration 2026092801–02; Step 5.1 2026092901, resample 2026092902; 3-member flows 2026093001, resample 2026093101; 4-member flows 2026094001; partition 2026095201; cross-validated mixture 2026095301; final proposal 2026095401; posterior resample 2026095501; curve resample 20260921. Not known: round-zero fit and parent-mixture draw |
| GPU steps | rented cloud GPUs (NVIDIA RTX 4090): round fits (38.6, 27.4, 29.4 s), round proposals (118.8, 119.8 s), flow ensembles (534.5, 827.7 s), final proposal (about 14 s); the other GPU stage times are not known |
| Exact evaluations | 780,000: 300,000 + 20,000 pilot + 20,000 + 20,000 + 45,000 + 30,000 + 75,000 + 120,000 + 150,000 |
| Exact hardware | two cluster nodes (60–62 workers each, one thread per worker) and a desktop workstation (22 workers), where stated in the steps above |
| Critical path | 92.7 min (5,560.5 s), production-solver parallel-equivalent; six small steps untimed (Route 1 `results/cost/table_VII_VIII_values.json`) |

**Hardware caution for a full rerun.** The 780,000 exact evaluations dominate. The shards of the published run
cost about 0.6–1.5 s per row per core; together they used roughly 200 core-hours (shard wall times × workers,
with 60 workers where the count is not known). With 120 cores that is about 1.5–2 hours;
on an 8-core desktop about a day. The timed GPU steps add at least 30 minutes on an RTX 4090 (much longer on
a CPU). A rerun reproduces the result only within its Monte Carlo errors: the round-zero estimator is not
available, two seeds are not known, and GPU random numbers and float32 arithmetic differ between machines.

## 5. The check

```bash
python -m route2.checks.tsnpe_hyperonic          # add --device cuda to use a GPU for the flow densities
```

It fetches the 13 frozen files (84.6 MB) if needed, runs in about 40 s on a CPU (3 threads, 1.1 GB memory) and
writes `build/route2/tsnpe_hyperonic_check.json`. It prints one line per step and a final `PASS` or `FAIL`.
`./reproduce.sh route2-check` runs it together with the other Route 2 checks.

| Step | What is replayed | Tolerance and why | Result on the test machine |
|---|---|---|---|
| Inputs | hash chain posterior → candidate → cross-validated and parent mixtures → both flow proposals → 7 network files; the imported TSNPE code equals `CODE_MANIFEST.json` | exact | chain closes |
| Weights | weights, ESS, log Z, its error and the largest weight from the stored log L and log q, with the formula of Step 9; compared with the published arrays and report | 1e-12 (float64; relative for the ESS) | difference 0 |
| Resamples | the 20,000-row posterior index (seed 2026095501) and the 6,000-draw curve resample (seed 20260921) | identical; Route 1's `tsnpe_resample6000.npz` must be the published file byte for byte | identical |
| Proposal density | 14 rows (the 4 largest weights, one accepted row of each of the 8 components spread over the three exact shards, 2 rejected rows; seed 20260925): `logq_repair` from the cross-validated mixture with the code of the Step 7 builder; `logq_parent_tsnpe` from the parent mixture with both flow ensembles loaded by the production `load_proposal`; their 0.9/0.1 mixture against the candidate and the posterior. Each ensemble also re-evaluates 10 of its own stored rows | `logq_repair` 1e-12 (float64). Flow parts 2e-3: the flows are float32 networks that the published run evaluated on an RTX 4090; re-evaluating 40,000 stored rows on a CPU differs by 3e-5 (median) and at most 9e-4 after weighting by the flow's share. A 2e-3 change in log q moves a weight by 0.2%, far below the log Z error | `logq_repair` 0; `logq_parent` 3.5e-5; log q 3.5e-5; self-replay 2.2e-4 |
| Exact log L | the A1 target is rebuilt from `data/observations`; the 14 rows are evaluated with the production evaluator (`fast_portable_hyperonic_likelihood`, as the hyperonic A-NET check) and with the portable target that computed the saved values; the frozen 64-row certificate is re-run; the nuclear term (no EOS/TOV solve needed) is recomputed on all 128,707 accepted rows | 1e-4, the pipeline's certification tolerance; accepted and rejected rows must agree | 12 accepted + 2 rejected agree; production vs saved 5.8e-7; portable vs saved 5.8e-7; production vs portable 5.1e-11; certificate PASS (5.8e-11); nuclear term on all rows 3.0e-5 |

**The two EOS implementations.** On the test machine the production evaluator and the portable target (a SciPy
port) give the same nuclear observables and agree to 5.1e-11 in log L (astrophysical terms only). Both differ
from the saved values by the same amount, only in the nuclear term. The nuclear observables come from a batched
nuclear-matter computation (JAX) whose result depends slightly on the array shape it is compiled for and on the
machine: on the test machine, the same rows computed in batches of 32 and of 2,500 differ by up to 7e-7
(relative), while the other rows in a batch have no effect. Recomputed in the production batches of 2,500 rows,
the nuclear term of all 12,878 accepted rows of part 00 (desktop workstation) equals the saved value exactly. For
the rows of parts 01 and 02 (cluster nodes), 90% differ: median 1.7e-7, largest 3.0e-5 (no row above 1e-4),
2.1e-7 when averaged with the posterior weights. So the offset is a machine and batch-shape effect, not an EOS
difference; it is at least 3 times below the tolerance everywhere and negligible for log Z.

The inputs of the check are built by `assemble_route2_tsnpe_hyperonic.py`, a build-time script kept outside the
repository. Before writing anything it verifies the files of the published run against their SHA-256 list,
closes the hash chain, and checks on all 150,000 rows that the candidate, the three exact shards and the
posterior agree, that the stored log q is exactly the 0.9/0.1 mixture of its parts, and that the weights,
summaries and both resamples replay. `route2/fixtures/tsnpe_hyperonic/MANIFEST.json` gives the SHA-256 of each
fixture.

## 6. Gaps

1. **Round-zero estimator.** The estimator fitted to the 947 round-zero rows (`283844cb…`), which generated the
   round-1 proposal, is not available and its seed is not known. Round 1 can be regenerated only
   statistically; its proposal and exact rows are frozen by hash in the round-1 reports.
2. **Flow density calibration.** The script that normalized the two round-2 flows to the prior box
   (`flow_density_calibration.json`, 1,000,000 draws per flow) is not available; only its output values are
   known (Step 4).
3. **Parent-mixture draw seed.** The seed of the 120,000-row draw of Step 5.4 is not known. The rows are
   frozen in its exact shards and in the partition of Step 6, so the result does not depend on it.
4. **Code identity.** The 7 scripts of Steps 6–10 are byte-identical to the versions that ran. Of the other
   22, 6 have the same SHA-256 as the versions that ran. For `combine_tsnpe_round0_exact_shards.py`, the SHA-256 of the
   version that ran is not known. The remaining scripts have no SHA-256 from their runs, so identity with the
   versions that ran is likely but not proven.
5. **Run environment.** The runtime code copies used by the cluster and desktop jobs and the Python 3.12
   environment of the cloud GPU have no SHA-256. The repository's target passes the frozen certificate, replays
   the saved log L of the 14 rows to 5.8e-7, reproduces the nuclear term of all desktop rows exactly and that of
   the cluster rows to at most 3.0e-5 (Section 5). A rerun on other hardware will differ at this level.
6. **Hosts.** The machines that ran the round 0–2 shards and the Step 5.1–5.2 shards are not known (their
   commands use 60 workers).
7. **Not included:** the script of the 195,000-row combination of Step 5.4, other diagnostic scripts, and
   further files that are not inputs of any step above.
8. **Downloads.** The six shared prior-generated physics caches of Step 1 (uniform-prior draws) are in the
   optional full-rerun download ([FULL_RERUN_FILES.md](FULL_RERUN_FILES.md)). The intermediate stage files of
   Steps 2–6 are not downloadable. The partition, the three final exact shards and the curves of the published
   run are listed with their hashes in Section 1; the check does not need them.
9. **Curve code.** The copy of `fast_curve_full_shard.py` that computed the published curves has no known
   SHA-256; the repository ships the production copy from the hyperonic pipeline. That copy imports the helper
   module `fast_newton_hyperon_audit`, which is not part of this package, so the curve command of Step 10 cannot
   run from this repository.
