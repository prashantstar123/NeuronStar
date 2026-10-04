# Hyperonic A-NET: how to rerun it from scratch

This page lists, step by step, how the hyperonic (DDB ΛΞ⁻, nine parameters) A-NET+IS results of the paper were
made: the Stage-A bridge that feeds the defensive proposal, the training caches, the scenario bank, the two
network heads, the A1 dual-head proposal with its exact importance sampling and certificate, and the seven
queries (four nuclear shifts and three held-out NICER sources). For each step it gives the command for a rerun
with the code in this repository, the settings, the seeds, the run time and machine where they are known, and
what is missing.

The commands use the package code with the settings listed; they are rebuilt from the programs' options and the
settings saved with the paper's files (for example, the scenario bank lists its nine caches and the counting
correction by hash, and every exact-likelihood shard stores its host, worker count and run time). Where a setting
of the paper's run is not known, this is said in the text.

**A full rerun is long.** It needs about 2.8 million exact nine-dimensional likelihood evaluations (Stage A alone
has 1.72 million), 600,000 uniform-bank EOS/TOV solves, and GPU training of two networks for 600 epochs. To see
that the code runs, use the smoke run; to check the published numbers, use the replay check (last section).
Neither needs a full rerun.

## Before you start

1. Use the Route 2 Python environment (Python 3.11, numpy 2.4.6, scipy 1.17.1, torch 2.5.1, jax 0.10.2,
   numba 0.65.1, joblib, scikit-learn). The frozen defensive Student checkpoint was pickled with scikit-learn
   1.9.1. Version 1.9.0 loads it with a version warning, and the replay check reproduces its density to 1e-14.
2. Work from the repository root with
   `PYTHONPATH=$PWD:$PWD/hyperonic_pipeline`, `CERTIFIED_DDB_DIR=$PWD/legacy_stack/validated_code_DDB`,
   `CERTIFIED_ASTRO_DIR=$PWD/legacy_stack/ddb_astro_mod` and `JAX_PLATFORMS=cpu`. The helper
   `route2.common.use_hyperonic_pipeline()` sets the same values from Python.
3. Make sure the observation files are unpacked in `data/observations/` (`./reproduce.sh check` does it).
4. The programs are in `hyperonic_pipeline/` (`hyperonic_pipeline/CODE_MANIFEST.json` gives each file's
   SHA-256; a few files differ from the code used for the paper only in a comment or a module name). The frozen
   heads and the defensive Student are in `route2/fixtures/hyperonic_anet/networks/`.
5. Machines of the paper's runs: two cluster nodes, called node 1 and node 2 below (62 workers per job), a
   desktop workstation (Intel i7-13700F), and rented cloud GPUs (an NVIDIA H100 NVL for the scenario bank and the
   nine-dimensional head, an RTX 4090 for the seven-dimensional head, and an H100 80 GB for the A1 proposal).
6. The large files named in the steps below (the Stage-A MIS file, the uniform banks, the nine training
   caches, the support screen, the Student draws, the counting correction and the scenario bank) are in the
   optional full-rerun download (`./reproduce.sh route2-fetch --group full_rerun`), so a rerun can start at any
   step. Their downloaded names start with `full_`; [FULL_RERUN_FILES.md](FULL_RERUN_FILES.md) gives both names.

## Step 1. The exact target and its evaluator

- Target convention: NICER likelihood on a 150 × 150 grid from a 20,000-row KDE subsample, bandwidth 0.08,
  seed 0; plus the nuclear, GW170817, maximum-mass and pQCD terms.
- Evaluator: `fast_portable_hyperonic_likelihood.py` (SHA-256 `e14beeb2…`) with the production EOS
  `ddb_hyperon_eos.py` (`0f4f7f98…` in production; the repository copy differs in one comment only).
- Target caches (SHA-256 as stored in the exact-likelihood shards): A1 `54e7bbfc…`, K0_200 `5709d968…`,
  K0_260 `4c7ca78d…`, Jsym_29 `cbf60c49…`, Jsym_36 `9934b7bc…`, J0614 `a3a2d826…`, J1231 `d20c64a4…`,
  J1614 `89c893d2…`.

Commands:

```bash
python hyperonic_pipeline/build_portable_hyperonic_target_cache.py --data-root data/observations \
    --output targets/A1.joblib
python hyperonic_pipeline/build_portable_hyperonic_target_cache.py --data-root data/observations \
    --base-cache targets/A1.joblib --nuclear-scenario K0_200 --output targets/K0_200.joblib   # K0_260, Jsym_29, Jsym_36
python hyperonic_pipeline/build_portable_hyperonic_target_cache.py --data-root data/observations \
    --source-data-root data/observations --source-scenario J0614 --output targets/J0614.joblib   # J1231, J1614
```

Each cache of the paper's runs was checked with `certify_fast_portable_target.py` before use. A rebuilt cache
holds the same target but need not be byte-identical, because the file also stores the absolute input paths.

## Step 2. Stage A: the tempered bridge (1.72 million exact rows)

**What it is.** A sequence of tempered, defensive importance-sampling rounds with fresh exact likelihoods. Its
only use in A-NET is as the input of the defensive Student fit (Step 5). The final product is one
deterministic-mixture (MIS) combination of eight rounds.

**The eight rounds** (listed in the MIS report `rounds/eval_mis_round8_round15_1720k.json`, SHA-256
`5cfc7da0…`). Each round's mixture fraction is its rows / 1,720,000.

| Round | Proposal (fit seed) | Draw seed(s) | Rows | β (temperature) |
|---|---|---|---|---|
| r8 | GMM, 32 components (20260951) | 20260952 and 20260956 (2 × 140,000) | 280,000 | 0.03183 |
| r9 | GMM, 32 components (20260958) | 20260959 | 200,000 | 0.03972 |
| r10 | GMM ensemble, 32 components (20260962) | 20260963 | 200,000 | 0.08207 |
| r11 | GMM ensemble, 32 components (20260964) | 20260965 | 220,000 | 0.11756 |
| r12 | GMM ensemble, 32 components (20260966) | 20260967 | 220,000 | 0.13879 |
| r13 | GMM ensemble, 48 components (20260970) | 20260971 | 200,000 | 0.16094 |
| r14 | GMM ensemble, 48 components (20260972) | 20260973 | 200,000 | 0.19539 |
| r15 | Student-t ensemble, df 8 (20260977) | 20260978 | 200,000 | 0.22088 |

**Result.** `rounds/eval_mis_round8_round15_1720k.npz` (SHA-256 `7740c7cd…`): 1,720,000 rows, 1,245,208 valid;
full-posterior (β = 1) ESS 3,282.44 against the required floor of 2,266; ESS at the last temperature 10,232.86;
density replay error 7.1e-15 (tolerance 1e-10).

**Programs.** `clean_gmm.py` (rounds 8–9), `clean_gmm_ensemble.py` (10–14) and `clean_student_t_ensemble.py` (15)
fit and draw each proposal (`fit`, then `sample`); `clean_flow.py suggest` proposes the next temperature;
`split_clean_proposal.py` → `fast_hyperonic_likelihood_shard.py` (cluster) → `assemble_likelihood_shards.py` →
`merge_exact_evaluation.py` produce each round's exact evaluation; `combine_clean_evaluations.py` joins the two
round-8 draws.

**Final combination.**

```bash
python hyperonic_pipeline/combine_mis_evaluations.py \
    --component rounds/eval_round8_combined_280k.npz checkpoints/gmm_round8_32comp_def25_seed_20260951.joblib \
    --component rounds/eval_round9_gmm32_seed_20260959.npz checkpoints/gmm_round9_32comp_def25_seed_20260958.joblib \
    --component rounds/eval_round10_ensemble_seed_20260963.npz checkpoints/gmm_ensemble_round10_32comp_seed_20260962.joblib \
    --component rounds/eval_round11_ensemble_seed_20260965.npz checkpoints/gmm_ensemble_round11_32comp_seed_20260964.joblib \
    --component rounds/eval_round12_ensemble_seed_20260967.npz checkpoints/gmm_ensemble_round12_32comp_seed_20260966.joblib \
    --component rounds/eval_round13_ensemble_seed_20260971.npz checkpoints/gmm_ensemble_round13_48comp_seed_20260970.joblib \
    --component rounds/eval_round14_ensemble_seed_20260973.npz checkpoints/gmm_ensemble_round14_48comp_seed_20260972.joblib \
    --component rounds/eval_round15_student_seed_20260978.npz checkpoints/student_ensemble_round15_conservative_df8_seed_20260977.joblib \
    --target-beta 0.2208750108267128 --output rounds/eval_mis_round8_round15_1720k.npz
```

Rounds 1–7 and a first uniform draw precede r8 in the sequence; they are not components of the final mixture.
Stage A was evaluated with `fast_hyperonic_likelihood_shard.py` on the full target without the NICER single-weight
correction. Stage A only shapes the defensive proposal, whose density enters the final weights exactly, so this
affects efficiency, not the final weights.

**Known defect of the stored Stage-A bank (no effect on any result).** In rounds r9–r14 the shard evaluated on
one desktop ran its EOS/TOV computations, but its likelihood-data step failed, so 291,082 of its rows are stored
as rejected although they have a finite (very low) likelihood. The Stage-A mixture, its stored ESS (3,282.4)
and the Student fit therefore saw those rows with zero weight. Because the Student is only a defensive proposal
component whose exact density enters the final A-NET weights, and every final row is weighted with the exact
likelihood, no posterior, evidence, figure or table depends on this; the `hyperonic_anet` check replays all eight
results. Details and the replay are in `STEPS_BANKS.md` (Section 3.7) and the `banks` check.

## Step 3. Uniform banks and uniform training caches

**Uniform banks** (settings stored in the bank files, which the caches name by SHA-256). Four banks of 150,000
uniform-prior rows, seeds 91, 92, 93 and 94, each made with 56 workers in 3,352–3,374 s (the machine is not
known); 103,707, 103,947, 104,094 and 103,933 rows have a valid EOS and TOV branch.

```bash
python hyperonic_pipeline/generate_ddbhy_bank.py --n 150000 --seed 91 --workers 56 --output hyp_bank_part1.npz
# parts 2-4: seeds 92, 93, 94
```

**Uniform training caches.** Every forward-valid bank row gets the exact target factors and fresh curves:

| Part | Rows | SHA-256 | Host | Workers | Run time |
|---|---|---|---|---|---|
| 1 | 103,707 | `e51cf2f1…` | node 1 | 62 | 2,406 s |
| 2 | 103,947 | `e7ac3193…` | node 2 | 62 | 1,243 s |
| 3 | 104,094 | `e13a8566…` | node 2 | 62 | 1,244 s |
| 4 | 103,933 | `575ebf36…` | node 1 | 62 | 2,412 s |

```bash
python hyperonic_pipeline/build_clean_uniform_hyperonic_training_cache.py --bank hyp_bank_part1.npz \
    --data-root data/observations --output uniform_training_cache_part1.npz --workers 62
```

## Step 4. Support screen and support caches

**Screen.** 160,000,000 uniform proposals in blocks of 2,000,000, seed 20260729, kept when all seven nuclear
observables lie within 3 σ of A1 (`cbox 3`): 83,764 rows (`support_screen_160m.npz`, `442da1a4…`), 827 s on the
desktop workstation.

```bash
python hyperonic_pipeline/screen_clean_hyperonic_support_enrichment.py --output support_screen_160m.npz \
    --proposals 160000000 --block 2000000
```

**Support caches.** Rows 0–41,881 on the desktop workstation (20 workers, 1,104 s; `479206d0…`) and rows
41,882–83,763 on node 2 (62 workers, 489 s; `28e0894d…`). Both use bank part 1 as the template (mass grid and
prior box).

```bash
python hyperonic_pipeline/evaluate_clean_hyperonic_support_cache.py --screen support_screen_160m.npz \
    --template-bank hyp_bank_part1.npz --data-root data/observations \
    --output support_training_cache_part1.npz --workers 20 --start 0 --stop 41882
# part 2: --workers 62 --start 41882 --stop 83764
```

## Step 5. The defensive Student proposal

**Fit** (settings stored in the checkpoint `student_beta1_48c_df5_s075_seed20260990.joblib`, SHA-256
`f5f4e11b…`). Input: the Stage-A MIS bank (`7740c7cd…`, ESS 3,282.4) at β = 1. A 48-component multivariate
Student-t mixture (5 degrees of freedom, shape scale 0.75), with weight 0.90 on the learned mixture and 0.10 on
the uniform prior; 500,000 training draws; 5-fold cross-validation with 300,000 draws (predicted ESS fraction
0.034–0.049 across the folds, median 0.038); regularization 1e-3; jitter 0.025; at most 300 iterations; seed
20260990; 856 s on a rented cloud CPU machine.

```bash
python hyperonic_pipeline/clean_student_t_ensemble.py --prior hyperonic_pipeline/prior_manifest.json fit \
    --input eval_mis_round8_round15_1720k.npz \
    --checkpoint student_beta1_48c_df5_s075_seed20260990.joblib \
    --beta 1.0 --learned-weight 0.90 --prior-weight 0.10 --components 48 --degrees-of-freedom 5 \
    --shape-scale 0.75 --training-draws 500000 --cross-validation-draws 300000 --cross-validation-folds 5 \
    --regularization 0.001 --jitter 0.025 --minimum-input-ess 2266 --minimum-cv-ess-fraction 0.02 \
    --source-weight-grid 0.1 --source-weight-grid 0.25 --source-weight-grid 0.5 \
    --maximum-iterations 300 --seed 20260990
```

`--minimum-cv-ess-fraction 0.02` is the value used for the paper (the default 0.10 would reject the
cross-validation fractions above), and 2,266 is the Stage-A ESS floor of Step 2.

**Draws.** 300,000 draws with seed 20260992 (`student_beta1_48c_300k_seed20260992.npz`, `2b2e7827…`):

```bash
python hyperonic_pipeline/clean_student_t_ensemble.py sample \
    --checkpoint student_beta1_48c_df5_s075_seed20260990.joblib \
    --output student_beta1_48c_300k_seed20260992.npz --draws 300000 --seed 20260992
```

## Step 6. Student proposal caches

Rows 0–124,999 on node 1 (62 workers, 2,867 s; `5a48a887…`), rows 125,000–249,999 on node 2 (62 workers,
1,474 s; `118acc32…`), rows 250,000–299,999 on the desktop workstation (20 workers, 1,339 s; `d1d80765…`).
Each cache also replays the Student density of its rows (tolerance 1e-10). The template bank of these caches is
the first uniform bank (`hyp_bank_part1.npz`, `c188c4d0…`, named in the `.json` report of each cache); it only
supplies the mass grid and the prior box, which are the same in all four banks.

```bash
python hyperonic_pipeline/evaluate_clean_hyperonic_proposal_cache.py \
    --proposal student_beta1_48c_300k_seed20260992.npz --checkpoint student_beta1_48c_df5_s075_seed20260990.joblib \
    --template-bank hyp_bank_part1.npz --data-root data/observations \
    --output proposal_training_cache_part1.npz --workers 62 --start 0 --stop 125000
# part 2: --start 125000 --stop 250000; part 3: --workers 20 --start 250000 --stop 300000
```

## Step 7. Exact counting correction

**Result** (`correction/augmented_counting_correction_300k.npz`, SHA-256 `a3a6f029…`). The nine caches
(799,445 rows) are one deterministic mixture of three streams: 600,000 uniform proposals (the four banks), the
160,000,000-proposal support screen, and the 300,000 Student draws. Conditional ESS at A1: 15,565 (at least
2,000 required); largest normalized weight 0.0025; Student density replay error 4.8e-13.

```bash
python hyperonic_pipeline/build_clean_hyperonic_augmented_correction.py \
    --base-cache uniform_training_cache_part1.npz --base-cache uniform_training_cache_part2.npz \
    --base-cache uniform_training_cache_part3.npz --base-cache uniform_training_cache_part4.npz \
    --support-cache support_training_cache_part1.npz --support-cache support_training_cache_part2.npz \
    --proposal-cache proposal_training_cache_part1.npz --proposal-cache proposal_training_cache_part2.npz \
    --proposal-cache proposal_training_cache_part3.npz \
    --proposal-checkpoint student_beta1_48c_df5_s075_seed20260990.joblib \
    --base-proposals 600000 --conditional-ess-gate 2000 --output augmented_counting_correction_300k.npz
```

The paper's run took 5.5 s.

## Step 8. The scenario bank

**Result** (`causal_scenarios_2000_seed9.npz`, SHA-256 `e942cb56…`): seed 9; 2,000 scenarios, of which 1,900 for
training and 100 for validation; 15 identity scenarios; identity, one-slot and all-slot fractions 0.25, 0.50 and
0.25; 201 to 6,000 posterior rows per scenario, 9,650,537 in total; built from the nine caches (799,445 rows,
568,420 target-valid) and the correction above. Only the three A1 NICER sources (J0030, J0740, J0437) appear;
J0614, J1231 and J1614 are absent from training.

```bash
python hyperonic_pipeline/build_clean_hyperonic_parity_scenarios.py \
    --cache uniform_training_cache_part1.npz --cache uniform_training_cache_part2.npz \
    --cache uniform_training_cache_part3.npz --cache uniform_training_cache_part4.npz \
    --cache support_training_cache_part1.npz --cache support_training_cache_part2.npz \
    --cache proposal_training_cache_part1.npz --cache proposal_training_cache_part2.npz \
    --cache proposal_training_cache_part3.npz \
    --counting-correction augmented_counting_correction_300k.npz --data-root data/observations \
    --output causal_scenarios_2000_seed9.npz --scenarios 2000 --cloud-points 256 \
    --validation-scenarios 100 --maximum-posterior-rows 6000 --minimum-bank-ess 250 --seed 9 --device cuda
```

The caches must be given in this order (the correction stores their hashes in order). `--identity-scenarios` is
left at its default, 15, as in the paper's bank; the nucleonic scenario bank has 200. The paper's build took
211 s on a rented cloud GPU (NVIDIA H100 NVL).

## Step 9. The two network heads

Both heads are trained on the same scenario bank with the same settings (stored in their training reports):
seed 2, 600 epochs, 1,900 training scenarios, 9,650,537 posterior rows, the four uniform caches for the nuclear
standardization, CUDA.

| Head | Learns | Checkpoint | Standardization | Run time | Best validation loss | GPU |
|---|---|---|---|---|---|---|
| head9 | all 9 parameters | `5f9eb01d…` | `d8d1d79b…` | 2,081.7 s | 0.64821 | H100 NVL |
| head7 | the 7 DDB parameters; the 2 hyperon couplings stay uniform | `7233139b…` | `aa0529c7…` | 3,806.1 s | 0.49744 | RTX 4090 |

```bash
python hyperonic_pipeline/train_clean_hyperonic_parity_anet.py --scenarios causal_scenarios_2000_seed9.npz \
    --base-cache uniform_training_cache_part1.npz --base-cache uniform_training_cache_part2.npz \
    --base-cache uniform_training_cache_part3.npz --base-cache uniform_training_cache_part4.npz \
    --output-dir training/seed2 --epochs 600 --learning-rate 0.001 --scenario-batch 4 --hidden 512 \
    --layers 4 --heun-steps 64 --seed 2 --device cuda
# head7: the same with --learned-dim 7 --output-dir training_factorized7/seed2
```

**Flow-density accuracy.** The fast analytic divergence agrees with autograd to 3.8e-6 (head9) and 5.7e-6
(head7). Comparing 1,024 with 2,048 Heun steps on 128 rows gives a 95th-percentile log q difference of 0.0036
(head9) and 0.0043 (head7); 512 against 1,024 steps gives 0.014 and 0.017. The proposal therefore uses 1,024
steps. The program for this test is `validate_parity_density_convergence.py` (options `--rows`,
`--coarse-steps`, `--fine-steps`).

## Step 10. The A1 dual-head proposal

**Result** (`A1_dual_25k_seed20260944.npz`, SHA-256 `c5227b32…`): 25,000 draws from the normalized mixture
0.35 head9 + 0.55 head7 (with the two hyperon couplings uniform) + 0.10 defensive Student; 1,024 Heun steps;
cloud seed 777001; proposal seed 20260944. Every draw is scored under all three components. Draws: 8,697 head9,
13,836 head7 and 2,467 Student. Time: 3.3 s of drawing and 69.6 s of density evaluation (73.5 s in total) on an
H100 80 GB GPU.

```bash
python hyperonic_pipeline/sample_clean_hyperonic_dual_parity_mixture.py \
    --head9-checkpoint training/seed2/anet_net.pt --head9-standardization training/seed2/anet_std.npz \
    --head9-report training/seed2/run_report.json \
    --head7-checkpoint training_factorized7/seed2/anet_net.pt \
    --head7-standardization training_factorized7/seed2/anet_std.npz \
    --head7-report training_factorized7/seed2/run_report.json \
    --student-checkpoint student_beta1_48c_df5_s075_seed20260990.joblib \
    --data-root data/observations --source-data-root data/observations \
    --source-scenario A1 --nuclear-scenario A1 --output A1_dual_25k_seed20260944.npz \
    --draws 25000 --head9-weight 0.35 --head7-weight 0.55 --student-weight 0.10 \
    --cloud-seed 777001 --proposal-seed 20260944 --flow-steps 1024 --device cuda
```

The frozen heads and Student in `route2/fixtures/hyperonic_anet/networks/` can be passed instead of retrained
ones. Inside the program, the component labels use `default_rng(proposal seed)`, head9 uses the torch generator
seed proposal seed + 10, head7 + 11, and the Student `default_rng(proposal seed + 12)`. The head draws come from
the GPU random generator, so a CPU run draws different (equally valid) samples.

## Step 11. A1 exact importance sampling and certificate

**The paper's run:**

1. Two shards of 12,500 rows each, evaluated with the A1 target cache and 62 workers per shard: one on node 1
   (174.8 s) and one on node 2 (90.4 s), with 180 GB of memory requested for each.
2. Assembled exact file `A1_dual_exact.npz` (`6e79fc4b…`): 25,000 rows, 23,619 valid.
3. Certificate `A1_dual_certificate.npz` (`df04ef1f…`): thresholds ESS ≥ 1,000 and largest normalized weight
   ≤ 0.01; ESS 5,584.65, largest weight 0.00442, log Z = −28.7269 ± 0.0118, so both thresholds are met.

```bash
python hyperonic_pipeline/split_clean_proposal.py --proposal A1_dual_25k_seed20260944.npz \
    --output exact_shards/A1_dual_shard0.npz --output exact_shards/A1_dual_shard1.npz
python hyperonic_pipeline/fast_portable_hyperonic_likelihood.py --input exact_shards/A1_dual_shard0.npz \
    --output exact_shards/A1_dual_shard0_exact.npz --target-cache targets/A1.joblib --workers 62 \
    --nuclear-scenario A1 --source-scenario A1        # and the same for shard 1
python hyperonic_pipeline/assemble_likelihood_shards.py --proposal A1_dual_25k_seed20260944.npz \
    --shard exact_shards/A1_dual_shard0_exact.npz --shard exact_shards/A1_dual_shard1_exact.npz \
    --output exact/A1_dual_exact.npz
python hyperonic_pipeline/certify_clean_anet_scenario.py --proposal A1_dual_25k_seed20260944.npz \
    --likelihood exact/A1_dual_exact.npz --output certified/A1_dual_certificate.npz \
    --minimum-posterior-ess 1000 --maximum-normalized-weight 0.01
```

## Step 12. After certification (A1)

**The paper's run:**

- Split halves: the certified rows are cut into two contiguous halves of 12,500 rows (posterior mass 0.4993 and
  0.5007; ESS 2,437.2 and 3,265.4); each is resampled to 3,000 draws with seed 20260945. Curves: on node 1
  (28.7 s) and node 2 (14.9 s). The 90% band edges of the two halves differ by 0.032 km (median), 0.090 km
  (95th percentile) and 0.122 km (largest), within the thresholds 0.05, 0.10 and 0.25 km. No UltraNest product
  is used in this test.
- Final resample: 6,000 draws (4,339 distinct rows) with seed 20260946 (`63af9c88…`); curves on node 2
  (62 cores, 27.5 s; `fb1acb10…`).
- Comparison with UltraNest, which does not enter the A-NET result (comparison data `0cfbd7ab…`): 90% band
  edges differ by 0.012 km (median), 0.024 km (95th percentile) and 0.099 km (largest) over 0.50–2.10 M☉.

> **The two `fast_curve_full_shard.py` commands below cannot run from this repository.** The script imports the
> helper module `fast_newton_hyperon_audit`, which is not part of this package (see Gaps). The resample, split
> and comparison commands run.

```bash
python hyperonic_pipeline/resample_certified_split_halves.py --certificate certified/A1_dual_certificate.npz \
    --output-prefix stability/A1_dual_split --draws-per-half 3000 --seed 20260945
python hyperonic_pipeline/fast_curve_full_shard.py --input stability/A1_dual_split_half0.npz \
    --output stability/A1_dual_split_half0_curves.npz --workers 62      # and half 1
python hyperonic_pipeline/compare_mass_radius_split_halves.py --paper-root . \
    --resample stability/A1_dual_split_half0.npz --resample stability/A1_dual_split_half1.npz \
    --curves stability/A1_dual_split_half0_curves.npz --curves stability/A1_dual_split_half1_curves.npz \
    --output stability/A1_dual_split_mass_radius_stability.json
python hyperonic_pipeline/resample_certified_posterior.py --certificate certified/A1_dual_certificate.npz \
    --output mass_radius/A1/dual_A1_resample6000.npz --draws 6000 --seed 20260946
python hyperonic_pipeline/fast_curve_full_shard.py --input mass_radius/A1/dual_A1_resample6000.npz \
    --output mass_radius/A1/dual_A1_curves.npz --workers 62
```

## Step 13. The seven queries

The frozen A1 networks and Student are reused without any retraining. Each query gets fresh proposal draws,
its own exact target (only the declared observation changes) and fresh exact importance weights.

**Proposals** (settings stored in the proposal files, which the certificates name by SHA-256). The A1 command of
Step 10 with `--nuclear-scenario <case>` (nuclear shifts) or `--source-scenario <case>` (held-out sources), 25,000
draws, cloud seed 777001, 1,024 steps, and these proposal seeds:

| Query | Proposal seed(s) | Rows | ESS | Largest weight | log Z | Resample seed | Certificate |
|---|---|---|---|---|---|---|---|
| K0 = 200 MeV | 2026092101 | 25,000 | 3,737.4 | 0.00914 | −29.1359 ± 0.0151 | 2026092301 | `1bd85fbd…` |
| K0 = 260 MeV | 2026092102 | 25,000 | 4,287.5 | 0.00832 | −28.5827 ± 0.0139 | 2026092302 | `64f3087f…` |
| Jsym = 29 MeV | 2026092103 + 2026092203 | 50,000 | 6,420.3 | 0.00625 | −28.6952 ± 0.0117 | 2026092303 | `13df6474…` |
| Jsym = 36 MeV | 2026092104 | 25,000 | 3,773.9 | 0.00955 | −29.0293 ± 0.0150 | 2026092304 | `fcde44be…` |
| J0614 | 2026092105 | 25,000 | 2,238.8 | 0.00433 | −31.3712 ± 0.0202 | 2026092305 | `d9848af7…` |
| J1231 | 2026092106 + 2026092206 | 50,000 | 6,707.8 | 0.00829 | −29.2254 ± 0.0114 | 2026092306 | `b9f82bef…` |
| J1614 | 2026092107 | 25,000 | 5,235.6 | 0.00205 | −28.7619 ± 0.0123 | 2026092307 | `f8925993…` |

Jsym = 29 and J1231 each have a second batch of 25,000 draws whose seed is the first seed + 100; their
certificates combine both batches. All seven meet the thresholds ESS ≥ 1,000 and largest weight ≤ 0.01. J0614
replaces J0437, J1231 replaces J0030 and J1614 replaces J0740.

**Exact likelihood** (run data stored in the shards). Every batch is split into two shards of 12,500 rows with
`split_locked_dual_query_proposal.py` (it keeps the query's scenario labels, which the evaluator checks), and each
shard ran with 62 workers: about 172–176 s on node 1 and 90–95 s on node 2 per shard.

```bash
python hyperonic_pipeline/split_locked_dual_query_proposal.py --proposal K0_200_dual_25k_seed2026092101.npz \
    --output exact_shards/K0_200_shard0.npz --output exact_shards/K0_200_shard1.npz
python hyperonic_pipeline/fast_portable_hyperonic_likelihood.py --input exact_shards/K0_200_shard0.npz \
    --output exact_shards/K0_200_shard0_exact.npz --target-cache targets/K0_200.joblib --workers 62 \
    --nuclear-scenario K0_200 --source-scenario A1                     # and shard 1
python hyperonic_pipeline/assemble_likelihood_shards.py --proposal K0_200_dual_25k_seed2026092101.npz \
    --shard exact_shards/K0_200_shard0_exact.npz --shard exact_shards/K0_200_shard1_exact.npz \
    --output exact/K0_200_exact.npz
python hyperonic_pipeline/certify_clean_anet_scenario.py --proposal K0_200_dual_25k_seed2026092101.npz \
    --likelihood exact/K0_200_exact.npz --output certified/K0_200_certificate.npz \
    --minimum-posterior-ess 1000 --maximum-normalized-weight 0.01
# two-batch queries: certify_clean_anet_scenario_batches.py --batch <proposal 1> <exact 1> \
#     --batch <proposal 2> <exact 2> --output certified/Jsym_29_combined_certificate.npz \
#     --minimum-posterior-ess 1000 --maximum-normalized-weight 0.01
python hyperonic_pipeline/resample_certified_posterior.py --certificate certified/K0_200_certificate.npz \
    --output mass_radius/K0_200/K0_200_resample6000.npz --draws 6000 --seed 2026092301
```

The mass–radius curves of each 6,000-draw resample are made with `fast_curve_full_shard.py`, as for A1; that
command cannot run from this repository (see Gaps). The query proposals were drawn on a rented cloud GPU (an
H100); their draw times are not available.

## Checks and smoke run

Replay check (no rerun; about 2–10 minutes on CPU):

```bash
PYTHONPATH=$PWD python -m route2.checks.hyperonic_anet
```

For A1 and the seven queries it recomputes the certificate (weights, ESS, largest weight, log Z) from the saved
log q and log L, redraws the 6,000-draw resample with its seed, redraws the clouds, component labels and first
Student draws from the saved seeds, recomputes log q with the frozen heads (1,024 steps) and the Student for
14 saved rows, and recomputes log L with the production evaluator for the same rows.

Smoke run (tiny bank → caches → correction → scenarios → 2-epoch head → dual sampler with the frozen networks →
exact IS → certificate; about 3 minutes on 3 CPU cores):

```bash
PYTHONPATH=$PWD python -m route2.smoke.anet_hyperonic_smoke
```

## Gaps

- The command lines of the paper's runs of Steps 2–13 are not available, including the per-round command lines
  of Stage A and the uniform-bank command line. The commands above are rebuilt from the saved settings and the
  programs' options.
- The run times of the Stage-A rounds and the draw times of the query proposals are not available.
- Two EOS implementations are used: the SciPy port (`eos/ddb_hyperonic`) for the training caches (and so the
  scenario bank), and the production solver `ddb_hyperon_eos.py` for the uniform banks, Stage A and the exact
  importance sampling. The two agree within 1e-6; `certify_fast_portable_target.py` compares the fast evaluator
  with the frozen target certificate.
- `fast_curve_full_shard.py` (the curve commands of Steps 12 and 13) imports the helper module
  `fast_newton_hyperon_audit`, which is not part of this package, so the curve commands cannot run from this
  repository.
- The head draws use the GPU random generator, so a rerun draws different samples (the check replays log q and
  log L at the saved rows instead).
- A rebuilt target cache is not byte-identical to the one used for the paper (it stores absolute input paths).
