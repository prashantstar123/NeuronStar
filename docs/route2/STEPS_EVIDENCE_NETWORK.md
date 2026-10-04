# Route 2 steps: Green Evidence Network (Table VI evidences, both sectors)

This page explains how the 16 Evidence Network (EN) values of Table VI were made (8 rows for the nucleonic DDB
model and 8 for the hyperonic DDBΛΞ⁻ model), how to check them quickly, and how to make them again from
scratch with the code in this repository.

The EN is a neural network that has learned the evidence Z of the paper's likelihood as a function of the data.
Once it is trained and frozen, it returns log Z for a new nuclear-data setting or for a new NICER mass-radius
source in seconds, with no new equation-of-state (EOS) or likelihood evaluation. Each sector has three
independently trained networks (seeds 9211, 9212, 9213 for the nucleonic sector; 9311, 9312, 9313 for the
hyperonic sector). Table VI quotes their combined value and an uncertainty.

The commands below use the package code with the settings listed; where a setting of the published run is not
known, this is said in the text.

Words used below:

| Word | Meaning |
|---|---|
| **frozen input** | a file whose SHA-256 is given in this guide and which this rerun takes as given |
| **S_ext** | the "support extension": parameter rows whose predicted nuclear observables lie inside the 3-sigma box of a shifted nuclear setting (K0 = 200, K0 = 260, Jsym = 29, Jsym = 36) but outside the A1 box, added to both banks so that the shifted settings are well covered. The extension region is defined by the nuclear-observable boxes alone; no posterior or UltraNest value enters it |
| **desktop** | the desktop workstation that ran the CPU assembly, the queries and the checks: Intel i7-13700F, RTX 3060 Ti |
| **cluster node** | a node of a PBS cluster, 64 cores |

## 1. Quick check and smoke run

**Matching check** (replays the saved result; CPU):

```bash
PYTHONPATH=$PWD python -m route2.checks.evidence --device cpu
```

It loads the six frozen networks (`route2/fixtures/evidence/checkpoints/`), queries each at the five nuclear
settings and the three held-out NICER sources (48 values), rebuilds the 16 Table VI values with the aggregation
of Section 8 and compares them with `results/evidence/green_en_vs_ultranest_16_rows.json`. On the desktop with 3
threads it took 15.9 s: PASS, largest differences 3.8e-6 (member value), 3.2e-6 (centre), 2.1e-6 (sigma), all
within the 1e-5 tolerance, and every printed Table VI value identical. On the production GPU stack the member
values are bit-identical.

**Smoke run** (runs the code end to end; CPU):

```bash
PYTHONPATH=$PWD python -m route2.smoke.en_smoke
```

It runs the label builder, the trainer (4 optimizer steps) and the frozen-network query engine, unchanged, on a
300-row bank written in the production bank format, with the production hyperonic settings except for the sizes.
It asserts only that every output is finite and has the right shape, and writes `build/route2/en_smoke.json`.
On the desktop with 3 threads: **PASS in 52 s** (label builder 38 s for 8,192 scenarios, trainer 1.4 s, query
11 s). Variants: `--sector nucleonic` (48 s); `--bank <frozen EN bank>` takes the 300 rows with the largest
base weight from a real frozen bank (hyperonic bank `f669d84f…`: 167 s; nucleonic bank `0421508c…`: 38 s).
The label builder and trainer are written for a CUDA GPU; the smoke runs them on the CPU by redirecting their
CUDA calls inside the smoke process (the program files are not changed). The smoke numbers mean nothing.

## 2. Where the code is

| Location | What it is |
|---|---|
| `inference/evidence_network/conditional/build_source_weight_cache_v2.py` | label builder (both sectors) |
| `inference/evidence_network/conditional/query_green_frozen_v5.py` | query engine: integrates a NICER density table against the frozen network. It also prints, next to each EN value, the difference from UltraNest evidences written into the script (`NUCLEONIC_ULTRANEST`, `HYPERONIC_ULTRANEST`). The EN values do not use them; the published comparison is in `results/evidence/green_en_vs_ultranest_16_rows.json` |
| `inference/evidence_network/conditional/query_green_frozen_v7.py`, `support_extension_freeze_manifest.py`, `green_en_support_extension_freeze_manifest_20260923.json` | nucleonic query gate: runs v5 only for a checkpoint listed (by SHA-256) in the frozen six-member manifest |
| `inference/evidence_network/conditional/query_green_frozen_v8.py` | hyperonic query gate: runs v5 only for a checkpoint in the hyperonic manifest, whose SHA-256 is given on the command line; removes v5's UltraNest comparison constants from its output |
| `green_en/solution/s2_green_en.py` + `green_en/train_onfly.py` | trainer (both sectors); the helper must stay one folder above the trainer |
| `green_en/scripts/` | the stage scripts: S_ext screen, exact rows and their certificates, bank assembly, parent-bank builder, post-freeze exact-bank check, query timing, and the timing wrapper `run_clean_timed_stage.py` |
| `fastsolver/` | the compiled hyperonic EOS solver and batched likelihood used by the S_ext exact rows (Section 10) |
| `hyperonic_pipeline/screen_clean_hyperonic_support_enrichment.py`, `evaluate_clean_hyperonic_support_cache.py` | screen and exact rows of the second A1-box stream of the hyperonic parent bank (step H1) |
| `route2/checks/evidence.py`, `route2/smoke/en_smoke.py` | the matching check and the smoke run of Section 1 |

`green_en/CODE_MANIFEST.json` and `fastsolver/CODE_MANIFEST.json` give the SHA-256 and role of every file.

## 3. The chain in short

```text
frozen physics banks (EOS/TOV rows + exact likelihood terms, shared with A-NET and TSNPE; not charged to EN)
  |
  |- [hyperonic only] H1: second A1-box support stream (seed 20260922) -> parent bank 02357702
  |- S_ext screen: 160 million full-prior draws per sector (seed 20260924 nucleonic, 20260923 hyperonic)
  |- S_ext exact EOS/TOV/likelihood rows (fastsolver/), then bank assembly with the proposal correction
  |      and the coverage gates
  |- labels: 8,192 synthetic NICER-source scenarios x exact bank weights (seed 20261011), one GPU
  |- training: 3 networks per sector, 12,000 steps each, one GPU per network
  |- freeze: checkpoint hashes written to a manifest and a separate freeze file BEFORE any held-out file is staged
  |- query: 5 nuclear settings + 3 held-out NICER sources per network (48 values), through the query gates
  |- post-freeze exact-bank check: direct importance estimate of every row on the same bank
  |- Table VI: mean evidence of the 3 networks and its uncertainty (Section 8)
```

| Final product | Nucleonic | Hyperonic |
|---|---|---|
| bank used for labels | `0421508c…` (428,342,848 B, 848,017 rows) | `f669d84f…` (265,184,413 B, 645,084 rows) |
| label index / source-weight matrix | `5cccc718…` (12,369,858 B) / `00ce1658…` (4,227,989,632 B) | `90ddb217…` (12,236,330 B) / `01a9714b…` (3,681,058,944 B) |
| checkpoints (1,903,469 B each) | 9211 `bb34849b…`, 9212 `1f5372dd…`, 9213 `7154e189…` | 9311 `d77d7984…`, 9312 `6382b5d2…`, 9313 `15704e15…` |
| freeze manifest (in this package) | `a657bae4…` (six members; written by hand) | `8429ec7c…` (three members) |
| separate freeze file | `020255fe…` | `17a6e868…` |
| post-freeze exact-bank check (in this package) | `a51778ee…` | `4d72223f…` |
| 16-row Table VI record (both sectors) | `results/evidence/green_en_vs_ultranest_16_rows.json` | same file |

Route 1 reads the 16 rows from `results/evidence/green_en_vs_ultranest_16_rows.json`, and the check fixtures
(`route2/fixtures/evidence/`) hold the six checkpoints, their training reports, both exact-bank checks, the stored
nucleonic uncertainties (`nucleonic_stored_sigma.json`) and the hyperonic manifest.

## 4. Before you start

1. Use the Route 2 environment (Python 3.11, numpy 2.4.6, scipy 1.17.1, torch 2.5.1, numba 0.65.1, jax 0.10.2
   on CPU, h5py, joblib). The production label and training runs used Python 3.12 with torch 2.9.1+cu128 and
   the same numpy/scipy/h5py/numba/joblib versions. The label builder, the trainer and (by default) the query
   and exact-bank scripts need a CUDA GPU; the other stages are CPU only.
2. Work in the repository root:

   ```bash
   ./reproduce.sh check                      # unpacks the compressed observation files once
   export PYTHONPATH=$PWD:$PWD/fastsolver DDB_REPO_ROOT=$PWD JAX_PLATFORMS=cpu JAX_ENABLE_X64=True
   G=$PWD/green_en/scripts                   # the stage scripts
   T="python $G/run_clean_timed_stage.py"    # the timing wrapper; writes the timing record given by --timing-output
   W=$PWD/runs/green_en                      # where a rerun writes its files
   D=$W/data/baseline                        # GW170817 + J0030 + J0740 + J0437 only
   H=$W/heldout                              # J0614 + J1231 + J1614; staged only after the freeze (step 7)
   mkdir -p $D $W/{nucleonic,hyperonic}/{cache,models,queries}   # the label builder and trainer do not create folders
   for f in GW170817_GWTC-1.hdf5 J0030_2spot_RM.txt J0740_NICERXMM_full_mr.txt J0437_post_equal_weights.dat; do
     ln -s $PWD/data/observations/$f $D/; done
   ```

   `fastsolver/` must be on `PYTHONPATH`: the exact-row scripts import `fast_exact_rows` and
   `fast_likelihood` from it. `DDB_REPO_ROOT` tells the trainer where this repository is. Every script refuses
   to overwrite an existing output.
3. The frozen inputs below are not in git. Five of the eight (nine files) are in the optional full-rerun download
   (`./reproduce.sh route2-fetch --group full_rerun`) under the names in the last column; see
   [FULL_RERUN_FILES.md](FULL_RERUN_FILES.md). The other three are not provided (see Gaps). Their SHA-256 must
   match.

| Frozen input | SHA-256 | Size | Used in | In the full-rerun download as (`route2/downloads/`) |
|---|---|---|---|---|
| `shared_prior_physics_bank.npz` (nucleonic, `data/shared_physics/`) | `06dda51aa7c1…` | 3,157,558,246 B | N4 (raw bank); parent of the template | `full_nucleonic_shared_prior_physics_bank.npz` |
| `shared_exact_astrophysical_table.npz` (same folder) | `e2dfb84d56a6…` | 43,919,890 B | N4 | `full_nucleonic_shared_exact_astrophysical_table.npz` |
| nucleonic base bank `nucleonic_conditional_physics.npz` | `1fd8830f…` | 683,538,572 B (806,053 rows) | N4 | not provided |
| fixed J0740 terms `tov_terms_outer_00..04.npz` | `c9b38228`, `a31635ae`, `df70edd3`, `12c07038`, `7a7b9a7a` | 34.5 MB in total | N4 | not provided |
| nucleonic exact template `nucleonic_exact_template.npz` | `6bfcdd73…` | 3,378 B | N3 | not provided |
| hyperonic template bank `hyp_bank_part1.npz` | `c188c4d0…` | 529,515,486 B | H1, H3 | `full_hyperonic_uniform_bank_part1.npz` |
| hyperonic uniform parts `uniform_training_cache_part1..4.npz` | `e51cf2f1`, `e7ac3193`, `e13a8566`, `575ebf36` | 713.6 MB in total | H1 | `full_hyperonic_uniform_training_cache_part1..4.npz` |
| hyperonic first A1-box stream `support_training_cache_part1..2.npz` (seed 20260729) | `479206d0`, `28e0894d` | 107.6 MB in total | H1 | `full_hyperonic_support_training_cache_part1..2.npz` |

The products of N1, N3 and N4 (the nucleonic screen, exact rows and bank) and of H1–H4 (the hyperonic parent bank,
screen, exact rows and bank) are also in the full-rerun download. A nucleonic rerun can therefore start at N5
without the three missing inputs.

## 5. Nucleonic DDB, step by step

### N1. S_ext screen (CPU)

```bash
$T --timing-output $W/nucleonic/screen_clean_timing.json --stage nucleonic_support_extension_screen -- \
  python $G/screen_green_en_support_extension.py --eos ddb \
  --output $W/nucleonic/nucleonic_Sext_seed20260924_160M.npz \
  --proposals 160000000 --block-size 2000000 --workers 48 --seed 20260924
```

- 160,000,000 draws from the full prior in 80 blocks; block b uses `SeedSequence([20260924, 1, b])`, so the
  result does not depend on the number of workers.
- Output `e43000e0…` (6,889,520 B): 62,486 retained rows (acceptance 3.9e-4).
- 265.095 s on a cluster node with 48 workers.
- Certification run before the production screen (not timed): the same screen with 400,000 draws in blocks of
  100,000, once with `--workers 1` and once with `--workers 4`, then
  `python $G/verify_support_extension_parallel.py --serial <serial.npz> --parallel <parallel.npz> --output <json>`:
  every array bitwise identical (both sectors).

### N2. Exact template (frozen input)

The template (`6bfcdd73…`, 3,378 B: the mass grid and prior bounds of the shared bank `06dda51a…`) is a frozen
input. The script that extracted it is not included in the package, so this step cannot be rerun. Use the frozen
file.

### N3. S_ext exact EOS/TOV/likelihood rows (CPU)

The driver starts
`scripts/evaluate_nucleonic_support_extension.py` relative to its working folder, so run it from `green_en/`:

```bash
cd green_en
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
$T --timing-output $W/nucleonic/exact_clean_timing.json \
  --stage nucleonic_support_extension_exact_rows_parallel8_chunk_aligned -- \
  python scripts/run_nucleonic_exact_parallel_stage.py \
  --screen $W/nucleonic/nucleonic_Sext_seed20260924_160M.npz --template <nucleonic_exact_template.npz> \
  --data-root $D --work-dir $W/nucleonic/shards \
  --output $W/nucleonic/nucleonic_Sext_exact_seed20260924_parallel8_aligned.npz \
  --start 0 --stop 62486 --shards 8 --workers-per-shard 7 --chunk-size 2048
cd ..
```

- Eight shards made of whole 2,048-row chunks, so every JAX batch is the same as in a serial run; the output
  is bitwise identical to the serial reference (`8e06c93c…`, not included in the package), as certified by
  `certify_nucleonic_exact_parallel_output.py`.
- EOS: the certified CPU-JAX nucleonic EOS; TOV: the unchanged exact CPU solver; likelihood: the batched
  `fastsolver/fast_likelihood.py`.
- Output `1fcd9396…` (107,317,349 B): 62,486 rows, 41,964 target-valid.
- 218.761 s on a cluster node, 8 shards x 7 workers (64 cores requested, peak about 31 GB).
- 200-row adapter certificate against the scalar reference path
  (`certify_nucleonic_support_extension_adapter.py`): PASS on all eight gates (its command is not known).

### N4. Bank assembly and coverage gates (CPU)

```bash
$T --timing-output $W/nucleonic/assembly_clean_timing.json \
  --stage nucleonic_support_extension_bank_assembly_and_coverage -- \
  python $G/assemble_green_en_support_extension.py nucleonic \
  --base-bank <nucleonic_conditional_physics.npz> --raw-bank <shared_prior_physics_bank.npz> \
  --exact-table <shared_exact_astrophysical_table.npz> --fixed-j0740-terms <folder with tov_terms_outer_00..04.npz> \
  --extension-screen $W/nucleonic/nucleonic_Sext_seed20260924_160M.npz \
  --extension-exact $W/nucleonic/nucleonic_Sext_exact_seed20260924_parallel8_aligned.npz \
  --output $W/nucleonic/nucleonic_conditional_physics_Sext_seed20260924.npz
```

- Proposal correction of every row: `log N0 − log(N0 + M_B·1_B + M_C·1_C + M_D·1_D + 160,000,000·1_Sext)`, with
  N0 = 660,000 and M_B, M_C, M_D read from the raw bank.
- Output `0421508c…` (428,342,848 B, 848,017 rows).
- Coverage gates (all pass): in-region effective sample size 461.43 / 401.75 / 512.55 / 560.66 / 417.01 for
  A1 / K0=200 / K0=260 / Jsym=29 / Jsym=36 (gate ≥ 50); prior-predictive kernel mass outside the covered region
  0.370 / 0.640 / 0.316 / 0.586 / 0.311 % (gate ≤ 2.5 %).
- 16.813 s on the desktop, one process.

### N5. Labels (GPU)

Run with 9 threads.

```bash
$T --timing-output $W/nucleonic/label_clean_timing.json --stage nucleonic_green_en_label_cache_support_extension -- \
  python inference/evidence_network/conditional/build_source_weight_cache_v2.py \
  --bank $W/nucleonic/nucleonic_conditional_physics_Sext_seed20260924.npz --baseline-root $D \
  --output $W/nucleonic/cache/nucleonic_green_cache_Sext_8192.npz \
  --scenarios 8192 --validation-scenarios 1024 --seed 20261011 --identity-fraction 0.02 --one-slot-fraction 0.98 \
  --scenario-batch 32 --minimum-tilt-ess 20 --maximum-restriction-delta 0.001 --nuclear-support-sigma 6 \
  --nuclear-designs 64 --mass-centre-min 0.75 --mass-centre-max 2.35 --radius-centre-min 8.0 --radius-centre-max 17.0 \
  --mass-sigma-min 0.005 --mass-sigma-max 0.5 --radius-sigma-min 0.05 --radius-sigma-max 2.5 --correlation-limit 0.97
```

- Output: the index `5cccc718…` (12,369,858 B) and, next to it, the source-weight matrix
  `nucleonic_green_cache_Sext_8192.sourcelogw.npy` (`00ce1658…`, 4,227,989,632 B). The trainer reads the matrix
  from that location.
- 8,192 scenarios accepted out of 8,800 drawn; 129,028 bank rows inside the 6-sigma support; largest
  restriction change 5.6e-4 (gate 1e-3); identity error 0.
- 227.789 s on one rented cloud GPU (NVIDIA A40, 48 GB).

### N6. Training, seeds 9211, 9212, 9213 (GPU)

Each member runs with 1 thread on one GPU.

```bash
for seed in 9211 9212 9213; do
  $T --timing-output $W/nucleonic/member_${seed}_clean_timing.json --stage nucleonic_green_en_member_${seed}_support_extension -- \
  python green_en/solution/s2_green_en.py --cache $W/nucleonic/cache/nucleonic_green_cache_Sext_8192.npz \
  --output $W/nucleonic/models/nucleonic_green_seed${seed}.pt \
  --bank $W/nucleonic/nucleonic_conditional_physics_Sext_seed20260924.npz --baseline-root $D --seed ${seed} \
  --steps 12000 --batch-scenarios 192 --designs-per-step 9 --lr 0.0006 \
  --nm 191 --nr 241 --mass-low 1.0 --mass-high 2.9 --radius-low 7.0 --radius-high 19.0
done
```

(The members ran concurrently, one per GPU, 9211 about 2.5 min before the other two; run the three commands in
parallel on three GPUs.)

| Seed | GPU | Clean wall | Checkpoint |
|---|---|---|---|
| 9211 | A40 | 1097.196 s | `bb34849b…` |
| 9212 | RTX 4090 | 664.984 s | `1f5372dd…` |
| 9213 | RTX 4090 | 665.638 s | `7154e189…` |

Critical path 1097.196 s. Pre-query gates (all three pass with margin): blind synthetic mean absolute error on
the axis ≤ 0.08, Gaussian ≤ 0.15, box ≤ 0.18 and paper ≤ 0.10 design families, baseline-cloud error ≤ 0.10,
identity residual ≤ 0.10. The trainer refuses to run if a held-out file is present in `--baseline-root`.

### N7. Freeze

The six-member manifest `inference/evidence_network/conditional/green_en_support_extension_freeze_manifest_20260923.json`
(written by hand at 15:53 KST on 2026-09-23) and a separate freeze file (`020255fe…`, 15:55) were written before
any held-out file was staged (16:00:29). The manifest also lists three hyperonic checkpoints (`0708eb54…`,
`f5b7d4e9…`, `3732bc76…`) that are not used for Table VI: the v7 gate requires exactly six members, and the
hyperonic members of Table VI are authorized by the hyperonic manifest (H7).
`support_extension_freeze_manifest.py` pins the manifest's SHA-256 (`a657bae4…`).

### N8. Query (GPU on the desktop)

```bash
mkdir -p $H; for f in J0614_mrsamples.dat J1231_wmrsamples.txt J1614_STU_mrsamples_post_equal_weights.dat; do
  ln -s $PWD/data/observations/$f $H/; done
for seed in 9211 9212 9213; do
  python inference/evidence_network/conditional/query_green_frozen_v7.py \
  --checkpoint route2/fixtures/evidence/checkpoints/nucleonic_green_seed${seed}.pt --heldout-root $H \
  --output $W/nucleonic/queries/nucleonic_green_seed${seed}_heldout_query.json --device cuda --sector nucleonic
done
```

- Output files `d17a8ea1…`, `d961f4e6…`, `3584165f…`, written about 12.6 s apart (this step was not run under
  the clean-timing wrapper).
- `--device cpu` also works. With the shipped checkpoints on the desktop's CPU the gate accepts each member and
  the eight values differ from the published GPU output by at most 3.8e-6.
- v7 keeps the UltraNest comparison constants that v5 copies into its output. They never enter `log_evidence`,
  and the Table VI aggregation does not use them.

### N9. Post-freeze exact-bank check (GPU on the desktop)

```bash
python $G/evaluate_support_extension_v2_exact_bank.py --bank $W/nucleonic/nucleonic_conditional_physics_Sext_seed20260924.npz \
  --heldout-root $H --output $W/nucleonic/nucleonic_support_extension_v2_exact_bank.json --device cuda
```

- Output `a51778ee…`: for each of the 8 rows, the direct importance estimate of log Z on the bank, its
  effective sample size (smallest 357.1) and the Monte Carlo error 1/√ESS used in Table VI. Internal wall
  15.57 s. This is a diagnostic, not part of the EN cost.

## 6. Hyperonic DDBΛΞ⁻ (parent bank 02357702), step by step

The S_ext screen and exact rows (H2, H3) do not depend on the parent bank.

### H1. Parent bank 02357702: second A1-box support stream (charged to EN in Table VIII)

The parent bank is 415,681 uniform-prior rows plus two A1-box support streams of 160,000,000 draws each (seed
20260729: 83,764 rows, frozen; seed 20260922: 83,106 rows, made here).

```bash
python hyperonic_pipeline/screen_clean_hyperonic_support_enrichment.py \
  --output $W/hyperonic/support_screen_seed20260922_160m.npz --proposals 160000000 --block 2000000 --cbox 3.0 --seed 20260922
# five exact shards (rows [start, stop) of the screen):
python hyperonic_pipeline/evaluate_clean_hyperonic_support_cache.py --screen $W/hyperonic/support_screen_seed20260922_160m.npz \
  --template-bank <hyp_bank_part1.npz> --data-root $D \
  --output $W/hyperonic/support_training_cache_seed20260922_part0.npz --workers 10 --start 0 --stop 15000
# ... likewise part1 [15000,21000), part_local2 [21000,27000), part3 [27000,45000), part4 [45000,83106)
python $G/prepare_expanded_hyperonic_green_bank.py \
  --uniform-part <uniform_training_cache_part1.npz> ... --uniform-part <uniform_training_cache_part4.npz> \
  --support-part <support_training_cache_part1.npz> --support-part <support_training_cache_part2.npz> \
  --support-part $W/hyperonic/support_training_cache_seed20260922_part0.npz ... (all five parts, in order) \
  --base-proposals 600000 --output $W/hyperonic/hyperonic_green_expanded_fullprior_bank_seed20260922.npz
```

| Piece | Rows | Machine, workers | Wall | SHA-256 |
|---|---|---|---|---|
| screen (seed 20260922) | 83,106 retained | desktop | 1024.891 s | `75f1d4ac…` |
| part0 | 0–15,000 | desktop, 10 | 688.9 s | `65b49dad…` |
| part1 | 15,000–21,000 | desktop, 6 | 513.5 s | `4bca1b08…` |
| part_local2 | 21,000–27,000 | desktop, 8 | 329.3 s | `e72cd8c5…` |
| part3 | 27,000–45,000 | cluster node, 22 | 1032.4 s | `6f62c757…` |
| part4 | 45,000–83,106 | cluster node, 46 | 518.2 s | `baaf83b2…` |
| parent bank | 582,551 | desktop | 9.9 s | `02357702…` (244,687,891 B) |

The five shards ran at the same time; the as-executed span was 1210 s. The builder groups the support parts
into streams by the screen hash stored in each part and stores all 11 parent hashes in the bank.

### H2. S_ext screen (CPU)

```bash
$T --timing-output $W/hyperonic/screen_clean_timing.json --stage hyperonic_support_extension_screen -- \
  python $G/screen_green_en_support_extension.py --eos ddb-hyperonic \
  --output $W/hyperonic/hyperonic_Sext_seed20260923_160M.npz \
  --proposals 160000000 --block-size 2000000 --workers 48 --seed 20260923
```

Output `629c156a…` (7,838,415 B): 62,533 retained rows. 267.492 s on a cluster node with 48 workers.

### H3. S_ext exact rows (CPU)

A 200-row certification (not timed) runs before the production command, which uses a fresh Numba cache:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=56 NUMBA_CACHE_DIR=$W/hyperonic/numba \
$T --timing-output $W/hyperonic/exact_clean_timing.json --stage hyperonic_support_extension_exact_rows -- \
  python $G/evaluate_hyperonic_support_extension.py --screen $W/hyperonic/hyperonic_Sext_seed20260923_160M.npz \
  --template-bank <hyp_bank_part1.npz> --data-root $D --output $W/hyperonic/hyperonic_Sext_exact_seed20260923.npz \
  --workers 56 --start 0 --stop 62533 --eos-backend compiled
```

- EOS: the compiled hyperonic solver of `fastsolver/`; the 476 rows it flags are solved again with the
  certified Python solver. TOV: the unchanged exact CPU solver.
- Output `aa5a8b23…` (76,455,064 B): 28,826 target-valid rows.
- 132.711 s on a cluster node with 56 workers.

### H4. Bank assembly and coverage gates (CPU)

```bash
$T --timing-output $W/hyperonic/assembly_clean_timing.json \
  --stage hyperonic_support_extension_parent02357702_assembly_and_coverage -- \
  python $G/assemble_green_en_support_extension.py hyperonic \
  --base-bank $W/hyperonic/hyperonic_green_expanded_fullprior_bank_seed20260922.npz \
  --extension-screen $W/hyperonic/hyperonic_Sext_seed20260923_160M.npz \
  --extension-exact $W/hyperonic/hyperonic_Sext_exact_seed20260923.npz \
  --output $W/hyperonic/hyperonic_conditional_physics_Sext_seed20260923_parent02357702.npz \
  --base-proposals 600000 --b3-proposals 320000000 --hyperonic-uniform-rows 415681
```

- Proposal correction of every row: `log 600000 − log(600000 + 320,000,000·1_B3 + 160,000,000·1_Sext)`.
- Output `f669d84f…` (265,184,413 B, 645,084 rows).
- Coverage gates (all pass): in-region effective sample size 154.97 / 120.10 / 202.87 / 202.32 / 92.55 for
  A1 / K0=200 / K0=260 / Jsym=29 / Jsym=36 (gate ≥ 50); kernel mass outside the covered region
  0.292 / 0.408 / 0.223 / 0.444 / 0.348 % (gate ≤ 2.5 %).
- 10.649 s on the desktop.

### H5. Labels (GPU)

Run with 10 threads.

```bash
$T --timing-output $W/hyperonic/label_clean_timing.json \
  --stage hyperonic_green_en_label_cache_support_extension_parent02357702 -- \
  python inference/evidence_network/conditional/build_source_weight_cache_v2.py \
  --bank $W/hyperonic/hyperonic_conditional_physics_Sext_seed20260923_parent02357702.npz --baseline-root $D \
  --output $W/hyperonic/cache/hyperonic_green_cache_Sext_8192_parent02357702.npz \
  --scenarios 8192 --validation-scenarios 1024 --seed 20261011 --identity-fraction 0.02 --one-slot-fraction 0.83 \
  --scenario-batch 32 --minimum-tilt-ess 20 --maximum-restriction-delta 0.001 --nuclear-support-sigma 6 \
  --nuclear-designs 64 --mass-centre-min 0.85 --mass-centre-max 2.25 --radius-centre-min 8.5 --radius-centre-max 16.5 \
  --mass-sigma-min 0.015 --mass-sigma-max 0.48 --radius-sigma-min 0.12 --radius-sigma-max 2.30 --correlation-limit 0.95
```

- Output: index `90ddb217…` (12,236,330 B) and matrix `…parent02357702.sourcelogw.npy` (`01a9714b…`,
  3,681,058,944 B) next to it.
- 8,192 scenarios accepted out of 13,408 drawn; 112,337 bank rows inside the 6-sigma support; largest
  restriction change 9.9e-4 (gate 1e-3); identity error 0.
- 160.241 s on one RTX 4090 (a rented cloud machine with 3 x RTX 4090, 64 vCPU, 125 GB RAM).

### H6. Training, seeds 9311, 9312, 9313 (GPU)

The three members run at the same time, one GPU each, 1 thread.

```bash
for seed in 9311 9312 9313; do
  $T --timing-output $W/hyperonic/member_${seed}_clean_timing.json \
  --stage hyperonic_green_en_member_${seed}_support_extension_parent02357702 -- \
  python green_en/solution/s2_green_en.py --cache $W/hyperonic/cache/hyperonic_green_cache_Sext_8192_parent02357702.npz \
  --output $W/hyperonic/models/hyperonic_green_seed${seed}.pt \
  --bank $W/hyperonic/hyperonic_conditional_physics_Sext_seed20260923_parent02357702.npz --baseline-root $D \
  --seed ${seed} --steps 12000 --batch-scenarios 192 --designs-per-step 9 --lr 0.0006 \
  --nm 161 --nr 205 --mass-low 1.0 --mass-high 2.6 --radius-low 8.2 --radius-high 18.4
done
```

| Seed | GPU | Clean wall | Checkpoint |
|---|---|---|---|
| 9311 | RTX 4090 | 484.966 s | `d77d7984…` |
| 9312 | RTX 4090 | 483.849 s | `6382b5d2…` |
| 9313 | RTX 4090 | 481.962 s | `15704e15…` |

The three started within 0.014 s; critical path 484.966 s. All pass the same pre-query gates as N6.

### H7. Freeze

The hyperonic manifest (17:01:44 KST) and a separate freeze file (`17a6e868…`, 17:02:07) were written before the
held-out files were staged (17:02:26). The manifest is `route2/fixtures/evidence/hyperonic_freeze_manifest.json`;
the script that wrote it is not included in the package.

### H8. Query (GPU on the desktop)

```bash
M=route2/fixtures/evidence/hyperonic_freeze_manifest.json
for seed in 9311 9312 9313; do
  $T --timing-output $W/hyperonic/query_${seed}_clean_timing.json \
  --stage hyperonic_green_en_member_${seed}_frozen_query_parent02357702 -- \
  python inference/evidence_network/conditional/query_green_frozen_v8.py \
  --checkpoint route2/fixtures/evidence/checkpoints/hyperonic_green_seed${seed}.pt --heldout-root $H \
  --output $W/hyperonic/queries/hyperonic_green_seed${seed}_heldout_query.json --device cuda \
  --freeze-manifest $M --freeze-manifest-sha256 $(sha256sum $M | cut -c1-64)
done
```

- With `--device cpu` the gate accepts each member and the values differ from the published GPU output by at most
  3.8e-6.
- Output files `08492a5a…`, `dc7856e9…`, `88465ed4…`; command walls 12.3 / 12.4 / 12.3 s on the desktop
  (RTX 3060 Ti), of which checkpoint loading 0.001 s, the five nuclear settings 0.047 s and each NICER density
  table 3.4–3.5 s. A second run of the three queries gives identical log-evidences.

### H9. Post-freeze exact-bank check (GPU on the desktop)

```bash
$T --timing-output $W/hyperonic/exact_bank_clean_timing.json --stage hyperonic_support_extension_exact_bank_postfreeze_check -- \
  python $G/evaluate_support_extension_v2_exact_bank.py \
  --bank $W/hyperonic/hyperonic_conditional_physics_Sext_seed20260923_parent02357702.npz \
  --heldout-root $H --output $W/hyperonic/hyperonic_support_extension_v2_exact_bank_parent02357702.json --device cuda
```

Output `4d72223f…`; 16.034 s on the desktop. Diagnostic only, not part of the EN cost.

## 7. Query timing (both sectors)

```bash
python $G/benchmark_green_en_query_timing.py \
  --nucleonic route2/fixtures/evidence/checkpoints/nucleonic_green_seed921{1,2,3}.pt \
  --hyperonic route2/fixtures/evidence/checkpoints/hyperonic_green_seed931{1,2,3}.pt \
  --heldout-root $H --output $W/query_timing.json
```

It reloads all six networks, reproduces all 48 member log-evidences (exactly, on the desktop's RTX 3060 Ti with
torch 2.5.1+cu124) and times the queries. Command wall 12.562 s. Table VIII quotes, per sector, the
load plus one warm batch of the five nuclear settings (0.0260 s nucleonic, 0.0273 s hyperonic) and, per
held-out source, one density table plus three warm network evaluations (nucleonic 3.381 / 3.511 / 3.548 s,
hyperonic 3.370 / 3.502 / 3.540 s for J0614 / J1231 / J1614).

## 8. Table VI: how the 16 values are formed

`route2/checks/evidence.py` forms the 16 values. For each sector and row (A1, K0_200, K0_260, Jsym_29, Jsym_36,
J0614, J1231, J1614), the printed EN value is the logarithm of the mean evidence of the three members,
`logsumexp(m1, m2, m3) − log 3`, where m1, m2, m3 are their query log-evidences. The printed uncertainty is
`sqrt(s² + e² + c²)`, with s the scatter of the three members, e the Monte Carlo error of the post-freeze exact-bank
estimate (N9, H9) and c the calibration term from the training reports. For the nucleonic rows it is the larger of
this value and the stored value in `route2/fixtures/evidence/nucleonic_stored_sigma.json`; for the three held-out
sources the difference between the network and exact-bank estimates is added in quadrature. Table VI prints both
rounded to 3 decimals. (Its UltraNest column and the separations come from the UltraNest steps; the separation is
`|centre − UN| / hypot(sigma, sigma_UN)`.)

For networks trained again (N6, H6) with their own exact-bank checks (N9, H9), give their files to the check, with
the same file names as in `route2/fixtures/evidence/`:

```bash
python -m route2.checks.evidence --checkpoints <folder> --training-reports <folder> --exact-bank <folder> \
  --report $W/table_vi_rebuilt.json
```

The report holds their 16 values; the printed comparison shows how far they are from the published networks
(retraining is not bit for bit, see Gaps).

Final values (centre ± sigma), which the matching check reproduces exactly:

| Row | Nucleonic | Hyperonic |
|---|---|---|
| A1 | −25.613 ± 0.050 | −28.726 ± 0.083 |
| K0=200 | −25.582 ± 0.053 | −29.121 ± 0.094 |
| K0=260 | −25.851 ± 0.048 | −28.608 ± 0.075 |
| Jsym=29 | −25.542 ± 0.127 | −28.692 ± 0.073 |
| Jsym=36 | −25.721 ± 0.114 | −28.923 ± 0.111 |
| PSR J0614−3329 | −27.170 ± 0.166 | −31.439 ± 0.197 |
| PSR J1231−1411 | −26.286 ± 0.234 | −29.117 ± 0.122 |
| PSR J1614−2230 | −25.585 ± 0.179 | −28.744 ± 0.117 |

## 9. Wall times and hardware (Table VIII)

Each value is the clean wall of the production command (process start to final output; concurrent shards or
members count once, by the longest). Queue, transfer, installation and idle time are not included.

| Stage | Nucleonic | Hyperonic |
|---|---|---|
| H1 parent-bank second stream: screen + shards + assembly | – (not needed) | 1024.891 + 1210.000 + 9.900 s (desktop and two cluster nodes) |
| S_ext screen | 265.095 s (cluster node, 48 workers) | 267.492 s (cluster node, 48 workers) |
| S_ext exact rows | 218.761 s (cluster node, 56 workers) | 132.711 s (cluster node, 56 workers) |
| bank assembly and coverage | 16.813 s (desktop) | 10.649 s (desktop) |
| **CPU preparation** | **500.669 s = 8.34 min** | **2655.643 s = 44.26 min** |
| labels | 227.789 s (one A40) | 160.241 s (one RTX 4090) |
| training (critical path) | 1097.196 s (A40 + 2 x RTX 4090) | 484.966 s (3 x RTX 4090) |
| **total before the first query** | **1825.654 s = 30.4 min** | **3300.850 s = 55.0 min** |
| query, five nuclear settings together | 0.026 s | 0.027 s |
| query, each held-out NICER source | 3.4–3.5 s | 3.4–3.5 s |

The shared EOS/TOV physics banks (shared with A-NET and TSNPE) are not charged to EN.

## 10. `fastsolver/`: what it is

`fastsolver/` holds a numba port of the certified hyperonic beta-equilibrium solver
(`hyp_solver_template.py` → `gen_backends.py` → `hyp_solver_cpu.py`, `hyp_solver_cuda.py`), the batched likelihood
`fast_likelihood.py` (bitwise equal to `JointLikelihood.evaluate` on 6,000 rows) and the exact-row evaluator
`fast_exact_rows.py` (compiled solver, with the certified Python solver for flagged rows). It is used only by the
S_ext exact-row stages (N3, H3). `gen_backends.py` regenerates `hyp_solver_cpu.py` and `hyp_solver_cuda.py`
byte-identically from the template. The CUDA backend (`hyp_solver_cuda.py`) was not used in production.
`fastsolver/README.md` describes the folder. Use the commands of steps N3 and H3.

## 11. Gaps

1. **Large files: most are in the optional full-rerun download, some are not.** The shared physics files, the
   hyperonic template bank and caches, the screens, the exact rows, the hyperonic parent bank and the two training
   banks are in the full-rerun download ([FULL_RERUN_FILES.md](FULL_RERUN_FILES.md); Section 4 gives the
   names). Not provided:
   - the label matrices (7.9 GB): rebuild them on a GPU (N5, H5);
   - the nucleonic base bank `1fd8830f…` and the fixed J0740 terms of N4, and the exact template `6bfcdd73…` of N3:
     start at N5 from the downloaded bank instead;
   - the parent-bank shards and second-stream files of H1: their product, the parent bank, is downloadable.

   The check and the smoke need none of these files.
2. **Two scripts of the chain are not included in the package:** the template extractor (N2) and the hyperonic
   manifest writer (H7). The cluster stages are given as complete commands (N1, N3, H1, H2, H3).
3. **The nucleonic base bank `1fd8830f…`** was built by a bank-preparation script whose version is not known, so
   the bank is a frozen input and the script is not included in the package.
4. **UltraNest constants in the nucleonic query outputs.** v7 keeps the hard-coded comparison constants that v5
   copies into its output; they never enter `log_evidence` and the Table VI aggregation does not use them.
5. **Retraining does not reproduce the checkpoints bit for bit.** Training uses TF32 without deterministic flags,
   the nucleonic members ran on two GPU types, and the trainer does not log the GPU name. Exact replay of Table VI
   needs the frozen checkpoints, which the check uses.
6. **GPU-only programs.** The label builder and the trainer hard-code a CUDA device. The smoke runs them on the
   CPU only through its in-process redirection.
7. **Other known limits:** the nucleonic manifest also lists three hyperonic checkpoints that Table VI does not
   use, and the v7 gate requires exactly that six-member file; the nucleonic bank's `AC[:,0]` column holds a
   radius, is invalid and is unused; bank `theta` is stored as float32; the serial nucleonic exact reference
   `8e06c93c…` is not included in the package; the cluster used numba 0.66.0, the desktop 0.65.1.
