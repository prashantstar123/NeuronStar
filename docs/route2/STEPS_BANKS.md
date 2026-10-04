# Route 2 steps: the prior and training banks

This page covers the stored parameter banks that every inference route of the paper starts from: what each
bank holds, how to make it again with the code in this repository, and how the replay check
`route2/checks/banks.py` tests it without a full rerun. It also lists every bank file with its size and
SHA-256.

**Banks covered.**

| Sector | Bank | Used by |
|---|---|---|
| nucleonic | shared prior physics bank + row-aligned exact astrophysical table | A-NET scenarios; parent of the three files below |
| nucleonic | uniform-prior support bank | TSNPE round zero |
| nucleonic | independent common-physics cache | Evidence Network (EN) caches; parent of the Green EN bank |
| hyperonic | four uniform-prior banks | Stage A; A-NET uniform training caches |
| hyperonic | nine A-NET training caches (4 uniform, 2 support, 3 proposal) | A-NET scenario bank (listed by hash in it) |
| hyperonic | Stage-A MIS evaluation (1.72 million exact rows) | fit of the defensive Student proposal |
| both | Green EN training banks (nucleonic and hyperonic, with their S_ext rows) | Green EN labels and training |

Every bank in this table can be downloaded instead of regenerated: it is in the optional full-rerun download
(`./reproduce.sh route2-fetch --group full_rerun`), under a name that starts with `full_`. Section 4 says which
files are in it, and [FULL_RERUN_FILES.md](FULL_RERUN_FILES.md) gives both names.

The commands use the package code with the settings listed. Many bank files store their own seed, row counts,
machine and run time. Where a setting of a published bank is not known, this is said in the text.

**Machines.** "The desktop" below is a desktop workstation, an Intel i7-13700F with 24 logical CPUs. "Cluster
node 1" and "cluster node 2" are two nodes of a CPU cluster (jobs with 62 workers unless stated).

**A full rerun is long.** The banks hold about 5 million EOS/TOV solutions in total (914,966 nucleonic shared-bank
rows, 600,000 hyperonic uniform rows, 1.72 million Stage-A rows, the caches and the S_ext rows). The replay check
below needs about one minute and none of the large files.

## 1. Quick check (CPU, about 75 s)

```bash
PYTHONPATH=$PWD python -m route2.checks.banks
```

- Report: `build/route2/banks_check.json` (one entry per bank, with every comparison and its largest difference).
- Inputs: only the fixtures in `route2/fixtures/banks/` (about 0.4 MB) and `data/observations/`.
- Measured on the desktop with 3 threads: 74 s wall, 118 CPU-seconds, 1.2 GB peak memory. It replays 299 rows of
  20 bank items.

**What it replays, for each bank.**

1. **Parameter draws.** Every row with a known seed is drawn again from that seed and must be bitwise equal:
   the nucleonic uniform streams (seeds 1234 and 2026) and enrichment streams (seed 20260729), the hyperonic
   uniform banks (seeds 91–94), the support screens (seeds 20260729 and 20260922), the Green EN S_ext screens
   (seeds 20260924 and 20260923). The Student proposal draws (seed 20260992) are compared to 1e-10, because
   they were made on another machine and the Student's linear algebra rounds differently there (1.2e-12 seen).
2. **Forward physics**, with the same functions that made the bank: nuclear-matter properties, R(M) and Λ(M)
   on the 200-point mass grid, the maximum mass, R1.4, R(Mmax), the pQCD and shifted-GW grids, and every stored
   log-likelihood column (NICER per source, GW170817, maximum mass, pQCD, total).
3. **Bookkeeping.** The support flags and counting corrections (`logw_prior`, `LPC`) are recomputed from the
   replayed physics; forward failures and rejected rows must be the same rows.

**Tolerances.** Draws, flags, failure codes and counting corrections: exact. Nuclear-matter properties:
1e-4 of the nuclear σ (the pipeline's own prediction tolerance; K0 is a finite-difference second derivative, so
its last digits depend on how JAX vectorizes the batch). Stellar curves: 1e-6 relative. Log-likelihood columns:
1e-4 (the pipeline's certification tolerance); values at or below −1e29 are "outside the support" markers and are
compared as markers. Where a bank stores float32, the float32 rounding of the stored value is allowed on top.

**Result (on the desktop).**

| Bank | Rows | Largest difference |
|---|---|---|
| nucleonic shared prior physics bank | 40 | X 2e-6 σ; curves 7e-11; log L grids 4e-9; draws, flags and `logw_prior` identical |
| nucleonic exact table | 40 | log L 2e-9; R(Mmax), Mmax 1e-14 |
| nucleonic uniform-prior support bank | 20 | X 1e-6 σ; Mmax 1e-15; draws identical |
| nucleonic independent cache | 30 | X 2e-6 σ; LA 8e-10; draws and `LPC` identical |
| Green EN nucleonic bank | 30 | X 4e-5 σ (float32); curves 6e-8 (float32); log L 4e-10; `LPC` identical |
| hyperonic uniform banks (4 × 8) | 32 | X 7e-13 σ; curves 8e-9; failure codes identical |
| hyperonic training caches (9 parts) | 41 | X 1e-6 σ; curves 1e-10; log L 7e-10; Student log q identical to 1e-10 |
| hyperonic Stage-A MIS evaluation | 36 | log L 1.8e-6 on stored values; **one documented defect** (Section 3.7) |
| Green EN hyperonic bank | 30 | X 3e-5 σ (float32); curves 6e-8 (float32); log L 3e-10; `LPC` identical |

The check reports the Stage-A defect described in Section 3.7 (the stored bank marks as rejected many rows that
the shipped code evaluates) and passes only if the mismatches are exactly that defect; any other mismatch fails.
No result of the paper depends on it (Section 3.7). With `--strict` the defect itself also counts as a failure.

## 2. Before you start (for a rerun)

1. Use the Route 2 Python environment (Python 3.11, numpy 2.4.6, scipy 1.17.1, jax 0.10.2 on CPU, numba 0.65.1,
   h5py, joblib, scikit-learn). The hyperonic Student checkpoint was pickled with scikit-learn 1.9.1; 1.9.0 loads it
   with a version warning and reproduces its draws and density.
2. `CompactObject-TOV==2.1` supplies the pQCD module used by the target code in `legacy_stack/`.
3. Work from the repository root with `PYTHONPATH=$PWD`. For the hyperonic programs also set
   `PYTHONPATH=$PWD:$PWD/hyperonic_pipeline`, `CERTIFIED_DDB_DIR=$PWD/legacy_stack/validated_code_DDB`,
   `CERTIFIED_ASTRO_DIR=$PWD/legacy_stack/ddb_astro_mod` and `JAX_PLATFORMS=cpu` (the helper
   `route2.common.use_hyperonic_pipeline()` does this from Python).
4. Unpack the observation files into `data/observations/` (`./reproduce.sh check` does it; so does the check).
5. The nucleonic programs are in `inference/anet/` (byte-identical to the published nucleonic code), the hyperonic
   ones in `hyperonic_pipeline/` (see its `CODE_MANIFEST.json`), and the Green EN stage scripts in
   `green_en/scripts/` and `fastsolver/` (see `docs/route2/STEPS_EVIDENCE_NETWORK.md`).

## 3. The banks, one by one

### 3.1 Nucleonic shared prior physics bank

**What it is.** 914,966 rows of the seven-parameter nucleonic DDB model. Each row holds the parameters
(`theta`), the nuclear-matter properties (`X`: ρ0, E0, K0, J, P at 0.08, 0.12 and 0.16 fm⁻³), R(M) and Λ(M) on
200 masses from 0.5 to 2.6 M☉ (`Rg`, `Lg`), the maximum mass (`MM`), R1.4, the pQCD log-likelihood at five
densities (`PQG`) and the GW170817 log-likelihood at nine tidal shifts (`GWG`). Rows 0–659,999 are uniform
prior draws; rows 660,000–914,965 are enrichment rows, and `logw_prior` is the exact counting correction that
brings every weight back to the uniform prior.

**File.** `shared_prior_physics_bank.npz`, downloaded as `full_nucleonic_shared_prior_physics_bank.npz` (size and
SHA-256 in Section 4).

**Settings.** The counts, the enrichment seed, the box and the radius windows are stored inside the file; the
uniform-stream seeds 1234 and 2026 are the defaults of `build_bank.py`. The check confirms all of them on all rows.

| Rows | Stream | Seed | Proposals |
|---|---|---|---|
| 0–199,999 | uniform prior | `default_rng(1234)` | 200,000 |
| 200,000–659,999 | uniform prior | `default_rng(2026)` | 460,000 |
| 660,000–743,399 (83,400) | B: nuclear box, all seven properties within 3 σ of A1 | `SeedSequence([20260729, 1, block])` | 160,000,000 in 80 blocks of 2,000,000 |
| 743,400–839,413 (96,014) | C: predicted R1.4 band, R1.4 in 9.4–11.2 km, Mmax ≥ 1.9 M☉ | `SeedSequence([20260729, 2, block])` | 2,640,000 in 44 blocks of 60,000 |
| 839,414–914,965 (75,552) | D: predicted R1.4 band, R1.4 in 14.8–18.0 km, Mmax ≥ 1.9 M☉ | `SeedSequence([20260729, 3, block])` | 1,320,000 in 22 blocks of 60,000 |

`logw_prior = log(660,000) − log(660,000 + 160,000,000·in_SB + 2,640,000·in_SC + 1,320,000·in_SD)`.

**Command.** The command line of the published bank is not known. The shipped builder's defaults are exactly the
settings above:

```bash
python inference/anet/build_bank.py --data-root data/observations \
    --output-dir runs/nucleonic_bank --phase all --workers <N>
```

It writes `final_bank.npz` and `final_exact.npz` (Section 3.2) with JSON reports, and is resumable.

**Run time and machine.** Building the published bank took 524.4 min. The machine and worker count are not
known. For scale, the check needs about 0.1–0.2 CPU-second per row for the TOV and likelihood terms on the
desktop.

**What the check confirms.** On all 914,966 rows (when the fixtures were made): every parameter row equals the
seeded draw of its stream, block and position; each enrichment row lies inside its own stream's support; the
support flags and `logw_prior` follow exactly from the stored columns; refitting the R1.4 ridge regression on the
uniform rows with `build_bank.py` gives the stored coefficients to 1.1e-10 and the prediction bands to 1.6e-12.
On 40 rows (8 per stream, including unsolved rows): all physics columns replay (table in Section 1).

**A rerun is not byte-identical.** The published file stores `OBS`, `SIG` and a note, while `build_bank.py` also
writes `logq` and `recipe`; and float results can differ in the last digits (about 1e-10 relative). The content
columns are the same.

### 3.2 Nucleonic exact astrophysical table

**What it is.** Row-aligned with the shared bank: `exact_j0030`, `exact_j0740`, `exact_j0437` (NICER), `exact_gw`,
`Rmm` (the radius at the maximum mass) and `Mmax_fresh`.

**File.** `shared_exact_astrophysical_table.npz`, downloaded as `full_nucleonic_shared_exact_astrophysical_table.npz`.

**Important.** The three NICER columns were made with the NICER grid of `legacy_stack/ddb_astro_mod/nicer_like.py`,
in which the J0740 sample weights enter twice. Only J0740 has sample weights, so only `exact_j0740` is affected:
it is the value before the single-weight correction. The corrected single-weight J0740 term
(`likelihoods/nicer/single_weight.py`) is what every downstream product uses (the independent cache in
Section 3.4 and the Green EN bank in Section 3.8). On the 40 replayed rows the two differ by up to 0.60 in log L.

**Command.** The command line of the published table is not known. `build_bank.py` (Section 3.1) makes this table
in its `exact` and `enrich` phases, but with the corrected J0740 term, so its `exact_j0740` column differs from the
published one; the other five columns agree. To reproduce the published J0740 column, evaluate
`build_bank.exact_row` with the grids of `ddb_ultranest_hyp.build_a1_likelihood_data()`, in which the J0740
weights enter twice (this is what the check does row by row).

**What the check confirms.** `Mmax_fresh` equals the bank's `MM` to 5.5e-13 on all rows, with the same unsolved
rows. On 40 rows, all six columns replay: log L to 2e-9 (with the double-weight grid for J0740), R(Mmax) and Mmax
to 1e-14.

### 3.3 Nucleonic uniform-prior support bank

**What it is.** `theta`, `X` and `MM` of rows 0–659,999 of the shared bank (the check confirms this on all rows).

**File.** `uniform_prior_support_bank.npz`, downloaded as `full_nucleonic_uniform_prior_support_bank.npz`.

**How it was made.** By the preparation script `prepare_nucleonic_clean_inputs.py` (SHA-256 `a2d2ad5a…`) on the
desktop as one Python process. The script is not included in the package; its command line and elapsed time are
not known.

**Rerun.** It is a slice of the shared bank:

```python
import json
import numpy as np
with np.load("shared_prior_physics_bank.npz") as bank:
    part = {key: bank[key][:660000] for key in ("theta", "X", "MM")}
np.savez_compressed("uniform_prior_support_bank.npz", **part, N0=np.int64(660000),
                    metadata=np.asarray(json.dumps({"rows": 660000, "fields": ["theta", "X", "MM"]})))
```

The arrays are identical to the published ones; the file bytes differ (compression and the metadata text).

### 3.4 Nucleonic independent common-physics cache

**What it is.** The 806,053 solved rows (finite `MM`) of the shared bank, in order: `theta`, `X`, the
astrophysical log-likelihood `LA` with the corrected single-weight J0740 term, and `LPC` (the bank's
`logw_prior`); plus `lpc_all_lse` (log-sum-exp of `logw_prior` over all 914,966 rows) and `n_proposed`.

**File.** `independent_common_physics_cache.npz`, downloaded as
`full_nucleonic_independent_common_physics_cache.npz`.

**How it was made.** By the same script, on the same desktop and in the same step as Section 3.3. That step
reports 914,966 proposed rows, 806,053 valid rows and a fresh J0740 single-weight replay discrepancy of at most
2.2e-9. The parent files are named by hash in the file's metadata (the shared bank, the exact table, another exact
cache and five fresh J0740 term files); the replay does not need them.

**Rerun with the shipped code.**

```python
import numpy as np
from workflows.a1_problem import A1Problem
with np.load("shared_prior_physics_bank.npz") as bank:
    solved = np.isfinite(bank["MM"])
    theta, X, logw = bank["theta"][solved], bank["X"][solved], bank["logw_prior"]
problem = A1Problem.from_data_root("data/observations", workers=<N>, model="ddb")
_, LA, _ = problem.evaluate_evidence_cache_batch(theta)   # max-mass + NICER + GW170817 + pQCD terms
np.savez_compressed("independent_common_physics_cache.npz", theta=theta, X=X, LA=LA,
                    LPC=logw[solved], lpc_all_lse=np.logaddexp.reduce(logw), n_proposed=np.int64(len(logw)))
```

`LA` for 806,053 rows is a full EOS/TOV and likelihood pass (about 0.1 CPU-second per row on the desktop, as
measured by the check). The check confirms `theta`, `X`, `LPC` and `lpc_all_lse` on all rows (the last to 2.8e-13)
and replays `LA` on 30 rows to 8e-10.

### 3.5 Hyperonic uniform-prior banks (four parts)

**What they are.** 150,000 draws each from the nine-parameter DDB ΛΞ⁻ prior with `X`, `Rg`, `Lg`, `MM`, `R14`,
a `valid` flag and a `failure_code`, computed with the production EOS `hyperonic_pipeline/ddb_hyperon_eos.py`
and the certified nuclear forward in `legacy_stack/`.

**Files.** `hyp_bank_part{1,2,3,4}.npz`, downloaded as `full_hyperonic_uniform_bank_part{1,2,3,4}.npz`; named by
SHA-256 in `hyperonic_pipeline/prior_manifest.json` and as `source_bank_sha256` by the uniform training caches
(Section 3.6).

**Settings (stored inside each file).**

| Part | Seed | Rows | Valid rows | Workers | Generation time |
|---|---|---|---|---|---|
| 1 | 91 | 150,000 | 103,707 | 56 | 3,364 s |
| 2 | 92 | 150,000 | 103,947 | 56 | 3,352 s |
| 3 | 93 | 150,000 | 104,094 | 56 | 3,359 s |
| 4 | 94 | 150,000 | 103,933 | 56 | 3,374 s |

TOV step 0.012; schema `ddbhy-bank-v1`. The machine and the command line are not known.

**Command.**

```bash
python hyperonic_pipeline/generate_ddbhy_bank.py --n 150000 --seed 91 --workers 56 --output hyp_bank_part1.npz
# parts 2-4: --seed 92, 93, 94
```

**What the check confirms.** All 600,000 draws equal `default_rng(seed)` uniform draws in the prior box; `valid`
equals `failure_code == 0`; the mass grid, density grid and TOV step equal the program's. On 8 rows per part
(6 valid, 2 failed): failure codes identical, X to 7e-13 σ, curves to 8e-9.

### 3.6 Hyperonic A-NET training caches (nine parts)

The four uniform and the two support caches are prior-only physics caches, shared by A-NET, TSNPE and the
Evidence Network; the three proposal caches belong to A-NET alone.

The A-NET scenario bank lists these nine caches by SHA-256. Each cache row holds the portable-EOS prediction, the
exact astrophysical log-likelihood with its four components and three NICER sources, a `target_valid` flag and
R(M), Λ(M), Mmax. Machine, workers and wall time are stored in each cache's JSON report. The commands below follow
those reports and the programs' options (the full sequence is in `docs/route2/STEPS_ANET_HYPERONIC.md`,
Steps 3–6).

| Cache | Source rows | Machine (workers) | Wall time |
|---|---|---|---|
| uniform part 1 | all solved rows of uniform bank 1 (103,707) | cluster node 1 (62) | 2,414 s |
| uniform part 2 | uniform bank 2 (103,947) | cluster node 2 (62) | 1,249 s |
| uniform part 3 | uniform bank 3 (104,094) | cluster node 2 (62) | 1,250 s |
| uniform part 4 | uniform bank 4 (103,933) | cluster node 1 (62) | 2,420 s |
| support part 1 | screen rows 0–41,881 | desktop (20) | 1,106 s |
| support part 2 | screen rows 41,882–83,763 | cluster node 2 (62) | 491 s |
| proposal part 1 | Student draws 0–124,999 | cluster node 1 (62) | 2,880 s |
| proposal part 2 | Student draws 125,000–249,999 | cluster node 2 (62) | 1,484 s |
| proposal part 3 | Student draws 250,000–299,999 | desktop (20) | 1,341 s |

Wall times are the `wall_seconds` of the JSON reports; the start and finish stamps stored inside the cache files
give 1.5–13.5 s less. The support and proposal reports name `hyp_bank_part1.npz` (`c188c4d0…`) as the template
bank (it supplies only the mass grid and the prior box).

```bash
python hyperonic_pipeline/build_clean_uniform_hyperonic_training_cache.py --bank hyp_bank_part1.npz \
    --data-root data/observations --output uniform_training_cache_part1.npz --workers 62
python hyperonic_pipeline/screen_clean_hyperonic_support_enrichment.py --output support_screen_160m.npz \
    --proposals 160000000 --block 2000000 --cbox 3 --seed 20260729
python hyperonic_pipeline/evaluate_clean_hyperonic_support_cache.py --screen support_screen_160m.npz \
    --template-bank hyp_bank_part1.npz --data-root data/observations \
    --output support_training_cache_part1.npz --workers 20 --start 0 --stop 41882
python hyperonic_pipeline/evaluate_clean_hyperonic_proposal_cache.py \
    --proposal student_beta1_48c_300k_seed20260992.npz --checkpoint student_beta1_48c_df5_s075_seed20260990.joblib \
    --template-bank hyp_bank_part1.npz --data-root data/observations \
    --output proposal_training_cache_part1.npz --workers 62 --start 0 --stop 125000
```

The support screen: 160,000,000 draws in 80 blocks of 2,000,000, box of ±3 σ, seed 20260729, 83,764 rows kept,
827 s on the desktop. The Student draws: 300,000 rows, seed 20260992.

**What the check confirms.** On all rows: the uniform caches hold exactly the solved rows of their bank; the support
caches hold their screen rows; every one of the 83,764 screen rows is the seeded draw of its block; the proposal
caches hold the Student draws and their log q. On 3–6 rows per part: draws, predictions, all likelihood columns,
curves and `target_valid` (Section 1).

### 3.7 Hyperonic Stage-A MIS evaluation

**What it is.** 1,720,000 rows from eight tempered proposal rounds (r8–r15), each with its proposal density
`logq` and exact log-likelihood `logl`, combined as one deterministic mixture. It is the input of the defensive
Student fit (the Student's fit report names this file as its input).

**Files.** `eval_mis_round8_round15_1720k.npz`, downloaded as `full_hyperonic_stage_a_mis_1720k.npz`, and its JSON
report `eval_mis_round8_round15_1720k.json`. The report lists the eight component evaluations and proposal
checkpoints by SHA-256; the check confirms that the bank is exactly their concatenation. 1,245,208 rows are
valid; full-posterior ESS 3,282.44; target β = 0.2208750108267128.

**Commands.** The command lines of the rounds are not known. The final combination and the round structure are
given in `docs/route2/STEPS_ANET_HYPERONIC.md` (Step 2). The exact likelihood of every round came from
`hyperonic_pipeline/fast_hyperonic_likelihood_shard.py` with the target of `hyperonic_pipeline/ddb_ultranest_hyp.py`,
in which the J0740 weight enters twice (as in Section 3.2). The check confirms this: with that target rebuilt from
`data/observations`, the stored values replay to 1.8e-6 (tolerance 1e-4). The run times and machines of the rounds
are not known.

**Documented defect of the stored bank.** In six of the eight rounds (r9–r14), a leading block of rows contains
no ordinary likelihood value at all:

| Round | Bank rows | Stored as rejected (−1e100) | Stored as −4e30 |
|---|---|---|---|
| r9 | 280,000–329,999 | 47,925 | 2,075 |
| r10 | 480,000–529,999 | 48,144 | 1,856 |
| r11 | 680,000–734,999 | 52,861 | 2,139 |
| r12 | 900,000–954,999 | 53,151 | 1,849 |
| r13 | 1,120,000–1,170,999 | 49,831 | 1,169 |
| r14 | 1,320,000–1,359,999 | 39,170 | 830 |

In these blocks 96–98% of the rows are stored as rejected, against 8–24% in every other 10,000-row window. The
−4e30 rows are the ones whose stable branch ends below 1 M☉, so their NICER and GW terms return −1e30 before any
density grid is read. Every row whose likelihood needs the NICER or GW density grids is stored as rejected. The shipped code
evaluates most of these rows: in the check, 5 of the 6 replayed rejected rows from these blocks get ordinary
values (up to −49.2), while rejected rows outside the blocks stay rejected. So the stored bank gives zero weight
to most of these 291,082 rows although they have a finite likelihood (elsewhere only 8–24% of the rows fail).

Effect: the Stage-A mixture and its stored ESS were computed with these rows at zero weight. Stage A only
shapes the defensive Student proposal; the Student's density enters the A-NET importance weights exactly, so the
A-NET posteriors and evidences are not biased by this (the Student's efficiency may be slightly lower). The check
reports this defect every time it runs (and fails on it only with `--strict`).

### 3.8 Green EN nucleonic training bank

**What it is.** 848,017 rows: the 806,053 solved rows of the shared bank (Section 3.4) followed by 41,964 S_ext rows
(rows whose nuclear properties fall in a shifted-scenario box but outside the A1 box). Columns: `X`, `theta`, `R`,
`MM` (float32), `LA`, `LPC`, `AC` (four likelihood components), `NIC` (three NICER sources).

**File.** `nucleonic_conditional_physics_Sext_seed20260924.npz`, downloaded as `full_green_en_nucleonic_bank.npz`
and named by the JSON reports of the three nucleonic training runs. Parent S_ext exact rows:
`nucleonic_Sext_exact_seed20260924_parallel8_aligned.npz`, downloaded as `full_green_en_nucleonic_sext_exact_rows.npz`
(named by hash in the bank's metadata).

**Command (16.8 s on the desktop).**

```bash
python green_en/scripts/assemble_green_en_support_extension.py nucleonic \
    --base-bank <nucleonic_conditional_physics.npz> --raw-bank <shared_prior_physics_bank.npz> \
    --exact-table <shared_exact_astrophysical_table.npz> --fixed-j0740-terms <folder of the five J0740 term files> \
    --extension-screen nucleonic_Sext_seed20260924_160M.npz \
    --extension-exact nucleonic_Sext_exact_seed20260924_parallel8_aligned.npz \
    --output nucleonic_conditional_physics_Sext_seed20260924.npz
```

The S_ext screen (seed 20260924) and the exact S_ext rows come first; their commands, machines and times are in
`docs/route2/STEPS_EVIDENCE_NETWORK.md` (Section 5, N1–N4).

**What the check confirms.** On all rows: the first 806,053 rows are the float32 view of the shared bank (`X`,
`theta`, `R`, `MM`) with the independent cache's `LA`, and with `NIC` and `AC` taken from the exact table where the
assembler says so; the S_ext rows are the valid rows of the exact S_ext file; `LPC` follows exactly from the stored
flags, box memberships and proposal counts; every S_ext row is the seeded draw of its block. On 30 rows (20 from
the shared part, 10 S_ext): all columns replay, including the corrected J0740 term in `NIC` (to 5e-13).
For the shared-bank rows, `AC[:,0]` holds R(Mmax), not the maximum-mass term; the check replays it as such.

### 3.9 Green EN hyperonic training bank

**What it is.** 645,084 rows: the 582,551 rows of the parent bank (the four uniform caches and seven support
caches of two screens, seeds 20260729 and 20260922) followed by the 62,533 S_ext rows (seed 20260923; rejected
rows have `LA = −∞`). Same columns as Section 3.8.

**File.** `hyperonic_conditional_physics_Sext_seed20260923_parent02357702.npz`, downloaded as
`full_green_en_hyperonic_bank.npz` and named by the JSON reports of the three hyperonic training runs. Parent bank:
`hyperonic_green_expanded_fullprior_bank_seed20260922.npz`, downloaded as `full_green_en_hyperonic_parent_bank.npz`.

**Command (10.6 s on the desktop).**

```bash
python green_en/scripts/assemble_green_en_support_extension.py hyperonic \
    --base-bank hyperonic_green_expanded_fullprior_bank_seed20260922.npz \
    --extension-screen hyperonic_Sext_seed20260923_160M.npz --extension-exact hyperonic_Sext_exact_seed20260923.npz \
    --output hyperonic_conditional_physics_Sext_seed20260923_parent02357702.npz \
    --base-proposals 600000 --b3-proposals 320000000 --hyperonic-uniform-rows 415681
```

The parent bank, the second support stream and the S_ext stages are in `docs/route2/STEPS_EVIDENCE_NETWORK.md`
(Section 6).

**What the check confirms.** On all rows: the parent rows equal the parent bank, which equals the concatenation of
its eleven parts in the listed order; the S_ext rows equal the exact S_ext file; `LPC` follows exactly from the
box memberships; all 83,106 rows of the second support screen and all 62,533 S_ext rows are seeded draws. On 30 rows
(including rejected ones): all columns replay with the portable EOS (the S_ext rows were made with the compiled
solver of `fastsolver/`; they agree to 3e-10 in log L).

## 4. Files, sizes and SHA-256

The replay check needs none of these files. They are needed only for a from-scratch rerun of a later stage, or to
re-extract the fixtures. The table gives each file under the name used in the steps above and, where it can be
downloaded, under its download name.

**Which of them can be downloaded.** All of them are in the optional full-rerun download
([FULL_RERUN_FILES.md](FULL_RERUN_FILES.md)), except these intermediate files:
- the Stage-A JSON report and the eight Stage-A round files r8–r15, whose rows are all contained in the
  downloadable Stage-A MIS evaluation;
- the second support screen (seed 20260922) and the five second-stream caches, whose product is the downloadable
  parent bank.

| Bank | File (download name in `route2/downloads/`) | Bytes | SHA-256 |
|---|---|---|---|
| nucleonic shared prior physics bank | `shared_prior_physics_bank.npz` (`full_nucleonic_shared_prior_physics_bank.npz`) | 3,157,558,246 | `06dda51aa7c1cbbf421f365cd782e6fb4dc814becfa6eb07a749ce1345c33a07` |
| nucleonic exact table | `shared_exact_astrophysical_table.npz` (`full_nucleonic_shared_exact_astrophysical_table.npz`) | 43,919,890 | `e2dfb84d56a6f015ac12d14ae00d11b3098ce43c34bb25a4c2bfccbf27c66fa9` |
| nucleonic uniform-prior support bank | `uniform_prior_support_bank.npz` (`full_nucleonic_uniform_prior_support_bank.npz`) | 75,307,495 | `4f50897beb3fdff046061229b694ca84d560462d4f29ab5b7f8c874b5969e8ae` |
| nucleonic independent cache | `independent_common_physics_cache.npz` (`full_nucleonic_independent_common_physics_cache.npz`) | 91,359,565 | `280f6f60bd93312ef34df821e037a6c4a61ea55bf1671c76f8efd852c7557a3d` |
| hyperonic uniform bank 1 | `hyp_bank_part1.npz` (`full_hyperonic_uniform_bank_part1.npz`) | 529,515,486 | `c188c4d0fc133271a384628b78c08928fc37a8a37577d9446079efb31c4299ef` |
| hyperonic uniform bank 2 | `hyp_bank_part2.npz` (`full_hyperonic_uniform_bank_part2.npz`) | 529,515,486 | `e9beaa28c8c90b9cbe275d5695e87fd90566a939744d5ddb16ed2c480be545d9` |
| hyperonic uniform bank 3 | `hyp_bank_part3.npz` (`full_hyperonic_uniform_bank_part3.npz`) | 529,515,482 | `2589bfec123f8d5b96003c7581183b7722ed2af880ae07d2c61941a7d3cedc06` |
| hyperonic uniform bank 4 | `hyp_bank_part4.npz` (`full_hyperonic_uniform_bank_part4.npz`) | 529,515,486 | `2880d62d8d0f64698f6ba4008bfcca8816781bd802eebb04a669c318e32e5851` |
| uniform cache 1 | `uniform_training_cache_part1.npz` (`full_hyperonic_uniform_training_cache_part1.npz`) | 177,742,685 | `e51cf2f181ba3a814a522c5ababa70853dccf25e3852e59c8cbb4f146d5eb2cf` |
| uniform cache 2 | `uniform_training_cache_part2.npz` (`full_hyperonic_uniform_training_cache_part2.npz`) | 178,852,452 | `e7ac3193ca790faf5e3e778e0c0bae03cbfde45394e426023716cbfeac0b098e` |
| uniform cache 3 | `uniform_training_cache_part3.npz` (`full_hyperonic_uniform_training_cache_part3.npz`) | 178,474,580 | `e13a85668f4f7e4b633b841d5ad80f2a0c1507113d04a9eb98c0e97f9730499c` |
| uniform cache 4 | `uniform_training_cache_part4.npz` (`full_hyperonic_uniform_training_cache_part4.npz`) | 178,557,033 | `575ebf362ad6b944aa24a0a39664d53c1d30b96be502cfd296e71706ea1a27f4` |
| support cache 1 | `support_training_cache_part1.npz` (`full_hyperonic_support_training_cache_part1.npz`) | 53,842,942 | `479206d0dd72be042eb90d341a4f1967d9535583f6b200775fee93ccabe0c016` |
| support cache 2 | `support_training_cache_part2.npz` (`full_hyperonic_support_training_cache_part2.npz`) | 53,792,694 | `28e0894d07b9a14da2fa27cc8594a7b3dd4ac13ee213e81848768a21ef6f6ee4` |
| support screen (seed 20260729) | `support_screen_160m.npz` (`full_hyperonic_support_screen_160m.npz`) | 10,224,977 | `442da1a47e447331166d51e1b83144ed67005739f1c90a90727c3cc0245907e3` |
| proposal cache 1 | `proposal_training_cache_part1.npz` (`full_hyperonic_proposal_training_cache_part1.npz`) | 262,169,814 | `5a48a8870930ed013d412ce30fbf8722e2e3188e39a6011d13b497651131e202` |
| proposal cache 2 | `proposal_training_cache_part2.npz` (`full_hyperonic_proposal_training_cache_part2.npz`) | 261,973,671 | `118acc3268f4852a7d8f1fa0dcecf3f3eb2f26f2e2e17865c74dd76dfca289a9` |
| proposal cache 3 | `proposal_training_cache_part3.npz` (`full_hyperonic_proposal_training_cache_part3.npz`) | 104,604,478 | `d1d80765914fd7ef1afb0846fb8032b9691ab64faa0c4f7121c9df5ac522086d` |
| Student proposal draws | `student_beta1_48c_300k_seed20260992.npz` (`full_hyperonic_student_proposal_draws_300k.npz`) | 22,753,264 | `2b2e7827d1f302e8404cec7fd4cf7354c77e10d02153559b644d170ab05f017f` |
| Stage-A MIS evaluation | `eval_mis_round8_round15_1720k.npz` (`full_hyperonic_stage_a_mis_1720k.npz`) | 139,000,136 | `7740c7cd625f89e49758e74f03635e67f1b74404916ac8ad8646faf1251c4e8d` |
| its JSON report | `eval_mis_round8_round15_1720k.json` (not downloadable) | 5,818 | `5cfc7da0324d62f2db1c2d33c42375fd0adc651cf9056a1829f14e60ec0ba5ea` |
| Stage-A round r8 | `eval_round8_combined_280k.npz` (not downloadable) | 22,717,235 | `5b3ed62c2e60887fa98183c34fb39538ff9a399d1caa4ceb92dae6a354412617` |
| Stage-A round r9 | `eval_round9_gmm32_seed_20260959.npz` (not downloadable) | 15,926,556 | `e91759ef4113ad4006b8497affc76ba72be93c22330e05b64cadc057de982722` |
| Stage-A round r10 | `eval_round10_ensemble_seed_20260963.npz` (not downloadable) | 15,967,060 | `25a9b7db866c2f6a76acd73d36c6c180681a26d60fc74421a6eb07ea2d39fc51` |
| Stage-A round r11 | `eval_round11_ensemble_seed_20260965.npz` (not downloadable) | 17,568,219 | `f9e92e9d7438b819496dee5b1b97b13d882bfb3da6cf4b035517f2584de5e9d9` |
| Stage-A round r12 | `eval_round12_ensemble_seed_20260967.npz` (not downloadable) | 17,589,533 | `274ef673a5c689684ff8c1a7670b175f66d0e6d99b028c5964c26fd3fd78495d` |
| Stage-A round r13 | `eval_round13_ensemble_seed_20260971.npz` (not downloadable) | 16,057,954 | `bc5919e822ed7e864f0432c155fc5a6c5ff14b51a3634987c703cf4a1497e3e4` |
| Stage-A round r14 | `eval_round14_ensemble_seed_20260973.npz` (not downloadable) | 16,151,328 | `31056d5cbe299a4bd4f36f9d73aec432989d3d8810a1974d732610acc2f38945` |
| Stage-A round r15 | `eval_round15_student_seed_20260978.npz` (not downloadable) | 16,491,836 | `68e897b6d18691f99292d42abdcd4dfe570090fb5f92caf8e0176645d14419ef` |
| Green EN nucleonic bank | `nucleonic_conditional_physics_Sext_seed20260924.npz` (`full_green_en_nucleonic_bank.npz`) | 428,342,848 | `0421508c9a15025080f57c036bf489df3a028ae951f4bac018b3ce6a19817929` |
| its S_ext screen | `nucleonic_Sext_seed20260924_160M.npz` (`full_green_en_nucleonic_sext_screen.npz`) | 6,889,520 | `e43000e05e088e94fe5cfe0f6e8d66db7b934bd6e46b4d98a232223fd7e5d514` |
| its S_ext exact rows | `nucleonic_Sext_exact_seed20260924_parallel8_aligned.npz` (`full_green_en_nucleonic_sext_exact_rows.npz`) | 107,317,349 | `1fcd9396075e44448b10ea8032f4f44e16dcdd205cc4cf17f2330cff40fb317c` |
| Green EN hyperonic bank | `hyperonic_conditional_physics_Sext_seed20260923_parent02357702.npz` (`full_green_en_hyperonic_bank.npz`) | 265,184,413 | `f669d84ffca0dd92d4164ec2c0cc4a30a4baa07c3cd8a3ad68a27647a382ba00` |
| its parent bank | `hyperonic_green_expanded_fullprior_bank_seed20260922.npz` (`full_green_en_hyperonic_parent_bank.npz`) | 244,687,891 | `02357702e85e86d8154d1641e7d4fb818c66145d0364e127ad520986c2fe1fce` |
| second support screen (seed 20260922) | `support_screen_seed20260922_160m.npz` (not downloadable) | 10,144,415 | `75f1d4ace983b4b5802ebcd109a5e46fccc1ce01048a85f6239c0b132777c3c4` |
| second-stream caches (5 parts) | `support_training_cache_seed20260922_part{0,1,_local2,3,4}.npz` (not downloadable) | 19,115,261; 7,740,777; 7,640,686; 23,012,256; 48,867,850 | `65b49dad…`, `4bca1b08…`, `e72cd8c5…`, `6f62c757…`, `baaf83b2…` (full values in `route2/fixtures/banks/expected.json`) |
| its S_ext screen | `hyperonic_Sext_seed20260923_160M.npz` (`full_green_en_hyperonic_sext_screen.npz`) | 7,838,415 | `629c156a4e7da32cd3b5d11a31660f5492f3351b24799e7d846989d4e9fca158` |
| its S_ext exact rows | `hyperonic_Sext_exact_seed20260923.npz` (`full_green_en_hyperonic_sext_exact_rows.npz`) | 76,455,064 | `aa5a8b23d7f02c6245d617c3b3bd01947fe7753f3f8b8eda3055d982a1af88d9` |

Totals: nucleonic shared-physics files 3.37 GB (the shared bank alone 3.16 GB); hyperonic uniform banks 2.12 GB;
the nine caches with the screen and Student draws 1.48 GB; Stage A 0.28 GB with its rounds; Green EN banks and
parents 1.25 GB.

## 5. How the fixtures were made

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=3 python assemble_route2_banks.py   # in the package build folder (a build-time script kept outside the repository)
```

The script reads the bank files of Section 4 (read-only), checks the SHA-256 of every file,
re-verifies all the relations listed above on all rows, and only then writes `route2/fixtures/banks/`: `rows.npz`
(the chosen rows; seed [20260925, k] per bank), `expected.json` (hashes, settings, relation results),
`MANIFEST.json` and a byte-identical copy of the defensive Student checkpoint (65 kB). It refuses to overwrite the
folder. It took 102 s on the desktop (3.9 GB peak memory).

## 6. Gaps

1. **Stage-A defect.** The stored Stage-A bank marks as rejected 291,082 rows in the leading blocks of rounds
   r9–r14, most of which the shipped code evaluates (Section 3.7). The final A-NET weights are not affected.
2. **Nucleonic shared bank and exact table.** The settings and a 524.4-minute build time are known, but not the
   command, machine or worker count. `build_bank.py` regenerates the same parameter rows and physics columns
   (verified), not byte-identical files.
3. **Double-weight J0740 column.** `exact_j0740` in the exact table uses the double-weight NICER grid. No shipped
   command writes that column in one step (Section 3.2 gives the route). Downstream products use the corrected
   term, which the shipped code computes.
4. **Support bank and independent cache.** Their preparation script (SHA-256 `a2d2ad5a…`) is not included in the
   package; its command line and run time are not known. Both are rebuilt from the shared bank with the steps
   above (verified on all rows; `LA` replayed on 30 rows).
5. **Hyperonic uniform banks.** The command line and machine are not known; the seed, worker count and time are
   stored in each file.
6. **Stage-A rounds.** The per-round commands, machines and times are not known, and the likelihood data object
   used by the rounds is not available. The check rebuilds that target from the observation files; the stored
   values then agree to 1.8e-6 (tolerance 1e-4), not bitwise.
7. **Student draws** are reproduced to 1.2e-12, not bitwise (made on another machine).
8. **Green EN banks** store `X`, `theta`, `R` and `MM` in float32, so they cannot be compared bitwise with a
   float64 rerun; the float32 view of the float64 parents is verified exactly on all rows.
9. **Not replayed here.** The Stage-A proposal densities (`logq`) and the hyperonic A-NET counting correction
   are not recomputed by this check; the A-NET checks replay the final importance weights.
