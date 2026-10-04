# Route 2 manual: run everything from scratch

This manual explains, step by step, how to check and rerun the complete inference pipeline of the paper:
- the prior physics banks;
- UltraNest;
- A-NET+IS;
- TSNPE+MIS;
- the Evidence Network.

It covers both equation-of-state models (nucleonic DDB and hyperonic DDBΛΞ⁻). If you only want to rebuild the figures and tables of the paper from the saved results, use Route 1 ([MANUAL_ROUTE1.md](MANUAL_ROUTE1.md)); it is much faster.

> ## ⚠ Read this first: hardware changes the timings, and reruns change the last digits
>
> **Timings.** The computing times in Tables VII and VIII were measured on the hardware used for the paper:
> - desktop CPUs and GPUs;
> - a CPU cluster;
> - cloud GPUs.
>
> UltraNest times are converted to a stated reference speed. On your hardware every time will be different: CPU model, number of cores, GPU model, memory, disk and library versions all matter. **The speed-up factors in Tables VII–VIII will therefore also be different.** Compare them only within one machine.
>
> **Results.** Retraining a network or rerunning a sampler does not give bit-identical numbers:
> - GPU and CPU arithmetic differs between hardware and library versions, so trained networks differ slightly;
> - nested sampling is random.
>
> What stays the same:
> - A-NET+IS and TSNPE+MIS always correct their neural proposal with the exact likelihood and the full prior. A fresh run therefore gives the same posterior up to Monte Carlo error, but its effective sample size (ESS) can differ.
> - Evidences and credible intervals from a fresh run should agree with the paper within the quoted uncertainties, not to the last digit.
> - Only the **replay checks** of step 3 are expected to match the paper's saved numbers to the stated tolerances, because they reuse the frozen networks and inputs.

---

## Step 1: download

```bash
git clone https://github.com/prashantstar123/NeuronStar.git
cd NeuronStar
```

## Step 2: install the exact Python packages (one time)

Route 2 needs more packages than Route 1: PyTorch, JAX, numba, sbi, zuko and UltraNest. The Route 2 environment also contains everything Route 1 needs. Use **one** option.

**Option A, conda or mamba:**

```bash
conda env create -f environment-route2.yml
conda activate ddb-route2
```

**Option B, plain Python 3.11:**

```bash
python3.11 -m venv .venv-route2
source .venv-route2/bin/activate
pip install -r requirements-route2.txt
```

- Route 2 needs Linux on x86-64 (on Windows, use WSL): the pinned CPU build of PyTorch exists only there.
- The installation compiles one package (UltraNest) and so needs a C compiler; on Ubuntu or Debian install it once with
  `sudo apt install build-essential`.
- This installs the CPU build of PyTorch. It took about 2 minutes on the test machine and uses about 2.5 GB.
- **With an NVIDIA GPU**, afterwards replace PyTorch by the CUDA build of the same version:

  ```bash
  pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
  ```

  All checks run on CPU unless you ask for the GPU (`--device cuda`).

## Step 3: check every stage (replay checks)

```bash
./reproduce.sh route2-check
```

**What it does.** It first downloads the large check inputs from the GitHub release of this repository (frozen flows and stage archives; about 130 MB) into `route2/downloads/` and verifies each file's SHA-256. Then each check replays one stage of the pipeline from frozen inputs and compares the result with the paper's saved numbers:

| Check | What it replays | CPU time on the test machine |
|---|---|---|
| `evidence` | Evidence Network: the 48 member values and the 16 Table VI evidences, from the six frozen networks | 15 s |
| `hyperonic_anet` | Hyperonic A-NET (A1 and the seven queries): importance weights, ESS and log Z of every certificate; the 6,000-draw resamples; proposal densities from the frozen networks; exact likelihoods from the data | 2 min |
| `nucleonic_anet` | Nucleonic A-NET (eight configurations): weights, NICER correction, pooling, resamples, curve selections, exact likelihoods, and the proposal density along each flow path | 1 min |
| `tsnpe_nucleonic` | Nucleonic TSNPE+MIS: weights, ESS and log Z; the two Gaussian proposal parts rebuilt from the stored rows; the frozen flow; exact likelihoods; the published posterior (bit for bit) | 20 s |
| `tsnpe_hyperonic` | Hyperonic TSNPE+MIS: weights, ESS and log Z; both resamples; proposal densities from the frozen mixture and flows; exact likelihoods | 45 s |
| `ultranest` | UltraNest: the UltraNest column of Table VI rebuilt from the 27 saved runs, the corrected posterior weights, 72 exact-likelihood rows and the drivers | 3.5 min |
| `banks` | Prior physics banks: 299 stored rows of 20 banks recomputed (draws, stellar curves, likelihood columns) | 1.5 min |

In total about 10 minutes on 3 CPU cores. The `banks` check also prints one **documented defect of an intermediate
bank**: in the hyperonic Stage-A bridge, one machine's shards stored 291,082 rows as rejected because its
likelihood-data step failed. That bank only shaped the defensive Student proposal component, whose exact density
enters every final A-NET weight, so no posterior, evidence, figure or table depends on it (the `hyperonic_anet`
check replays all eight results). Details: `docs/route2/STEPS_BANKS.md`, Section 3.7.

A successful run ends with:

```text
ROUTE 2 PASSED: 7/7 checks in 8.1 min
```

**Why this is enough to trust the pipeline without rerunning it.** Each stage is replayed exactly where it is deterministic. For example:
- importance weights, ESS and evidences are recomputed from the saved likelihoods and proposal densities;
- the resamples behind the figures are redrawn with their saved seeds;
- the exact likelihood is recomputed for sample rows from the data files;
- the proposal densities are recomputed from the frozen networks; the two Gaussian parts of the nucleonic TSNPE
  mixture are rebuilt from the stored rows.

Where floating-point hardware differences are unavoidable, the tolerance used is stated and justified in each check's description (`python -m route2.checks.<name> --help`).

## Step 4: run the code end to end on tiny inputs (smoke runs)

```bash
./reproduce.sh route2-smoke
```

Each smoke run executes a stage with tiny settings: a few hundred prior draws, a few training steps, a few hundred proposals. It confirms that the code, the environment and the data work together. On the test machine the four smoke runs took 7.2 minutes on 3 CPU cores and ended with:

```text
ROUTE 2 PASSED: 4/4 smoke runs in 7.2 min
```

Their numbers are not scientific results.

## Step 5: the full rerun from scratch (optional, long)

The full pipeline took days of CPU and GPU time for the paper (Tables VII–VIII). The commands and settings for every stage, with the seeds where available, are in `docs/route2/`:

| Order | Stage | Document | Needs |
|---|---|---|---|
| 1 | Prior physics banks and caches (both models) | `docs/route2/STEPS_BANKS.md` | CPU cluster or many cores |
| 2 | UltraNest reference runs | `docs/route2/STEPS_ULTRANEST.md` | CPU cluster |
| 3 | Nucleonic A-NET+IS | `docs/route2/STEPS_ANET_NUCLEONIC.md` | one GPU + CPU cores |
| 4 | Hyperonic A-NET+IS | `docs/route2/STEPS_ANET_HYPERONIC.md` | one GPU + CPU cluster |
| 5 | Nucleonic TSNPE+MIS | `docs/route2/STEPS_NUCLEONIC_TSNPE.md` | one GPU + CPU cores |
| 6 | Hyperonic TSNPE+MIS | `docs/route2/STEPS_HYPERONIC_TSNPE.md` | one GPU + CPU cluster |
| 7 | Evidence Network | `docs/route2/STEPS_EVIDENCE_NETWORK.md` | one GPU + CPU cores |

UltraNest (stage 2) needs only the data and the likelihood code; stages 3–7 start from the prior banks of stage 1.
No stage uses another method's results: the methods are independent by construction, which is what makes their
agreement in the paper meaningful.

**Skip stage 1 if you like.** The banks, caches and other large intermediate files used for the paper (about 11 GB)
are attached to the GitHub release. Download and verify them with:

```bash
./reproduce.sh route2-fetch --group full_rerun
```

They are saved in `route2/downloads/`. Files larger than 1.9 GB are downloaded in parts, joined and checked
automatically.

**The downloaded files have different names.** In the release every file has a unique name that starts with `full_`,
while the step guides use the file names of the paper's runs (for example, the guides' `shared_prior_physics_bank.npz` is
`route2/downloads/full_nucleonic_shared_prior_physics_bank.npz`). [docs/route2/FULL_RERUN_FILES.md](docs/route2/FULL_RERUN_FILES.md)
gives both names for all 32 files, shows how to make a folder of links with the names used in the guides, and lists what is
not in the download.

Each document gives the commands with the settings used for the paper, says where a setting of the paper's run is not known, and lists any known gap.

Run the stages in the order of the table. After each stage, compare your outputs with the paper's saved results in the same way the replay checks do, keeping the caution at the top of this manual in mind.

## What cannot be rerun from this package

Route 1 checks the published figures and tables against the saved results, and step 3 replays the calculations behind them. A complete rerun from scratch, however, meets these gaps. Each is
explained in the guide named in brackets.

1. **Hyperonic mass–radius and mass–tidal curves.** `hyperonic_pipeline/fast_curve_full_shard.py` imports a
   helper, `fast_newton_hyperon_audit`, that is not part of this package. The curve
   commands of hyperonic A-NET steps 12–13 and hyperonic TSNPE step 10 stop at that import. The paper's hyperonic
   bands were computed from the saved curve files of these steps, and Route 1 rebuilds the figures from those saved
   results, so no published number is affected ([STEPS_ANET_HYPERONIC.md](docs/route2/STEPS_ANET_HYPERONIC.md), Gaps).
2. **Part of the nucleonic Evidence Network chain.** Three frozen inputs are not provided: the base bank and the
   fixed J0740 terms of step N4, and the exact template of step N3. The products of those steps are in the
   full-rerun download, so a rerun can start at step N5. The label matrices of steps N5 and H5 (7.9 GB) are not
   provided either; they take a few GPU-minutes to rebuild
   ([STEPS_EVIDENCE_NETWORK.md](docs/route2/STEPS_EVIDENCE_NETWORK.md), Gaps).
3. **Missing seeds and command lines.** Some commands, seeds and logs of the paper's runs are not available. The
   guides give commands rebuilt from the code defaults and the saved metadata for them, so those steps can be
   repeated only statistically. Examples: the nucleonic TSNPE training seeds, and the round-zero estimator of
   hyperonic TSNPE, which is not available
   ([STEPS_NUCLEONIC_TSNPE.md](docs/route2/STEPS_NUCLEONIC_TSNPE.md),
   [STEPS_HYPERONIC_TSNPE.md](docs/route2/STEPS_HYPERONIC_TSNPE.md), Gaps).
4. **Hyperonic Stage-A bank.** In rounds r9–r14, 291,082 rows are stored as rejected, most of them wrongly (see
   step 3 above). No published number depends on them ([STEPS_BANKS.md](docs/route2/STEPS_BANKS.md), Gaps).

---

## What is in the repository for Route 2

| Folder or file | Contents |
|---|---|
| `eos/`, `tov/`, `likelihoods/`, `workflows/` | the equations of state, TOV solver, likelihoods and the exact targets (portable production code) |
| `inference/anet/`, `inference/tsnpe/`, `inference/ultranest/` | nucleonic A-NET, TSNPE and UltraNest code, with the frozen nucleonic networks (identical to those used for the paper) |
| `inference/evidence_network/` | the Evidence Network library (query engines, label builder, models) |
| `hyperonic_pipeline/` | the hyperonic A-NET pipeline; `hyperonic_pipeline/tsnpe/` is the hyperonic TSNPE pipeline |
| `legacy_stack/` | the certified DDB forward model and astrophysical likelihood modules that the hyperonic pipeline imports |
| `green_en/` | the Evidence Network trainer and stage scripts |
| `fastsolver/` | the compiled hyperonic solver used for the Evidence Network support-extension rows |
| `route2/checks/`, `route2/smoke/` | the replay checks and smoke runs |
| `route2/fixtures/` | small frozen inputs of the checks (including the frozen hyperonic A-NET and Evidence Network networks) |
| `route2/downloads.d/` | lists of the large files on the GitHub release, with their SHA-256 |
| `docs/route2/` | step-by-step rerun instructions for every stage, and `FULL_RERUN_FILES.md`, the list of the full-rerun download |
| `environment-route2.yml`, `requirements-route2.txt` | the exact package versions |

The code folders (`inference/`, `hyperonic_pipeline/`, `hyperonic_pipeline/tsnpe/`, `legacy_stack/`,
`green_en/`, `fastsolver/`) each have a manifest giving every file's SHA-256 (in `inference/`:
`NUCLEONIC_CODE_MANIFEST.json`).

## Troubleshooting

| Message | What to do |
|---|---|
| `No module named 'fast_newton_hyperon_audit'` | Expected for the hyperonic curve commands: that helper is not shipped (see *What cannot be rerun from this package*). |
| `No module named ...` (any other module) | Activate the Route 2 environment (step 2) in the same terminal. |
| `FAIL: ... does not match its recorded SHA-256` | The download was interrupted. Run the same `./reproduce.sh route2-fetch` command again (with `--group full_rerun` if that was the download). |
| `FAIL: cannot download ...` | Check the internet connection and run the same `./reproduce.sh route2-fetch` command again (with `--group full_rerun` if that was the download). Files that are already present and verified are not downloaded again. |
| `An NVIDIA GPU may be present ... Falling back to cpu` | Harmless: JAX runs on CPU on purpose. |
| A check prints `FAIL` | Keep `build/route2/<check>_check.json` and open an issue on GitHub with it. |

For any other problem, open an issue on GitHub.
