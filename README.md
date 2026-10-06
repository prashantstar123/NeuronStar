# NeuronStar

**NeuronStar** is the open-source code and reproduction package of the paper **"Fast Bayesian Updating of the Neutron-Star Equation of State with Neural
Posterior and Evidence Estimation"** by **Prashant Thakur** (Department of Physics, Yonsei University).
Paper: [arXiv:2610.05121](https://arxiv.org/abs/2610.05121).

The package has three parts:

| Route | What it does | Status |
|---|---|---|
| **1. Reproduce the paper** | Rebuilds the ten data figures and Tables I and III–XII of the paper from the saved final results and checks each one against the paper. | **Ready** |
| **2. Run everything from scratch** | Checks every stage of the pipeline against the paper's saved results, and documents every step for regenerating the training data, retraining the networks and rerunning all inference (UltraNest, A-NET+IS, TSNPE+MIS, Evidence Network). The few pieces that cannot be rerun from this package are listed in the manual. | **Ready** |
| **3. Use your own model** | Other equations of state, other inference methods, new likelihoods. | Planned |

---

## Route 1: reproduce the paper (about 2 minutes, no GPU)

**New to this? Follow the step-by-step manual: [MANUAL_ROUTE1.md](MANUAL_ROUTE1.md).**

You need Linux or macOS (on Windows, use WSL), `git`, and either **conda/mamba** or **Python 3.11**.
The package contains everything else: code, saved results and all observational data.
About 1.5 GB of disk space is needed including the Python environment.

### Step 1: download

```bash
git clone https://github.com/prashantstar123/NeuronStar.git
cd NeuronStar
```

### Step 2: install the exact Python packages (one time)

Use **one** of the two options.

**Option A, with conda or mamba:**

```bash
conda env create -f environment-route1.yml
conda activate ddb-route1
```

**Option B, with plain Python 3.11:**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-route1.txt
```

Every package version is fixed to the one that drew the published figures, so nothing needs to be chosen.

### Step 3: check

```bash
./reproduce.sh check
```

This checks the Python environment and compares every data and result file with its recorded
fingerprint (SHA-256). It also unpacks the three compressed data files. A correct installation prints:

```text
[check] PASS Python environment (exact package versions)
[check] PASS all 144 data and result files match their recorded checksums
```

### Step 4: run

```bash
./reproduce.sh route1
```

A successful run ends with:

```text
ROUTE 1 PASSED in 1.1 min: all 14 data-figure files are pixel-identical to the paper, the workflow diagram matches its SHA-256, and every number in Tables I and III-XII agrees.
```

### What Route 1 checks

1. **Files.** Every supplied data and result file matches its fingerprint.
2. **Figures.** The ten data figures, Figs. 2–11 (14 PDF files), are redrawn from the saved results and compared with the published figures. Both are turned into images, which must be identical pixel for pixel. Fig. 1, the workflow diagram, is checked by its SHA-256.
3. **Tables.** Every number in Tables I and III–XII is recomputed from the saved results and compared with the number printed in the paper. For Tables VII and VIII, the complete table text is compared; for the Appendix C Tables X–XII, every number must equal the record in `results/numerical_checks/` at the printed precision. Table II lists the likelihood inputs; their sources are in `data/SOURCES.md`.
4. **Paper.** The manuscript is compiled with the rebuilt figures. This step needs LaTeX. Without LaTeX it is skipped with a message, and the figure and table checks still run.

The output is written to `build/`:

| Folder or file | Contents |
|---|---|
| `build/figures/` | the rebuilt figures |
| `build/tables/` | the recomputed table numbers; Tables VII–VIII as text |
| `build/reproduced_manuscript.pdf` | the paper compiled with the rebuilt figures |
| `build/logs/` | the log of each figure job and of the LaTeX run; the other steps print to the terminal |

If anything differs, the run stops and names the figure, the table and the first differing number.

### Optional: LaTeX for the PDF

| System | Install command |
|---|---|
| Ubuntu or Debian | `sudo apt install texlive-latex-extra texlive-publishers texlive-science` |
| macOS | install [MacTeX](https://www.tug.org/mactex/) |

---

## Route 2: run everything from scratch (checks about 10 minutes on CPU; full rerun takes days)

**Step-by-step manual: [MANUAL_ROUTE2.md](MANUAL_ROUTE2.md).**

> **Caution.** The computing times in Tables VII–VIII depend on the hardware; on your machine every time, and so
> every speed-up factor, will be different. Retrained networks and new sampler runs are not bit-identical to the
> paper's; their results agree within the quoted uncertainties. Only the replay checks below reproduce the paper's
> saved numbers to stated tolerances.

```bash
conda env create -f environment-route2.yml && conda activate ddb-route2   # or, in a Python 3.11 virtual environment: pip install -r requirements-route2.txt
./reproduce.sh route2-check    # replays every stage against the saved results (downloads about 130 MB once)
./reproduce.sh route2-smoke    # tiny end-to-end runs of the code
```

- **Checks.** `route2-check` replays every stage from its frozen inputs: Evidence Network, A-NET+IS and TSNPE+MIS
  for both models, UltraNest, and the prior banks. It recomputes the importance weights, evidences and resamples,
  the proposal densities, and the exact likelihoods from the data.
- **Full rerun.** The commands and settings of every stage, with the seeds where available, are in
  `docs/route2/`; each guide gives the settings used for the paper and says where a setting of the paper's run is
  not known. The banks, caches and other large inputs used for the paper (32 files,
  about 11 GB) can be downloaded with `./reproduce.sh route2-fetch --group full_rerun` instead of being regenerated;
  [docs/route2/FULL_RERUN_FILES.md](docs/route2/FULL_RERUN_FILES.md) lists them and the few files that are not
  provided. A few steps cannot be rerun from this package as it stands; the manual lists them
  ([MANUAL_ROUTE2.md](MANUAL_ROUTE2.md#what-cannot-be-rerun-from-this-package)).

---

## Which figure or table comes from which saved result

| Paper item | Saved results used (in `results/`) |
|---|---|
| Fig. 2, A1 parameter posterior | `nucleonic/a1_parameter_posteriors.npz` |
| Figs. 3(a), 5, nucleonic mass–radius (A1, J0614, J1231) | `nucleonic/nucleonic_mass_radius_bands.npz` |
| Fig. 3(b), A1 mass–tidal | `nucleonic/a1_mass_tidal_bands.npz` |
| Fig. 4, nuclear-data changes | `nucleonic/nuclear_amortization_bands.npz` |
| Fig. 6, nucleonic J1614 | `nucleonic/j1614_*_mass_radius_curves.npz` |
| Fig. 7, hyperonic A1 | `hyperonic/A1/*` (6,000-draw posterior selections, stellar curves, bands) |
| Figs. 8–9, hyperonic J0614, J1231, J1614 | `hyperonic/J0614/`, `hyperonic/J1231/`, `hyperonic/J1614/` and `hyperonic/hyperonic_mass_radius_tail_support.npz` |
| Fig. 10 and Table IX, hyperonic nuclear-data changes (appendix) | `appendix/hyperonic_nuclear_amortization_*` |
| Fig. 11, P–P calibration (appendix) | `appendix/pp_calibration_ranks.npz` |
| Table I, priors | `tables/table_I_priors.json` (from the EOS model definitions) |
| Table III, A1 summary | `nucleonic/a1_scalar_summary.json`, `nucleonic/tsnpe_mis_a1_summary.json`, `nucleonic/a1_mass_tidal_bands.npz`, `hyperonic/table_II_IV_hyperonic_values.json` |
| Table IV, nuclear-data changes | `nucleonic/nuclear_shift_scalar_summary.json` |
| Table V, unseen NICER sources | `nucleonic/substituted_source_scalar_summary.json`, `nucleonic/j1614_mass_radius_comparison.json`, `hyperonic/table_II_IV_hyperonic_values.json` |
| Table VI, evidence | `evidence/green_en_vs_ultranest_16_rows.json`, `evidence/table_VI_independent_ultranest_rows.json` |
| Tables VII–VIII, computing cost | `cost/table_VII_VIII_values.json`, `cost/en_and_tsnpe_timing_constants.json` |
| Tables X–XII and the other numbers of Appendix C | `numerical_checks/appendix_c/` (see `numerical_checks/README.md`) |
| Sec. III D, time per effective sample and the reweighting test | `numerical_checks/cost/`, `numerical_checks/reweighting_vs_fresh/` |

`results/MANIFEST.json` lists every result file with its fingerprint and the calculation it comes from.

## What is in the repository

| Folder | Contents |
|---|---|
| `paper/` | manuscript source (`main.tex`, `refs.bib`, `main.bbl`) and the published figure files |
| `results/` | saved final results used to rebuild the figures and tables |
| `data/` | all observational data (NICER, XMM-Newton, GW170817); sources and citations in `data/SOURCES.md` |
| `plotting/` | the plotting code that drew the published figures |
| `route1/` | the Route 1 steps: check, figures, comparison, tables, manuscript |
| `route2/`, `docs/route2/` | the Route 2 checks, smoke runs, small frozen inputs and per-stage rerun instructions |
| `eos/`, `tov/`, `likelihoods/`, `workflows/`, `inference/` | the production code of the inference pipeline |
| `hyperonic_pipeline/`, `legacy_stack/`, `green_en/`, `fastsolver/` | the hyperonic pipeline, its certified physics modules, the Evidence Network trainer and the fast hyperonic solver |
| `reproduce.sh` | the only command you need |

Some saved JSON records contain the file locations of the paper's calculations as plain text. They
are kept only as a record of where each result was produced, and are never used as paths.

## Troubleshooting

| Message | What to do |
|---|---|
| `Python 3.11 is required` or `missing Python package` | Do step 2, then activate the environment (`conda activate ddb-route1` or `source .venv/bin/activate`) in the same terminal. |
| `... is required, found ...` | A different package version is installed. Create a fresh environment as in step 2. |
| `does not match its recorded checksum` or `is damaged` | The download is incomplete or changed. Clone the repository again. |
| `pdflatex was not found` | LaTeX is not installed. Figures and tables are still checked. Install LaTeX as shown above to build the PDF. |
| `./reproduce.sh: Permission denied` | Run `bash reproduce.sh route1`. |
| Route 2: `No module named torch` (or jax, sbi, ultranest) | Create and activate the Route 2 environment (`environment-route2.yml`). |

For any other problem, open an issue on GitHub and attach the files in `build/logs/`.

## Citation

If you use NeuronStar, please cite the paper:

> P. Thakur, *Fast Bayesian Updating of the Neutron-Star Equation of State with Neural Posterior and
> Evidence Estimation*, [arXiv:2610.05121](https://arxiv.org/abs/2610.05121) [astro-ph.HE] (2026).

```bibtex
@article{Thakur2026NeuronStar,
    author = "Thakur, Prashant",
    title = "{Fast Bayesian Updating of the Neutron-Star Equation of State with Neural Posterior and Evidence Estimation}",
    eprint = "2610.05121",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.HE",
    month = "10",
    year = "2026"
}
```

Please also cite the observational data releases listed in `data/SOURCES.md` and the original papers of
the inference methods, as cited in the paper.

## License

NeuronStar is released under the MIT License (see `LICENSE`). The license covers the code and the results
produced for the paper. The observational data in `data/` are redistributed from the public releases listed in
`data/SOURCES.md` and remain under their original terms.

## Contact

**Prashant Thakur**, Department of Physics, Yonsei University

- Email: [prashant@yonsei.ac.kr](mailto:prashant@yonsei.ac.kr), [prashantthakur1921@gmail.com](mailto:prashantthakur1921@gmail.com)
- ORCID: [0000-0003-4189-6176](https://orcid.org/0000-0003-4189-6176)
