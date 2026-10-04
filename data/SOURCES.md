# Observational data: sources and citations

All observational files used by this package are included here, so nothing has to be downloaded
separately. They are the published posterior samples and contours, redistributed unchanged, except the PSR J0740+6620 68%
contour, which is computed from the published samples (see below). The three
largest files are stored gzip-compressed; `./reproduce.sh check` unpacks them and verifies that each
unpacked file is byte-identical to the original (SHA-256 in `MANIFEST.json`).

If you use these data, please cite the original papers and data releases below.

## `observations/`: inputs of the likelihood

| File | Source | Paper | Data release |
|---|---|---|---|
| `J0030_2spot_RM.txt` | PSR J0030+0451 mass–radius samples | Miller et al. 2019, ApJL 887, L24 (arXiv:1912.05705) | doi:10.5281/zenodo.3473466 |
| `J0740_NICERXMM_full_mr.txt` (stored as `.gz`) | PSR J0740+6620, joint NICER + XMM-Newton samples | Dittmann et al. 2024, ApJ 974, 295 (arXiv:2406.14467) | doi:10.5281/zenodo.10215109 |
| `J0437_post_equal_weights.dat` (stored as `.gz`) | PSR J0437−4715 posterior samples | Choudhury et al. 2024, ApJL 971, L20 (arXiv:2407.06789) | doi:10.5281/zenodo.13766753 |
| `GW170817_GWTC-1.hdf5` | GW170817 posterior samples, GWTC-1 parameter-estimation release | Abbott et al. 2019, PRX 9, 031040 (arXiv:1811.12907) | LIGO document P1800370 |
| `J0614_mrsamples.dat` | PSR J0614−3329 mass–radius samples | Mauviard et al. 2025, ApJ 995, 60 (arXiv:2506.14883) | doi:10.5281/zenodo.17380576 |
| `J1231_wmrsamples.txt` (stored as `.gz`) | PSR J1231−1411 weighted mass–radius samples | Salmi et al. 2024, ApJ 976, 58 (arXiv:2409.14923) | doi:10.5281/zenodo.13358349 |
| `J1614_STU_mrsamples_post_equal_weights.dat` | PSR J1614−2230 mass–radius samples (headline model) | Mauviard et al. 2026, arXiv:2609.00172 | doi:10.5281/zenodo.22163155 |

J0614, J1231 and J1614 are the three NICER sources that were left out of A-NET training. They are
used only in the unseen-source tests of the paper.

## `plotting_overlays/`: drawn in the figures only (never used in a likelihood)

| File | Shows | Paper |
|---|---|---|
| `GW170817_50.csv`, `GW170817_90.csv` | GW170817 50% and 90% mass–radius regions | Abbott et al. 2018, PRL 121, 161101 (arXiv:1805.11581) |
| `J0437_4715_posterior_RM.dat` | PSR J0437−4715 mass–radius posterior | Choudhury et al. 2024 |
| `Miller_68_3_J0030_0451.csv`, `Riley_68_3_J0030_0451.csv` | PSR J0030+0451 credible-region contours | Miller et al. 2019; Riley et al. 2019, ApJL 887, L21 (arXiv:1912.05702) |
| `Dittmann_68_J0740_6620.csv` | PSR J0740+6620 68% contour, computed from the samples in `observations/J0740_NICERXMM_full_mr.txt` with a weighted kernel density estimate | Dittmann et al. 2024, ApJ 974, 295 (arXiv:2406.14467) |
| `Miler_6620_lower1.png.dat`, `Miler_6620_upper1.png.dat`, `Riley_68_J0740_6620.csv` | PSR J0740+6620 contours of the 2021 analyses (kept for reference; the figures draw the 2024 contour above) | Miller et al. 2021, ApJL 918, L28 (arXiv:2105.06979); Riley et al. 2021, ApJL 918, L27 (arXiv:2105.06980) |

The fingerprints (SHA-256) of all files are listed in `MANIFEST.json`, and those of the overlay and
substituted-pulsar files also in `plotting-manifest.json`.
