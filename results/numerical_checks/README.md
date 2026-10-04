# Records behind Appendix C and Sec. III D

These records hold the numerical checks behind Appendix C (Tables X–XII) and parts of Sec. III D of the paper.
Route 1 recomputes Tables X–XII from these files (`route1/tables.py`), and `./reproduce.sh check` verifies every
file against its SHA-256 in `results/MANIFEST.json`.

| Folder or file | What it holds | Where the paper uses it |
|---|---|---|
| `appendix_c/conv_{nuc,hyp}_{A1,J0614,J1231,J1614}.json` | Likelihood-construction changes on the posterior samples of eight configurations: finer quadrature and grids, other KDE subsets, GW prior divided out, all refinements combined | Table X, rows 1–3 and 8; Appendix C text |
| `appendix_c/prior_*.json` | NICER analysis-prior shapes divided out (four configurations) | Table X, last row |
| `appendix_c/extra_likelihood_checks/` | NICER and GW bandwidths, chirp mass integrated over, 80 stars per sequence: per-configuration results (`extra_*.json`, `dense_*.json`) and their summary (`extra_likelihood_checks.json`) | Table X, rows 4–7; Appendix C text |
| `appendix_c/appendix_c_aggregate.json` | Largest changes over the configurations | Table X |
| `appendix_c/mmax_factor_removal.json` | Maximum-mass factor removed | Appendix C; Sec. II B |
| `appendix_c/step_posteriors.json`, `appendix_c/A1_steps*.npz` | Nucleonic A1 A-NET+IS with 64, 256 and 1024 Heun steps, with bootstrap errors | Table XI |
| `appendix_c/flow_exact/` | Exact density of the discrete Heun map (nucleonic and hyperonic) | Appendix C, flow integration |
| `appendix_c/ultranest_seed_scatter.json` | Three nucleonic UltraNest seeds at J_sym = 36 MeV | Appendix C, repeated runs |
| `appendix_c/hyperonic_9d_parameter_summary.json` | Medians and 90% intervals of the nine hyperonic parameters (two UltraNest seeds, A-NET+IS, TSNPE+MIS) | Appendix C, repeated runs |
| `appendix_c/ev1024_*.npz`, `appendix_c/is_table.json` | Nucleonic 1024-step A-NET+IS runs with 4000 draws, and the importance-sampling evidences of both models | Table XII |
| `cost/per_ess_ratios.json` | Time per effective sample for all 16 configurations | Sec. III D |
| `cost/one_time_with_preparation_values.json` | One-time A-NET rows including the A-NET training-data preparation | Table VII |
| `cost/en_bank_timing/` | EN bank estimate for a new NICER source, timed against the network query | Sec. III D (the further 4–9 s) |
| `reweighting_vs_fresh/` | The A1 UltraNest posterior reweighted to each new configuration, compared with fresh UltraNest and A-NET+IS runs | Sec. III D, reweighting test |
| `software_versions.txt` | Pinned package versions of the released environment files | Data availability |
