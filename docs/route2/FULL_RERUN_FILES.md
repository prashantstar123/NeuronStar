# Route 2: the full-rerun download (32 files, about 11 GB)

A from-scratch rerun starts from large input files: prior banks, training caches, screens, scenario banks and a
few intermediate products. They are attached to the GitHub release `route2-data-v1` of this repository, so they
can be downloaded instead of regenerated. They are optional: the replay checks (`./reproduce.sh route2-check`)
and the smoke runs (`./reproduce.sh route2-smoke`) do not need them.

## Download

```bash
./reproduce.sh route2-fetch --group full_rerun
```

- 32 files, 10.9 GB in total. Keep about 13 GB of free disk space: the largest file (3.16 GB) arrives in
  two parts that are joined on disk.
- Every file, and every part, is checked against its SHA-256. A file that is already present and correct is not
  downloaded again, so an interrupted download can simply be started again.
- The files are saved in `route2/downloads/`, which git ignores.
- On the test machine the whole group took 29 minutes to download and verify.

## Names in the step guides

The step guides name each of these files by the file name that their commands write and read (for example
`hyp_bank_part1.npz`). In the release every file has a unique name that starts with `full_`. The table below
gives both names; the file and its SHA-256 are the same.

To use the names of the guides, make a folder of links after the download:

```bash
mkdir -p runs/inputs
while read -r download guide; do
    if [ -e "route2/downloads/$download" ] && [ ! -e "runs/inputs/$guide" ]; then
        ln -s "$PWD/route2/downloads/$download" "runs/inputs/$guide"
    fi
done <<'EOF'
full_nucleonic_shared_prior_physics_bank.npz          shared_prior_physics_bank.npz
full_nucleonic_shared_exact_astrophysical_table.npz   shared_exact_astrophysical_table.npz
full_nucleonic_uniform_prior_support_bank.npz         uniform_prior_support_bank.npz
full_nucleonic_independent_common_physics_cache.npz   independent_common_physics_cache.npz
full_nucleonic_anet_training_scenarios.npz            anet_training_scenarios.npz
full_nucleonic_tidal_direct_theta_proposal_320k.npz   direct_theta_proposal_320k.npz
full_nucleonic_tidal_direct_theta_curves_320k.npz     direct_theta_curves_320k.npz
full_green_en_nucleonic_bank.npz                      nucleonic_conditional_physics_Sext_seed20260924.npz
full_green_en_nucleonic_sext_screen.npz               nucleonic_Sext_seed20260924_160M.npz
full_green_en_nucleonic_sext_exact_rows.npz           nucleonic_Sext_exact_seed20260924_parallel8_aligned.npz
full_hyperonic_uniform_bank_part1.npz                 hyp_bank_part1.npz
full_hyperonic_uniform_bank_part2.npz                 hyp_bank_part2.npz
full_hyperonic_uniform_bank_part3.npz                 hyp_bank_part3.npz
full_hyperonic_uniform_bank_part4.npz                 hyp_bank_part4.npz
full_hyperonic_uniform_training_cache_part1.npz       uniform_training_cache_part1.npz
full_hyperonic_uniform_training_cache_part2.npz       uniform_training_cache_part2.npz
full_hyperonic_uniform_training_cache_part3.npz       uniform_training_cache_part3.npz
full_hyperonic_uniform_training_cache_part4.npz       uniform_training_cache_part4.npz
full_hyperonic_support_training_cache_part1.npz       support_training_cache_part1.npz
full_hyperonic_support_training_cache_part2.npz       support_training_cache_part2.npz
full_hyperonic_support_screen_160m.npz                support_screen_160m.npz
full_hyperonic_proposal_training_cache_part1.npz      proposal_training_cache_part1.npz
full_hyperonic_proposal_training_cache_part2.npz      proposal_training_cache_part2.npz
full_hyperonic_proposal_training_cache_part3.npz      proposal_training_cache_part3.npz
full_hyperonic_student_proposal_draws_300k.npz        student_beta1_48c_300k_seed20260992.npz
full_hyperonic_augmented_counting_correction_300k.npz augmented_counting_correction_300k.npz
full_hyperonic_anet_training_scenarios.npz            causal_scenarios_2000_seed9.npz
full_hyperonic_stage_a_mis_1720k.npz                  eval_mis_round8_round15_1720k.npz
full_green_en_hyperonic_bank.npz                      hyperonic_conditional_physics_Sext_seed20260923_parent02357702.npz
full_green_en_hyperonic_parent_bank.npz               hyperonic_green_expanded_fullprior_bank_seed20260922.npz
full_green_en_hyperonic_sext_screen.npz               hyperonic_Sext_seed20260923_160M.npz
full_green_en_hyperonic_sext_exact_rows.npz           hyperonic_Sext_exact_seed20260923.npz
EOF
ls runs/inputs
```

Then use `runs/inputs/<name in the guides>` wherever a guide names one of these files. (`runs/` is ignored by git.)

## The 32 files

Guides: **Banks** = [STEPS_BANKS.md](STEPS_BANKS.md); **A-NET N** = [STEPS_ANET_NUCLEONIC.md](STEPS_ANET_NUCLEONIC.md);
**A-NET H** = [STEPS_ANET_HYPERONIC.md](STEPS_ANET_HYPERONIC.md); **TSNPE N** =
[STEPS_NUCLEONIC_TSNPE.md](STEPS_NUCLEONIC_TSNPE.md); **TSNPE H** = [STEPS_HYPERONIC_TSNPE.md](STEPS_HYPERONIC_TSNPE.md);
**EN** = [STEPS_EVIDENCE_NETWORK.md](STEPS_EVIDENCE_NETWORK.md).

| Downloaded file (in `route2/downloads/`) | Size | SHA-256 | Name in the guides | Where the guides use it |
|---|---|---|---|---|
| `full_nucleonic_shared_prior_physics_bank.npz` | 3.16 GB | `06dda51aa7c1…` | `shared_prior_physics_bank.npz` | Banks 3.1; A-NET N Step 1; EN Section 4, N2, N4 |
| `full_nucleonic_shared_exact_astrophysical_table.npz` | 43.9 MB | `e2dfb84d56a6…` | `shared_exact_astrophysical_table.npz` | Banks 3.2; A-NET N Step 1; EN Section 4, N4 |
| `full_nucleonic_uniform_prior_support_bank.npz` | 75.3 MB | `4f50897beb3f…` | `uniform_prior_support_bank.npz` | Banks 3.3; TSNPE N Step 1 |
| `full_nucleonic_independent_common_physics_cache.npz` | 91.4 MB | `280f6f60bd93…` | `independent_common_physics_cache.npz` | Banks 3.4 |
| `full_nucleonic_anet_training_scenarios.npz` | 1.08 GB | `269581450402…` | `anet_training_scenarios.npz` | A-NET N Step 2 |
| `full_nucleonic_tidal_direct_theta_proposal_320k.npz` | 41.5 MB | `16096dda3499…` | `direct_theta_proposal_320k.npz` | A-NET N Step 7 (the 320,000 proposal rows) |
| `full_nucleonic_tidal_direct_theta_curves_320k.npz` | 366.6 MB | `1eec4b4b2c19…` | `direct_theta_curves_320k.npz` | A-NET N Step 7 (their exact curves) |
| `full_green_en_nucleonic_bank.npz` | 428.3 MB | `0421508c9a15…` | `nucleonic_conditional_physics_Sext_seed20260924.npz` | Banks 3.8; EN N4–N6, N9 |
| `full_green_en_nucleonic_sext_screen.npz` | 6.9 MB | `e43000e05e08…` | `nucleonic_Sext_seed20260924_160M.npz` | Banks 3.8; EN N1, N3, N4 |
| `full_green_en_nucleonic_sext_exact_rows.npz` | 107.3 MB | `1fcd9396075e…` | `nucleonic_Sext_exact_seed20260924_parallel8_aligned.npz` | Banks 3.8; EN N3, N4 |
| `full_hyperonic_uniform_bank_part1.npz` | 529.5 MB | `c188c4d0fc13…` | `hyp_bank_part1.npz` | Banks 3.5, 3.6; A-NET H Steps 3, 4, 6; EN Section 4, H1, H3 |
| `full_hyperonic_uniform_bank_part2.npz` | 529.5 MB | `e9beaa28c8c9…` | `hyp_bank_part2.npz` | Banks 3.5; A-NET H Step 3 |
| `full_hyperonic_uniform_bank_part3.npz` | 529.5 MB | `2589bfec123f…` | `hyp_bank_part3.npz` | Banks 3.5; A-NET H Step 3 |
| `full_hyperonic_uniform_bank_part4.npz` | 529.5 MB | `2880d62d8d0f…` | `hyp_bank_part4.npz` | Banks 3.5; A-NET H Step 3 |
| `full_hyperonic_uniform_training_cache_part1.npz` | 177.7 MB | `e51cf2f181ba…` | `uniform_training_cache_part1.npz` | Banks 3.6; A-NET H Steps 3, 7–9; TSNPE H Step 1; EN Section 4, H1 |
| `full_hyperonic_uniform_training_cache_part2.npz` | 178.9 MB | `e7ac3193ca79…` | `uniform_training_cache_part2.npz` | Banks 3.6; A-NET H Steps 3, 7–9; TSNPE H Step 1; EN Section 4, H1 |
| `full_hyperonic_uniform_training_cache_part3.npz` | 178.5 MB | `e13a85668f4f…` | `uniform_training_cache_part3.npz` | Banks 3.6; A-NET H Steps 3, 7–9; TSNPE H Step 1; EN Section 4, H1 |
| `full_hyperonic_uniform_training_cache_part4.npz` | 178.6 MB | `575ebf362ad6…` | `uniform_training_cache_part4.npz` | Banks 3.6; A-NET H Steps 3, 7–9; TSNPE H Step 1; EN Section 4, H1 |
| `full_hyperonic_support_training_cache_part1.npz` | 53.8 MB | `479206d0dd72…` | `support_training_cache_part1.npz` | Banks 3.6; A-NET H Steps 4, 7, 8; TSNPE H Step 1; EN Section 4, H1 |
| `full_hyperonic_support_training_cache_part2.npz` | 53.8 MB | `28e0894d07b9…` | `support_training_cache_part2.npz` | Banks 3.6; A-NET H Steps 4, 7, 8; TSNPE H Step 1; EN Section 4, H1 |
| `full_hyperonic_support_screen_160m.npz` | 10.2 MB | `442da1a47e44…` | `support_screen_160m.npz` | Banks 3.6; A-NET H Step 4 |
| `full_hyperonic_proposal_training_cache_part1.npz` | 262.2 MB | `5a48a8870930…` | `proposal_training_cache_part1.npz` | Banks 3.6; A-NET H Steps 6–8 |
| `full_hyperonic_proposal_training_cache_part2.npz` | 262.0 MB | `118acc3268f4…` | `proposal_training_cache_part2.npz` | Banks 3.6; A-NET H Steps 6–8 |
| `full_hyperonic_proposal_training_cache_part3.npz` | 104.6 MB | `d1d80765914f…` | `proposal_training_cache_part3.npz` | Banks 3.6; A-NET H Steps 6–8 |
| `full_hyperonic_student_proposal_draws_300k.npz` | 22.8 MB | `2b2e7827d1f3…` | `student_beta1_48c_300k_seed20260992.npz` | Banks 3.6; A-NET H Steps 5, 6 |
| `full_hyperonic_augmented_counting_correction_300k.npz` | 4.1 MB | `a3a6f0296d58…` | `augmented_counting_correction_300k.npz` | A-NET H Steps 7, 8 |
| `full_hyperonic_anet_training_scenarios.npz` | 1.15 GB | `e942cb56f167…` | `causal_scenarios_2000_seed9.npz` | A-NET H Steps 8, 9 |
| `full_hyperonic_stage_a_mis_1720k.npz` | 139.0 MB | `7740c7cd625f…` | `eval_mis_round8_round15_1720k.npz` | Banks 3.7; A-NET H Steps 2, 5 |
| `full_green_en_hyperonic_bank.npz` | 265.2 MB | `f669d84ffca0…` | `hyperonic_conditional_physics_Sext_seed20260923_parent02357702.npz` | Banks 3.9; EN H4–H6, H9 |
| `full_green_en_hyperonic_parent_bank.npz` | 244.7 MB | `02357702e85e…` | `hyperonic_green_expanded_fullprior_bank_seed20260922.npz` | Banks 3.9; EN H1, H4 |
| `full_green_en_hyperonic_sext_screen.npz` | 7.8 MB | `629c156a4e7d…` | `hyperonic_Sext_seed20260923_160M.npz` | Banks 3.9; EN H2–H4 |
| `full_green_en_hyperonic_sext_exact_rows.npz` | 76.5 MB | `aa5a8b23d7f0…` | `hyperonic_Sext_exact_seed20260923.npz` | Banks 3.9; EN H3, H4 |

## What is not in the download

- **The Evidence Network label matrices** (7.9 GB; steps N5 and H5 of the EN guide). Rebuild them on a GPU; for the
  paper this took 228 s on one A40 (nucleonic) and 160 s on one RTX 4090 (hyperonic).
- **Three frozen inputs of the nucleonic Evidence Network chain:** the base bank `nucleonic_conditional_physics.npz`
  (`1fd8830f…`, 684 MB) and the fixed J0740 terms `tov_terms_outer_00..04.npz` (34.5 MB), both used in step N4, and
  the exact template `nucleonic_exact_template.npz` (`6bfcdd73…`, 3.4 kB), used in step N3. The products of N1, N3
  and N4 (the screen, the exact rows and the bank) are in the download, so a nucleonic EN rerun can start at N5.
- **Intermediate files** that each guide lists under *Gaps*, for example the Stage-A rounds r8–r15 (Banks 3.7;
  the combined Stage-A file is in the download), the second-stream support files of EN step H1 (their product, the
  parent bank, is in the download) and the UltraNest output folders.
