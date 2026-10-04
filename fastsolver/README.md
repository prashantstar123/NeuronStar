# fastsolver: fast exact-row evaluation

This folder contains the compiled hyperonic EOS solver and the batched likelihood that compute the exact rows of
the Evidence Network's nuclear-support extension (steps N3 and H3 of `docs/route2/STEPS_EVIDENCE_NETWORK.md`).
Their purpose is to make this computation as fast as possible without changing any result.

## Files

- `hyp_solver_template.py` -> `gen_backends.py` -> `hyp_solver_cpu.py` / `hyp_solver_cuda.py`: compiled port of the
  certified accelerated Newton hyperonic solver (`hyperonic_pipeline/ddb_hyperon_eos.py`). One algorithmic
  substitution: the SciPy least_squares fallback is replaced by the same boundary-exit rule applied to Newton's
  best iterate; any other fallback situation is flagged (status 1) and re-solved with the certified Python solver
  by the driver. `hyp_solver_cuda.py` is the optional numba CUDA backend.
- `fast_likelihood.py`: batched evaluation of the joint likelihood, bitwise equal to `JointLikelihood.evaluate`
  (6000 rows).
- `fast_exact_rows.py`: drop-in replacement for `evaluate_clean_hyperonic_support_cache.py` (same CLI, same output
  keys, extra provenance keys `evaluator` / `hyp_solver_template_sha256` / `eos_backend` / `fast_fallback_rows` /
  `fast_rows_with_emulated_exit`). `--eos-backend compiled` (default, fast) or `certified` (original solver, bitwise
  identical to the original shard).
- `CODE_MANIFEST.json`: the SHA-256 of every file in this folder.

## Run

On CPU with the default backend (as for the paper), from the top folder of the package, with the Route 2
environment active:

```bash
PYTHONPATH=.:fastsolver JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python fastsolver/fast_exact_rows.py \
  --screen <screen.npz> --template-bank <template_bank.npz> --data-root <data_root> \
  --output <out.npz> --workers 24 --start <a> --stop <b> [--eos-backend compiled|certified]
```

The commands that rebuild the paper's support-extension rows are steps N3 and H3 of
`docs/route2/STEPS_EVIDENCE_NETWORK.md`.

Before any GPU use, check which processes already use the GPU:

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```
