# LAION-fMRI baseline sweep

Reproduces the 5-model × 20-cell baseline reported in `../METHODS.md` § "Difficulty relative to Allen2022 / NSD".

## Files

| File | Purpose |
|---|---|
| `baseline_sweep.py` | Single-process sweep runner. Iterates 6 default models × 20 headline benchmarks, writes per-cell scores to `baseline_sweep_results.csv` (long) + `baseline_sweep_pivot.csv` (wide). Idempotent via Brain-Score's `result_caching` — re-running skips completed cells. |
| `sweep_per_cell.sh` | Wrapper that spawns one python process per (model, benchmark) cell. Use this on macOS — jetsam SIGKILLs cluster_k5 cells on big models when run in-process; spawning fresh resets memory pressure between cells. |
| `baseline_sweep_results.csv` | Published baseline. 113 rows = (model, benchmark, score, ceiling, error, seconds). |
| `baseline_sweep_pivot.csv` | Same data pivoted: 20 headline cells × 5 models. |

## Running it yourself

```bash
# In-process (Linux / when memory headroom is plentiful)
python -m brainscore_vision.benchmarks.laion_fmri.baselines.baseline_sweep

# One-process-per-cell (macOS / lower RAM)
bash brainscore_vision/benchmarks/laion_fmri/baselines/sweep_per_cell.sh
```

Prereqs: complete the stimulus + AWS setup from `../README.md` first. The sweep auto-resumes — kill and restart it at any point.

## Reproducing the published numbers

The 5 baseline models all have shipped `region_layer_map.json` files in
`brainscore_vision/models/`, so no auto-commit step is required. Scores should
match `baseline_sweep_results.csv` within ridge-fit randomness (`alpha_coord`
CV adds small variation per run).

| Model | Brain-Score leaderboard rank (at time of sweep) |
|---|---|
| `alexnet` | 95 |
| `alexnet_random` | n/a (null baseline) |
| `resnet50_tutorial` | 30 |
| `convnext_tiny_imagenet1K_torchvisionV1` | top-50 |
| `resnext101_32x8d_wsl` | top-30 |
