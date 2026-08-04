# LAION-fMRI Brain-Score Benchmark

Score vision models against the LAION-fMRI 7T dataset (Zerbe et al., VSS 2026): a densely-sampled 5-subject fMRI dataset over ~25K natural images, with built-in out-of-distribution stress tests.

For dataset, ROIs, splits, ceilings, and scoring see [METHODS.md](METHODS.md). For runnable cells covering every variant see [usage_examples.ipynb](usage_examples.ipynb).

## Quick start

```python
from brainscore_vision import _run_score
score = _run_score("alexnet", "Zerbe2026_fmri_persubject.IT-tau-ridgecv")
print(float(score.values))   # ceiled per-voxel correlation, averaged across 5 subjects
```

If this fails with `FileNotFoundError` on `~/laion-fmri/stimuli/...`, do the stimulus setup below — stimuli are gated by the dataset's Data Use Agreement.

## Prerequisites 

1. Brain-Score Vision installed.
2. AWS credentials (`aws configure`) for the S3-served neural assemblies.
3. A signed LAION-fMRI DUA at <https://laion-fmri.hebartlab.com/request> — returns a `LAION_FMRI_REQUEST_ID`.

## Stimulus setup (once per machine)

```bash
pip install "laion-fmri @ git+https://github.com/ViCCo-Group/LAION-fMRI.git@main"
laion-fmri config --data-dir ~/laion-fmri
LAION_FMRI_REQUEST_ID=<your_id> laion-fmri download-stimuli
python -m brainscore_vision.data.laion_fmri.data_packaging.get_local_stimuli \
    --stimuli-h5    ~/laion-fmri/stimuli/task-images_stimuli.h5 \
    --metadata-csv  ~/laion-fmri/stimuli/task-images_metadata.csv
```

Override location with `LAION_FMRI_STIMULI_DIR=/custom/path`. Neural assemblies fetch themselves from S3 on first call (~400-800 MB per subject, cached at `~/.brainio/`).

## Registered benchmarks (20 headline variants)

Identifier pattern: `{family}.{region}-{split}-{metric}`.

| Family | Regions | Splits | Metric | Count |
|---|---|---|---|---|
| `Zerbe2026_fmri` (shared 1,492-stim pool) | V1, V2, V4, IT | tau, ood | ridgecv | 8 |
| `Zerbe2026_fmri_persubject` (5,833 stim/subj) | V1, V2, V4, IT | tau, ood | ridgecv | 8 |
| `Zerbe2026_fmri` (shared) | V1, V2, V4, IT | — | rdm-pearson | 4 |

`ridgecv` is kernel/dual ridge with per-fit CV alpha selection over a 21-value log-spaced sweep (1e-10 to 1e10). The dual form keeps the `(n_features, n_targets)` coefficient matrix from being materialized — important for wide-feature models on the persubject pool.

Non-headline variants — fixed-alpha ridge, `cluster_k5` CV, per-OOD-category sub-splits, and the `IT_full` (V4 ∪ IT) alias — are constructible via the factory API:

```python
from brainscore_vision.benchmarks.laion_fmri.benchmark import (
    LAIONfMRI, LAIONfMRIRSA, LAIONfMRIClusterCV,
)
LAIONfMRI("IT", "tau", metric_type="ridge")  # fixed alpha=1, faster
LAIONfMRIClusterCV("V4")                      # 5-fold CLIP-cluster CV, shared pool
LAIONfMRI("IT", "ood_gabor")                  # one OOD category
LAIONfMRI("IT_full", "tau")                   # V4 ∪ IT region alias
```

## Score interpretation

Each cell returns a **ceiled per-voxel correlation**:

- `0` = no better than chance
- `1` = model fully explains the explainable variance (within noise ceiling)
- Negative possible with random nets on hard cells

`score.attrs` carries:
- `ceiling` — noise ceiling used for normalization
- `raw` — disaggregated per-subject (or per-fold) scores
- `error` — bootstrap SE on the ceiled scale; `error_over` + `n_bootstrap` document the resample axis
- `sub-XX` keys — per-subject ceiled scores (multi-subject wrappers)

## Reproducing the full pipeline (maintainer-only)

One command rebuilds both the stimuli and the neural assemblies from the raw upstream GLMsingle output, with semantic verification against the published S3 artifacts:

```bash
python -m brainscore_vision.data.laion_fmri.data_packaging.rebuild_assemblies \
    --request-id <YOUR_LAION_FMRI_REQUEST_ID>
```

See [`../../data/laion_fmri/data_packaging/README.md`](../../data/laion_fmri/data_packaging/README.md) for per-script detail.

## See also

- [METHODS.md](METHODS.md) — dataset, ROI definitions, splits, ceilings, scoring
- [usage_examples.ipynb](usage_examples.ipynb) — runnable examples
- [baselines/](baselines/) — sweep runner + 5-model published baseline CSVs
- [LAION-fMRI website](https://laion-fmri.hebartlab.com/) — paper + DUA request form
