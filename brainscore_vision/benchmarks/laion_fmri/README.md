# LAION-fMRI Brain-Score Benchmark

Score vision models against the LAION-fMRI 7T dataset (Zerbe et al., VSS 2026) — a densely-sampled multi-subject fMRI dataset spanning the natural-image distribution with built-in OOD stress tests.

This README walks you from a clean checkout to your first score. For methodology details see [METHODS.md](METHODS.md). For runnable examples of every benchmark family see [usage_examples.ipynb](usage_examples.ipynb).

---

## Quick start

```python
from brainscore_vision import _run_score
score = _run_score("alexnet", "LAION_fMRI_persubject.IT-tau-ridge")
print(float(score.values))   # ceiled per-voxel correlation, averaged across 5 subjects
```

If that fails with `FileNotFoundError` on `~/laion-fmri/stimuli/...`, you need to do the [stimulus setup](#stimulus-setup) below — the stimuli are gated by a Data Use Agreement and aren't shipped automatically.

---

## Prerequisites

1. **Brain-Score Vision installed** with its standard env (see the repo-root README).
2. **AWS credentials** configured (`aws configure`) — needed to fetch the neural assemblies from Brain-Score's private S3 bucket. Same gating as every other Brain-Score benchmark.
3. **A LAION-fMRI Data Use Agreement** signed at <https://laion-fmri.hebartlab.com/request> — you'll receive a `LAION_FMRI_REQUEST_ID`. This authorizes you to download the stimulus images locally.

---

## Stimulus setup

The stimuli are gated by the LAION-fMRI DUA, which prohibits redistribution. Brain-Score does not mirror them publicly. You download once per machine. (If you want to fully rebuild the *neural assemblies* from raw GLMsingle output too — not just download stimuli — skip ahead to the [Full reproduction](#full-reproduction) section, which wraps all of this into one command.)

```bash
# 1. Install the upstream Hebart-lab CLI
pip install "laion-fmri @ git+https://github.com/ViCCo-Group/LAION-fMRI.git@main"

# 2. Point the CLI at where you want stimuli to land
laion-fmri config --data-dir ~/laion-fmri

# 3. Download the gated archive (~3.2 GB JPEGs + ~1.6 MB metadata)
LAION_FMRI_REQUEST_ID=<your_id> laion-fmri download-stimuli

# 4. Extract into the layout the Brain-Score loader expects
python -m brainscore_vision.data.laion_fmri.data_packaging.get_local_stimuli \
    --stimuli-h5    ~/laion-fmri/stimuli/task-images_stimuli.h5 \
    --metadata-csv  ~/laion-fmri/stimuli/task-images_metadata.csv
# (default --out-dir is ~/laion-fmri/stimuli/images_extracted/, which is where
#  the runtime loader probes first)
```

After this you should have:

```
~/laion-fmri/
└── stimuli/
    ├── task-images_stimuli.h5          (downloaded archive)
    ├── task-images_metadata.csv
    └── images_extracted/
        ├── manifest.csv
        └── <25,052 JPEGs>
```

Override the location with `LAION_FMRI_STIMULI_DIR=/custom/path` if you can't use `~/laion-fmri/`.

The neural assemblies download themselves from S3 on first call (~400-800 MB per subject, cached at `~/.brainio/`).

---

## Full reproduction

If you want to **reproduce both the stimulus set and the neural assemblies from raw upstream data** — e.g. for an audit, a schema migration, or because you don't trust the published S3 artifacts — one command does the whole pipeline:

```bash
python -m brainscore_vision.data.laion_fmri.data_packaging.rebuild_assemblies \
    --request-id <YOUR_LAION_FMRI_REQUEST_ID>
```

This:
1. Downloads all 5 subjects' GLMsingle output from `s3://laion-fmri/` (CC0, no DUA needed)
2. Builds per-subject NeuroidAssemblies (`build_assembly.py`)
3. Slices into shared (1,492 stim) + persubject (6,204 stim/subj) pools
4. Repackages to S3 schema (single `data` data_var)
5. Downloads + extracts the DUA-gated stimuli (uses `--request-id`)
6. Semantic-verifies every rebuilt assembly against the published S3 copy (data + every coord element-wise)

Runs ~2-3 hours total, idempotent. Peak RSS ~4 GB during the verification step (5-subject sweep). See [`../../data/laion_fmri/data_packaging/README.md`](../../data/laion_fmri/data_packaging/README.md) for per-script detail, flags (`--skip-stimuli`, `--skip-semantic-check`, `--subjects sub-01 sub-03`), and a known cosmetic divergence in `subject_id_pres` values that the verifier will flag.

---

## Available benchmarks

20 registered "headline" variants. Identifier pattern: `{family}.{region}-{split}-{metric}`.

| Family | Regions | Splits | Metric | Count |
|---|---|---|---|---|
| `LAION_fMRI` (shared 1,492 stim) | V1, V2, V4, IT | tau, ood | ridge | 8 |
| `LAION_fMRI_persubject` (5,833 stim/subj) | V1, V2, V4, IT | tau, ood | ridge | 8 |
| `LAION_fMRI` (shared) | V1, V2, V4, IT | (no split) | rdm-pearson | 4 |

Persubject ridge is the **headline** family (most discriminative — see METHODS § "Difficulty relative to Allen2022 / NSD"). Shared ridge is for direct comparison with Allen2022/Hebart2023. Shared RSA complements ridge encoding.

Non-headline variants (`cluster_k5`, per-OOD-category, `IT_full` ablation) are accessible via the factory API — see [usage_examples.ipynb § 2-3](usage_examples.ipynb).

---

## Running scores

### Single benchmark, single model

```python
from brainscore_vision import _run_score
score = _run_score("resnet50_tutorial", "LAION_fMRI_persubject.IT-tau-ridge")
print(f"ceiled: {float(score.values):.3f}")
print(f"ceiling: {float(score.attrs['ceiling'].values):.3f}")
for sub in ["sub-01","sub-03","sub-05","sub-06","sub-07"]:
    if sub in score.attrs:
        print(f"  {sub}: {float(score.attrs[sub].values):+.3f}")
```

### Sweep multiple models across multiple benchmarks

See [baselines/baseline_sweep.py](baselines/baseline_sweep.py) for the script that produced the 5-model × 20-cell baseline pivot ([baselines/baseline_sweep_pivot.csv](baselines/baseline_sweep_pivot.csv)).

### Non-headline variants (cluster_k5, per-OOD-category)

```python
from brainscore_vision import load_model
from brainscore_vision.benchmarks.laion_fmri.benchmark import LAIONfMRIClusterCV

model = load_model("alexnet")
bench = LAIONfMRIClusterCV("V4")               # 5-fold CLIP-cluster CV, shared pool
score = bench(model)
```

[usage_examples.ipynb](usage_examples.ipynb) has end-to-end runnable cells for every non-headline variant + how to tune the noise-ceiling threshold + per-subject scoring.

---

## What does the score mean?

Each cell returns a **ceiled per-voxel correlation**, computed as:

```
raw   = Pearson r between model prediction and held-out neural response, per voxel
ceiled = raw / sqrt(nc_12rep/100)
score = median ceiled across voxels, averaged across subjects
```

- `0` = no better than chance
- `1` = model fully explains the explainable variance (within noise ceiling)
- Negative scores happen with random nets on hard cells (per-OOD-category IT)

See METHODS.md § "Noise ceilings" + § "Scoring" for the exact formulas.

---

## Troubleshooting

**`FileNotFoundError` on `~/laion-fmri/stimuli/...`** — Do the [stimulus setup](#stimulus-setup). The benchmark refuses to silently re-download because the DUA requires user-initiated consent.

**`NoCredentialsError` from boto3** — Run `aws configure`. Same as any Brain-Score benchmark.

**`AssertionError: No registrations found for ...`** — You're trying to score a variant that's not in the lean registry (e.g. `LAION_fMRI.IT-ood_gabor-ridge`). Use the factory API directly: `LAIONfMRI('IT', 'ood_gabor')(model)`. See usage_examples.ipynb § 7.

**Disk fills mid-scoring** — Activations get cached to `~/.result_caching/` and assemblies to `~/.brainio/`. The persubject pool extracts ~30 GB per model. If your home partition is small, symlink these to a larger volume:
```bash
mv ~/.result_caching /path/to/big/volume/result_caching
ln -s /path/to/big/volume/result_caching ~/.result_caching
```

**macOS jetsam SIGKILL on big models** — Heavy 5-fold cluster_k5 cells can hit ~30 GB RAM during ridge fit. Two options: run cells one-per-process via [baselines/sweep_per_cell.sh](baselines/sweep_per_cell.sh), or run on a Linux box with more RAM headroom (EC2 r6i.8xlarge worked well in our sweep).

**Score differs from the baseline_sweep_pivot.csv numbers** — Activations cache at `~/.result_caching/` is keyed by (model_id, stim_set_identifier). If you upgraded the model package or changed stim IDs, the cache key invalidates and you re-extract. Score values should still match within rounding.

---

## See also

- **[METHODS.md](METHODS.md)** — Full methodology: dataset, ROI definitions, splits, preprocessing, noise ceilings, scoring, NSD comparison, benchmark variants.
- **[usage_examples.ipynb](usage_examples.ipynb)** — 9 runnable sections covering registry, factory API, per-OOD-category, cluster_k5, RSA, per-subject scoring, noise-ceiling tuning, bulk-register, sweep result loading.
- **[baselines/](baselines/)** — Sweep runner + per-cell driver + 5-model published baseline CSVs.
- **[../../data/laion_fmri/data_packaging/](../../data/laion_fmri/data_packaging/)** — Full rebuild pipeline from raw GLMsingle output to S3-ready assemblies, including semantic verification against published artifacts. Only needed if you're re-uploading, auditing, or schema-migrating; normal scoring runs use the pre-built S3 assemblies automatically.
- **[../../data/laion_fmri/nc_reproduction/](../../data/laion_fmri/nc_reproduction/)** — Independent reproduction of the dataset's published `nc_12rep` noise ceiling (Spearman ρ = 1.000 vs. published per-voxel map).
- **[LAION-fMRI website](https://laion-fmri.hebartlab.com/)** — Dataset documentation, DUA request form, paper preprint.
