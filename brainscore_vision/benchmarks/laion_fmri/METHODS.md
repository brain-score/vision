# LAION-fMRI Brain-Score Benchmark — Methods

## Dataset

LAION-fMRI (Zerbe, Roth, Mell, Herholz, Knapen, & Hebart, 2026, *VSS*) is a densely-sampled 7T fMRI dataset acquired at the Max-Planck Institute CBS Leipzig. Five participants viewed 25,052 unique natural images plus 1,492 shared images across 150 image-viewing sessions (30 sessions per subject in the launch release; 15 supplemental sessions will follow). Stimuli were drawn primarily from LAION-natural (Roth & Hebart, 2025), with a 371-image out-of-distribution subset spanning abstract shapes, visual illusions, Gabor patches, and other deliberately atypical content.

**Acquisition.** Multi-echo 7T MRI at 1.8 mm isotropic resolution, TR = 1.9 s. Stimuli were rendered at 1000 × 1000 pixels and presented via a PROpixx MRI/MEG DLP LED projector at a viewing distance of ~165 cm, subtending 9.2 × 9.2 degrees of visual angle. Each trial: 2.5 s stimulus presentation followed by 0.5 s inter-stimulus interval. Participants performed a continuous old/new recognition memory task. Shared images were repeated 4 (n=611) or 12 (n=881) times across sessions; subject-unique images were repeated 4 times each. Total: ~31,856 image presentations per subject.

**Preprocessing & GLM.** Single-trial betas were estimated with GLMsingle (Prince et al., 2022) following tedana-based multi-echo denoising. Beta estimates use fractional ridge regression in volumetric T1w 1.8 mm space, indexed per-voxel inside the brain mask (272,080 voxels for sub-01). Per the dataset documentation, beta magnitudes are reliable in a relative sense per voxel but should not be interpreted as raw percent signal change — different voxels receive different regularization strengths, so within-session z-scoring is required before pooling across sessions.

## Space

This benchmark uses the **volumetric T1w 1.8 mm** space published by the dataset authors. The dataset publishes no surface betas, and the `.h5` GLMsingle model files are also flat-voxel-indexed. Surface ROI masks (`.func.gii`) exist for users who wish to project, but no surface BOLD data ship with the release. We do not introduce a surface projection step; this would diverge from the published methodology and reintroduce interpolation noise (`mri_vol2surf`).

The dataset's noise ceilings in volumetric space are exceptionally high — 50% in IT for sub-01 — substantially exceeding NSD volumetric (10% in IT). Multi-echo denoising and the 12-rep shared protocol provide the SNR that, in NSD, only surface reconstruction recovered.

## Regions

Five regions are exposed as benchmark variants:

| Region | Source ROI(s) | sub-01 voxel count |
|---|---|---|
| V1 | retinotopy.V1v ∪ V1d | 1300 |
| V2 | retinotopy.V2v ∪ V2d | 1305 |
| V3 | retinotopy.V3v ∪ V3d | 1016 |
| V4 | retinotopy.hV4 | 372 |
| IT | laionventral \ (V1v ∪ V1d ∪ V2v ∪ V2d ∪ V3v ∪ V3d ∪ hV4) | 3401 |

V1–V4 use the dataset's manually-delineated retinotopic ROIs (mutually exclusive within the retinotopy category by construction). IT is defined as `laionventral` (general visual selectivity in ventral cortex from the LAION-fMRI experiment's own GLMsingle R²) minus the retinotopic ROIs. This anatomically reasonable definition matches NSD's "streams ventral" parcel, which is the IT-equivalent ROI used by the Algonauts 2023 challenge.

ROIs are bilateral (left + right hemisphere combined). Category-selective ROIs (FFA, PPA, EBA, MT, etc.) are available in the dataset but not exposed by this benchmark.

## Train/test splits

Three split methods from the re:vision generalization initiative are exposed:

1. **`tau` — within-distribution.** A single fixed 80/20 partition that matches the train and test distributions at the population level (low train/test MMD) but maximizes per-image nearest-neighbour distance in CLIP feature space. Train: 4,666 images per subject + 897 shared. Test: 1,167 per subject + 224 shared.

2. **`cluster_k5_0` ... `cluster_k5_4` — five-fold cluster CV.** k-means on CLIP features (k = 5); each fold holds out one cluster as test. The held-out cluster is feature-space-different from training, so each fold tests semantic-cluster extrapolation. The benchmark variant `LAION_fMRI.{region}-cluster_k5-ridge` averages across the five folds.

3. **`ood` — held-out out-of-distribution images.** A single test set of 371 deliberately atypical stimuli (shapes, illusions, Gabors, cropped objects, etc.) shared across all subjects. The training set is the pool's regular (non-OOD) images. We additionally expose nine per-category sub-variants (`ood_<type>`) for fine-grained generalization analysis: `shape` (n=82), `relations` (72), `unusual` (64), `cropped` (60), `selfmade` (30), `gaudy` (24), `illusion-classic` (21), `gabor` (10), `illusion-natural` (8).

Splits are computed at benchmark-load time via `laion_fmri.splits.get_split_masks` using the per-subject pool semantics; one full assembly is registered with the data registry and sliced at runtime. This avoids materializing 22+ split-keyed assemblies on S3 and makes adding new split flavors a zero-data-rebuild change.

## Beta preprocessing

For each subject, single-trial volumetric betas are loaded for the combined region mask across all 30 sessions, then **z-scored within each session per voxel**. This step is essential: ridge-regression GLMsingle betas have voxel-specific regularization, so cross-session pooling without normalization conflates trial signal with regularization variance.

The assembly preserves individual trial repetitions. The benchmark loader applies the following per the Brain-Score `TrainTestNeuralBenchmark` pattern:
- **Train assembly**: individual trials, no averaging
- **Test assembly (for scoring)**: averaged across repetitions per stimulus
- **Test assembly (for ceiling)**: individual repetitions preserved, fed to `internal_consistency`

## Noise ceilings

Three noise-ceiling estimates are surfaced as neuroid coordinates on the assembly:

- `nc_12rep` — ncsnr-based ceiling computed from the 881 shared 12-rep images only. Highest-signal estimate; serves as the default reliability filter.
- `nc_4rep` — ncsnr-based ceiling from the 611 shared 4-rep images. Substantially lower than 12-rep by the `NC = 100·ncsnr²/(ncsnr² + 1/n)` formula.
- `nc_allrep` — ncsnr-based ceiling pooled across all repeated stimuli.

Neuroids with `nc_12rep < 30` are dropped before scoring (Brain-Score precedent: matches `NOISE_CEILING_THRESHOLD = 0.3 * 100` in `Hebart2023_fmri` and `Allen2022_fmri`).

In addition to the published ncsnr-based estimators above, the benchmark's `ceiling_func` computes an **internal-consistency** ceiling at evaluation time from the rep-preserved test assembly via `brainscore_vision.load_metric('internal_consistency')`. This is the value applied to ceil the raw score. Comparing the internal-consistency ceiling against the ncsnr-based ceilings is a diagnostic for the assembly's noise model — strong agreement across both estimators is the gold-standard expectation.

### Per-region NC validation (validated 2026-05-18, 2 subjects packaged so far)

Median NC across voxels per (subject, region):

| Region | Subject | n_voxels | nc_12rep | nc_4rep | nc_allrep |
|---|---|---|---|---|---|
| V1 | sub-01 | 1300 | 34.2 | 13.6 | 14.6 |
| V1 | sub-03 | 1584 | 54.8 | 32.1 | 33.7 |
| V2 | sub-01 | 1305 | 51.2 | 24.4 | 26.5 |
| V2 | sub-03 | 1660 | 52.8 | 31.8 | 33.1 |
| V3 | sub-01 | 1016 | 54.8 | 30.2 | 31.6 |
| V3 | sub-03 | 1237 | 59.2 | 35.7 | 37.2 |
| V4 | sub-01 | 372 | 61.5 | 34.2 | 36.9 |
| V4 | sub-03 | 362 | 59.4 | 35.6 | 37.2 |
| IT | sub-01 | 3401 | 50.4 | 24.2 | 25.8 |
| IT | sub-03 | 2817 | 56.6 | 27.6 | 29.6 |

Cross-subject medians: V1 44 / V2 52 / V3 57 / V4 60 / IT 54 (nc_12rep).

For comparison, NSD volumetric medians (from Allen2022_fmri packaging): V1 37 / V2 31 / V4 26 / IT 10. LAION-fMRI's V4 and IT are 2.3x and 5.4x NSD's respectively — almost entirely attributable to multi-echo denoising and the 12-rep shared protocol. The sub-01/sub-03 NC gap (V1: 34 vs 55) is the typical between-subject variability also seen in NSD.

### Independent reproduction of `nc_12rep` (sub-01)

To validate that we are consuming the published noise ceiling faithfully, we
independently re-derive `nc_12rep` from the shipped TYPED betas
(`SingletrialBetas_statmap.nii.gz`) for sub-01 within V1/V2/V4/IT, then compare
voxel-by-voxel to the dataset's
`sub-01_task-images_space-T1w_desc-Noiseceiling12rep_statmap.nii.gz`. The
recipe follows the Allen2022 NSD variance-based estimator described in the
published NC sidecar JSON (878 images × 12 reps = 10,536 trials matched). See
`../../data/laion_fmri/nc_reproduction/reproduce_nc12rep.py`.

| Region | n voxels | Reproduced median NC | Published median NC | Voxel Pearson r | Voxel Spearman ρ |
|---|---|---|---|---|---|
| V1 | 1,300 | 62.4 | 34.2 | 0.993 | **1.000** |
| V2 | 1,305 | 69.1 | 51.2 | 0.991 | **1.000** |
| V4 | 372   | 73.9 | 61.5 | 0.992 | **1.000** |
| IT | 3,401 | 68.7 | 50.4 | 0.991 | **1.000** |

The Spearman ρ = 1.000 per region confirms our re-derivation and the dataset's
shipped map place **identical voxels in identical reliability rank**. The
systematic positive offset on the absolute scale (our reproduced ncsnr ~1.8×
the dataset's) is attributable to a normalization step that isn't reproducible
from the shipped data alone: NSD/GLMsingle internally divides betas by an
estimated voxel-wise *noise* standard deviation (not the trial-level SD that a
straight z-score uses), which lowers the resulting ncsnr. The noise-SD
estimates ship inside the GLMsingle `_model.h5` files, which the LAION-fMRI
release does not distribute. We therefore use the dataset's published
`nc_12rep` as the ground-truth reliability map; the reproduction confirms our
interpretation is rank-consistent with theirs.

## Scoring

The benchmark uses a Brain-Score `TrainTestNeuralBenchmark` with `alpha_coord='subject'` — a separate ridge alpha is cross-validated per subject, and the final score is the mean of per-subject ceiled scores. Per-subject raw scores are preserved in `score.attrs`. The similarity metric is `ridge_split` (linear ridge regression with cross-validated alpha selection); other linear regressors are supported by passing `metric_type=` to the benchmark factory.

Cluster CV (`cluster_k5`) uses the `KFoldNeuralBenchmark` wrapper, which runs each of the five fold-benchmarks as an independent `TrainTestNeuralBenchmark`, then averages the per-fold ceiled scores. Per-fold detail is preserved in `score.attrs["raw_folds"]`, and `score.attrs["sem_folds"]` reports cross-fold standard error.

## Difficulty relative to Allen2022 / NSD

Side-by-side alexnet ridge scores establish that LAION-fMRI is meaningfully harder than the field-standard NSD-derived `Allen2022_fmri` benchmark, particularly in IT and especially in the per-subject pool.

| Region | Allen2022 ridge | LAION shared tau-ridge | LAION shared ood-ridge | LAION persubject tau-ridge | LAION persubject ood-ridge |
|---|---|---|---|---|---|
| V1 | 0.402 | 0.334 (0.83×) | 0.231 (0.57×) | 0.147 (0.37×) | 0.156 (0.39×) |
| V2 | 0.437 | 0.335 (0.77×) | 0.290 (0.66×) | 0.190 (0.43×) | 0.230 (0.53×) |
| V4 | 0.348 | 0.277 (0.80×) | 0.294 (0.84×) | 0.139 (0.40×) | 0.223 (0.64×) |
| IT | 0.294 | **0.136 (0.46×)** | **0.079 (0.27×)** | **0.031 (0.11×)** | 0.019 (0.06×) |

Three factors stack to make LAION-fMRI harder:

1. **Noise ceilings are roughly 2× NSD's** (5.4× in IT — see the NC table above). A fixed model captures a smaller fraction of a larger explainable-variance pool, so ceiled scores look lower.
2. **Stimulus distribution is broader.** Allen2022/NSD uses 73K curated COCO scenes; LAION-fMRI samples 25K mixed images from a much wider LAION distribution. Models trained on ImageNet-style data are closer to NSD's stylistic norm than to LAION's wider draw.
3. **Per-subject design is dramatically harder.** Each LAION-fMRI subject saw 4,712 subject-unique images on top of the 1,121 shared ones. Predicting subject-specific responses to subject-specific stimuli is genuinely harder than predicting cross-subject responses to shared stimuli.

**Important nuance: at the raw correlation level it's mixed.** Back-computing raw r (= ceiled × ceiling):

| Region | Allen2022 raw r | LAION shared tau raw r |
|---|---|---|
| V1 | 0.193 | ~0.270 (LAION higher) |
| V2 | 0.195 | ~0.278 (LAION higher) |
| V4 | 0.147 | ~0.221 (LAION higher) |
| IT | 0.126 | ~0.102 (Allen higher) |

LAION-fMRI individual voxel predictions are *more accurate* in V1/V2/V4 thanks to higher-SNR data and multi-echo denoising. The reason ceiled scores look lower is the higher ceiling, not worse fits. Only in IT does both raw and ceiled drop relative to Allen2022 — which is exactly where the broader stimulus distribution + persubject design bite hardest.

**For leaderboard purposes** (where ceiled scores are what gets ranked), this means LAION-fMRI is a useful next-generation benchmark: strong models still differentiate, but weak models collapse to noise much faster. Our full 5-model sweep showed resnet50_tutorial IT-tau persubject = 0.188 vs alexnet = 0.031 (6× gap), versus shared IT where the same comparison is only 2× (0.284 vs 0.136).

## Benchmark variants

**Headline registry: 20 variants.** The registered set is deliberately lean to keep the leaderboard focused on the most informative cells:

- `LAION_fMRI_persubject.{V1,V2,V4,IT}-{tau,ood}-ridge` — 8 persubject ridge variants (drive the leaderboard rank — most discriminative on hard stimuli)
- `LAION_fMRI.{V1,V2,V4,IT}-{tau,ood}-ridge` — 8 shared-pool ridge variants (cross-subject comparison with Allen2022 / Hebart2023)
- `LAION_fMRI.{V1,V2,V4,IT}-rdm-pearson` — 4 shared-pool RSA variants (representational alignment; complements encoding-based ridge)

Identifiers follow the pattern `{family}.{region}-{split}-{metric}` (e.g. `LAION_fMRI_persubject.IT-tau-ridge`, `LAION_fMRI.V4-rdm-pearson`).

**Non-headline variants accessible via factory API**: cluster_k5 cross-validation (re:vision Method 2), per-OOD-category sub-variants (`ood_shape`, `ood_gabor`, ...), and the `IT_full` runtime alias (V4 ∪ IT). These are intentionally not in `benchmark_registry` but constructible directly via `LAIONfMRI(region, split)`, `LAIONfMRIClusterCV(region)`, and `LAIONfMRIRSA(region)`.

**See `usage_examples.ipynb`** in this directory for runnable examples covering: scoring via the registry, calling the factory for non-headline variants, per-subject and per-fold detail, tuning the noise-ceiling threshold, and bulk-registering the non-headline variants into your session.

## Data distribution

- **Neural assemblies** (`LAION_fMRI_full`): CC0 1.0, uploaded to Brain-Score S3 (`brainscore-storage/brainscore-vision/benchmarks/LAION_fMRI/`).
- **Stimulus images**: gated by the LAION-fMRI Data Use Agreement (prohibits redistribution, commercial use, and use for training general-purpose AI models). Brain-Score does **not** mirror the stimuli. Users obtain them via `laion-fmri request-access` (signs DUA) and `laion-fmri download-stimuli`. The local stimulus loader (`data_packaging/get_local_stimuli.py`) extracts JPEGs from the resulting `task-images_stimuli.h5` and bundles them into the Brain-Score format at `~/.brainio/<sha1>/`.

### Assembly layout: per-subject files vs combined assembly

Allen2022 and Hebart2023 ship one combined `.nc` per benchmark variant — all subjects stacked along the neuroid dim, presentation aligned across subjects. LAION-fMRI instead ships one `.nc` per (pool, subject) — 5 files per pool, 10 total. The benchmark loader stitches them block-diagonally at scoring time.

This divergence is deliberate. Combining requires that every subject's `(stimulus_id, repetition)` rows align cleanly across subjects so that neuroid-axis concatenation produces a dense matrix. In Allen2022/Hebart2023 that holds because every subject saw the exact same stim list with the exact same rep counts. In LAION-fMRI it doesn't: per-subject row counts differ slightly (shared pool: 13,012 / 13,013 / 13,014 trials across our 5 subjects; persubject pool: 23-32K depending on completed sessions per subject). The mismatch is real — it comes from partially-completed final trials per session — not a packaging artifact.

The three combining strategies we evaluated all forced a tradeoff we weren't willing to take:

1. **Concat on neuroid with `(stim_id, rep)` alignment** — drops the ~2-3 misaligned trials per subject and shifts ridge scores at the last-decimal-place level.
2. **Pre-average reps per stim before concat** — loses individual repetition data needed for internal-consistency ceiling computation at scoring time.
3. **Concat on presentation with neuroid NaN-padding** — preserves all data but produces a ~2 GB matrix (vs the 1.9 GB total across 5 per-subject files) that is ~75% NaN, with worse memory characteristics for streaming.

Per-subject files preserve every subject's exact trial count without padding and without dropping rows. The convention difference adds 4 entries to the data registry per pool, which we judge a small price for bit-exact fidelity to the upstream dataset.

All tests are gated with `@pytest.mark.private_access` (Brain-Score's standard marker for benchmarks requiring private S3 / restricted resources). This benchmark is private-eval only — models are scored on our compute and never submitted externally.

## References

- Zerbe, J., Roth, J., Mell, M. M., Herholz, P., Knapen, T., & Hebart, M. N. (2026). LAION-fMRI: A densely sampled 7T-fMRI dataset providing broad coverage of natural image diversity. *Vision Sciences Society Annual Meeting*.
- Prince, J. S., Charest, I., Kurzawski, J. W., Pyles, J. A., Tarr, M. J., & Kay, K. N. (2022). Improving the accuracy of single-trial fMRI response estimates using GLMsingle. *eLife*, 11, e77599.
- Roth, J., & Hebart, M. N. (2025). LAION-natural: 120M curated natural-image–text pairs.
- Allen, E. J., St-Yves, G., Wu, Y., Breedlove, J. L., Prince, J. S., Dowdle, L. T., ... Kay, K. (2022). A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial intelligence. *Nature Neuroscience*, 25(1), 116–126.
- Gifford, A. T., Lahner, B., Saba-Sadiya, S., Vilas, M. G., Lascelles, A., Oliva, A., Kay, K., Roig, G., & Cichy, R. M. (2023). The Algonauts Project 2023 Challenge. *arXiv:2301.03198*.
