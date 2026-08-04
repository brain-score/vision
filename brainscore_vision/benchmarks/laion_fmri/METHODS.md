# LAION-fMRI Brain-Score Benchmark — Methods

## Dataset

LAION-fMRI (Zerbe, Roth, Mell, Herholz, Knapen, & Hebart, 2026, *VSS*) is a 7T fMRI dataset acquired at the Max-Planck Institute CBS Leipzig. Five participants viewed 25,052 unique natural images plus 1,492 shared images across 150 sessions. Stimuli were drawn primarily from LAION-natural (Roth & Hebart, 2025), with a 371-image out-of-distribution subset spanning abstract shapes, illusions, Gabor patches, and other deliberately atypical content.

**Acquisition.** Multi-echo 7T MRI at 1.8 mm isotropic, TR 1.9 s. Stimuli rendered 1000 × 1000 px on a PROpixx projector at ~165 cm viewing distance → **9.2 × 9.2 degrees of visual angle**. 2.5 s stimulus + 0.5 s ISI. Shared images repeated 4 or 12 times; subject-unique images repeated 4 times each (~31.9K presentations per subject).

**Preprocessing.** Single-trial betas via GLMsingle (Prince et al., 2022) on tedana-denoised multi-echo data, in volumetric T1w 1.8 mm space. Per-voxel betas are reliable in a relative sense but require within-session z-scoring before cross-session pooling (GLMsingle's per-voxel ridge regularization makes raw magnitudes incomparable across voxels).

## Space

**Volumetric T1w 1.8 mm**, the only space published by the dataset authors. No surface betas ship with the release; we do not introduce a surface projection step.

## Regions

| Region | Source ROI(s) |
|---|---|
| V1 | retinotopy.V1v ∪ V1d |
| V2 | retinotopy.V2v ∪ V2d |
| V4 | retinotopy.hV4 |
| IT | laionventral \ (V1v ∪ V1d ∪ V2v ∪ V2d ∪ V3v ∪ V3d ∪ hV4) |
| IT_full | V4 ∪ IT (factory-only alias for the authors' broader ventral mask) |

ROIs are bilateral. IT matches NSD's "streams ventral" parcel (Algonauts 2023 convention).

## Splits

Three split methods from the re:vision generalization initiative:

1. **`tau`** — within-distribution, fixed 80/20. Matches train/test population statistics but maximizes per-image nearest-neighbour distance in CLIP feature space. ~4,666 train + 1,167 test images per subject (plus ~897/224 shared).
2. **`ood`** — 371 deliberately atypical held-out stimuli (shapes, illusions, Gabors, cropped objects, etc.). Nine per-category sub-variants (`ood_shape`, `ood_gabor`, ...) accessible via factory.
3. **`cluster_k5_{0..4}`** — five-fold cluster CV over k-means on CLIP features. Each fold holds out one cluster as test. Aggregated into a single `cluster_k5` benchmark via `LAIONfMRIClusterCV`.

Splits are computed at benchmark-load time via `laion_fmri.splits.get_split_masks` and applied to one registered full assembly per subject.

### Note on `ood` direction at V4

In the persubject pool, ceiled scores for `ood` test trials in V1-V4 are typically *higher* than the same model's `tau` test. The persubject `ood` test is the 371 categorically-distinct images (Gabors, illusions, isolated shapes) — early/mid cortex predicts these unusually well from clean low-level features. At IT, where object identity dominates, the expected `ood ≤ tau` direction returns.

In the shared pool, the same effect attenuates with region: V1 and V2 show the conventional `ood < tau` (1-2 of 100 scored models invert), IT is also `ood < tau` (1 of 100 inverts), but **V4 shows `ood > tau` in 89 of 100 scored models** (median Δceiled +0.061, median Δraw +0.046). This is a real signature of the dataset design, not a benchmark error — the inversion has been audited end-to-end: train/test splits are fully disjoint (no OOD image appears in any train mask, all 5 subjects); the inversion holds independently across 4/5 subjects in alexnet's per-subject breakdown; alexnet's 9 per-OOD-category sub-cells all score *lower* than V4-tau when scored in isolation, so the inflation is not "OOD is V4-friendly"; and inflation magnitude is **flat across model-quality quartiles** (median Δraw +0.034 / +0.051 / +0.047 / +0.040 from weakest to strongest V4-tau-raw quartile of 100 production models), ruling out a model-capacity story.

The likely mechanism is between-image response variance: V4 is the smallest ROI by voxel count (`hV4` from retinotopy, ~1,485 voxels across 5 subjects post NC≥30 filter), and the heterogeneous 371-image OOD test set spans a broader V4 response range than the smaller, more homogeneous natural-image tau test holdout. Two outliers consistent with this story: `hmax` (handcrafted V4-like features) inflates dramatically (+0.174 Δraw) and `pixels` (random baseline) is the only model whose V4 goes the expected `ood < tau` direction substantially (−0.087 Δraw). Reviewers and downstream consumers should treat V4-ood as a complementary signal to V4-tau rather than as a "harder" test.

## Beta preprocessing

Single-trial volumetric betas are loaded per subject for the combined region mask, then **z-scored within each session per voxel**. The train assembly preserves individual trials; the test assembly is averaged across repetitions per stimulus for scoring (with a rep-preserved copy retained for the internal-consistency ceiling).

## Noise ceilings

Three published estimates ride as neuroid coordinates on every assembly:

- `nc_12rep` — ncsnr-based, computed from 881 shared 12-rep images. Default reliability filter.
- `nc_4rep` — ncsnr-based, from 611 shared 4-rep images.
- `nc_allrep` — ncsnr-based, pooled across all repeated stimuli.

Voxels with `nc_12rep < 30` are dropped before scoring (`NOISE_CEILING_THRESHOLD = 0.3 * 100`).

Per cross-subject medians of `nc_12rep`: V1 44 / V2 52 / V4 60 / IT 54 — roughly 2× NSD volumetric medians (5× in IT), attributable to multi-echo denoising and the 12-rep shared protocol.

The benchmark's `ceiling_func` additionally computes an **internal-consistency** ceiling at scoring time from the rep-preserved test assembly — this is the value applied to ceil the raw score.

## Scoring

The benchmark uses a Brain-Score `TrainTestNeuralBenchmark` per subject, wrapped in a `MultiSubjectNeuralBenchmark` that averages per-subject ceiled scores. Each cell returns:

- `score.values` — mean of per-subject ceiled scores
- `score.attrs['ceiling']` — noise ceiling
- `score.attrs['raw']` — per-subject (or per-fold) disaggregated values
- `score.attrs['error']`, `error_over`, `n_bootstrap` — bootstrap SE on the ceiled scale
- `score.attrs['sub-XX']` — per-subject scores

### Regression

Headline ridge cells use `dual_ridgecv_split` — kernel/dual ridge with per-fit CV alpha selection over a 21-value log-spaced sweep (`1e-10` to `1e10`, defined locally as `LAION_ALPHA_LIST`). The dual form keeps the `(n_features, n_targets)` coefficient matrix from being materialized — important for wide-feature models on the persubject pool.

Fixed-alpha ridge (`dual_ridge_split`, α=1) is accessible via the factory by passing `metric_type='ridge'`, but is not on the leaderboard.

### RSA flavour

`-rdm-pearson` variants run `RSABenchmark` per subject: per-subject neural RDM × model RDM Spearman r, averaged across 5 subjects. Per-subject ncsnr-derived RDM-reliability ceiling. Shared-pool only (cross-subject RDMs require shared stimuli).

### Memory footprint

Persubject ridgecv variants use ~4× the memory of shared-pool variants — each subject's 5,833-stim activation set is non-overlapping so there are no cross-subject activation cache hits. Per-cell peak scales with model layer width: resnet50-class peaks ~128 GB on persubject cells, resnext101-class ~256 GB. Production scoring of persubject variants for medium/large vision models should provision worker pools with ≥256 GB RAM.

## Registered variants

20 headline cells:

- `Zerbe2026_fmri_persubject.{V1,V2,V4,IT}-{tau,ood}-ridgecv` (8) — most discriminative
- `Zerbe2026_fmri.{V1,V2,V4,IT}-{tau,ood}-ridgecv` (8) — cross-subject comparable to Allen2022 / Hebart2023
- `Zerbe2026_fmri.{V1,V2,V4,IT}-rdm-pearson` (4) — RSA on shared pool

Non-headline (factory-only): fixed-alpha ridge (`metric_type='ridge'`), `cluster_k5` CV, per-OOD-category sub-splits, `IT_full` ablation.

## Data distribution

- **Neural assemblies** (`Zerbe2026_fmri_full_sub-XX`): CC0 1.0, served from Brain-Score S3.
- **Stimulus images**: gated by the LAION-fMRI DUA (prohibits redistribution, commercial use, training general-purpose AI). Brain-Score does not mirror them; users obtain them via `laion-fmri request-access` + `laion-fmri download-stimuli`. The local stimulus loader extracts JPEGs and bundles them into Brain-Score format.

All tests gated with `@pytest.mark.private_access`.

### Assembly layout

One `.nc` per (pool, subject) — 10 files total (5 subjects × shared + persubject pools). The benchmark loader stitches them block-diagonally at scoring time. This preserves per-subject trial counts exactly without forcing dropped rows or NaN padding that the combined-assembly convention used by Allen2022 / Hebart2023 would require.

## References

- Zerbe, J., Roth, J., Mell, M. M., Herholz, P., Knapen, T., & Hebart, M. N. (2026). LAION-fMRI: A densely sampled 7T-fMRI dataset providing broad coverage of natural image diversity. *Vision Sciences Society Annual Meeting*.
- Prince, J. S., Charest, I., Kurzawski, J. W., Pyles, J. A., Tarr, M. J., & Kay, K. N. (2022). Improving the accuracy of single-trial fMRI response estimates using GLMsingle. *eLife*, 11, e77599.
- Roth, J., & Hebart, M. N. (2025). LAION-natural: 120M curated natural-image–text pairs.
- Allen, E. J., St-Yves, G., Wu, Y., Breedlove, J. L., Prince, J. S., Dowdle, L. T., ... Kay, K. (2022). A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial intelligence. *Nature Neuroscience*, 25(1), 116–126.
