# Li2026 — Triple-N macaque Neuropixels on NSD shared1000

Macaque electrophysiology data plugin. Wraps the **Triple-N** dataset
(Li, Liu, …, Bao. *Triple-N dataset: large-scale fMRI-guided dense
recordings of nonhuman primate neural responses to natural scenes*,
**Nature Neuroscience 2026**) into Brain-Score's `StimulusSet` +
`NeuronRecordingAssembly` format.

* Paper / DOI: <https://doi.org/10.1038/s41593-026-02322-z>
* Project docs: <https://liyipeng-moon.github.io/Triple-N-Docs/>
* Upstream code: <https://github.com/liyipeng-moon/Triple-N>

---

## What's in the registry

| Identifier            | Type                       | Shape                                        | Notes |
|-----------------------|----------------------------|----------------------------------------------|-------|
| `Li2026`              | StimulusSet (PNG + CSV)    | 1000 images                                  | Full-resolution PNGs sourced from `nsd_stimuli.hdf5`. |
| `Li2026`              | NeuronRecordingAssembly    | `presentation=1000 × neuroid=47,503 × time_bin=1` | Static response — mean firing rate in a **fixed 70–170 ms post-onset window** applied uniformly to every unit. Derived from `Li2026.temporal`; matches the MajajHong2015 static convention so this assembly is directly comparable to other Brain-Score primate-IT benchmarks. |
| `Li2026.temporal`     | NeuronRecordingAssembly    | `presentation=1000 × neuroid × time_bin=30`  | 10 ms PSTH bins covering 0–300 ms post stimulus onset. |

All three are loaded from `s3://brainscore-storage/brainscore-vision/benchmarks/Li2026/`
with hashes pinned in `__init__.py`.

---

## Stimulus set

* **1000 natural scene images**, sourced verbatim from the NSD `nsd_stimuli.hdf5`
  `imgBrick`. Pixel-identical to NSD's *shared1000* — the subset shown to every
  NSD subject — and to Brain-Score's `Allen2022` stimulus set.
* **Visual angle**: ~11° at the macaque eye.
* Triple-N also includes 72 *localizer* stimuli (presentation indices 1001–1072
  in the upstream `.mat` files). These are **not** in this plugin's
  StimulusSet — only the 1000 shared natural-scene images are packaged here.

### Stimulus coords

| coord                | meaning |
|----------------------|---------|
| `stimulus_id`        | `nsd_<NNNNN>` where `NNNNN` is the **0-indexed** NSD 73k id (5-digit, zero-padded). Identical to `Allen2022`'s `stimulus_id`, so the same model activation extraction can serve both benchmarks. |
| `nsd_id`             | 0-indexed NSD 73k id as an int (matches `Allen2022.nsd_id`). |
| `nsd_id_1indexed`    | 1-indexed NSD 73k id — convenience for users coming from NSD's MATLAB tooling, which is 1-based. |
| `coco_id`            | MS-COCO image id, joined from `nsd_stim_info_merged.csv`. |
| `tn_index`           | Triple-N's 1-based presentation index (1..1000). Use this when cross-referencing the upstream `.mat`/`.h5` files. |

---

## Neural data

* **Subjects**: 5 macaques (`M1`–`M5`).
* **Probes**: dense Neuropixels recordings, fMRI-guided to visual ROIs.
* **Sessions**: 90 total — 71 in IT, 19 across V1/V2/V4.
* **Presentation**: 150 ms on / 150 ms off, 4–8 repetitions per image, free
  viewing with central fixation.
* **Unit count after packaging**: 47,503 across all four regions and all five
  animals. ~26,700 IT units cross the paper's best-window `reliability_best > 0.4`
  (the `reliability` coord); the benchmark instead selects on the 70-170 ms
  `reliability_window > 0.4`, giving ~21,059 IT (thresholding is applied at benchmark
  time, not baked into the assembly).

### Neuroid coords (both static and temporal)

| coord                 | meaning |
|-----------------------|---------|
| `neuroid_id`          | `M{1..5}_ses{NN}_unit{kkkk}` — animal × session × within-session index. |
| `region`              | `V1`, `V2`, `V4`, or `IT`. IT is session-level; V1/V2/V4 are pos-level (resolved from `exclude_area.xls`). |
| `animal`              | `M1`–`M5`. |
| `unittype`            | `1`=single unit, `2`=multi-unit activity, `3`=non-somatic. |
| `reliability`         | Paper-canonical per-neuroid split-half Spearman-Brown reliability at each unit's **best** window (Triple-N's `reliability_best`). Kept for provenance and to reproduce the paper's reliable-unit counts (IT ~26.7k); **not** the benchmark ceiling. |
| `reliability_window`  | Split-half Spearman-Brown reliability recomputed at the **fixed 70–170 ms window** (the response actually scored). The static benchmark **selects and ceils on this** so the noise ceiling matches the scored response. Built by `data_packaging/build_li2026_reliability_70_170ms.py`. |
| `arealabel`           | Category-region patch label (e.g. `MBody`, `MFace`, `AObject`) or `IT-other`/`V1`/… |
| `pos`, `F_SI`, `B_SI`, `O_SI`, `snr`, `best_time_start/end`, `tn_index` | Carried through from the upstream Processed files for downstream slicing / diagnostics. |

### Temporal assembly extras

| coord                 | meaning |
|-----------------------|---------|
| `time_bin_start`/`time_bin_end` | Window edges in ms, 0–300 ms in 10 ms steps (30 bins). |

The temporal assembly is computed from the per-session `raster_matrix_img`
tensors (`ses*.h5 /raster_matrix_img`, units × trials × 1 ms) by rebinning
to 10 ms and averaging across repetitions per image. The static assembly
collapses the time axis by picking each unit's individually-best 30 ms
response window.

---

## Relationship to NSD (Allen2022)

* **Image identity**: 1:1. Both Triple-N and Allen2022 source their 1000-image
  stimulus set from the same NSD `nsd_stimuli.hdf5`. Pixel-match was verified
  during packaging (`r ≈ 1.000` per-image cross-correlation).
* **Index convention**: Triple-N's upstream `.mat` uses a 1-based
  presentation index that aliases NSD's `sharedix` field
  (`tn_index = sharedix`). When packaged for Brain-Score, the canonical
  `stimulus_id` is rewritten to match Allen2022's `nsd_<NNNNN>` 5-digit
  zero-padded form on the 0-indexed 73k id (i.e. `nsd_id = sharedix - 1`).
  The "off-by-one" between Triple-N's MATLAB tooling and Allen2022's
  Python tooling is therefore handled at packaging time — downstream
  consumers only see the unified `stimulus_id`.
* **Cross-study comparisons enabled by shared identifiers**:
  * `Allen2022_fmri[.surface].{V1,V2,V4,IT}-{ridge,rdm}` — human fMRI on
    the same 1000 images.
  * `Hebart2023_fmri.{V1,V2,V4,IT}-ridgecv` — human fMRI on a related
    set with overlapping NSD images.
  * Any model's activation extraction over the `Li2026` stimulus set is
    reusable as-is for the Allen2022 / Hebart2023 benchmarks.
* **What Triple-N adds that NSD does not**: single-unit / multi-unit
  macaque electrophysiology at native temporal resolution, plus the 72
  localizer stimuli (held back from this plugin, in the upstream
  release). NSD itself has no spike data.

---

## Layout

```
data/li2026/
├── __init__.py                          # registry entries (stimulus + assembly + temporal)
├── data_packaging/
│   ├── build_li2026_static.py           # MAT → static NeuronRecordingAssembly (presentation × neuroid × 1)
│   ├── build_li2026_temporal.py         # H5 rasters → 10 ms PSTH assembly (presentation × neuroid × 30)
│   ├── package_li2026.py                # upload stimulus set + static assembly to S3
│   ├── package_li2026_temporal.py       # upload temporal assembly to S3
│   ├── notebooks/                       # exploratory + validation notebooks
│   └── uploaded_data_info/              # JSON manifests with sha1/version_id pinned in __init__.py
└── README.md                            # this file
```

Packaging scripts read the raw upstream files from
`/Volumes/Hagibis/triple-n/` (local; not part of this repo). The packaged
artifacts on S3 are the deployable surface.

---

## Methodology — fixed-window static, why and how

The shipping `Li2026` static is **not** the upstream `response_best` matrix.
Triple-N's per-unit best-window method is appropriate for the paper's
unit-characterization analyses (reliability, SNR, selectivity) and is
cross-validated for those uses, but it is the **wrong convention** for
Brain-Score's model-vs-brain population scoring:

* Other primate-IT static benchmarks (MajajHong2015, Sanghavi2020) use a
  single fixed window applied uniformly to every unit, so scoring against
  Li2026 needs to use the same convention to make leaderboard numbers
  comparable across the suite.
* Per-unit windowing means different units are measured in different
  temporal slices — a population code reconstructed from those slices is
  not a coherent snapshot for comparison against a model's
  single-forward-pass representation.
* The paper itself falls back to fixed/binned windows for its
  cross-population analyses (Fig 4 RSA uses 20 ms peak-aligned bins; Fig 5
  encoding models use a fixed peak time lag), so the methodology used
  here is consistent with how the paper handles the same question.

**Window choice — 70–170 ms post-onset.** Empirically the reliable-IT
population PSTH peaks at ~130–140 ms with the >50%-of-peak band spanning
100–220 ms, so 70–170 ms covers the leading edge through peak and avoids
the post-peak decay tail. This matches MajajHong2015's static window.

**Window-matched ceiling.** Because the response is now the 70–170 ms window, the
noise ceiling must be the reliability *of that window* — not the best-window
`reliability_best` (which is ~0.08 higher and would make the ceiling optimistic and
scores systematically low). `data_packaging/build_li2026_reliability_70_170ms.py`
recomputes per-unit split-half Spearman-Brown reliability from the raw GoodUnit
rasters using the 70–170 ms per-trial rate, and stores it as `reliability_window`.
The benchmark **selects and ceils on `reliability_window`** (mirroring how
MajajHong2015/Allen2022 threshold and ceil on the scored-response reliability).

**Build pipeline.** `build_li2026_static_70_170ms.py` averages the ten 10-ms bins
covering 70–170 ms from the temporal assembly, merges the metadata coords
(`arealabel`, `snr`, …) back from the original static and the `reliability_window`
column, and writes a `(presentation × neuroid × time_bin=1)` NetCDF.
`package_li2026_static_70_170ms.py` uploads it and prints the version_id + sha1 used
in `__init__.py`. `build_li2026_static.py` / `package_li2026.py` are retained for
reference (the upstream best-window methodology) but are not the deployed pipeline.

## Sequencing

The data plugin is delivered in three tiers (see `tasks/todo.md` Triple-N
section for the full plan):

1. **v1 — static, light tier, upstream best-window** *(superseded by v2)*.
   Per-unit best-window means from the `Processed_ses*.mat` files. Kept
   in `build_li2026_static.py` for documentation; not currently exposed
   through the registry.
2. **v2 — static, fixed 70–170 ms window** *(shipped — this is what
   `Li2026` resolves to)*. Derived from the temporal assembly, see
   "Methodology" above. Selection and ceiling use `reliability_window`
   (split-half SB recomputed at 70–170 ms); best-window `reliability` is
   retained as a provenance coord. Window-matched reliable counts:
   IT ≈ 21.1k, V1 ≈ 2.3k, V2 ≈ 2.5k, V4 ≈ 3.4k (vs the paper's best-window
   IT 26.7k).
3. **v3 — temporal** *(shipped as data only, `Li2026.temporal`)*. 10 ms PSTH
   bins. Useful for recurrent / temporal models (CORnet-S, etc.). Ceiling is
   per-bin and is most informative in the 0–200 ms range. No benchmark is
   registered on it yet: `spantime_pls` stacks time into the sample dimension
   (1000 × 30 = 30k rows per fit), and `TemporalIgnore` repeats a feedforward
   model's single response across all 30 bins, so non-recurrent models would pay
   ~38× the static cost to be scored on 30 copies of the same activations.

## Benchmarks

Built on this plugin in `brainscore_vision/benchmarks/li2026/`. Scored leaves hang
off a dataset-level node rather than the region directly, so Li2026 contributes one
child per region however many metrics it carries — the convention Geirhos2021,
Scialom2024 and Baker2022 already follow:

```text
V1 → Li2026.V1 → {Li2026.V1-ridgecv, Li2026.V1-rdm}
V2 → Li2026.V2 → {Li2026.V2-ridgecv, Li2026.V2-rdm}
V4 → Li2026.V4 → {Li2026.V4-ridgecv, Li2026.V4-rdm}
IT → Li2026.IT → {Li2026.IT-ridgecv, Li2026.IT-rdm}
```

The four `Li2026.{region}` nodes must exist as `BenchmarkType` rows before the first
submission. `benchmarkinstance_from_benchmark` looks the parent up and, if it is
missing, logs a warning and leaves the parent NULL — the leaf is still scored but
hangs off nothing.


* `Li2026.{V1,V2,V4,IT}-ridgecv` — scored leaves, one per region. RidgeCV over
  `ALPHA_LIST`, matching the other high-neuroid-count datasets in the suite
  (Allen2022, Hebart2023, Papale2025). Alpha selection is leave-one-out GCV off a
  single gram decomposition, so it costs about the same as a fixed-alpha ridge fit.
* `Li2026.{V1,V2,V4,IT}-rdm` — scored leaves. Representational geometry over the
  same units, one RDM per animal, averaged. Near-orthogonal to the encoding leaves
  rather than a restatement of them, and the measure that compares directly with
  `Allen2022_fmri.{region}-rdm` on the same 1000 NSD images: same stimuli, same
  metric, human fMRI against macaque electrophysiology. An encoding model cannot
  support that comparison because the fitted mapping absorbs the species and
  modality difference.

  **Use the raw score for cross-study comparison, not the ceiled score.**
  `RSACeiling` returns the Nili upper bound, which correlates each subject against
  the mean of *all* subjects including itself, so it does not converge in subject
  count. Measured on IT by subsampling animals: 0.767 (N=2), 0.678 (N=3), 0.631
  (N=4), 0.601 (N=5) — still falling at N=5. Ceiled scores are therefore only
  comparable between datasets with matched subject counts, which is rarely the
  case: Allen2022 has 8 subjects, so even `Li2026.IT-rdm` sits above it. The raw
  model-to-neural RDM correlation carries no normalisation and is directly
  comparable. The leave-one-out lower bound is kept in
  `ceiling.attrs['lower_bound_loo']`.

  | leaf | animals | ceiling (Nili) | LOO lower |
  | --- | --- | --- | --- |
  | `Li2026.V1-rdm` | 2 | 0.822 | 0.396 |
  | `Li2026.V2-rdm` | 2 | 0.912 | 0.673 |
  | `Li2026.V4-rdm` | 2 | 0.780 | 0.264 |
  | `Li2026.IT-rdm` | 5 | 0.601 | 0.337 |

  Mean pairwise inter-animal RDM agreement in IT is 0.223 (range 0.101-0.449), so
  the reliably shared geometry across animals is modest; small differences between
  models on these leaves should not be over-read.
* `Li2026.{V1,V2,V4,IT}-pls` — unparented (runnable, not in the scored tree).
  Kept for comparability against MajajHong2015, which is PLS. PLS is not the
  headline here because its cross-covariance is (n_features × n_neuroids) and no
  PLS benchmark in the suite runs above ~1k neuroids; Li2026 IT has ~21k.
