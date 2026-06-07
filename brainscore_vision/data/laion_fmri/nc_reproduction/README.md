# Independent reproduction of the published `nc_12rep` noise ceiling

Validates that our benchmark's use of the dataset's shipped per-voxel noise-ceiling
map (`Noiseceiling12rep_statmap.nii.gz`) is faithful to the underlying GLMsingle
betas — by reimplementing the Allen2022 NSD variance-based estimator from scratch
and comparing voxel-by-voxel.

See `../../benchmarks/laion_fmri/METHODS.md` § "Independent reproduction of `nc_12rep`"
for the headline results and discussion.

## Files

| File | Purpose |
|---|---|
| `reproduce_nc12rep.py` | The reproduction script. Loads per-session TYPED betas + trial TSVs, z-scores within session, computes per-voxel variance across the 12 repetitions of each 12-rep image, and applies the NSD ncsnr formula `NC = 100·ncsnr²/(ncsnr² + 1/12)`. Compares to the dataset's `Noiseceiling12rep_statmap.nii.gz` per voxel within V1/V2/V4/IT. |
| `nc_reproduction_sub-XX.csv` | Per-voxel output (one row per voxel: region, reproduced NC, published NC). Generated only when the script is run. |
| `summary.csv` | Per (subject, region) summary: voxel count, median reproduced NC, median published NC, Pearson r, Spearman ρ. Generated when the script is run. |

## Running it

Prereq: the LAION-fMRI raw derivatives (`derivatives/glmsingle-tedana/sub-XX/...`)
and ROIs (`derivatives/rois/sub-XX/...`) must be locally accessible. Defaults
match the layout that `laion-fmri config --data-dir ~/laion-fmri` produces.

```bash
# Default ~/laion-fmri/
python -m brainscore_vision.data.laion_fmri.nc_reproduction.reproduce_nc12rep

# Custom data directory + subject subset:
python -m brainscore_vision.data.laion_fmri.nc_reproduction.reproduce_nc12rep \
    --data-dir /mnt/big-volume/laion-fmri \
    --subjects sub-01 sub-03
```

Takes ~10 min per subject (33 sessions × ~3 GB beta volume each), ~50 min for all 5.

## Headline finding

For sub-01 within V1/V2/V4/IT (6,379 voxels), reproduced vs published `nc_12rep`:

| Region | Spearman ρ | Pearson r | Median offset (reproduced ÷ published, on ncsnr scale) |
|---|---|---|---|
| V1 | **1.000** | 0.993 | 1.83× |
| V2 | **1.000** | 0.991 | 1.35× |
| V4 | **1.000** | 0.992 | 1.20× |
| IT | **1.000** | 0.991 | 1.36× |

Spearman ρ = 1.000 confirms our reproduction identifies the **same voxels as more/less reliable** as the published map. The systematic positive scale offset on absolute NC values is attributable to the noise-SD-vs-trial-SD normalization step in NSD/GLMsingle that we can't fully replicate without access to the GLMsingle `_model.h5` files (which the dataset doesn't distribute).

Bottom line: structural reliability is reproduced; absolute values defer to the published map, which the benchmark consumes directly.
