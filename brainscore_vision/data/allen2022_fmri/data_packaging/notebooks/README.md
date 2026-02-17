# Allen2022 NSD fMRI Benchmark Notebooks

These notebooks document the full pipeline for building the Allen2022 NSD fMRI
benchmarks in brain-score. Starting from raw NSD data, they cover data exploration,
beta extraction, assembly packaging, benchmark scoring, and scientific validation.

Both volumetric (func1pt8mm voxel space) and fsaverage surface pipelines are included.

## Notebooks

### Volumetric (`volumetric/`)

| Notebook | Description |
|----------|-------------|
| 01_nsd_data_exploration | Explore raw betas, ROI masks, and reproduce published noise ceilings |
| 02_nsd_data_preparation | Extract shared-image betas for all subjects and regions |
| 03a_nsd_packaging_prep | Build train/test NeuroidAssembly with global z-score |
| 03b_nsd_stimulus_packaging | Extract stimulus PNGs from nsd_stimuli.hdf5 |
| 04_nsd_benchmark_smoke_test | Score AlexNet on all 16 volumetric benchmarks |
| 05_nsd_assembly_validation | End-to-end validation: spot check, raw bypass, standalone ridge |

### Surface (`surface/`)

| Notebook | Description |
|----------|-------------|
| 01_surface_data_exploration | Explore fsaverage MGH betas, ROI labels, validate NC |
| 02_surface_data_preparation | Extract surface betas for shared images across subjects |
| 03_surface_packaging_prep | Build surface train/test assemblies with global z-score |
| 04_surface_smoke_test | Score AlexNet on all 16 surface benchmarks |
| 05_surface_assembly_validation | Validate surface assemblies against raw MGH files |

### Cross-domain

| Notebook | Description |
|----------|-------------|
| 06_scientific_validation | IT ROI sensitivity analysis, split stability, surface vs volumetric ceiling comparison, per-category diagnostics |

## Prerequisites

- **Data:** NSD dataset (~500 GB) from https://naturalscenesdataset.org
- **Environment:** `conda activate vision-2026`
- **Execution:** Run 01-05 sequentially per pipeline, then 06. Each notebook defines
  `NSD_ROOT` at the top -- update this to your local NSD data path.
- **Shared stimuli:** The stimulus images are produced by `volumetric/03b_nsd_stimulus_packaging`.
  The surface pipeline reuses these same images. If you are only running the surface
  pipeline, you still need to run 03b from the volumetric folder first.

## Algonauts 2023 Tutorial

For additional context on NSD encoding models, the Algonauts 2023 Challenge Tutorial
is a useful reference. It's not included here due to size (64 MB), but you can view or
download it from:
https://colab.research.google.com/drive/1bLJGP3bAo_hAOwZPHpiSHKlt97X9xsUw

## Note on Outputs

Interactive HTML outputs (nilearn surface views) have been stripped to keep repo size
manageable. Static plots and text outputs are preserved. Re-run notebooks with NSD data
mounted to regenerate all visualizations.
