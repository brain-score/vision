# Package contents

The wheel contains the `brainscore_vision` namespace and its runtime resources,
including reusable dataset builders and examples. Top-level repository tests and
scripts remain in Git but are not installed as global Python packages. Python
bytecode is excluded.

## Reusable notebooks

NSD notebooks use `NSD_ROOT` for the authorized local dataset directory. The
VOneNet example uses `IMAGENET_VAL`. Install Brain-Score and the notebook-specific
dependencies before starting Jupyter; local source-path overrides are removed.
Li2026 validation uses `VAL_DATA` and `VAL_OUT`, defaulting to `./validation` and
`./validation/figs`. Saved outputs were cleared so local paths and old results
are not presented as a fresh execution. These edits do not qualify full dataset
reproduction; real-data runs remain separate.

## Archived exploration notebooks

The following notebooks depend on personal directories, old APIs, or undocumented
intermediate files. They are preserved at the same paths on
[`archive/pre-packaging-review-2026-10-21`](https://github.com/brain-score/vision/tree/archive/pre-packaging-review-2026-10-21).
Current runtime loaders are retained; builder exceptions are listed below.

- `brainscore_vision/data/cadena2017/data_packaging/2018-08-07_tolias_v1.ipynb`
- `brainscore_vision/data/david2004/data_packaging/2018-06-05_gallant_data.ipynb`
- `brainscore_vision/data/david2004/data_packaging/2018-05-23_gallant_data.ipynb`
- `brainscore_vision/data/david2004/data_packaging/2018-05-10_gallant_data.ipynb`
- `brainscore_vision/data/freemanziemba2013/data_packaging/2020-02-21_movshon_aperture.ipynb`
- `brainscore_vision/data/freemanziemba2013/data_packaging/2018-10-05_movshon.ipynb`
- `brainscore_vision/data/freemanziemba2013/data_packaging/2020-02-26_movshon_aperture_test.ipynb`
- `brainscore_vision/data/rust2012/2020-12-28_rust.ipynb`
- `brainscore_vision/data/seibert2019/data_packaging/2020-10-13_juvenile.ipynb`
- `brainscore_vision/data/seibert2019/data_packaging/2020-11-22_juvenile_dldata.ipynb`
- `brainscore_vision/data/igustibagus2024/investigation_consistency.ipynb`
- `brainscore_vision/data/igustibagus2024/oleo_pico_exploration.ipynb`
- `brainscore_vision/data/igustibagus2024/merged_assembly/create_merged_assembly.ipynb`
- `brainscore_vision/data/geirhos2021/creating_geirhos_ids.ipynb`

## Dataset builder defaults

Retained builders now use paths below `./datasets` instead of an author's home
or mounted drive. Run from a writable working directory. Supply the authorized
source data in the layout specified by the builder, or override its CLI arguments
where available. Edit the clearly named path constants in scripts without a CLI.
These scripts may require plugin-specific dependencies and some upload artifacts
to S3 when executed; inspect their entry point before running. Path cleanup does
not establish scientific reproduction or authorize uploads.

Defaults include `datasets/nsd`, `datasets/triple-n`, `datasets/hebart2023`,
`datasets/kuzovkin2018`, `datasets/ImageNetSlim15000`, `datasets/imagenet`,
`datasets/coggan2024`, `datasets/BashivanKar2019`, and `datasets/marques2020`.

Four additional historical scripts were archived because they depend on old
imports or missing intermediate-file workflows:

- `brainscore_vision/data/rust2012/rust305.py`
- `brainscore_vision/data/david2004/data_packaging/xr_align_debug.py`
- `brainscore_vision/data/rajalingham2018/rajalingham2018objectome.py`
- `brainscore_vision/data/seibert2019/data_packaging/juvenile.py`
