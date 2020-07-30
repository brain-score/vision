[![Build Status](https://travis-ci.com/brain-score/brain-score.svg?token=vqt7d2yhhpLGwHsiTZvT&branch=master)](https://travis-ci.com/brain-score/brain-score)
[![Documentation Status](https://readthedocs.org/projects/brain-score/badge/?version=latest)](https://brain-score.readthedocs.io/en/latest/?badge=latest)

# Brain-Score

`brainscore` standardizes the interface between neuroscience metrics
and the data they operate on.
Brain recordings (termed "assemblies", e.g. neural or behavioral)
are packaged in a [standard format](http://xarray.pydata.org/).
This allows metrics (e.g. neural predictivity, RDMs) to operate
on many assemblies without having to be re-written.
Together with http://github.com/brain-score/candidate_models, `brainscore`
allows scoring candidate models of the brain on a range of assemblies and metrics.


## Quick setup

Recommended for most users. Use Brain-Score as a library. You will need Python >= 3.6 and pip >= 18.1.

`pip install git+https://github.com/brain-score/brain-score`

To contribute to Brain-Score, please [send in a pull request](https://github.com/brain-score/brain-score/pulls).


## Basic Usage

```python
import brainscore
data = brainscore.get_assembly("dicarlo.MajajHong2015")
data
> <xarray.NeuronRecordingAssembly 'dicarlo.MajajHong2015' (neuroid: 296, presentation: 268800, time_bin: 1)>
> array([[[ 0.060929],
>         [-0.686162],
>         ...,
> Coordinates:
>   * neuroid          (neuroid) MultiIndex
>   - neuroid_id       (neuroid) object 'Chabo_L_M_5_9' 'Chabo_L_M_6_9' ...
>   ...

from brainscore.metrics.rdm import RDM
metric = RDM()
score = metric(assembly1=data, assembly2=data)
> Score(aggregation: 2)>
> array([1., 0.])
> Coordinates:
>   * aggregation    'center' 'error'
```

Some steps may take minutes because data has to be downloaded during first-time use.

For more details, see [the documentation](https://brain-score.readthedocs.io).


## Environment Variables

| Variable               | Description                                                                                                                           |
|------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| RESULTCACHING_HOME     | directory to cache results (benchmark ceilings) in, `~/.result_caching` by default (see https://github.com/mschrimpf/result_caching) |



## License
MIT license


## Troubleshooting
<details>
<summary>`ValueError: did not find HDF5 headers` during netcdf4 installation</summary>
pip seems to fail properly setting up the HDF5_DIR required by netcdf4.
Use conda: `conda install netcdf4`
</details>

<details>
<summary>repeated runs of a benchmark / model do not change the outcome even though code was changed</summary>
results (scores, activations) are cached on disk using https://github.com/mschrimpf/result_caching.
Delete the corresponding file or directory to clear the cache.
</details>
