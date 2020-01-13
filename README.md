[![Build Status](https://travis-ci.com/brain-score/brain-score.svg?token=vqt7d2yhhpLGwHsiTZvT&branch=master)](https://travis-ci.com/brain-score/brain-score)

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

To contribute code to Brain-Score, see the [Development Setup](#development-setup).


## Basic Usage

```python
$ import brainscore
$ hvm = brainscore.get_assembly("dicarlo.Majaj2015")`
$ hvm
<xarray.NeuronRecordingAssembly 'dicarlo.Majaj2015' (neuroid: 296, presentation: 268800, time_bin: 1)>
array([[[ 0.060929],
        [-0.686162],
        ...,
Coordinates:
  * neuroid          (neuroid) MultiIndex
  - neuroid_id       (neuroid) object 'Chabo_L_M_5_9' 'Chabo_L_M_6_9' ...
  ...
$ ...
$ metric = RDM()
$ score = metric(assembly1=hvm, assembly2=hvm)
Score(aggregation: 2)>
array([1., 0.])
Coordinates:
  * aggregation    'center' 'error'
```

Some steps may take minutes because data has to be downloaded during first-time use.

More examples can be found in the [examples](examples/) directory.


## Environment Variables

| Variable               | Description                                                                                                                           |
|------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| RESULTCACHING_HOME     | directory to cache results (benchmark ceilings) in, `~/.result_caching` by default (see https://github.com/mschrimpf/result_caching) |


## Development setup

Only necessary if you plan to change code.

1. If you want to access private S3 data, get permissions for the DiCarlo Lab Amazon S3 account
    1. The lab has several S3 accounts. You need to have access to the one numbered 613927419654. Ask [Chris Shay](cshay@mit.edu) to grant access to you
    2. Configure your AWS credentials files using awscli:
      1. Install awscli using `pip install awscli`
      2. Run `aws configure`: region: `us-east-1`, output format: `json`
2. Clone the Git repository to wherever you keep repositories:
    * `cd ~/dev`
    * `git clone git@github.com:dicarlolab/brain-score.git`
3. Install the depencies (we suggest doing this in a [conda environment](https://conda.io/docs/user-guide/tasks/manage-environments.html)):
    * `pip install -e .`


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
