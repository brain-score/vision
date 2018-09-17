# Brain-Score

A framework for the quantitative comparison of mindlike systems.

## Introduction

`brainscore` is a simple framework
for standardizing the interface between neuroscience metrics
and the data they operate on.
It is based on the package [`xarray`](http://xarray.pydata.org/),
a project affiliated with NumFOCUS,
which extends the capabilities of `pandas`
to multi-dimensional `numpy` arrays.


## Quick setup

Recommended for most users. Use Brain-Score as a library. You will need Python >= 3.6.

`pip install --process-dependency-links git+https://github.com/dicarlolab/brain-score`

To contribute code to Brain-Score, see the [Development Setup](#development-setup).


## Basic Usage

Try it in IPython:
```python
ipython
import brainscore
hvm = brainscore.get_assembly("dicarlo.Majaj2015")`
hvm
# The IPython output should show a representation of a `NeuronRecordingAssembly`,
# including a snippet of the 3-dimensional numeric data,
# and a list of the metadata coordinates attached to it.
hvm.attrs["stimulus_set"]
# The output displays the `StimulusSet` object (a subclass of a pandas `DataFrame`)
# associated with this `NeuronRecordingAssembly`.
# The same `StimulusSet` can be obtained directly:
hvm_images = brainscore.get_stimulus_set("dicarlo.hvm")
# You can also obtain the local path of any individual image via its `image_id`:
hvm_images.get_image("8a72e2bfdb8c267b57232bf96f069374d5b21832")
```

Some steps may take minutes because data has to be downloaded.

More examples can be found in the `examples/` directory.


## Environment Variables

| Variable               | Description                                                            |
|------------------------|------------------------------------------------------------------------|
| BSC_BOTO3_SIGN         | 0 (default) to not sign S3 requests, 1 to sign and access private data |


## Development setup

Only necessary if you plan to change code.

1. If you want to access private S3 data, get permissions for the DiCarlo Lab Amazon S3 account
    1. There are several accounts, you want the one numbered 848242192475. [Chris Shay](cshay@mit.edu) can get you access
    2. Configure your AWS credentials files: using awscli `aws configure` (options e.g. region `us-east-1`, format `json`)
2. Clone the Git repository to wherever you keep repositories:
    * `cd ~/dev`
    * `git clone git@github.com:dicarlolab/brain-score.git`
3. Create and activate a Conda environment with relevant packages:
    * `conda env create -f environment.yml`
    * `conda activate brainscore`


## License
MIT license


## Troubleshooting
<details>
<summary>`ValueError: did not find HDF5 headers` during netcdf4 installation</summary>
pip seems to fail properly setting up the HDF5_DIR required by netcdf4.
Use conda: `conda install netcdf4`
</details>
