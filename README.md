# Brain-Score

A framework for the quantitative comparison of mindlike systems.

#### Introduction

`brainscore` is a simple framework
for standardizing the interface between neuroscience metrics
and the data they operate on.
It is based on the package [`xarray`](http://xarray.pydata.org/),
a project affiliated with NumFOCUS,
which extends the capabilities of `pandas`
to multi-dimensional `numpy` arrays.


#### Setup

1. Get permissions for the DiCarlo Lab Amazon S3 account (There are several accounts, you want the one numbered 848242192475) and configure your AWS credentials files.
2. Clone the Git repository to wherever you keep repositories:
    * `cd ~/dev`
    * `git clone git@github.com:dicarlolab/brain-score.git`
3. Create and activate a Conda environment with relevant packages:
    * `conda env create -f environment.yml`
    * `conda activate brainscore`


#### Basic Usage

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

#### License
MIT license
