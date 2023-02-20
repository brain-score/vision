Model Metamers
==============

## Contents
* [Overview](#overview)
* [Repo Directories](#repo-directories)
* [Installation Guide](#installation-guide)
* [Feather et al. 2022 figure replications](#feather-et-al-2022-figure-replications)
    * [Human recognition of metamers from Feather et al. 2022 models](#human-recognition-of-metamers-from-feather-et-al-2022-models)
    * [Model-model comparisons from Feather et al. 2022 models](#model-model-comparisons-from-feather-et-al-2022-models)
    * [Auditory fMRI voxel regression analysis from Feather et al. 2022](#auditory-fmri-voxel-regression-analysis-from-feather-et-al-2022)
* [Metamer generation from the command line](#metamer-generation-from-the-command-line)
* [Null distribution generation from the command line](#null-distribution-generation-from-the-command-line)
* [Setup configuration for your own model](#setup-configuration-for-your-own-model)
* [Citation](#citation)
* [Authors](#authors)
* [Acknowledgments](#acknowledgments)
* [License](#license)


Overview
========
Deep neural network models of sensory systems are often proposed to learn representational transformations with invariances like those in the brain. To reveal these invariances we generated "model metamers" — stimuli whose activations within a model stage are matched to those of a natural stimulus. In the paper ["Model metamers illuminate divergences between biological and artificial neural networks"](https://www.biorxiv.org/content/10.1101/2022.05.19.492678v1), we demonstated that metamers for state-of-the-art supervised and unsupervised neural network models of vision and audition were often completely unrecognizable to humans when generated from deep model stages, suggesting differences between model and human invariances. Targeted model changes improved human-recognizability of model metamers, but did not eliminate the overall human-model discrepancy. The human-recognizability of a model's metamers was well predicted by their recognizability by other models, suggesting that models learn idiosyncratic invariances in addition to those required by the task. Metamer recognition dissociated from both traditional brain-based benchmarks and adversarial vulnerability, revealing a distinct failure mode of existing sensory models and providing a complementary benchmark for model assessment.  

Here, we provide code for replicating the main analyses in the paper, including instructions to download pytorch checkpoints for the analyzed models. We additionally provide code and a tutorial for generating metamers from a custom model, comparing the metamers to a "null distribution" to validate optimization success, and example experiments to run on Amazon Mechanical Turk to test human recogognition of generated model metamers. 

Repo Directories
================
* [model_analysis_folders](model_analysis_folders): folders containing the files to build the pre-trained neural networks, and configuration files specifying the analysis parameters used in Feather et al. 2022. 
* [analysis_scripts](analysis_scripts): python files used to run each of the analyses for visual and audio metamer generation, null distribution construction, network evaluation (on clean and adversarial data), network-network comparisons, and fMRI prediction analysis for an auditory dataset.  
* [AuditoryBehavioralExperiments](AuditoryBehavioralExperiments): Human data for each of the word-recognition human behavioral experiments. Also includes an html file as an example experiment for Amazon Mechanical Turk and the matlab scripts to replicate the ANOVA statistics.
* [VisionBehavioralExperiments](VisionBehavioralExperiments): Human data for each of the 16-way image classification human behavioral experiments. Also includes an html file as an example experiment for Amazon Mechanical Turk and the matlab scripts to replicate the ANOVA statistics.
* [robustness](robustness): Scripts for training and model evaluation, based on [https://github.com/MadryLab/robustness](https://github.com/MadryLab/robustness), with changes for metamer generation, model architectures, and training with auditory stimuli.
* [notebooks](notebooks): Jupyter notebooks for Figure replication, plotting the human and model recognition of metamers and the auditory fMRI analysis. Also includes notebooks that show example model metamers for each experiment. 
* [matlab_statistics_functions](matlab_statistics_functions): matlab functions for running the anovas used for statistical comparisons. 

Installation Guide
==================
The python module versions used for the analysis in Feather et al. 2022 are output into [feather_metamers_conda_2022.yml](feather_metamers_conda_2022.yml). Using conda, this environmental can be created with the following: 
`conda env create -f feather_metamers_conda_2022.yml`

Currently, the repository is not set up for pip installation.

ANOVA statistics were performed in MATLAB 2021a (not included in the conda installation). 

## Downloading Assets
Some files are too large for uploading to github and are hosted elsewhere. A helper script is included for downloading. 
* Visual model checkpoints
* Auditory model checkpoints
* Base stimulus sets necessary for metamer generation
* Auditory fMRI data

The helper script can be run with `python download_large_files.py`. Note: Downloading all of the assets will take ~15GB of storage! If you only need some of the files, only run the necessary lines from `download_large_files.py`. 

## Testing Environment
All code for Feather et al. 2022 was run and tested on the [MIT OpenMind Computing Cluster](https://openmind.mit.edu/) with the CentOS Linux 7 (Core) operating system. Model training and metamer generation relies on GPUs. Metamer generation was performed on NVIDIA GPUs with a minimum of 11 GB of RAM. GPUs used for model training are further documented for [visual models](model_analysis_folders/visual_networks/pytorch_checkpoints/README) and [auditory models](model_analysis_folders/audio_networks/pytorch_checkpoints/README).

## Hardware Dependencies
Metamer generation and model training work best on machines with GPUs. Running the full 24000 iterations of metamer generation used in Feather et al. for one image or sound can take between 5 minutes and 2 hours depending on the model architecture and the generation layer. Statistical tests, fMRI regression, and exploring the human behavioral data do not require a GPU. 

Feather et al. 2022 replications
================================
In addition to the code released for metamer generation and null distribution comparison, we include code for replication of
results in Feather et al. 2022. Figure generation for data presented in main-text figures are included in notebooks documented below. 
In addition, within each network analysis directory (ie [model_analysis_folders/visual_models/cochresnet50](model_analysis_folders/visual_models/cochresnet50))
we include the scripts used to submit the jobs to our computing cluster (all scripts of the form `submit_*.sh`). The python commands within these 
submission scripts contain the necessary information to replicate metamer generation, null distribution comparison, and fMRI regression analysis. 

## Human recognition of metamers from Feather et al. 2022 models
For convenience, a notebook is included at [notebooks/All_Experiments_Metamer_Vs_Network_Predictions.ipynb](notebooks/All_Experiments_Metamer_Vs_Network_Predictions.ipynb) which loads in and plots
the human performance and generating network performance for audio and visual behavioral experiments in Feather et al. 2022.

Notebooks are also included that load example metamers for each experiment using jupyter widgets. The audio examples are at
[notebooks/AudioExperimentsListenToMetamers.ipynb](notebooks/AudioExperimentsListenToMetamers.ipynb) and visual examples are at [notebooks/VisualExperimentsViewMetamers.ipynb](notebooks/VisualExperimentsViewMetamers.ipynb).
Note that these, especially audio examples, may take time to load (samples are loaded from the mcdermott lab website).

The data for proportion correct for each participant for each experiment are included in the repo in VisionBehavioralExperiments and AudioBehavioralExperiments
directories. Also included is the proportion correct for the generating model on the set of stimuli matched to each human participant. For example, the
file `VisionBehavioralExperiments/EXP1_ANALYSIS/VisionExperiment1_network_vs_humans_datamatrix_alexnet_public.mat` contains the porportion correct for
each participant for metamers generated from each layer of the AlexNet architecture (corresponding to the 5th plot in Feather et al. 2022 Figure 1c).
It also contains the alexnet proportion correct for the same sets of metamers. Note that experiments are numbered based on the scheme in
[model_analysis_folders/all_model_info.py](model_analysis_folders/all_model_info.py).

## Model-model comparisons from Feather et al. 2022 models
A summary of other network's predictions of metamers generated from one model is included within each model analysis folder within `network_network_evaluations`. Notebooks are included at [notebooks/AudioNetworkNetworkPredictions.ipynb](notebooks/AudioNetworkNetworkPredictions.ipynb) and at [notebooks/VisualNetworkNetworkPredictions.ipynb](notebooks/VisualNetworkNetworkPredictions.ipynb) which load in the network-network predictions and make the plots included in Figure 9 of Feather et al. 2022.

## Auditory fMRI voxel regression analysis from Feather et al. 2022
The computed variance explained for each voxel for each split of data is saved within the `regression_results/natsound_activations/` directory for each model included in Figure 8 of Feather et al. 2022. The notebook included at [notebooks/fMRIPredictionPlotsAndStatistics.ipynb](notebooks/fMRIPredictionPlotsAndStatistics.ipynb) walks through the analysis and statistics based on these regression values. 
To recompute variance explained for our models (or your own model) run the following:
1) Compute the activations for the set of natural sounds with: ```python measure_layer_activations_165_natural_sounds_pytorch.py 'build_network.py'```
2) Build a mapping between the model activations and the fMRI response, and test on held out sounds ```python run_regressions_all_voxels_om_natsounddata.py <LAYER_IDX> 'natsound_activations.h5' -Z```

Metamer generation from the command line
========================================
Using model directory and `build_network.py` file that is already assembed, you can generate metamers with the following:

Audio Networks:
```
python make_metamers_wsj400_behavior_only_save_metamer_layers <IDX> -D 'model_analysis_folders/audio_networks/<model_name>'
```

Visual Networks:
```
python make_metamers_imagenet_16_category_val_400_only_save_metamer_layers <IDX> -D 'model_analysis_folders/visual_networks/<model_name>'
```

To generate metamers with fewer iterations (recommended for debugging, quick model evalulation) reduce the number of iterations (number of
updates at each learning rate) and the number of iteration repetitions (the number of learning rate drops). Total number of gradient steps
used for metamer generation is ITERATIONS * NUMREPITERATIONS

Example:
```
python make_metamers_wsj400_behavior_only_save_metamer_layers <IDX> -I 100 -N 4 -D 'model_analysis_folders/audio_networks/<model_name>'
python make_metamers_imagenet_16_category_val_400_only_save_metamer_layers <IDX> -I 100 -N 4 -D 'model_analysis_folders/visual_networks/<model_name>'
```
The default datasets for generating metamers (used in Feather et al. 2022) have IDX values from 0-399. See the metamer generation scripts
for additional options (including initialization changes, using custom sets of images or audio etc).

Metamers will be output into the `metamers` folder in the model directory that is specified.

Null distribution generation from the command line
==================================================
For checking metamer optimization we recommend comparison to a null distribution generated from random pairs of image and audio.
Example script uses the training dataset specified within the `build_network.py` file assembled for the model.

Null distribution will be output into the `null_dist` folder in the model directory that is specified.

Example (100 null samples using training dataset specified in `build_network.py`):
```
python make_null_distributions -N 100 -I 0 -R 0 --shuffle -D 'model_analysis_folders/visual_networks/<model_name>'
python make_null_distributions -N 100 -I 0 -R 0 -D 'model_analysis_folders/audio_networks/<model_name>'
```

Setup configuration for your own model
======================================
1) Create a new directory to store all of the model analysis files. Example directories to copy and modify are at the following: 
* Visual Example: [model_analysis_folders/visual_networks/example_blank_directory](model_analysis_folders/visual_networks/example_blank_directory)
* Audio Example: [model_analysis_folders/audio_networks/example_blank_directory](model_analysis_folders/audio_networks/example_blank_directory)

2) If you are using a new architecture, place the architecture in robustness/imagenet_models or robustness/audio_models. If you are
using an included architecture, you can move onto step 3. Otherwise, the architecture should have the following: 
Compared to typical pytorch models, the forward pass must have additional kwargs: `with_latent=False`, `fake_relu=False`, `no_relu=False`. 
* The `with_latent` argument allows returning a dictionary of `all_outputs`, which contains the outputs from intermediate stages of the 
pytorch model. The keys for this dictionary will be used to specify the layers used for metamer generation. 
* The `fake_relu` argument allows for using a relu with a modified gradient for the metamer generation layer, which can improve optimization.
To use this argument, the architecture will need to have flags to build in the appropriate relus. 

The included architectures have these modifications. 

3) Modify the `build_network.py` file in the directory that you created to include the name of the architecture to generate metamers from, 
a list of layers for metamer generation (these are the keys in the `all_outputs` dictionary), and any additional dataset or model loading 
parameters that must be specified. You can also load a checkpoint by specifiying a filepath in this code, if it is not included in the 
architecture file. There are TODO flags in the `example_blank_directory/build_network.py` files for these steps. 

Note: If the model was trained with a different repository, you might need to remap some of the checkpoint keys. There are options for this in the 
model loader file (see [model_analysis_folders/visual_models/resnet50_l2_3_robust/build_network.py](model_analysis_folders/visual_models/resnet50_l2_3_robust/build_network.py) for an example)

4) Once the `build_network.py` file is set up, you can use any of the analysis_scripts included in the repo for metamer generation, null 
distibution measuring, or analysis! We typically place the output from the analysis in the model directory with the `build_network.py` file, so that 
everything for a given model stays together. 

# Citation
This repository was released with the following pre-print. If you use this repository in your research, please cite as:

[Feather, J., Leclerc, G., Mądry, A., & McDermott, J. H. (2022). Model metamers illuminate divergences between biological and artificial neural networks. bioRxiv.](https://www.biorxiv.org/content/10.1101/2022.05.19.492678v1)

```
@article{feather2022model,
  title={Model metamers illuminate divergences between biological and artificial neural networks},
  author={Feather, Jenelle and Leclerc, Guillaume and M{\k{a}}dry, Aleksander and McDermott, Josh H},
  journal={bioRxiv},
  year={2022},
  publisher={Cold Spring Harbor Laboratory}
}
```

# Authors
* **Jenelle Feather** (https://github.com/jfeather)

# Acknowledgments
* McDermott Lab: [https://github.com/mcdermottLab](https://github.com/mcdermottLab)
* MadryLab/Robustness Repo: [https://github.com/MadryLab/robustness](https://github.com/MadryLab/robustness)

# License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
