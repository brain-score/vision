This directory contains the analysis folders for each model.

The metamers were generated within each of these directories, and directory structure remains, however the png and .pckl files are omited from the github repository due to size.

The general structure follows: 
```
metamers_paper_model_analysis_folders
│   README
│   all_model_info.py (contains analysis parameters for each model, and experiment list)    
└───audio_networks
│   └───pytorch_checkpoints (empty in repository, download model checkpoints and place them here). 
│       │   README (documents training parameters for the models)
│       │   <model_1_checkpoint.pt>
│       │   <model_2_checkpoint.pt>
│       │   ...
│   └───<model_1_directory>
│       │   build_network.py (Unique for each model. Loads in the model checkpoint and specifies architecture and other parameters)
│       │   submit_*.sh (contains the calls to the scripts in the directory. Here for documentation and reproducibility). 
│       └───metamers (folder where the metamers are output. Not uploaded to github repo due to to size.) 
│       └───null_dist
│           │   distance_hist_*.pdf (summary histograms of the distances measured at each layer for each stage of metamer generation). 
│   └───<model_2_directory>
│       │   ...
│   └───... 
└───visual_networks
│   └───pytorch_checkpoints (empty in repository, download model checkpoints and place them here).
│       │   README (documents training parameters for the models)
│       │   <model_1_checkpoint.pt>
│       │   <model_2_checkpoint.pt>
│       │   ...
│   └───<model_1_directory>
│       │   build_network.py (Unique for each model. Loads in the model checkpoint and specifies architecture and other parameters)
│       │   submit_*.sh (contains the calls to the scripts in the directory. Here for documentation and reproducibility).
│       └───metamers (folder where the metamers are output. Not uploaded to github repo due to to size.)
│       └───null_dist
│           │   distance_hist_*.pdf (summary histograms of the distances measured at each layer for each stage of metamer generation).
│   └───<model_2_directory>
│       │   ...
│   └───...
```
