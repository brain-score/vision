from data_packaging import create_stimulus_set_and_upload, DATASETS
from brainio.packaging import package_stimulus_set
import pandas as pd
from pathlib import Path
from brainio.stimuli import StimulusSet
from brainscore_vision import load_dataset, load_stimulus_set

# package and upload all 14 training stimuli sets
all_stimulus_sets = []
paths = {}
for dataset in DATASETS:

    """
    For the fitting stimuli, each stimulus set will have 1920 images in them:
    320 images x 3 distractors (low, medium, high) x 2 types (target on distractor, distractor on target)
    """

    stimulus_set = create_stimulus_set_and_upload("Ferguson2024", f"{dataset}_training_stimuli", upload_to_s3=False)
    all_stimulus_sets.append(stimulus_set)
    paths[dataset] = list(stimulus_set.stimulus_paths)


# label each dataset with the name of the dataset
for df, name in zip(all_stimulus_sets, DATASETS):
    df['experiment'] = name


# merge all 14 into one large stimulus_set
merged_dataframe = pd.concat(all_stimulus_sets, axis=0, ignore_index=True)
merged_dataframe = StimulusSet(merged_dataframe)

merged_dataframe.stimulus_paths = []
merged_dataframe.name = "Ferguson2024_merged_training_stimuli"

init_data_merged = package_stimulus_set(catalog_name=None, proto_stimulus_set=merged_dataframe,
                                        stimulus_set_identifier=merged_dataframe.name,
                                        bucket_name="brainio-brainscore")

print(f"Merged init data: {init_data_merged}")
