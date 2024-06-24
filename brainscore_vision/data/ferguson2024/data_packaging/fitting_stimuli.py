from data_packaging import create_stimulus_set_and_upload, DATASETS
from brainio.packaging import package_stimulus_set
import xarray as xr


all_stimulus_sets = []
for dataset in DATASETS:

    """
    Each stimulus set will have 960 images in them:
    
    160 images x 3 distractors (low, medium, high) x 2 types (target on distractor, distractor on target)
    
    
    """

    stimulus_set = create_stimulus_set_and_upload("Ferguson2024", f"{dataset}_fitting_stimuli", upload_to_s3=False)

# merge all 14 into one large stimulus_set
merged_dataset = xr.concat(all_stimulus_sets, dim='experiment')
merged_dataset['experiment'] = DATASETS
init_data_merged = package_stimulus_set(catalog_name=None, proto_stimulus_set=merged_dataset,
                                        stimulus_set_identifier="Ferguson2024_merged_training_data",
                                        bucket_name="brainio-brainscore")

print(f"Merged init data: {init_data_merged}")
