from data_packaging import create_stimulus_set_and_upload, DATASETS

# package and upload all 14 training stimuli sets
all_stimulus_sets = []
paths = {}
for dataset in DATASETS:

    """
    For the fitting stimuli, each stimulus set will have 1920 images in them:
    320 images x 3 distractors (low, medium, high) x 2 types (target on distractor, distractor on target)
    """

    stimulus_set = create_stimulus_set_and_upload("Ferguson2024", f"{dataset}_training_stimuli", upload_to_s3=True)
    all_stimulus_sets.append(stimulus_set)


# label each dataset with the name of the dataset
for df, name in zip(all_stimulus_sets, DATASETS):
    df['experiment'] = name

