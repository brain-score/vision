from brainio.packaging import package_data_assembly
from pathlib import Path
from shutil import copy
from brainio.stimuli import StimulusSet
from brainio.packaging import package_stimulus_set
from brainio.assemblies import BehavioralAssembly
import pandas as pd

DATASETS = ['circle_line', 'color', 'convergence', 'eighth',
            'gray_easy', 'gray_hard', 'half', 'juncture',
            'lle', 'llh', 'quarter', 'round_f',
            'round_v', 'tilted_line']


# Packages the stimulus_sets for the Ferguson2024 experiment. There are 14 in all.
def create_stimulus_set_and_upload(name: str, experiment: str, upload_to_s3=True) -> StimulusSet:
    """

    Sample image from dataset:
    first_block_0.png

    1) first_block -> what block the stimuli belong two (which image is target, which is distractor)
    2) 0 -> a number, 0-23 indicating which variation the image is

    There are 24 images in the first block, and 24 in the second block, so the combined stimulus_set is length 48.
    The packaged stimuli were structured so that the root folder (tilted_line) had two subfolders, /first_block and /second_block.

    :param name: the name of the experiment, usually Ferguson2024
    :param experiment: the dataset, i.e. color
    :param upload_to_s3: True if you want to upload this to BrainIO on S3
    :return: the Stimulus Set
    """

    stimuli = []
    stimulus_paths = {}
    stimuli_directory = f'{experiment}'
    combine_block_images(stimuli_directory)
    for filepath in Path(f"{stimuli_directory}/final").glob('**/*.png'):
        stimulus_id = filepath.stem
        parts_list = stimulus_id.split("_")
        block = parts_list[0]
        image_number = parts_list[2]

        stimulus_paths[stimulus_id] = filepath
        stimuli.append({
            'stimulus_id': stimulus_id,
            'image_number': image_number,
            'block': block,
        })

    stimuli = StimulusSet(stimuli)
    stimuli.stimulus_paths = stimulus_paths
    stimuli.name = f'{name}_{experiment}'  # give the StimulusSet an identifier name

    # upload to S3
    if upload_to_s3:
        init_data = package_stimulus_set(catalog_name=None, proto_stimulus_set=stimuli,
                                         stimulus_set_identifier=stimuli.name, bucket_name="brainscore-storage/brainio-brainscore")
        print(f"{experiment} stimulus_set\n{init_data}")
    return stimuli


# Packages the assemblies for the Ferguson2024 experiment. There are 14 in all.
def create_assembly_and_upload(name: str, experiment: str, upload_to_s3=True) -> BehavioralAssembly:
    """
    Takes in a sanity-processed csv file, converts to an assembly, and uploads it to BrainIO

    :param name: the name of the experiment, usually Ferguson2024
    :param experiment: the dataset, i.e. color
    :param upload_to_s3: True if you want to upload this to BrainIO on S3
    :return: the assmebly
    """
    all_subjects = pd.read_csv(f'csvs/{experiment}_sanity_processed.csv')

    # only look at testing data (no warmup or sanity data):
    all_subjects = all_subjects[all_subjects["trial_type"] == "normal"]
    all_subjects = bool_to_int(all_subjects, ['correct', 'target_present'])  # cast bool to int for NetCDF

    # create an ID that is equal to the stimulus_set ID
    all_subjects['stimulus_id'] = all_subjects['stimulus'].apply(extract_and_concatenate)

    assembly = BehavioralAssembly(all_subjects['correct'],
                                  coords={
                                      'stimulus_id': ('presentation', all_subjects['stimulus_id']),
                                      'stimulus_id_long': ('presentation', all_subjects['stimulus']),
                                      'participant_id': ('presentation', all_subjects['participant_id']),
                                      'response_time_ms': ('presentation', all_subjects['response_time_ms']),
                                      'correct': ('presentation', all_subjects['correct']),
                                      'target_present': ('presentation', all_subjects['target_present']),
                                      'distractor_nums': ('presentation', all_subjects['distractor_nums']),
                                      'block': ('presentation', all_subjects['block']),
                                      'keypress_response': ('presentation', all_subjects['response']),
                                      'trial_type': ('presentation', all_subjects['trial_type']),
                                  },
                                  dims=['presentation']
                                  )

    assembly.name = f"{name}_{experiment}"

    # upload assembly to S3
    if upload_to_s3:
        init_data = package_data_assembly(None, assembly, assembly_identifier=assembly.name,
                                          stimulus_set_identifier=f"{name}_{experiment}",
                                          assembly_class_name="BehavioralAssembly",
                                          bucket_name="brainscore-storage/brainio-brainscore")
        print(f"{experiment} assembly\n{init_data}")
    return assembly


# helper function to take in a folder with the structure outlined in the above file docs, and move them
# all into one folder
def combine_block_images(stimuli_directory: str) -> None:
    """

    :param stimuli_directory: the path where your stimuli are located. This folder has two subfolders, /first_block and
           /second_block
    """
    final_directory_path = Path(stimuli_directory) / 'final'
    final_directory_path.mkdir(exist_ok=True)
    subfolders = ['first_block', 'second_block']
    for subfolder in subfolders:
        current_folder_path = Path(stimuli_directory) / subfolder
        if not current_folder_path.exists():
            continue
        for filepath in current_folder_path.glob('*.png'):
            stimulus_id = filepath.stem
            new_file_name = f"{subfolder}_{stimulus_id}.png"
            new_file_path = final_directory_path / new_file_name
            copy(filepath, new_file_path)


# helper function to get the stimulus_set stimulus_id from the assembly stimulus:
def extract_and_concatenate(url):
    parts = url.split('/')
    block_part = parts[-3]
    file_name = parts[-1].replace(".png", "")
    return f"{block_part}_{file_name}"


# Converts boolean values to integers in specified columns of a DataFrame.
def bool_to_int(df, columns):
    for column in columns:
        if column in df.columns:
            df[column] = df[column].map({'True': 1, 'False': 0, True: 1, False: 0}).fillna(df[column])
        else:
            print(f"Column '{column}' not found in DataFrame.")
    return df


# wrapper function to loop over all datasets
def package_all_stimulus_sets(name):
    for experiment in DATASETS:
        create_stimulus_set_and_upload(name, experiment)


# wrapper function to loop over all datasets:
def package_all_assemblies(name):
    for experiment in DATASETS:
        create_assembly_and_upload(name, experiment)


if __name__ == '__main__':
    package_all_stimulus_sets(name='Ferguson2024')
    package_all_assemblies(name='Ferguson2024')
