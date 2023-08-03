import numpy as np
import pandas as pd
from brainio.assemblies import BehavioralAssembly
import os
from brainio.packaging import package_data_assembly

"""
Takes in a .csv from Mturk and converts into an assembly. Tests the assembly and uploads to BrainIO.

    :param stimulus_set_name: name of matching stimulus set
    :param assembly_name: desired name of assembly
    :param csv_file: path to csv file
    :param num_subjects: number of subjects in experiment (for assembly testing purposes)
    :param num_reps:  number of image repetitions in experiment (for assembly testing purposes)
    :param num_images: number of *unique* images in experiment (for assembly testing purposes)
    :return: BehavioralAssembly
    
"""


def csv_to_assembly(stimulus_set_name: str, assembly_name: str, csv_file: str) -> BehavioralAssembly:

    all_subjects = pd.read_csv(csv_file)

    # parse stimuli from stimuli paths:
    all_subjects['stimulus_id'] = all_subjects['stimulus'].apply(lambda x: os.path.basename(x))
    all_subjects['target_id'] = all_subjects['target_dir'].apply(lambda x: os.path.basename(x))
    all_subjects['distractor_id'] = all_subjects['distractor_dir'].apply(lambda x: os.path.basename(x))

    # construct the assembly
    assembly = BehavioralAssembly(all_subjects['correct'],
                                  coords={
                                      'stimulus_id': ('presentation', all_subjects['stimulus_id']),
                                      'stimulus_full_path': ('presentation', all_subjects['stimulus']),
                                      'response_time': ('presentation', all_subjects['rt']),
                                      'response_key': ('presentation', all_subjects['response']),
                                      'task': ('presentation', all_subjects['task']),
                                      'correct_key': ('presentation', all_subjects['correct_response']),
                                      'trial_type': ('presentation', all_subjects['trial_type']),
                                      'trial_index': ('presentation', all_subjects['trial_index']),
                                      'time_elapsed': ('presentation', all_subjects['time_elapsed']),
                                      'internal_node_id': ('presentation', all_subjects['internal_node_id']),
                                      'correct': ('presentation', all_subjects['correct']),
                                      'subject_id': ('presentation', all_subjects['participant_id']),
                                      'target_present': ('presentation', all_subjects['have_target']),
                                      'number_distractors': ('presentation', all_subjects['distractor_nums']),
                                      'type_rep': ('presentation', all_subjects['type_rep']),
                                      'target_image': ('presentation', all_subjects['target_dir']),
                                      'distractor_image': ('presentation', all_subjects['distractor_dir']),
                                      'target_id': ('presentation', all_subjects['target_id']),
                                      'distractor_id': ('presentation', all_subjects['distractor_id']),
                                  },
                                  dims=['presentation']
                                  )

    # give the assembly an identifier name
    assembly.name = assembly_name

    # make sure there are only two key presses:
    assert len(np.unique(assembly['response_key'].values)) == 2
    assert len(np.unique(assembly['correct_key'].values)) == 2

    # make sure there are only two types of target present:
    assert len(np.unique(assembly['target_present'].values)) == 2

    # check number of distractors
    assert set(np.unique(assembly['number_distractors'].values)) == {0, 5, 11}

    # upload to S3
    # package_data_assembly('brainio_brainscore', assembly, assembly_identifier=assembly.name,
    #                       stimulus_set_identifier=stimulus_set_name,
    #                       assembly_class_name="BehavioralAssembly", bucket_name="brainio-brainscore")

    return assembly
