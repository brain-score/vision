from pathlib import Path

import numpy as np
import pandas as pd

from brainio.assemblies import BehavioralAssembly
from brainio.packaging import package_data_assembly


'''
Experiment Information:

 - 25 or 50 subjects depending on condition 
    - rgb & contours have 50, others have 25
 - 48 images each, except in the composites, where all conditions are shown (all phosphene/segment + contours = rgb)
    - 48 * 11 = 528 images each in composites
 - 1200 or 2400 total images shown 
    - rgb & contours have 2400, composites have 13200, others have 1200
 - 12-AFC categorization task
 - 12 image categories; one of:
        'Banana', 
        'Beanie', 
        'Binoculars', 
        'Boot', 
        'Bowl', 
        'Cup', 
        'Glasses',
        'Lamp', 
        'Pan', 
        'Sewing machine', 
        'Shovel', 
        'Truck'
'''

SUBJECT_GROUPS = ['rgb', 'contours', 'phosphenes-12', 'phosphenes-16', 'phosphenes-21', 'phosphenes-27',
                  'phosphenes-35', 'phosphenes-46', 'phosphenes-59', 'phosphenes-77', 'phosphenes-100', 'segments-12',
                  'segments-16', 'segments-21', 'segments-27', 'segments-35', 'segments-46', 'segments-59',
                  'segments-77', 'segments-100', 'phosphenes-all', 'segments-all']
PERCENTAGE_ELEMENTS = {'rgb': 'RGB', 'contours': 'contours', 'phosphenes-12': 12, 'phosphenes-16': 16,
                       'phosphenes-21': 21, 'phosphenes-27': 27, 'phosphenes-35': 35, 'phosphenes-46': 46,
                       'phosphenes-59': 59, 'phosphenes-77': 77, 'phosphenes-100': 100, 'segments-12': 12,
                       'segments-16': 16, 'segments-21': 21, 'segments-27': 27, 'segments-35': 35, 'segments-46': 46,
                       'segments-59': 59, 'segments-77': 77, 'segments-100': 100, 'phosphenes-all': 'all',
                       'segments-all': 'all'}


def collect_scialom_behavioral_assembly(data_path, subject_group, percentage_elements, which_composite):
    # load and filter the data to only take this benchmark
    data = pd.read_csv(data_path)
    data['percentage_elements'] = data['percentage_elements'].astype(str)
    percentage_elements = str(percentage_elements)
    subject_group = subject_group.split('-')[0]
    if which_composite is not None:
        filtered_data = data[(data['subject_group'] == which_composite) |
                             (data['subject_group'] == 'RGB') |
                             (data['subject_group'] == 'contours')]
    elif subject_group in ['phosphenes', 'segments']:
        filtered_data = data[(data['subject_group'] == subject_group) &
                             (data['percentage_elements'] == percentage_elements)]
    else:
        filtered_data = data[(data['percentage_elements'] == percentage_elements)]

    # construct the assembly
    if which_composite is not None:
        assembly = BehavioralAssembly(filtered_data['subject_answer'],
                                      coords={
                                          'subject': ('presentation', filtered_data['subject_id']),
                                          'subject_group': ('presentation', filtered_data['subject_group']),
                                          'visual_degrees': ('presentation', filtered_data['visual_degrees']),
                                          'image_duration': ('presentation', filtered_data['image_duration']),
                                          'is_correct': ('presentation', filtered_data['is_correct']),
                                          'subject_answer': ('presentation', filtered_data['subject_answer']),
                                          'condition': ('presentation', filtered_data['subject_group']),
                                          'percentage_elements': ('presentation', filtered_data['percentage_elements']),
                                          'stimulus_id': ('presentation', filtered_data['stimulus_id'].astype(int)),
                                          'truth': ('presentation', filtered_data['correct_answer'])
                                      },
                                      dims=['presentation']
                                      )
    else:
        assembly = BehavioralAssembly(filtered_data['subject_answer'],
                                      coords={
                                          'subject': ('presentation', filtered_data['subject_id']),
                                          'subject_group': ('presentation', filtered_data['subject_group']),
                                          'visual_degrees': ('presentation', filtered_data['visual_degrees']),
                                          'image_duration': ('presentation', filtered_data['image_duration']),
                                          'is_correct': ('presentation', filtered_data['is_correct']),
                                          'subject_answer': ('presentation', filtered_data['subject_answer']),
                                          'condition': ('presentation', filtered_data['percentage_elements']),
                                          'percentage_elements': ('presentation', filtered_data['percentage_elements']),
                                          'stimulus_id': ('presentation', filtered_data['stimulus_id'].astype(int)),
                                          'truth': ('presentation', filtered_data['correct_answer'])
                                      },
                                      dims=['presentation']
                                      )

    # give the assembly an identifier name
    if which_composite is not None:
        assembly.name = f'Scialom2024_{which_composite}-all'
    elif subject_group in ['phosphenes', 'segments']:
        assembly.name = f'Scialom2024_{subject_group}-{percentage_elements}'
    else:
        assembly.name = f'Scialom2024_{subject_group}'
    return assembly


if __name__ == '__main__':
    data_path = Path(r'../Data_Results_experiment.csv')
    for subject_group in SUBJECT_GROUPS:
        percentage_elements = PERCENTAGE_ELEMENTS[subject_group]
        if subject_group == 'rgb' or subject_group == 'contours':
            num_dims = 2400
            num_subjects = 50
            which_composite = None
        elif subject_group == 'phosphenes-all' or subject_group == 'segments-all':
            num_dims = 13200
            num_subjects = 25
            which_composite = subject_group[:-4]
        else:
            num_dims = 1200
            num_subjects = 25
            which_composite = None

        assembly = collect_scialom_behavioral_assembly(data_path, subject_group, percentage_elements,
                                                       which_composite=which_composite)

        # make sure assembly dims are correct length
        assert len(assembly['presentation']) == num_dims

        # make sure assembly coords are correct length
        assert len(assembly['subject']) == num_dims
        assert len(assembly['subject_group']) == num_dims
        assert len(assembly['visual_degrees']) == num_dims
        assert len(assembly['image_duration']) == num_dims
        assert len(assembly['is_correct']) == num_dims
        assert len(assembly['subject_answer']) == num_dims
        assert len(assembly['truth']) == num_dims
        assert len(assembly['condition']) == num_dims
        assert len(assembly['stimulus_id']) == num_dims

        if subject_group == 'phosphenes-all' or subject_group == 'segments-all':
            # all stimuli within-group shown to all subjects (11 conditions * 48 stimuli = 528 stimuli)
            assert len(np.unique(assembly['stimulus_id'].values)) == 528
        else:
            # make sure there are 48 unique images (shown 1 time for each of 25 or 50 subjects, total of
            #  25 * 48 = 1200 images shown or 50 * 48 = 2400 images shown)
            assert len(np.unique(assembly['stimulus_id'].values)) == 48

        # make sure there are the correct number of unique subjects:
        assert len(np.unique(assembly['subject'].values)) == num_subjects

        # make sure there are 12 unique object categories (ground truths):
        assert len(np.unique(assembly['truth'].values)) == 12
        assert len(np.unique(assembly['subject_answer'].values)) == 12

        # upload to S3
        prints = package_data_assembly(catalog_identifier=None,
                              proto_data_assembly=assembly,
                              assembly_identifier=assembly.name,
                              stimulus_set_identifier=assembly.name,
                              assembly_class_name="BehavioralAssembly",
                              bucket_name="brainscore-storage/brainio-brainscore")

        print(prints)