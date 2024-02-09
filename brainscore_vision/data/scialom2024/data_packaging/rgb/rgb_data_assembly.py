from pathlib import Path

import numpy as np
import pandas as pd

from brainio.assemblies import BehavioralAssembly
from brainio.packaging import package_data_assembly


'''
Experiment Information:

 - 50 subjects 
 - 48 images each
 - 2400 total images shown 
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


def collect_scialom_behavioral_assembly(data_path, percentage_elements):
    # load and filter the data to only take this benchmark
    data = pd.read_csv(data_path)
    filtered_data = data[data['percentage_elements'] == percentage_elements]

    # construct the assembly
    assembly = BehavioralAssembly(filtered_data['subject_answer'],
                                  coords={
                                      'subject': ('presentation', filtered_data['subject_id']),
                                      'subject_group': ('presentation', filtered_data['subject_group']),
                                      'visual_degrees': ('presentation', filtered_data['visual_degrees']),
                                      'image_duration': ('presentation', filtered_data['image_duration']),
                                      'is_correct': ('presentation', filtered_data['is_correct']),
                                      'subject_answer': ('presentation', filtered_data['subject_answer']),
                                      'condition': ('presentation', filtered_data['percentage_elements']),
                                      'stimulus_id': ('presentation', filtered_data['stimulus_id']),
                                      'truth': ('presentation', filtered_data['correct_answer'])
                                  },
                                  dims=['presentation']
                                  )

    # give the assembly an identifier name
    assembly.name = f'Scialom2024-{percentage_elements}'
    return assembly


if __name__ == '__main__':
    percentage_elements = 'RGB'
    data_path = Path(r'../../Data_Results_experiment.csv')
    assembly = collect_scialom_behavioral_assembly(data_path, percentage_elements)

    # make sure assembly dims are correct length
    assert len(assembly['presentation']) == 2400

    # make sure assembly coords are correct length
    assert len(assembly['subject']) == 2400
    assert len(assembly['subject_group']) == 2400
    assert len(assembly['visual_degrees']) == 2400
    assert len(assembly['image_duration']) == 2400
    assert len(assembly['is_correct']) == 2400
    assert len(assembly['subject_answer']) == 2400
    assert len(assembly['truth']) == 2400
    assert len(assembly['condition']) == 2400
    assert len(assembly['stimulus_id']) == 2400

    # make sure there are 48 unique images (shown 1 time for each of 50 subjects, total of 50 * 48 = 2400 images shown)
    assert len(np.unique(assembly['stimulus_id'].values)) == 48

    # make sure there are 50 unique subjects:
    assert len(np.unique(assembly['subject'].values)) == 50

    # make sure there are 12 unique object categories (ground truths):
    assert len(np.unique(assembly['truth'].values)) == 12
    assert len(np.unique(assembly['subject_answer'].values)) == 12

    # upload to S3
    # package_data_assembly('brainio_brainscore', assembly, assembly_identifier=assembly.name,
    #                       stimulus_set_identifier="assembly.name",
    #                       assembly_class="BehavioralAssembly", bucket_name="brainio-brainscore")