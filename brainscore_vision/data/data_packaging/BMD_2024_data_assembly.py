from pathlib import Path

import pandas as pd

from brainio.assemblies import BehavioralAssembly
from brainio.packaging import package_data_assembly

'''
Experiment information

- 4 experimental conditions
    - Texturized stimuli 1 --> 'texture_1'
    - Texturized stimuli 2 --> 'texture_2'
    - Dotted outlines 1 --> 'dotted_1'
    - Dotted outlines 2 --> 'dotted_2'

Texture stimuli are silhouettes filled with a repeating character that produces the texture.
texture_1 --> character is "3"
texture_2 --> character is "%"

Dotted outlines can have dots at various distances and the two conditions correspond to a smaller
and larger distance.

There were 10 images per condition per eahch of the 10 categories
- 'airplane'
- 'bear'
- 'bicycle'
- 'bird'
- 'bottle'
- 'car'
- 'cat'
- 'chair'
- 'elephant'
- 'knife'

Between 51 and 54 partticipants completed each of the conditions (a total of 211 participants).
'''

CONDITIONS = ['texture_1', 'texture_2', 'dotted_1', 'dotted_2', 'all']

def collect_BMD2024_behavioral_assembly(data_path,condition):
    data = pd.read_csv(data_path)
    if condition == 'all':
        filtered_data = data
    else:
        filtered_data = data[data['condition'] == condition]
    
    assembly = BehavioralAssembly(filtered_data['subject_answer'],
                                          coords={
                                              'subject': ('presentation', filtered_data['subject']),
                                              'subject_answer': ('presentation', filtered_data['subject_answer']),
                                              'stimulus_id': ('presentation', filtered_data['stimulus_id']),
                                              'truth': ('presentation', filtered_data['truth']),
                                              'condition': ('presentation', filtered_data['condition'])
                                          },
                                          dims=['presentation']
                                          )
    
    assembly.name = f'BMD_2024_{condition}'
    return assembly
    

if __name__ == '__main__':
    data_path = Path('Data/BDM_2024_behavioral_data.csv')
    for condition in CONDITIONS:
        if condition == 'texture_1':
            num_dims = 5100
            num_subjects = 51
            num_stimuli = 100
        elif condition == 'texture_2':
            num_dims = 5200
            num_subjects = 52
            num_stimuli = 100
        elif condition == ('dotted_1' or 'dotted_2'):
            num_dims = 5400
            num_subjects = 54
        elif condition == 'all':
            num_dims = 21100
            num_subjects = 211
            num_stimuli = 400
        num_categories = 10
        
        assembly = collect_BMD2024_behavioral_assembly(data_path, condition)

        # make sure assembly dims are correct length
        assert len(assembly['presentation']) == num_dims    
        assert len(set(assembly['subject'].values)) == num_subjects
        assert len(set(assembly['stimulus_id'].values)) == num_stimuli
        assert len(set(assembly['truth'].values)) == num_categories

        prints = package_data_assembly(catalog_identifier=None,
                              proto_data_assembly=assembly,
                              assembly_identifier=assembly.name,
                              stimulus_set_identifier=assembly.name,
                              assembly_class_name="BehavioralAssembly",
                              bucket_name="brainio-brainscore")

        print(prints)