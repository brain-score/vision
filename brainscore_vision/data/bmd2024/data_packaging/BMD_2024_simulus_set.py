import csv

from brainio.stimuli import StimulusSet
from brainio.packaging import package_stimulus_set

CONDITIONS = ['texture_1', 'texture_2', 'dotted_1', 'dotted_2']


def collect_BMD_2024_stimulus_set(condition, stimuli_directory, meta_filepath):
    stimuli = []
    stimulus_paths = {}
    
    with open(meta_filepath, 'r') as metadata:
        reader = csv.DictReader(metadata)
        for row in reader:
            stimulus_meta = {
                'stimulus_id' : row['stimulus_id'],
                'condition' : row['condition'],
                'truth' : row['truth']
                }
            if row['condition'] == condition:
                stimuli.append(stimulus_meta)
                stimulus_paths[row['stimulus_id']] = f'{stimuli_directory}/{row["stimulus_id"]}.png'
        
        stimuli_assembly = StimulusSet(stimuli)
        stimuli_assembly.stimulus_paths = stimulus_paths
        stimuli_assembly.name = f'BMD_2024_{condition}'
        stimuli_assembly.identifier = f'BMD_2024_{condition}'
    
    return stimuli_assembly


if __name__ == '__main__':
    stimuli_directory = 'Stimuli_set'
    meta_filepath = 'stim_meta.csv'
    
    for condition in CONDITIONS:
        assembly = collect_BMD_2024_stimulus_set(condition, stimuli_directory, meta_filepath)
        
        assert len(assembly) == 100
        
        prints = package_stimulus_set(catalog_name=None,
                             proto_stimulus_set=assembly,
                             stimulus_set_identifier=assembly.name,
                             bucket_name="brainio-brainscore")
        print(prints)


