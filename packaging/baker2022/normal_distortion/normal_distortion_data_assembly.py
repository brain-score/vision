import numpy as np

from brainio.assemblies import BehavioralAssembly
from brainio.packaging import package_data_assembly
import pandas as pd

'''
Experiment Information:
https://www.sciencedirect.com/science/article/pii/S2589004222011853#sec9
 - (Data/subjects are from experiment 1 in paper above)

 - 32 subjects 
 - However there are 33 subjects, as some subject's results were invalid.
 - 40 images for each condition (whole, fragmented, frankenstein) = 120 images/subject shown
 - 120 * 32 = 3840 total images shown 
 - However there are only 3120 total images, 20 trials were thrown out.
 - 9-way AFC, from set: {bear, bunny, cat, elephant, frog, lizard, tiger, turtle, wolf}
 
'''

# initial csv to dataframe processing:
all_subjects = pd.read_csv('../human_data_normal/human_data.csv')

# parse df for the image lookup id. This relates the data assembly with the stimulus set.
condition = all_subjects['Frankensteinonfig']
abbreviated = []

for condition in condition:
    if condition == "whole":
        abbreviated.append("w")
    elif condition == "fragmented":
        abbreviated.append("o")
    else:
        abbreviated.append("f")
all_subjects["condition"] = abbreviated
all_subjects["condition_image"] = all_subjects['condition'] + all_subjects['Animal']

# ignore subject 33, per Nick Baker himself:
all_subjects = all_subjects.drop(all_subjects[all_subjects.Subj == 33].index)

# construct the assembly
assembly = BehavioralAssembly(all_subjects['Correct?'],
                              coords={
                                  'stimulus_id': ('presentation', all_subjects['condition_image']),
                                  'truth': ('presentation', all_subjects['Animal']),
                                  'condition': ('presentation', all_subjects['condition']),
                                  'condition_full': ('presentation', all_subjects['Frankensteinonfig']),
                                  'condition_image': ('presentation', all_subjects['condition_image']),
                                  'correct': ('presentation', all_subjects['Correct?']),
                                  'subject': ('presentation', all_subjects['Subj']),
                              },
                              dims=['presentation']
                              )

# give the assembly an identifier name
assembly.name = 'kellmen.Baker2022_local_configural'

# make sure assembly dims are correct length
assert len(assembly['presentation']) == 3706

# make sure assembly coords are correct length
assert len(assembly['image_id']) == 3706
assert len(assembly['truth']) == 3706
assert len(assembly['condition']) == 3706
assert len(assembly['condition_full']) == 3706
assert len(assembly['correct']) == 3706
assert len(assembly['subject']) == 3706

# make sure there are 9 unique image categories
assert len(np.unique(assembly['truth'].values)) == 9

# make sure there are 32 unique subjects
assert len(np.unique(assembly['subject'].values)) == 32

# make sure there are 3 unique conditions
assert len(np.unique(assembly['condition'].values)) == 3
assert len(np.unique(assembly['condition_full'].values)) == 3

# make sure there are 27  (9 categories * 3 conditions) unique image types:
assert len(np.unique(assembly['condition_image'].values)) == 27


# upload to S3
# package_data_assembly('brainio_brainscore', assembly, assembly_identifier=assembly.name,
#                       stimulus_set_identifier='kellmen.Baker2022_local_configural',
#                       assembly_class="BehavioralAssembly", bucket_name="brainio-brainscore")
