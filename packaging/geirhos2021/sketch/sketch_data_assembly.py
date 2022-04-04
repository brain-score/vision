import numpy as np

from brainio.assemblies import BehavioralAssembly
from brainio.packaging import package_data_assembly
import pandas as pd

'''
Experiment Information:
https://arxiv.org/pdf/1706.06969.pdf

 - 7 subjects 
 - 800 trials each
 - 5600 total trials 
 - match to sample task 
 - 16 image categories
 - for the this benchmark (sketch) subjects saw the EXACT image indicated with the variable/column name
   image_lookup_id, and not a variation of it (no distortions, editing, etc).

'''

# initial csv to dataframe processing:
subject_1 = pd.read_csv('data_assemblies/sketch_subject-01_session_1.csv')
subject_2 = pd.read_csv('data_assemblies/sketch_subject-02_session_1.csv')
subject_3 = pd.read_csv('data_assemblies/sketch_subject-03_session_1.csv')
subject_4 = pd.read_csv('data_assemblies/sketch_subject-04_session_1.csv')
subject_5 = pd.read_csv('data_assemblies/sketch_subject-05_session_1.csv')
subject_6 = pd.read_csv('data_assemblies/sketch_subject-06_session_1.csv')
subject_7 = pd.read_csv('data_assemblies/sketch_subject-07_session_1.csv')
all_subjects = pd.concat([subject_1, subject_2, subject_3, subject_4, subject_5, subject_6, subject_7])

# parse for the image lookup id. This relates the data assembly with the stimulus set.
all_subjects['image_lookup_id'] = all_subjects['imagename'].str.split("_").str[-1]

# construct the assembly
assembly = BehavioralAssembly(all_subjects['object_response'],
                              coords={
                                  'image_id': ('presentation', all_subjects['imagename']),
                                  'image_lookup_id': ('presentation', all_subjects['image_lookup_id']),
                                  'ground_truth': ('presentation', all_subjects['category']),
                                  'category': ('presentation', all_subjects['category']),
                                  'condition': ('presentation', all_subjects['condition']),
                                  'response_time': ('presentation', all_subjects['rt']),
                                  'trial': ('presentation', all_subjects['trial']),
                                  'subject': ('presentation', all_subjects['subj']),
                                  'session': ('presentation', all_subjects['session']),
                              },
                              dims=['presentation']
                              )

# give the assembly an identifier name
assembly.name = 'brendel.Geirhos2021_sketch'

# make sure assembly dims are correct length
assert len(assembly['presentation']) == 5600

# make sure assembly coords are correct length
assert len(assembly['image_id']) == 5600
assert len(assembly['image_lookup_id']) == 5600
assert len(assembly['ground_truth']) == 5600
assert len(assembly['category']) == 5600
assert len(assembly['condition']) == 5600
assert len(assembly['response_time']) == 5600
assert len(assembly['trial']) == 5600
assert len(assembly['subject']) == 5600
assert len(assembly['session']) == 5600


# make sure there are 800 unique images (shown 1 time for each  of 7 subjects, total of 7 * 800 = 5600 images shown)
assert len(np.unique(assembly['image_lookup_id'].values)) == 800

# make sure there are 7 unique subjects
assert len(np.unique(assembly['subject'].values)) == 7

# make sure there are 16 unique object categories (ground truths)
assert len(np.unique(assembly['ground_truth'].values)) == 16
assert len(np.unique(assembly['category'].values)) == 16

# make sure there is only one condition (i.e. no image variations added after presentation)
assert len(np.unique(assembly['condition'].values)) == 1

# upload to S3
# package_data_assembly(assembly, assembly_identifier=assembly.name, ,
#                       assembly_class='BehavioralAssembly'
#                       stimulus_set_identifier=stimuli.name)  # link to the StimulusSet
