import numpy as np

from brainio.assemblies import BehavioralAssembly
from brainio.packaging import package_data_assembly
import pandas as pd

'''
** NOTE: This benchmark (edge) has a very different stimulus sets/assemblies then others in Geirhos set!! **

Experiment Information:
https://arxiv.org/pdf/1706.06969.pdf


 - 10 subjects 
 - 160 images each
 - 1600 total images shown 
 - match to sample task, 16AFC
 - 16 image categories
 - for the this benchmark (edge) subjects saw the EXACT image indicated with the variable/column name
   image_lookup_id, and not a variation of it (no distortions, editing, etc). Condition is o for all 1600 trials. 

'''

# initial csv to dataframe processing:
subject_1 = pd.read_csv('data_assemblies/edge_subject-01_session_1.csv')
subject_2 = pd.read_csv('data_assemblies/edge_subject-02_session_1.csv')
subject_3 = pd.read_csv('data_assemblies/edge_subject-03_session_1.csv')
subject_4 = pd.read_csv('data_assemblies/edge_subject-04_session_1.csv')
subject_5 = pd.read_csv('data_assemblies/edge_subject-05_session_1.csv')
subject_6 = pd.read_csv('data_assemblies/edge_subject-06_session_1.csv')
subject_7 = pd.read_csv('data_assemblies/edge_subject-07_session_1.csv')
subject_8 = pd.read_csv('data_assemblies/edge_subject-08_session_1.csv')
subject_9 = pd.read_csv('data_assemblies/edge_subject-09_session_1.csv')
subject_10 = pd.read_csv('data_assemblies/edge_subject-10_session_1.csv')

all_subjects = pd.concat([subject_1, subject_2, subject_3, subject_4, subject_5,
                          subject_6, subject_7, subject_8, subject_9, subject_10])

# parse df for the image lookup id. This relates the data assembly with the stimulus set.
split_cols = all_subjects['imagename'].str.split("_", expand=True)
all_subjects["image_id_long"] = split_cols[6].str.replace(".png", "")


# construct the assembly
assembly = BehavioralAssembly(all_subjects['object_response'],
                              coords={
                                  'image_id': ('presentation', all_subjects['image_id_long']),
                                  'image_id_long': ('presentation', all_subjects['imagename']),
                                  'truth': ('presentation', all_subjects['category']),
                                  'choice': ('presentation', all_subjects['object_response']),
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
assembly.name = 'brendel.Geirhos2021_edge'

# make sure assembly dim is correct length
assert len(assembly['presentation']) == 1600

# make sure assembly coords are correct length
assert len(assembly['image_id']) == 1600
assert len(assembly['image_id_long']) == 1600
assert len(assembly['truth']) == 1600
assert len(assembly['category']) == 1600
assert len(assembly['condition']) == 1600
assert len(assembly['response_time']) == 1600
assert len(assembly['trial']) == 1600
assert len(assembly['subject']) == 1600
assert len(assembly['session']) == 1600


# # make sure there are 160 unique images (shown 1 time for each  of 10 subjects, total of 16 * 10 = 160 images shown)
assert len(np.unique(assembly['image_id'].values)) == 160

# make sure there are 10 unique subjects:
assert len(np.unique(assembly['subject'].values)) == 10

# make sure there is only a single condition (0)
assert len(np.unique(assembly['condition'].values)) == 1

# make sure there are 16 unique object categories (ground truths):
assert len(np.unique(assembly['truth'].values)) == 16
assert len(np.unique(assembly['category'].values)) == 16


# upload to S3
package_data_assembly('brainio_brainscore', assembly, assembly_identifier=assembly.name,
                      stimulus_set_identifier="Geirhos2021_edge",
                      assembly_class="BehavioralAssembly", bucket_name="brainio-brainscore")
