import numpy as np

from brainio.assemblies import BehavioralAssembly
from brainio.packaging import package_data_assembly
import pandas as pd

'''

** NOTE: This benchmark (cue-conflict) has a very different stimulus sets then ALL others,
**


Experiment Information:
https://arxiv.org/pdf/1706.06969.pdf


 - 10 subjects 
 - 1280images each
 - 12800 total images shown 
 - match to sample task, 16AFC
 - 16 image categories
 - for the this benchmark (cue-conflict) subjects saw the EXACT image indicated with the variable/column name
   image_id, and not a variation of it (no distortions, editing, etc). Condition is o for all 12800 trials. The 
   subject had to choose the original_image, not the conflict_image.

'''

# initial csv to dataframe processing:
subject_1 = pd.read_csv('data_assemblies/cue-conflict_subject-01_session_1.csv')
subject_2 = pd.read_csv('data_assemblies/cue-conflict_subject-02_session_1.csv')
subject_3 = pd.read_csv('data_assemblies/cue-conflict_subject-03_session_1.csv')
subject_4 = pd.read_csv('data_assemblies/cue-conflict_subject-04_session_1.csv')
subject_5 = pd.read_csv('data_assemblies/cue-conflict_subject-05_session_1.csv')
subject_6 = pd.read_csv('data_assemblies/cue-conflict_subject-06_session_1.csv')
subject_7 = pd.read_csv('data_assemblies/cue-conflict_subject-07_session_1.csv')
subject_8 = pd.read_csv('data_assemblies/cue-conflict_subject-08_session_1.csv')
subject_9 = pd.read_csv('data_assemblies/cue-conflict_subject-09_session_1.csv')
subject_10 = pd.read_csv('data_assemblies/cue-conflict_subject-10_session_1.csv')

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
assembly.name = 'brendel.Geirhos2021_cue-conflict'

# make sure assembly dim is correct length
assert len(assembly['presentation']) == 12800

# make sure assembly coords are correct length
assert len(assembly['image_id']) == 12800
assert len(assembly['image_id_long']) == 12800
assert len(assembly['truth']) == 12800
assert len(assembly['category']) == 12800
assert len(assembly['condition']) == 12800
assert len(assembly['response_time']) == 12800
assert len(assembly['trial']) == 12800
assert len(assembly['subject']) == 12800
assert len(assembly['session']) == 12800


# make sure there are 1280 unique images (shown 1 time for each of 10 subjects. Total of 1280*10 = 12800 images shown)
assert len(np.unique(assembly['image_id'].values)) == 1280

# make sure there are 10 unique subjects:
assert len(np.unique(assembly['subject'].values)) == 10

# make sure there is only a single condition (0)
assert len(np.unique(assembly['condition'].values)) == 1

# make sure there are 16 unique object categories (ground truths):
assert len(np.unique(assembly['truth'].values)) == 16
assert len(np.unique(assembly['category'].values)) == 16


# upload to S3
package_data_assembly('brainio_brainscore', assembly, assembly_identifier=assembly.name,
                      stimulus_set_identifier="Geirhos2021_cue-conflict",
                      assembly_class="BehavioralAssembly", bucket_name="brainscore-storage/brainio-brainscore")
