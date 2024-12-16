import numpy as np

from brainio.assemblies import BehavioralAssembly
from brainio.packaging import package_data_assembly
import pandas as pd

'''
Experiment Information:
https://arxiv.org/pdf/1706.06969.pdf

 - 4 subjects 
 - 1280 images each
 - 5120 total images shown 
 - match to sample task, 16AFC
 - 16 image categories
 - for the this benchmark (high-pass) subjects saw the EXACT image indicated with the variable/column name
   image_lookup_id, and not a variation of it (no distortions, editing, etc). Condition is unclear based on 
   documentation fromm source repo, needs follow up. 
'''

# initial csv to dataframe processing:
subject_1 = pd.read_csv('data_assemblies/high-pass_subject-01_session_1.csv')
subject_2 = pd.read_csv('data_assemblies/high-pass_subject-02_session_1.csv')
subject_3 = pd.read_csv('data_assemblies/high-pass_subject-03_session_1.csv')
subject_4 = pd.read_csv('data_assemblies/high-pass_subject-04_session_1.csv')

all_subjects = pd.concat([subject_1, subject_2, subject_3, subject_4])

# parse df for the image lookup id. This relates the data assembly with the stimulus set.
split_cols = all_subjects['imagename'].str.split("_", expand=True)
drop_cols = split_cols.drop(split_cols.columns[[0, 1, 2]], axis=1)
all_subjects['image_lookup_id'] = drop_cols.agg("_".join, axis=1).str.replace(".png", "")


# construct the assembly
assembly = BehavioralAssembly(all_subjects['object_response'],
                              coords={
                                  'image_id': ('presentation', all_subjects['image_lookup_id']),
                                  'image_id_long': ('presentation', all_subjects['imagename']),
                                  'truth': ('presentation', all_subjects['category']),
                                  'choice': ('presentation', all_subjects['object_response']),
                                  'category': ('presentation', all_subjects['category']),
                                  'condition': ('presentation', all_subjects['condition']),
                                  'response_time': ('presentation', all_subjects['rt']),
                                  'trial': ('presentation', all_subjects['trial']),
                                  'subject': ('presentation', all_subjects['subj']),
                                  'session': ('presentation', all_subjects['Session']),
                              },
                              dims=['presentation']
                              )

# give the assembly an identifier name
assembly.name = 'brendel.Geirhos2021_high-pass'

# make sure assembly dim is correct length
assert len(assembly['presentation']) == 5120

# make sure assembly coords are correct length
assert len(assembly['image_id']) == 5120
assert len(assembly['image_id_long']) == 5120
assert len(assembly['truth']) == 5120
assert len(assembly['category']) == 5120
assert len(assembly['condition']) == 5120
assert len(assembly['response_time']) == 5120
assert len(assembly['trial']) == 5120
assert len(assembly['subject']) == 5120
assert len(assembly['session']) == 5120


# make sure there are 1280 unique images (shown 1 time for each  of 4 subjects, total of 4 * 1280 = 5120 images shown)
assert len(np.unique(assembly['image_id'].values)) == 1280

# make sure there are 4 unique subjects:
assert len(np.unique(assembly['subject'].values)) == 4

# make sure there are 16 unique object categories (ground truths):
assert len(np.unique(assembly['truth'].values)) == 16
assert len(np.unique(assembly['category'].values)) == 16


# upload to S3
package_data_assembly('brainio_brainscore', assembly, assembly_identifier=assembly.name,
                      stimulus_set_identifier="Geirhos2021_high-pass",
                      assembly_class="BehavioralAssembly", bucket_name="brainscore-storage/brainio-brainscore")
