import numpy as np

from brainio.assemblies import BehavioralAssembly
from brainio.packaging import package_data_assembly
import pandas as pd

'''
Experiment Information:
https://arxiv.org/pdf/1706.06969.pdf

 - 5 subjects 
 - 800 trials each
 - 4000 total trials 
 - match to sample task 
 - 16 image categories
 - for the this benchmark (stylized) subjects saw the EXACT image indicated with the variable/column name
   image_lookup_id, and not a variation of it (no distortions, editing, etc).

'''

# initial csv to dataframe processing:
subject_1 = pd.read_csv('data_assemblies/stylized_subject-01_session_1.csv')
subject_2 = pd.read_csv('data_assemblies/stylized_subject-02_session_1.csv')
subject_3 = pd.read_csv('data_assemblies/stylized_subject-03_session_1.csv')
subject_4 = pd.read_csv('data_assemblies/stylized_subject-04_session_1.csv')
subject_5 = pd.read_csv('data_assemblies/stylized_subject-05_session_1.csv')

all_subjects = pd.concat([subject_1, subject_2, subject_3, subject_4, subject_5])

# parse for the image lookup id. This relates the data assembly with the stimulus set.
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
                                  'session': ('presentation', all_subjects['session']),
                              },
                              dims=['presentation']
                              )

# give the assembly an identifier name
assembly.name = 'brendel.Geirhos2021_stylized'

# make sure assembly dims are correct length
assert len(assembly['presentation']) == 4000

# make sure assembly coords are correct length
assert len(assembly['image_id']) == 4000
assert len(assembly['image_id_long']) == 4000
assert len(assembly['truth']) == 4000
assert len(assembly['choice']) == 4000
assert len(assembly['category']) == 4000
assert len(assembly['condition']) == 4000
assert len(assembly['response_time']) == 4000
assert len(assembly['trial']) == 4000
assert len(assembly['subject']) == 4000
assert len(assembly['session']) == 4000


# make sure there are 800 unique images (shown 1 time for each  of 5 subjects, total of 5 * 800 = 4000 images shown)
assert len(np.unique(assembly['image_id'].values)) == 800

# make sure there are 5 unique subjects
assert len(np.unique(assembly['subject'].values)) == 5

# make sure there are 16 unique object categories (ground truths)
assert len(np.unique(assembly['truth'].values)) == 16
assert len(np.unique(assembly['category'].values)) == 16

# make sure there is only one condition (i.e. no image variations added after presentation)
assert len(np.unique(assembly['condition'].values)) == 1

# upload to S3
package_data_assembly('brainio_brainscore', assembly, assembly_identifier=assembly.name,
                      stimulus_set_identifier="brendel.Geirhos2021_stylized",
                      assembly_class="BehavioralAssembly", bucket_name="brainio.contrib")
