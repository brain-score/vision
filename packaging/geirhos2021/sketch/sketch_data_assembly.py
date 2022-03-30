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
assembly = BehavioralAssembly(all_subjects.to_numpy(),
                              coords={
                                  'image_id_long': ('presentation', all_subjects['imagename'].to_numpy()),
                                  'image_lookup_id': ('presentation', all_subjects['image_lookup_id'].to_numpy()),
                                  'subject_response': ('presentation', all_subjects['object_response'].to_numpy()),
                                  'ground_truth': ('presentation', all_subjects['category'].to_numpy()),
                                  'condition': ('presentation', all_subjects['condition'].to_numpy()),
                                  'response_time': ('presentation', all_subjects['rt'].to_numpy()),
                                  'trial': ('presentation', all_subjects['trial'].to_numpy()),
                                  'subject': ('presentation', all_subjects['subj'].to_numpy()),
                                  'session': ('presentation', all_subjects['session'].to_numpy()),
                              },
                              dims=['presentation', 'info']
                              )

# give the assembly an identifier name
assembly.name = 'Geirhos2021_Sketch'

# make sure assembly dims are correct length
assert len(assembly['presentation']) == 5600
assert len(assembly['info']) == 8

# make sure assembly coords are correct length
assert len(assembly['image_id_long']) == 5600
assert len(assembly['image_lookup_id']) == 5600
assert len(assembly['subject_response']) == 5600
assert len(assembly['ground_truth']) == 5600
assert len(assembly['condition']) == 5600
assert len(assembly['response_time']) == 5600
assert len(assembly['trial']) == 5600
assert len(assembly['subject']) == 5600
assert len(assembly['session']) == 5600


# make sure there are 800 unique images (shown 1 time for each  of 7 subjects, total of 7 * 800 = 5600 images shown)
assert len(all_subjects['image_lookup_id'].unique()) == 800

# make sure there are 7 unique subjects
assert len(all_subjects['subj'].unique()) == 7

# make sure there are 16 unique object categories (ground truths)
assert len(all_subjects['category'].unique()) == 16

# make sure there is only one condition (i.e. no image variations added after presentation)
assert len(all_subjects['condition'].unique()) == 1


# upload to S3
# package_data_assembly(assembly, assembly_identifier=assembly.name, ,
#                       assembly_class='BehavioralAssembly'
#                       stimulus_set_identifier=stimuli.name)  # link to the StimulusSet
