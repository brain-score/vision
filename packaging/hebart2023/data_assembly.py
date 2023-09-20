import numpy as np

from brainio.assemblies import BehavioralAssembly
from brainio.packaging import package_data_assembly
import pandas as pd

'''
Experiment Information:
 - Link
 - Data/subjects are from experiment 1 

 - oo task
 - n subjects 
 - 120 * 32 = 3840 total triplets shown 
'''

# initial csv to dataframe processing:
all_subjects = pd.read_csv('human_data/validationset.txt')

# construct the assembly TODO
assembly = BehavioralAssembly(all_subjects['Correct?'],
                              coords={
                                  'stimulus_id': ('presentation', all_subjects['image_shown']),
                                  'subject': ('presentation', all_subjects['Subj']),
                                  'condition': ('presentation', all_subjects['Frankensteinonfig']),
                                  'truth': ('presentation', all_subjects['Animal']),
                                  'correct': ('presentation', all_subjects['Correct?']),
                                  'orientation': ('presentation', all_subjects['orientation']),
                              },
                              dims=['presentation']
                              )

# give the assembly an identifier name
assembly.name = 'Hebart2023'


# upload to S3
package_data_assembly('brainio_brainscore', assembly, assembly_identifier=assembly.name,
                      stimulus_set_identifier='Hebart2023',
                      assembly_class_name="BehavioralAssembly", bucket_name="brainio-brainscore")