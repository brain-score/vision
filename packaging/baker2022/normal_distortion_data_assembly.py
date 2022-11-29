import numpy as np

from brainio.assemblies import BehavioralAssembly
from brainio.packaging import package_data_assembly
import pandas as pd

'''
Experiment Information:
 - https://www.sciencedirect.com/science/article/pii/S2589004222011853#sec9
 - Data/subjects are from experiment 1 in paper above

 - 32 subjects 
 - 40 images for each condition (whole, fragmented, frankenstein) = 120 images/subject shown
 - 120 * 32 = 3840 total images shown 
 - However there are only 3706 trials total, for data issues (according to Baker)
 - 9-way AFC, from set: {bear, bunny, cat, elephant, frog, lizard, tiger, turtle, wolf}
 
'''

# initial csv to dataframe processing:
all_subjects = pd.read_csv('human_data/human_data.csv')

# all images presented in this assembly are normal (non-inverted)
all_subjects["orientation"] = "normal"

# construct the assembly
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
assembly.name = 'kellmen.Baker2022_normal_distortion'


# upload to S3
package_data_assembly('brainio_brainscore', assembly, assembly_identifier=assembly.name,
                      stimulus_set_identifier='kellmen.Baker2022_normal_distortion',
                      assembly_class_name="BehavioralAssembly", bucket_name="brainio-brainscore")
