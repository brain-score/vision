import numpy as np

from brainio.assemblies import BehavioralAssembly
from brainio.packaging import package_data_assembly
import pandas as pd

'''
Experiment Information:
https://arxiv.org/pdf/1905.04598.pdf
 - Collected on Mturk, 1,250 Human Intelligence Tasks (HITs)
 - 25 subjects, each did 50 repetitions of 20 stimuli -> 25 * 50 * 20 = 25,000 data points
 - Word string responses were recorded, then filtered by hand to 1 of 5 categories
 - Even though 25,000 data points (trials) were collected, only 19,341 made it into final analysis.
   See paper (page 3) for details. 
'''

# initial csv to dataframe processing:
all_subjects = pd.read_csv('human_data/final_results.csv')

# grab the image_id from 1st column:
image_field = all_subjects['Input.image'].str.split("/", expand=True)
all_subjects["stimulus_id"] = image_field[5].str.replace(".png", "")

# drop answer is nan columns
all_subjects = all_subjects[all_subjects['final_answer'].notna()]

# construct the assembly
assembly = BehavioralAssembly(all_subjects['final_answer'],
                              coords={
                                  'stimulus_id': ('presentation', all_subjects['stimulus_id']),
                                  'truth': ('presentation', all_subjects['ground truth answer']),
                                  'choice': ('presentation', all_subjects['final_answer']),
                                  'correct': ('presentation', all_subjects['correct']),
                                  'subject': ('presentation', all_subjects['subject']),
                              },
                              dims=['presentation']
                              )

# give the assembly an identifier name
assembly.name = 'Zhu2019_extreme_occlusion'

# upload to S3
package_data_assembly('brainio_brainscore', assembly, assembly_identifier=assembly.name,
                      stimulus_set_identifier='Zhu2019_extreme_occlusion',
                      assembly_class_name="BehavioralAssembly", bucket_name="brainio-brainscore")