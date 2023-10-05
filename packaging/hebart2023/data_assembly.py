import numpy as np

from brainio.assemblies import BehavioralAssembly
from brainio.packaging import package_data_assembly
import pandas as pd

'''
Experiment Information:
 - https://elifesciences.org/articles/82580

 - odd one out task
 - 470.000 total triplets shown 
'''

# initial csv to dataframe processing:
all_subjects = pd.read_csv('Desktop/Brain-Score-Data/DataAssembly/validationset.txt ')

# construct the assembly TODO
assembly = BehavioralAssembly(None,
                              coords={
                                  'stimulus_id': ('presentation', None), # TODO there are multiple stimuly?
                                  'odd_one_out': ('presentation', None), # TODO id of last element in triplet
                              },
                              dims=['presentation']
                              )

# give the assembly an identifier name
assembly.name = 'Hebart2023'


# upload to S3
#package_data_assembly('brainio_brainscore', assembly, assembly_identifier=assembly.name,
#                      stimulus_set_identifier='Hebart2023',
#                      assembly_class_name="BehavioralAssembly", bucket_name="brainio-brainscore")