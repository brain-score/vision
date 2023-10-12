import numpy as np
from brainio.assemblies import BehavioralAssembly
from brainio.packaging import package_data_assembly
import pandas as pd

'''
Experiment Information:
 - https://elifesciences.org/articles/82580

 - Participants were shown three images and had to choose the odd one out.
 - The dataset consisted of 12,340 workers and 4,699,160 triplets, of which 4,574,059 
   triplets comprised the training and test data for computational modeling and 125,101 
   triplets the four datasets used for computing noise ceilings
'''

# initial csv to dataframe processing:
df = pd.read_csv('/Users/linussommer/Desktop/Brain-Score-Data/DataAssembly/validationset.csv')

assert len(df) == 453642

# the last image is the odd one in the human trial
assembly = BehavioralAssembly(df['ooo'],
                               coords={
                                   'stimulus_id': ('presentation', [f"triplet_{i}" for i in range(len(df))]),
                                   'image_1': ('presentation', df['img1']),
                                   'image_2': ('presentation', df['img2']),
                                   'image_3': ('presentation', df['ooo']),  
                               },
                               dims=['presentation'])

# give the assembly an identifier name
assembly.name = 'Hebart2023'

# upload to S3
package_data_assembly('brainio_brainscore', assembly, assembly_identifier=assembly.name,
                      stimulus_set_identifier='Hebart2023',
                      assembly_class_name="BehavioralAssembly", bucket_name="brainio-brainscore")