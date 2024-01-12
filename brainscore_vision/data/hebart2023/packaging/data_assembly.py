from brainio.assemblies import BehavioralAssembly
from brainio.packaging import package_data_assembly
import pandas as pd

'''
Experiment Information:
 - From Hebart 2023: https://elifesciences.org/articles/82580v1

 - Participants were shown three images and had to choose the odd one out.
 - The dataset consisted of 12,340 workers and 4,699,160 triplets, of which 
   4,574,059 triplets comprised the training and test data for computational 
   modeling and 125,101 triplets the four datasets used for computing noise 
   ceilings
'''

assembly_path = '/Users/linussommer/Desktop/Brain-Score-Data/DataAssembly/'
df = pd.read_csv(assembly_path + 'validationset.csv')
n = len(df)

# The last image is the odd-one-out in the human trial.
assembly = BehavioralAssembly(
    df['ooo'],
    coords={
        'stimulus_id': ('presentation', df['ooo']),
        'triplet_id': ('presentation', [f"triplet_{i}" for i in range(n)]),
        'image_1': ('presentation', df['img1']),
        'image_2': ('presentation', df['img2']),
        'image_3': ('presentation', df['ooo']),
    },
    dims=['presentation']
)

assembly.name = 'Hebart2023'
assert len(assembly) == 453642

package_data_assembly('brainio_brainscore', 
                      assembly, assembly_identifier=assembly.name,
                      stimulus_set_identifier='Hebart2023',
                      assembly_class_name="BehavioralAssembly", 
                      bucket_name="brainio-brainscore")
