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
all_subjects = pd.read_csv('human_data/Mturk_results.csv')

# grab the image_id from 1st column:
image_field = all_subjects['Input.image'].str.split("/", expand=True)
all_subjects["image_id"] = image_field[5].str.replace(".png", "")

# construct the assembly
assembly = BehavioralAssembly(all_subjects['ground truth answer'],
                              coords={
                                  'image_id': ('presentation', all_subjects['image_id']),
                                  'truth': ('presentation', all_subjects['ground truth answer']),
                                  'choice': ('presentation', all_subjects['Answer.Img_Main']),
                                  #'subject': ('presentation', all_subjects['subj']),
                              },
                              dims=['presentation']
                              )

# give the assembly an identifier name
assembly.name = 'yuille.Zhu2019_extreme_occlusion'

# make sure assembly dim is correct length
assert len(assembly['presentation']) == 25000

# make sure assembly coords are correct length
assert len(assembly['image_id']) == 25000
assert len(assembly['truth']) == 25000
assert len(assembly['choice']) == 25000

# # make sure there are 500 unique images
assert len(np.unique(assembly['image_id'].values)) == 500

# # make sure there are 25 unique subjects:
# assert len(np.unique(assembly['subject'].values)) == 25

# make sure there are 16 unique object categories (ground truths):
# assert len(np.unique(assembly['truth'].values)) == 5
# assert len(np.unique(assembly['category'].values)) == 5


# upload to S3
# package_data_assembly('brainio_brainscore', assembly, assembly_identifier=assembly.name,
#                       stimulus_set_identifier="brendel.Geirhos2021_contrast",
#                       assembly_class="BehavioralAssembly", bucket_name="brainio-brainscore")