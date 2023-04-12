import numpy as np

from brainio.assemblies import BehavioralAssembly
from brainio.packaging import package_data_assembly
import pandas as pd

'''

Experiment Information:

    1) This assembly is slightly different from normal assemblies, as the data is very sparse. 
        It comes from this 1998 paper: https://www.sciencedirect.com/science/article/pii/S0042698998000510#FIG2
  
    2) In this case, the primary data is simply mean response time and SEMs (see below for breakdown)
  
    3) Fields (from raw data.csv):
        - stimulus_id: name of image
        - mean_rt: response time average of 10 subjects when shown that image against a field of copies of another 
                   image. That other image  is given by the column displayed_against, and the number of copies of that 
                   image is given by the column display_size.
        - rt_error: response time error (standard error of the mean, not (!) STD) of the mean_rt
        - num_subjects: the number of subjects, fixed at 10 for all rows
        - target_trial: indicates whether or not the image was actually present in the field of distractors
        - display_size: number of distractor objects in visual search task
        - mean_accuracy: mean accuracy across 10 subjects as a percentage
        - reported_slope: response time measured as a function of display size (2, 8, 14).
        - displayed_against: what the distractor images were in the visual search task.
     
'''


# initial csv to dataframe processing:
all_subjects = pd.read_csv('data.csv')


# construct the assembly
assembly = BehavioralAssembly(all_subjects['mean_rt'],
                              coords={
                                  'stimulus_id': ('presentation', all_subjects['stimulus_id']),
                                  'response_time': ('presentation', all_subjects['mean_rt']),
                                  'response_time_error': ('presentation', all_subjects['rt_error']),
                                  'num_subjects': ('presentation', all_subjects['num_subjects']),
                                  'target_trial': ('presentation', all_subjects['target_trial']),
                                  'display_size': ('presentation', all_subjects['display_size']),
                                  'mean_accuracy': ('presentation', all_subjects['mean_accuracy']),
                                  'reported_slope': ('presentation', all_subjects['reported_slope']),

                                  'displayed_against': ('presentation', all_subjects['displayed_against'])
                              },
                              dims=['presentation']
                              )

# give the assembly an identifier name
assembly.name = 'Jacob2020_occlusion_depth_ordering'

# upload to S3
package_data_assembly('brainio_brainscore', assembly, assembly_identifier=assembly.name,
                      stimulus_set_identifier='Jacob2020_occlusion_depth_ordering',
                      assembly_class_name="BehavioralAssembly", bucket_name="brainio-brainscore")
