import numpy as np

from brainio.assemblies import BehavioralAssembly
from brainio.packaging import package_data_assembly
import pandas as pd

'''
Experiment Information:

- This assembly is slightly different from normal assemblies, as the data is very sparse. 
  It comes from the papers below, from 1990 and 1991. 
  
  1990: https://www.jstor.org/stable/40062736?seq=2#metadata_info_tab_contents
  1991: https://www2.psych.ubc.ca/~rensink/publications/download/E&R-PsychRev-91.pdf
  
  In this case, the data is simply means and SEMs reported in Figure 1 of the 1990 paper
  and Figure 2 of the 1991 paper. 
  
  - 10 subjects
  - Response time is measured as a function of display size. 
 
'''

# initial csv to dataframe processing:
all_subjects = pd.read_csv('human_data/data.csv')


# construct the assembly
assembly = BehavioralAssembly(all_subjects['mean_rt'],
                              coords={
                                  'stimulus_id': ('presentation', all_subjects['stimuli']),
                                  'response_time': ('presentation', all_subjects['mean_rt']),
                                  'response_time_error': ('presentation', all_subjects['rt_error']),
                                  'num_subjects': ('presentation', all_subjects['num_subjects']),
                                  'target_trial': ('presentation', all_subjects['target_trial']),
                                  'display_size': ('presentation', all_subjects['display_size']),
                                  'error_rate': ('presentation', all_subjects['error_rate']),
                              },
                              dims=['presentation']
                              )

# give the assembly an identifier name
assembly.name = 'Jacob2020_3d_processing'

# upload to S3
package_data_assembly('brainio_brainscore', assembly, assembly_identifier=assembly.name,
                      stimulus_set_identifier='Jacob2020_3d_processing',
                      assembly_class_name="BehavioralAssembly", bucket_name="brainio-brainscore")
