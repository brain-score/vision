from brainio.assemblies import BehavioralAssembly
from brainio.packaging import package_data_assembly
import pandas as pd

'''
Experiment Information:
https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006613

 - 10 subjects 
 - 6 images each
 - 60 total images shown 
 - subject wrote down what object they saw, of the 6
 - 6 image categories
 
'''

# initial csv to dataframe processing:
all_subjects = pd.read_csv('../human_data/local_human_data.csv')

images = ['microphone', 'violin', 'camel',
          'warplane', 'hammer', 'jersey']

# construct the assembly
assembly = BehavioralAssembly(all_subjects.values,
                              coords={
                                  'category': ('presentation', images),
                                  'stimulus_id': ('presentation', [image + "_point" for image in images]),
                                  'subject_number': ('subject', list(range(1, 11))),
                                  'species': ('subject', ['human'] * 10),
                              },
                              dims=['presentation', 'subject']
                              )

# # give the assembly an identifier name
assembly.name = 'kellmen.Baker2018_local_distortion'

# make sure assembly dims are correct length
assert len(assembly['presentation']) == 6
assert len(assembly['subject']) == 10

# random peace of mind checks:
assert assembly.values[0][0] == 1
assert assembly.values[4][4] == 1
assert assembly.values[3][0] == 0
assert assembly.values[1][8] == 0


# upload to S3
package_data_assembly('brainio_brainscore', assembly, assembly_identifier=assembly.name,
                      stimulus_set_identifier="kellmen.Baker2018_local_distortion",
                      assembly_class="BehavioralAssembly", bucket_name="brainio-brainscore")
