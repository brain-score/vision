import pandas as pd

from brainio.assemblies import PropertyAssembly
from brainio.packaging import package_data_assembly

'''
Experiment Information:
    - 5 subjects
    - 2AFC left/right offset discrimination task
    - PEST staircase to 75% correct responses
    - thresholds measured with a cumulative gaussian psychometric function with a likelihood fit

'''

num_subjects = 5
all_subjects = pd.read_excel('./metadata_human.xlsx')
this_experiment_subjects = all_subjects.dropna(subset=['threshold'], axis=0)
assembly = PropertyAssembly(this_experiment_subjects['threshold'],
                            coords={
                                'subject_unique_id': ('subject', this_experiment_subjects['subject_unique_id']),
                                'condition': ('subject', ['long-16', ] * num_subjects),
                            },
                            dims=['subject']
                            )

# assign assembly an identifier name
assembly.name = 'Malania2007_long-16'

# make sure assembly dims are correct length
assert len(assembly['subject']) == 5

# make sure assembly coords are correct length
assert len(assembly['subject_unique_id']) == 5
assert len(assembly['condition']) == 5

# make sure assembly coords are correct values
assert (assembly['condition'].values == 'long-16').all()
assert (assembly['subject_unique_id'].values == [1, 3, 4, 5, 6]).all()

# package_data_assembly('brainio_brainscore', assembly, assembly_identifier=assembly.name,
#                       stimulus_set_identifier=f"Malania2007_long-16",
#                       assembly_class_name="PropertyAssembly", bucket_name="brainio-brainscore")
