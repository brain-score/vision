from brainio.assemblies import BehavioralAssembly
from brainio.packaging import package_data_assembly

assembly = BehavioralAssembly(['dog', 'dog', 'cat', 'dog', ...],
                               coords={
                                   'image_id': ('presentation', ['image1', 'image2', 'image3', 'image4', ...]),
                                   'sample_object': ('presentation', ['dog', 'cat', 'cat', 'dog', ...]),
                                   'distractor_object': ('presentation', ['cat', 'dog', 'dog', 'cat', ...]),
                                   # ...more meta
                                   # Note that meta from the StimulusSet will automatically be merged into the
                                   #  presentation dimension:
                                   #  https://github.com/brain-score/brainio/blob/d0ac841779fb47fa7b8bdad3341b68357c8031d9/brainio/fetch.py#L125-L132
                               },
                               dims=['presentation'])
assembly.name = 'Geirhos2021_Sketch'  # give the assembly an identifier name

# make sure the assembly is what you would expect
assert len(assembly['presentation']) == 179660
assert len(set(assembly['image_id'].values)) == 1600
assert len(set(assembly['choice'].values)) == len(set(assembly['sample_object'].values)) \
       == len(set(assembly['distractor_object'].values)) == 2

# upload to S3
# package_data_assembly(assembly, assembly_identifier=assembly.name, ,
#                       assembly_class='BehavioralAssembly'
#                       stimulus_set_identifier=stimuli.name)  # link to the StimulusSet