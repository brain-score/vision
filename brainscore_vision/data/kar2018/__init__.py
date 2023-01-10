from brainio.assemblies import NeuronRecordingAssembly

from brainscore_vision import data_registry, stimulus_set_registry
from brainscore_vision.metrics.ceiling import InternalConsistency
from brainscore_vision.metrics.transformations import CrossValidation
from brainscore_vision.utils.s3 import load_assembly_from_s3, load_stimulus_set_from_s3
from brainscore_vision.data.data_helpers.helper import version_id_df, build_filename


# assemblies: hvm
data_registry['dicarlo.Kar2018hvm'] = lambda: load_assembly_from_s3(
    identifier="dicarlo.Kar2018hvm",
    version_id=version_id_df.at[build_filename('dicarlo.Kar2018hvm', '.nc'), 'version_id'],
    sha1="96ccacc76c5fa30ee68bdc8736d1d43ace93f3e7",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly)

# assemblies: cocogray
data_registry['dicarlo.Kar2018cocogray'] = lambda: load_assembly_from_s3(
    identifier="dicarlo.Kar2018cocogray",
    version_id=version_id_df.at[build_filename('dicarlo.Kar2018cocogray', '.nc'), 'version_id'],
    sha1="4202cb3992a5d71f71a7ca9e28ba3f8b27937b43",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly)

# stimulus set: cocogray
stimulus_set_registry['dicarlo.Kar2018cocogray'] = lambda: load_stimulus_set_from_s3(
    identifier="dicarlo.Kar2018cocogray",
    bucket="brainio-brainscore",
    csv_sha1="be9bb267b80fd7ee36a88d025b73ae8a849165da",
    zip_sha1="1457003ee9b27ed51c018237009fe148c6e71fd3",
    csv_version_id=version_id_df.at[build_filename('dicarlo.Kar2018cocogray', '.csv'), 'version_id'],
    zip_version_id=version_id_df.at[build_filename('dicarlo.Kar2018cocogray', '.zip'), 'version_id'])


def filter_neuroids(assembly, threshold):
    ceiler = InternalConsistency()
    ceiling = ceiler(assembly)
    ceiling = ceiling.raw
    ceiling = CrossValidation().aggregate(ceiling)
    ceiling = ceiling.sel(aggregation='center')
    pass_threshold = ceiling >= threshold
    assembly = assembly[{'neuroid': pass_threshold}]
    return assembly
