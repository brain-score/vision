from brainio.assemblies import BehavioralAssembly

from brainscore_vision import data_registry, stimulus_set_registry, load_stimulus_set
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3, load_stimulus_set_from_s3


data_registry['Scialom2024_rgb'] = lambda: load_assembly_from_s3(
    identifier='',
    version_id='',
    sha1='',
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set(''))

stimulus_set_registry[''] = lambda: load_stimulus_set_from_s3(
    identifier='',
    bucket="brainio-brainscore",
    csv_sha1='',
    zip_sha1='',
    csv_version_id='',
    zip_version_id='',
    filename_prefix='stimulus_')