from brainio.assemblies import BehavioralAssembly
from brainscore_vision import data_registry, stimulus_set_registry, load_stimulus_set
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3, load_stimulus_set_from_s3


# normal distortion:
stimulus_set_registry['Baker2022_normal_distortion'] = lambda: load_stimulus_set_from_s3(
    identifier='Baker2022_normal_distortion',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="17d4db7458a29a787d12bb29c34e91daef1872bf",
    zip_sha1="2c726abaf081c8a9828269a559222f8c6eea0e4f",
    csv_version_id="null",
    zip_version_id="null")

data_registry['Baker2022_normal_distortion'] = lambda: load_assembly_from_s3(
    identifier='Baker2022_normal_distortion',
    version_id="null",
    sha1="46c79a48bf2dbd995a9305d8fbc03a134a852e17",
    bucket="brainscore-storage/brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Baker2022_normal_distortion'),
)

# inverted distortion:
stimulus_set_registry['Baker2022_inverted_distortion'] = lambda: load_stimulus_set_from_s3(
    identifier='Baker2022_inverted_distortion',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="91e452e4651024c2b382694edfcbc7bdc6c3189b",
    zip_sha1="4740a096af994c2232350469c664e53796f17a05",
    csv_version_id="null",
    zip_version_id="null")

data_registry['Baker2022_inverted_distortion'] = lambda: load_assembly_from_s3(
    identifier='Baker2022_inverted_distortion',
    version_id="null",
    sha1="b76fb57b25a58ca68db78d188fd0a783e1dcaf73",
    bucket="brainscore-storage/brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Baker2022_inverted_distortion'),
)