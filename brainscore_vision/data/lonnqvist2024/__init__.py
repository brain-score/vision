from brainio.assemblies import BehavioralAssembly

from brainscore_vision import data_registry, stimulus_set_registry, load_stimulus_set
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3, load_stimulus_set_from_s3


data_registry['Lonnqvist2024_inlab-instructions'] = lambda: load_assembly_from_s3(
    identifier='Lonnqvist2024_inlab-instructions',
    version_id='null',
    sha1='64ec603ebc852d193e7437980eaabe8fc482d88b',
    bucket="brainscore-storage/brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Lonnqvist2024_test'))

data_registry['Lonnqvist2024_inlab-no-instructions'] = lambda: load_assembly_from_s3(
    identifier='Lonnqvist2024_inlab-no-instructions',
    version_id='null',
    sha1='ff248ca2058d4e36eee44dbc6f8ea6a79c70b715',
    bucket="brainscore-storage/brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Lonnqvist2024_test'))

data_registry['Lonnqvist2024_online-no-instructions'] = lambda: load_assembly_from_s3(
    identifier='Lonnqvist2024_online-no-instructions',
    version_id='null',
    sha1='04240330eaf371d160ab418fd5560a72ed42cecb',
    bucket="brainscore-storage/brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Lonnqvist2024_test'))

stimulus_set_registry['Lonnqvist2024_train'] = lambda: load_stimulus_set_from_s3(
    identifier='Lonnqvist2024_train',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1='2d6a95a8239aa647ddc6aedd449eabcebdf882cf',
    zip_sha1='8adbf4de94524892042d3e43629a4be2beeedcaf',
    csv_version_id='null',
    zip_version_id='null',
    filename_prefix='stimulus_')

stimulus_set_registry['Lonnqvist2024_test'] = lambda: load_stimulus_set_from_s3(
    identifier='Lonnqvist2024_test',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1='8bc98dfc9f334e5c21b68f6787b3255da0d8644a',
    zip_sha1='cf94b5341d956d250e7f7798044cf71bbd100721',
    csv_version_id='null',
    zip_version_id='null',
    filename_prefix='stimulus_')