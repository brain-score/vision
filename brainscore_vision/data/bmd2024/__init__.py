from brainio.assemblies import BehavioralAssembly

from brainscore_vision import data_registry, stimulus_set_registry, load_stimulus_set
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3, load_stimulus_set_from_s3

data_registry['BMD2024.texture_1'] = lambda: load_assembly_from_s3(
    identifier='BMD_2024_texture_1',
    version_id='null',
    sha1='050cef2bd38fe0e0c6d55c9a4ba0b1c57550a072',
    bucket="brainscore-storage/brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('BMD2024.texture_1'))

data_registry['BMD2024.texture_2'] = lambda: load_assembly_from_s3(
    identifier='BMD_2024_texture_2',
    version_id='null',
    sha1='1f9f4ee938df509c0cbeaec7fdfe0f40997da331',
    bucket="brainscore-storage/brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('BMD2024.texture_2'))

data_registry['BMD2024.dotted_1'] = lambda: load_assembly_from_s3(
    identifier='BMD_2024_dotted_1',
    version_id='null',
    sha1='eb16feffe392087b4c40ef249850825f702e7911',
    bucket="brainscore-storage/brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('BMD2024.dotted_1'))

data_registry['BMD2024.dotted_2'] = lambda: load_assembly_from_s3(
    identifier='BMD_2024_dotted_2',
    version_id='null',
    sha1='297833a094513b99ae434e581df09ac64cd6582f',
    bucket="brainscore-storage/brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('BMD2024.dotted_2'))


stimulus_set_registry['BMD2024.texture_1'] = lambda: load_stimulus_set_from_s3(
    identifier='BMD_2024_texture_1',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1='395911b2933d675b98dda7bae422f11648d8e86d',
    zip_sha1='cfde36c93dc9070ef5dfaa0a992c9d2420af3460',
    csv_version_id='null',
    zip_version_id='null')

stimulus_set_registry['BMD2024.texture_2'] = lambda: load_stimulus_set_from_s3(
    identifier='BMD_2024_texture_2',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1='98ff8e3a1ca6f632ebc2daa909804314bc1b7e31',
    zip_sha1='31c9d8449b25da8ad3cb034eee04db9193027fcb',
    csv_version_id='null',
    zip_version_id='null')

stimulus_set_registry['BMD2024.dotted_1'] = lambda: load_stimulus_set_from_s3(
    identifier='BMD_2024_dotted_1',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1='de4214666237a0be39810ec4fefd6ec8d2a2e881',
    zip_sha1='b4ab1355665b5bf3bf81b7aa6eccfd396c96bda2',
    csv_version_id='null',
    zip_version_id='null')

stimulus_set_registry['BMD2024.dotted_2'] = lambda: load_stimulus_set_from_s3(
    identifier='BMD_2024_dotted_2',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1='4555daa5257dee10c6c6a5625d3bb2d94452e294',
    zip_sha1='20337c1fac66ed0eec16410c6801cca830e6c20c',
    csv_version_id='null',
    zip_version_id='dzELAKHsBx1DKkrWAR9uteJ7K1.FtlAm')