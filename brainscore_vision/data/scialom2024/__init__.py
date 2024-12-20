from brainio.assemblies import BehavioralAssembly

from brainscore_vision import data_registry, stimulus_set_registry, load_stimulus_set
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3, load_stimulus_set_from_s3


def force_stimulus_set_column_to_str(stimulus_set, columns=('percentage_elements', 'condition')):
    # a function to convert the data type of an entire column to string
    for column in columns:
        stimulus_set[column] = stimulus_set[column].astype(str)
    return stimulus_set


data_registry['Scialom2024_rgb'] = lambda: load_assembly_from_s3(
    identifier='Scialom2024_rgb',
    version_id='null',
    sha1='b79217a6b700760b96ffe60a948d6c2af9e7a615',
    bucket="brainscore-storage/brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Scialom2024_rgb'))

data_registry['Scialom2024_contours'] = lambda: load_assembly_from_s3(
    identifier='Scialom2024_contours',
    version_id='null',
    sha1='9487edf9f10d019968a77792908fb853fd73f818',
    bucket="brainscore-storage/brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Scialom2024_contours'))

data_registry['Scialom2024_phosphenes-12'] = lambda: load_assembly_from_s3(
    identifier='Scialom2024_phosphenes-12',
    version_id='null',
    sha1='1badbc7d08e2d135b2383a25e39ca7d22a3cd7ff',
    bucket="brainscore-storage/brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Scialom2024_phosphenes-12'))

data_registry['Scialom2024_phosphenes-16'] = lambda: load_assembly_from_s3(
    identifier='Scialom2024_phosphenes-16',
    version_id='null',
    sha1='1059a8126039ca01e8f293bfbe90539ea829b86f',
    bucket="brainscore-storage/brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Scialom2024_phosphenes-16'))

data_registry['Scialom2024_phosphenes-21'] = lambda: load_assembly_from_s3(
    identifier='Scialom2024_phosphenes-21',
    version_id='null',
    sha1='ac28102c6b165a759b102f670a99b33a67c7fc9a',
    bucket="brainscore-storage/brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Scialom2024_phosphenes-21'))

data_registry['Scialom2024_phosphenes-27'] = lambda: load_assembly_from_s3(
    identifier='Scialom2024_phosphenes-27',
    version_id='null',
    sha1='3bb9adfcdb294a9ecfec14af381f3dbdc7f6dfeb',
    bucket="brainscore-storage/brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Scialom2024_phosphenes-27'))

data_registry['Scialom2024_phosphenes-35'] = lambda: load_assembly_from_s3(
    identifier='Scialom2024_phosphenes-35',
    version_id='null',
    sha1='fa27c3931f76a696b1f1dde4c67a6733d682bcb2',
    bucket="brainscore-storage/brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Scialom2024_phosphenes-35'))

data_registry['Scialom2024_phosphenes-46'] = lambda: load_assembly_from_s3(
    identifier='Scialom2024_phosphenes-46',
    version_id='null',
    sha1='11f42e777a0938879473baa7a4efaefd27681c54',
    bucket="brainscore-storage/brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Scialom2024_phosphenes-46'))

data_registry['Scialom2024_phosphenes-59'] = lambda: load_assembly_from_s3(
    identifier='Scialom2024_phosphenes-59',
    version_id='null',
    sha1='f9e6ecb9013871f357fa56a9bfaa43c80eb0d9f5',
    bucket="brainscore-storage/brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Scialom2024_phosphenes-59'))

data_registry['Scialom2024_phosphenes-77'] = lambda: load_assembly_from_s3(
    identifier='Scialom2024_phosphenes-77',
    version_id='null',
    sha1='f3e3c50983b859eff094b75ca53939bf156b0f3f',
    bucket="brainscore-storage/brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Scialom2024_phosphenes-77'))

data_registry['Scialom2024_phosphenes-100'] = lambda: load_assembly_from_s3(
    identifier='Scialom2024_phosphenes-100',
    version_id='null',
    sha1='b97bda0f9ea10a684b81cf2118578edb483d9e27',
    bucket="brainscore-storage/brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Scialom2024_phosphenes-100'))

data_registry['Scialom2024_segments-12'] = lambda: load_assembly_from_s3(
    identifier='Scialom2024_segments-12',
    version_id='null',
    sha1='fec2104fc53f04af727174ca54fec2c9ad3b553d',
    bucket="brainscore-storage/brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Scialom2024_segments-12'))

data_registry['Scialom2024_segments-16'] = lambda: load_assembly_from_s3(
    identifier='Scialom2024_segments-16',
    version_id='null',
    sha1='4e30fc27bf98fa374af29a3bbb0de27c95ff845c',
    bucket="brainscore-storage/brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Scialom2024_segments-16'))

data_registry['Scialom2024_segments-21'] = lambda: load_assembly_from_s3(
    identifier='Scialom2024_segments-21',
    version_id='null',
    sha1='280dde1bb8ad226307ce25505a54f852197d0686',
    bucket="brainscore-storage/brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Scialom2024_segments-21'))

data_registry['Scialom2024_segments-27'] = lambda: load_assembly_from_s3(
    identifier='Scialom2024_segments-27',
    version_id='null',
    sha1='249148aaf4bed26ac2dcf07e4ab1d4dd19fc6531',
    bucket="brainscore-storage/brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Scialom2024_segments-27'))

data_registry['Scialom2024_segments-35'] = lambda: load_assembly_from_s3(
    identifier='Scialom2024_segments-35',
    version_id='null',
    sha1='67b729721928c5f1e85fc5f48e0c574ecfa196e4',
    bucket="brainscore-storage/brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Scialom2024_segments-35'))

data_registry['Scialom2024_segments-46'] = lambda: load_assembly_from_s3(
    identifier='Scialom2024_segments-46',
    version_id='null',
    sha1='2e8e7f937407eb74a77e006d09df4364945f24bf',
    bucket="brainscore-storage/brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Scialom2024_segments-46'))

data_registry['Scialom2024_segments-59'] = lambda: load_assembly_from_s3(
    identifier='Scialom2024_segments-59',
    version_id='null',
    sha1='ba161343f424673e2dbc123bc058b49ffa16af07',
    bucket="brainscore-storage/brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Scialom2024_segments-59'))

data_registry['Scialom2024_segments-77'] = lambda: load_assembly_from_s3(
    identifier='Scialom2024_segments-77',
    version_id='null',
    sha1='1f29d37fa6b84defd268cb35d8e26c6798c63714',
    bucket="brainscore-storage/brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Scialom2024_segments-77'))

data_registry['Scialom2024_segments-100'] = lambda: load_assembly_from_s3(
    identifier='Scialom2024_segments-100',
    version_id='null',
    sha1='298823d65ceacccd3247fe05e21b2df85c46343d',
    bucket="brainscore-storage/brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Scialom2024_segments-100'))

data_registry['Scialom2024_phosphenes-all'] = lambda: load_assembly_from_s3(
    identifier='Scialom2024_phosphenes-all',
    version_id='null',
    sha1='ae0fd1095846f0c637e54ad8ff96e44bdac8117d',
    bucket="brainscore-storage/brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Scialom2024_phosphenes-all'))

data_registry['Scialom2024_segments-all'] = lambda: load_assembly_from_s3(
    identifier='Scialom2024_segments-all',
    version_id='null',
    sha1='0ecd8e45b4eb5a2afba91b5fe06cacc8696e5925',
    bucket="brainscore-storage/brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Scialom2024_segments-all'))

stimulus_set_registry['Scialom2024_rgb'] = lambda: force_stimulus_set_column_to_str(load_stimulus_set_from_s3(
    identifier='Scialom2024_rgb',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1='c7e66eca214ffd9e38a5ebc1de159ef5f7755df9',
    zip_sha1='6dce0513eb20c6d501a4e2e574028d12f910a1da',
    csv_version_id='null',
    zip_version_id='null',
    filename_prefix='stimulus_'))

stimulus_set_registry['Scialom2024_contours'] = lambda: force_stimulus_set_column_to_str(load_stimulus_set_from_s3(
    identifier='Scialom2024_contours',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1='378ea511a906a1c440d556bd9b6232d433ba186e',
    zip_sha1='70a2277a327b7aa150654f46f46ea4bd247c5273',
    csv_version_id='null',
    zip_version_id='null',
    filename_prefix = 'stimulus_'))

stimulus_set_registry['Scialom2024_phosphenes-12'] = lambda: force_stimulus_set_column_to_str(load_stimulus_set_from_s3(
    identifier='Scialom2024_phosphenes-12',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1='762a114b716b8414322d0de620bd800f35beb02b',
    zip_sha1='701ecb4e2c87fac322a9d06b238e361d4c45b5bd',
    csv_version_id='null',
    zip_version_id='null',
    filename_prefix='stimulus_'))

stimulus_set_registry['Scialom2024_phosphenes-16'] = lambda: force_stimulus_set_column_to_str(load_stimulus_set_from_s3(
    identifier='Scialom2024_phosphenes-16',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1='40979a641232dab1515b7061171921221d86a13a',
    zip_sha1='26563eff7a42561204fb6c158ae00ef4acf62e4d',
    csv_version_id='null',
    zip_version_id='null',
    filename_prefix='stimulus_'))

stimulus_set_registry['Scialom2024_phosphenes-21'] = lambda: force_stimulus_set_column_to_str(load_stimulus_set_from_s3(
    identifier='Scialom2024_phosphenes-21',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1='72cffd4c0c8d30b815825475d864a87b7eb6e386',
    zip_sha1='22f9ecd333fcc177c16ed6c005989fe04f63700c',
    csv_version_id='null',
    zip_version_id='null',
    filename_prefix='stimulus_'))

stimulus_set_registry['Scialom2024_phosphenes-27'] = lambda: force_stimulus_set_column_to_str(load_stimulus_set_from_s3(
    identifier='Scialom2024_phosphenes-27',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1='94e91373299e82bcaa413e626bcaab6c914a9649',
    zip_sha1='69f17f8b1121a19738625da78b4b73de2b151926',
    csv_version_id='null',
    zip_version_id='null',
    filename_prefix='stimulus_'))

stimulus_set_registry['Scialom2024_phosphenes-35'] = lambda: force_stimulus_set_column_to_str(load_stimulus_set_from_s3(
    identifier='Scialom2024_phosphenes-35',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1='dc5657b53cf5d77dc6b62db856d7c2fd6151284c',
    zip_sha1='e26e6e5fabca6bd52d85b38416df92de0b16b81e',
    csv_version_id='null',
    zip_version_id='null',
    filename_prefix='stimulus_'))

stimulus_set_registry['Scialom2024_phosphenes-46'] = lambda: force_stimulus_set_column_to_str(load_stimulus_set_from_s3(
    identifier='Scialom2024_phosphenes-46',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1='5838fabb81840bba4fc667e48c1e82569fcbff29',
    zip_sha1='6bb41a20bea70b3e521320258682dba1bef7a0e6',
    csv_version_id='null',
    zip_version_id='null',
    filename_prefix='stimulus_'))

stimulus_set_registry['Scialom2024_phosphenes-59'] = lambda: force_stimulus_set_column_to_str(load_stimulus_set_from_s3(
    identifier='Scialom2024_phosphenes-59',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1='126123ae0596ebe93bb8b0fb985b3fd6e2db1aea',
    zip_sha1='3e7d106796050a0fd4868567c8ef2325f294a13c',
    csv_version_id='null',
    zip_version_id='null',
    filename_prefix='stimulus_'))

stimulus_set_registry['Scialom2024_phosphenes-77'] = lambda: force_stimulus_set_column_to_str(load_stimulus_set_from_s3(
    identifier='Scialom2024_phosphenes-77',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1='04eccad03672f7122b2821c667e8ba352eca8262',
    zip_sha1='5b5ae2437193d75d26f5ffa87245ee479ce2674a',
    csv_version_id='null',
    zip_version_id='null',
    filename_prefix='stimulus_'))

stimulus_set_registry['Scialom2024_phosphenes-100'] = lambda: force_stimulus_set_column_to_str(load_stimulus_set_from_s3(
    identifier='Scialom2024_phosphenes-100',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1='f75998b7e2624edbdf511a4f88d4665a0258e377',
    zip_sha1='b416d47663cd6e5d747f75f2f15eb52637313509',
    csv_version_id='null',
    zip_version_id='null',
    filename_prefix='stimulus_'))

stimulus_set_registry['Scialom2024_segments-12'] = lambda: force_stimulus_set_column_to_str(load_stimulus_set_from_s3(
    identifier='Scialom2024_segments-12',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1='c9de15f5aec2d2fbbd3669e8f3620ae4e884556b',
    zip_sha1='0e04090ea7cc6cabc9efa7695dab5d217ebad70b',
    csv_version_id='null',
    zip_version_id='null',
    filename_prefix='stimulus_'))

stimulus_set_registry['Scialom2024_segments-16'] = lambda: force_stimulus_set_column_to_str(load_stimulus_set_from_s3(
    identifier='Scialom2024_segments-16',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1='dc66edcd3f6e051418878e07002c06d2a936d142',
    zip_sha1='2aa3cb8c89bf5956ea5c7a6c0f86ce8759056b41',
    csv_version_id='null',
    zip_version_id='null',
    filename_prefix='stimulus_'))

stimulus_set_registry['Scialom2024_segments-21'] = lambda: force_stimulus_set_column_to_str(load_stimulus_set_from_s3(
    identifier='Scialom2024_segments-21',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1='c461e47b42f17c05399387c16b3d76db6389d05f',
    zip_sha1='e8116c9b1b2822bf2235b7ef91193baf2a7f47fb',
    csv_version_id='null',
    zip_version_id='null',
    filename_prefix='stimulus_'))

stimulus_set_registry['Scialom2024_segments-27'] = lambda: force_stimulus_set_column_to_str(load_stimulus_set_from_s3(
    identifier='Scialom2024_segments-27',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1='31e35368dbe090fe2748765b82bc0dd956818887',
    zip_sha1='607d8589a270220046455e150d3589095d8270b1',
    csv_version_id='null',
    zip_version_id='null',
    filename_prefix='stimulus_'))

stimulus_set_registry['Scialom2024_segments-35'] = lambda: force_stimulus_set_column_to_str(load_stimulus_set_from_s3(
    identifier='Scialom2024_segments-35',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1='3758af9739943817254ed79e12b441bb055858b9',
    zip_sha1='1df725d457f8092b2cc75a7f1509359383ae5420',
    csv_version_id='null',
    zip_version_id='null',
    filename_prefix='stimulus_'))

stimulus_set_registry['Scialom2024_segments-46'] = lambda: force_stimulus_set_column_to_str(load_stimulus_set_from_s3(
    identifier='Scialom2024_segments-46',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1='5c2982d3b079f5dd15dc18b5f127624256ca75b8',
    zip_sha1='1f0bd6bcb476bff6937dc2bb0bf9222b381e41d5',
    csv_version_id='null',
    zip_version_id='null',
    filename_prefix='stimulus_'))

stimulus_set_registry['Scialom2024_segments-59'] = lambda: force_stimulus_set_column_to_str(load_stimulus_set_from_s3(
    identifier='Scialom2024_segments-59',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1='22df0dcfec63af39a3df97555c085ccd57a5ec09',
    zip_sha1='63cd9ee774ed4aeca54b8aa54c037037a67537d6',
    csv_version_id='null',
    zip_version_id='null',
    filename_prefix='stimulus_'))

stimulus_set_registry['Scialom2024_segments-77'] = lambda: force_stimulus_set_column_to_str(load_stimulus_set_from_s3(
    identifier='Scialom2024_segments-77',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1='350f3c1ad3ae52150783e8be0135d79ef1f88c9f',
    zip_sha1='be2c2039e5eea7c3dcd90b885243fb6a1ebd40b9',
    csv_version_id='null',
    zip_version_id='null',
    filename_prefix='stimulus_'))

stimulus_set_registry['Scialom2024_segments-100'] = lambda: force_stimulus_set_column_to_str(load_stimulus_set_from_s3(
    identifier='Scialom2024_segments-100',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1='d2d7e231c7420730d5b3118c82b859a684d89a5b',
    zip_sha1='9d7ccb70f2231c8e62aecf18f1f48c160046c98b',
    csv_version_id='null',
    zip_version_id='null',
    filename_prefix='stimulus_'))

stimulus_set_registry['Scialom2024_phosphenes-all'] = lambda: force_stimulus_set_column_to_str(load_stimulus_set_from_s3(
    identifier='Scialom2024_phosphenes-all',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1='16276359d917b16fc84c736881cf70b5de3d9c5c',
    zip_sha1='c2c26b17a4b4d152d1f20e78fe085e321080bcf6',
    csv_version_id='null',
    zip_version_id='null',
    filename_prefix='stimulus_'))

stimulus_set_registry['Scialom2024_segments-all'] = lambda: force_stimulus_set_column_to_str(load_stimulus_set_from_s3(
    identifier='Scialom2024_segments-all',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1='4e66191b81517f3601e7bb27c9c3a84b9ee51a89',
    zip_sha1='59b279b2c65be4cdab50bf17b7295da2426744ee',
    csv_version_id='null',
    zip_version_id='null',
    filename_prefix='stimulus_'))
