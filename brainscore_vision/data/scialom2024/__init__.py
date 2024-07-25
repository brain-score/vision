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
    version_id='g.9QO4x6dLeBLSOAjUfUmgOPliyWLVwT',
    sha1='b79217a6b700760b96ffe60a948d6c2af9e7a615',
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Scialom2024_rgb'))

data_registry['Scialom2024_contours'] = lambda: load_assembly_from_s3(
    identifier='Scialom2024_contours',
    version_id='Ri4iQZmgxzUKvHDETkafbrnarzMrAVO8',
    sha1='9487edf9f10d019968a77792908fb853fd73f818',
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Scialom2024_contours'))

data_registry['Scialom2024_phosphenes-12'] = lambda: load_assembly_from_s3(
    identifier='Scialom2024_phosphenes-12',
    version_id='dYCCMwM0Pf5yiK9dmYLRYyijtV9ZoGdx',
    sha1='1badbc7d08e2d135b2383a25e39ca7d22a3cd7ff',
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Scialom2024_phosphenes-12'))

data_registry['Scialom2024_phosphenes-16'] = lambda: load_assembly_from_s3(
    identifier='Scialom2024_phosphenes-16',
    version_id='p5NzMB.QzH612GaztXi8EMvYL3Js1R70',
    sha1='1059a8126039ca01e8f293bfbe90539ea829b86f',
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Scialom2024_phosphenes-16'))

data_registry['Scialom2024_phosphenes-21'] = lambda: load_assembly_from_s3(
    identifier='Scialom2024_phosphenes-21',
    version_id='TbXg6.Xcf8.ssN7tkWfWpF0x.5lnolVN',
    sha1='ac28102c6b165a759b102f670a99b33a67c7fc9a',
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Scialom2024_phosphenes-21'))

data_registry['Scialom2024_phosphenes-27'] = lambda: load_assembly_from_s3(
    identifier='Scialom2024_phosphenes-27',
    version_id='5Sp9DU3CgIaZrnzzq6EntKPnKlgsHrpm',
    sha1='3bb9adfcdb294a9ecfec14af381f3dbdc7f6dfeb',
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Scialom2024_phosphenes-27'))

data_registry['Scialom2024_phosphenes-35'] = lambda: load_assembly_from_s3(
    identifier='Scialom2024_phosphenes-35',
    version_id='3rZBFqfPkSUJV6GTqlbn6x7JV8E65LoB',
    sha1='fa27c3931f76a696b1f1dde4c67a6733d682bcb2',
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Scialom2024_phosphenes-35'))

data_registry['Scialom2024_phosphenes-46'] = lambda: load_assembly_from_s3(
    identifier='Scialom2024_phosphenes-46',
    version_id='GsjfhIatQ2wlUMol8jtK9RiC5cCVnTg7',
    sha1='11f42e777a0938879473baa7a4efaefd27681c54',
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Scialom2024_phosphenes-46'))

data_registry['Scialom2024_phosphenes-59'] = lambda: load_assembly_from_s3(
    identifier='Scialom2024_phosphenes-59',
    version_id='9Mp6HAugSkKeYUVczRDWYMoYI2Qb8K0r',
    sha1='f9e6ecb9013871f357fa56a9bfaa43c80eb0d9f5',
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Scialom2024_phosphenes-59'))

data_registry['Scialom2024_phosphenes-77'] = lambda: load_assembly_from_s3(
    identifier='Scialom2024_phosphenes-77',
    version_id='h4HdLUn8NEM6mzYf45rYuUnukampVN0l',
    sha1='f3e3c50983b859eff094b75ca53939bf156b0f3f',
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Scialom2024_phosphenes-77'))

data_registry['Scialom2024_phosphenes-100'] = lambda: load_assembly_from_s3(
    identifier='Scialom2024_phosphenes-100',
    version_id='NFmHKPwWEDkQiCexKX2ROBchfDgwOHLw',
    sha1='b97bda0f9ea10a684b81cf2118578edb483d9e27',
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Scialom2024_phosphenes-100'))

data_registry['Scialom2024_segments-12'] = lambda: load_assembly_from_s3(
    identifier='Scialom2024_segments-12',
    version_id='2ezI3rg_t1nycV_8FF9_N.mC8hjT.Abz',
    sha1='fec2104fc53f04af727174ca54fec2c9ad3b553d',
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Scialom2024_segments-12'))

data_registry['Scialom2024_segments-16'] = lambda: load_assembly_from_s3(
    identifier='Scialom2024_segments-16',
    version_id='7q9AyMlmD4oHWYCDpYK5UlwidCrdP7kc',
    sha1='4e30fc27bf98fa374af29a3bbb0de27c95ff845c',
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Scialom2024_segments-16'))

data_registry['Scialom2024_segments-21'] = lambda: load_assembly_from_s3(
    identifier='Scialom2024_segments-21',
    version_id='k2EC3vY8ZT_qv_0ErMErRISxu7uKtzkH',
    sha1='280dde1bb8ad226307ce25505a54f852197d0686',
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Scialom2024_segments-21'))

data_registry['Scialom2024_segments-27'] = lambda: load_assembly_from_s3(
    identifier='Scialom2024_segments-27',
    version_id='LLZMZjrx6sqmbxc16i6rXsB3IR4q48rx',
    sha1='249148aaf4bed26ac2dcf07e4ab1d4dd19fc6531',
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Scialom2024_segments-27'))

data_registry['Scialom2024_segments-35'] = lambda: load_assembly_from_s3(
    identifier='Scialom2024_segments-35',
    version_id='OZLLP4b953wlOiFD3PqCQWXc4Hp2391W',
    sha1='67b729721928c5f1e85fc5f48e0c574ecfa196e4',
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Scialom2024_segments-35'))

data_registry['Scialom2024_segments-46'] = lambda: load_assembly_from_s3(
    identifier='Scialom2024_segments-46',
    version_id='XsmgXxqEAXaz7luA4pWgCp3CWZ_FWMe3',
    sha1='2e8e7f937407eb74a77e006d09df4364945f24bf',
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Scialom2024_segments-46'))

data_registry['Scialom2024_segments-59'] = lambda: load_assembly_from_s3(
    identifier='Scialom2024_segments-59',
    version_id='oUJYFbkNdRRnL_vwDa18tsVtyg3kS47E',
    sha1='ba161343f424673e2dbc123bc058b49ffa16af07',
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Scialom2024_segments-59'))

data_registry['Scialom2024_segments-77'] = lambda: load_assembly_from_s3(
    identifier='Scialom2024_segments-77',
    version_id='fm5hSgpwgftQoAiyc5sj0mkiZ8qooJGM',
    sha1='1f29d37fa6b84defd268cb35d8e26c6798c63714',
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Scialom2024_segments-77'))

data_registry['Scialom2024_segments-100'] = lambda: load_assembly_from_s3(
    identifier='Scialom2024_segments-100',
    version_id='kDsLxv8JxenqL79Uwzvtz5STE5TvDrYW',
    sha1='298823d65ceacccd3247fe05e21b2df85c46343d',
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Scialom2024_segments-100'))

data_registry['Scialom2024_phosphenes-all'] = lambda: load_assembly_from_s3(
    identifier='Scialom2024_phosphenes-all',
    version_id='U_i4FlNK4GNohBOFPpUwb7EZZ35Z_EWW',
    sha1='ae0fd1095846f0c637e54ad8ff96e44bdac8117d',
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Scialom2024_phosphenes-all'))

data_registry['Scialom2024_segments-all'] = lambda: load_assembly_from_s3(
    identifier='Scialom2024_segments-all',
    version_id='La4EBnFDur5GkyI.zKu4QjusTUpXM4sy',
    sha1='0ecd8e45b4eb5a2afba91b5fe06cacc8696e5925',
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Scialom2024_segments-all'))

stimulus_set_registry['Scialom2024_rgb'] = lambda: force_stimulus_set_column_to_str(load_stimulus_set_from_s3(
    identifier='Scialom2024_rgb',
    bucket="brainio-brainscore",
    csv_sha1='c7e66eca214ffd9e38a5ebc1de159ef5f7755df9',
    zip_sha1='6dce0513eb20c6d501a4e2e574028d12f910a1da',
    csv_version_id='5bqWqpuVrl4St01_lRg3utq5cdOOGGJs',
    zip_version_id='kdGhqj0cDkpWdgMVLadpZGrf6PK8Dirz',
    filename_prefix='stimulus_'))

stimulus_set_registry['Scialom2024_contours'] = lambda: force_stimulus_set_column_to_str(load_stimulus_set_from_s3(
    identifier='Scialom2024_contours',
    bucket="brainio-brainscore",
    csv_sha1='378ea511a906a1c440d556bd9b6232d433ba186e',
    zip_sha1='70a2277a327b7aa150654f46f46ea4bd247c5273',
    csv_version_id='SqXFodaAoW7GbxjeDinX4PrOJP6lybUh',
    zip_version_id='JgLMb99lgnHd8qW0dAdp4dzqeZW4sOFE',
    filename_prefix = 'stimulus_'))

stimulus_set_registry['Scialom2024_phosphenes-12'] = lambda: force_stimulus_set_column_to_str(load_stimulus_set_from_s3(
    identifier='Scialom2024_phosphenes-12',
    bucket="brainio-brainscore",
    csv_sha1='762a114b716b8414322d0de620bd800f35beb02b',
    zip_sha1='701ecb4e2c87fac322a9d06b238e361d4c45b5bd',
    csv_version_id='QpBcMWbziLzLiZCbF7WlIx7BxO0Gnpef',
    zip_version_id='TzjrwhkU37r7Ek9SvzzK5xVlYZJxv_bE',
    filename_prefix='stimulus_'))

stimulus_set_registry['Scialom2024_phosphenes-16'] = lambda: force_stimulus_set_column_to_str(load_stimulus_set_from_s3(
    identifier='Scialom2024_phosphenes-16',
    bucket="brainio-brainscore",
    csv_sha1='40979a641232dab1515b7061171921221d86a13a',
    zip_sha1='26563eff7a42561204fb6c158ae00ef4acf62e4d',
    csv_version_id='EPTbc8QroMufOV85_XwB7WqPQfvqaGTl',
    zip_version_id='Yvr7I7PDyjqUoCsC0IVau374r0aYSupx',
    filename_prefix='stimulus_'))

stimulus_set_registry['Scialom2024_phosphenes-21'] = lambda: force_stimulus_set_column_to_str(load_stimulus_set_from_s3(
    identifier='Scialom2024_phosphenes-21',
    bucket="brainio-brainscore",
    csv_sha1='72cffd4c0c8d30b815825475d864a87b7eb6e386',
    zip_sha1='22f9ecd333fcc177c16ed6c005989fe04f63700c',
    csv_version_id='klnmUb0aDEnfzk5UPE.6V7XEMN6sQKRm',
    zip_version_id='YMQqJmwbEzRTrdm2ZoMy9rmRTsPnqm1j',
    filename_prefix='stimulus_'))

stimulus_set_registry['Scialom2024_phosphenes-27'] = lambda: force_stimulus_set_column_to_str(load_stimulus_set_from_s3(
    identifier='Scialom2024_phosphenes-27',
    bucket="brainio-brainscore",
    csv_sha1='94e91373299e82bcaa413e626bcaab6c914a9649',
    zip_sha1='69f17f8b1121a19738625da78b4b73de2b151926',
    csv_version_id='6daseIjtYi.w.bqa4xxVwY74494T.IUr',
    zip_version_id='AtqEH__oIbcA76nSn6BqRxxVsffdqIoo',
    filename_prefix='stimulus_'))

stimulus_set_registry['Scialom2024_phosphenes-35'] = lambda: force_stimulus_set_column_to_str(load_stimulus_set_from_s3(
    identifier='Scialom2024_phosphenes-35',
    bucket="brainio-brainscore",
    csv_sha1='dc5657b53cf5d77dc6b62db856d7c2fd6151284c',
    zip_sha1='e26e6e5fabca6bd52d85b38416df92de0b16b81e',
    csv_version_id='yh3oZ1E7TP1WrwEv2kzBqweu9R9soly5',
    zip_version_id='wjSfkL4RamMzbt7nlOgTiXp5r468yosf',
    filename_prefix='stimulus_'))

stimulus_set_registry['Scialom2024_phosphenes-46'] = lambda: force_stimulus_set_column_to_str(load_stimulus_set_from_s3(
    identifier='Scialom2024_phosphenes-46',
    bucket="brainio-brainscore",
    csv_sha1='5838fabb81840bba4fc667e48c1e82569fcbff29',
    zip_sha1='6bb41a20bea70b3e521320258682dba1bef7a0e6',
    csv_version_id='aRLe.zBrwTyU6rv4TvobKW0bkW4wWVZp',
    zip_version_id='5glt.ZHhImjlCM7AVNXvDsbucvpIVTU7',
    filename_prefix='stimulus_'))

stimulus_set_registry['Scialom2024_phosphenes-59'] = lambda: force_stimulus_set_column_to_str(load_stimulus_set_from_s3(
    identifier='Scialom2024_phosphenes-59',
    bucket="brainio-brainscore",
    csv_sha1='126123ae0596ebe93bb8b0fb985b3fd6e2db1aea',
    zip_sha1='3e7d106796050a0fd4868567c8ef2325f294a13c',
    csv_version_id='j4w9iSTF2R4z6lxGUVVOrW2QWy5l0E7J',
    zip_version_id='aI3UVaw9ewMenhJnAxsf38VJ6HYLBNHb',
    filename_prefix='stimulus_'))

stimulus_set_registry['Scialom2024_phosphenes-77'] = lambda: force_stimulus_set_column_to_str(load_stimulus_set_from_s3(
    identifier='Scialom2024_phosphenes-77',
    bucket="brainio-brainscore",
    csv_sha1='04eccad03672f7122b2821c667e8ba352eca8262',
    zip_sha1='5b5ae2437193d75d26f5ffa87245ee479ce2674a',
    csv_version_id='pEBfaM_9kwUxagDYVazs3Bk1RpHUUOxA',
    zip_version_id='AQczJ9zAtfNNS7qenBw6dyXnLEoNnYBo',
    filename_prefix='stimulus_'))

stimulus_set_registry['Scialom2024_phosphenes-100'] = lambda: force_stimulus_set_column_to_str(load_stimulus_set_from_s3(
    identifier='Scialom2024_phosphenes-100',
    bucket="brainio-brainscore",
    csv_sha1='f75998b7e2624edbdf511a4f88d4665a0258e377',
    zip_sha1='b416d47663cd6e5d747f75f2f15eb52637313509',
    csv_version_id='EZQCfqMrjcQo0gT1w4nDp.yDyQkRpabH',
    zip_version_id='RdS_d8Ld2tN8PCmllvU9vPGhDMIfAo8E',
    filename_prefix='stimulus_'))

stimulus_set_registry['Scialom2024_segments-12'] = lambda: force_stimulus_set_column_to_str(load_stimulus_set_from_s3(
    identifier='Scialom2024_segments-12',
    bucket="brainio-brainscore",
    csv_sha1='c9de15f5aec2d2fbbd3669e8f3620ae4e884556b',
    zip_sha1='0e04090ea7cc6cabc9efa7695dab5d217ebad70b',
    csv_version_id='btMm1X3.9pzkXrd4GPHoBav8ONyyOeBg',
    zip_version_id='pc4Q2pK.tv6XKF5RlY8vZnEk8_Bx349v',
    filename_prefix='stimulus_'))

stimulus_set_registry['Scialom2024_segments-16'] = lambda: force_stimulus_set_column_to_str(load_stimulus_set_from_s3(
    identifier='Scialom2024_segments-16',
    bucket="brainio-brainscore",
    csv_sha1='dc66edcd3f6e051418878e07002c06d2a936d142',
    zip_sha1='2aa3cb8c89bf5956ea5c7a6c0f86ce8759056b41',
    csv_version_id='jXnfBBor.RNtD_Bg53eSdCLCTgJKv1LO',
    zip_version_id='TAGJKzpndcfo8ckUyMv6OWnLUlV3OiN8',
    filename_prefix='stimulus_'))

stimulus_set_registry['Scialom2024_segments-21'] = lambda: force_stimulus_set_column_to_str(load_stimulus_set_from_s3(
    identifier='Scialom2024_segments-21',
    bucket="brainio-brainscore",
    csv_sha1='c461e47b42f17c05399387c16b3d76db6389d05f',
    zip_sha1='e8116c9b1b2822bf2235b7ef91193baf2a7f47fb',
    csv_version_id='cMvCmdKyyXsMDsgDDIJl.A1VEpDV6k5a',
    zip_version_id='y6bAR8Y5Kbi0B7rlxEvk0j5qNO2VrZph',
    filename_prefix='stimulus_'))

stimulus_set_registry['Scialom2024_segments-27'] = lambda: force_stimulus_set_column_to_str(load_stimulus_set_from_s3(
    identifier='Scialom2024_segments-27',
    bucket="brainio-brainscore",
    csv_sha1='31e35368dbe090fe2748765b82bc0dd956818887',
    zip_sha1='607d8589a270220046455e150d3589095d8270b1',
    csv_version_id='MdeJ2KQtCsN5oLa3.lQDJFwGm02R4BiX',
    zip_version_id='yF.EUfpHFbMJHAtTCrnEhLQ9D1zD5M.T',
    filename_prefix='stimulus_'))

stimulus_set_registry['Scialom2024_segments-35'] = lambda: force_stimulus_set_column_to_str(load_stimulus_set_from_s3(
    identifier='Scialom2024_segments-35',
    bucket="brainio-brainscore",
    csv_sha1='3758af9739943817254ed79e12b441bb055858b9',
    zip_sha1='1df725d457f8092b2cc75a7f1509359383ae5420',
    csv_version_id='ABPjQCWfRRaa0xy4ESKahIG1_I5eIhM4',
    zip_version_id='dpbgFyI3s_BynU62DML3V866JXH5Q_aw',
    filename_prefix='stimulus_'))

stimulus_set_registry['Scialom2024_segments-46'] = lambda: force_stimulus_set_column_to_str(load_stimulus_set_from_s3(
    identifier='Scialom2024_segments-46',
    bucket="brainio-brainscore",
    csv_sha1='5c2982d3b079f5dd15dc18b5f127624256ca75b8',
    zip_sha1='1f0bd6bcb476bff6937dc2bb0bf9222b381e41d5',
    csv_version_id='jhH9v7SieY6y6ff83XUPmLh95QPjrtIK',
    zip_version_id='88I4SDLBlhlNKBTZ0yRgZurknooVObUA',
    filename_prefix='stimulus_'))

stimulus_set_registry['Scialom2024_segments-59'] = lambda: force_stimulus_set_column_to_str(load_stimulus_set_from_s3(
    identifier='Scialom2024_segments-59',
    bucket="brainio-brainscore",
    csv_sha1='22df0dcfec63af39a3df97555c085ccd57a5ec09',
    zip_sha1='63cd9ee774ed4aeca54b8aa54c037037a67537d6',
    csv_version_id='eOxRFJhOKIm.mk78V0vjuhmd5FUBTzgh',
    zip_version_id='_6ThdN0H9c0bOvqH0MfWKPuGSwWYkJef',
    filename_prefix='stimulus_'))

stimulus_set_registry['Scialom2024_segments-77'] = lambda: force_stimulus_set_column_to_str(load_stimulus_set_from_s3(
    identifier='Scialom2024_segments-77',
    bucket="brainio-brainscore",
    csv_sha1='350f3c1ad3ae52150783e8be0135d79ef1f88c9f',
    zip_sha1='be2c2039e5eea7c3dcd90b885243fb6a1ebd40b9',
    csv_version_id='_X.oelHKQFdyqEmurki57Zf49L6CRJII',
    zip_version_id='Qn10b4obb0UCLezVGCwjzaSkbUyIeInp',
    filename_prefix='stimulus_'))

stimulus_set_registry['Scialom2024_segments-100'] = lambda: force_stimulus_set_column_to_str(load_stimulus_set_from_s3(
    identifier='Scialom2024_segments-100',
    bucket="brainio-brainscore",
    csv_sha1='d2d7e231c7420730d5b3118c82b859a684d89a5b',
    zip_sha1='9d7ccb70f2231c8e62aecf18f1f48c160046c98b',
    csv_version_id='bacyQYdD0g4Puj4MB0d5qkdmanvy8RG7',
    zip_version_id='PXGICnHlx0mNo5MU_xW4v708e411SVdQ',
    filename_prefix='stimulus_'))

stimulus_set_registry['Scialom2024_phosphenes-all'] = lambda: force_stimulus_set_column_to_str(load_stimulus_set_from_s3(
    identifier='Scialom2024_phosphenes-all',
    bucket="brainio-brainscore",
    csv_sha1='16276359d917b16fc84c736881cf70b5de3d9c5c',
    zip_sha1='c2c26b17a4b4d152d1f20e78fe085e321080bcf6',
    csv_version_id='ar1W5DUVRQNZeUFCFp0OFKQYPlfr4004',
    zip_version_id='NM9aLLdkv0cZuUFFiKK7NJgh8_DAcrOB',
    filename_prefix='stimulus_'))

stimulus_set_registry['Scialom2024_segments-all'] = lambda: force_stimulus_set_column_to_str(load_stimulus_set_from_s3(
    identifier='Scialom2024_segments-all',
    bucket="brainio-brainscore",
    csv_sha1='4e66191b81517f3601e7bb27c9c3a84b9ee51a89',
    zip_sha1='59b279b2c65be4cdab50bf17b7295da2426744ee',
    csv_version_id='Nl3ywB0CEEjEZmB3NH2yQkYGqxAiAD4y',
    zip_version_id='N5bg4NWu7qbJASRULKgJsarEYRyLHlnt',
    filename_prefix='stimulus_'))
