from brainio.assemblies import PropertyAssembly

from brainscore_vision import data_registry, stimulus_set_registry, load_stimulus_set
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3, load_stimulus_set_from_s3


BIBTEX = """@article{malania2007,
            author = {Malania, Maka and Herzog, Michael H. and Westheimer, Gerald},
            title = "{Grouping of contextual elements that affect vernier thresholds}",
            journal = {Journal of Vision},
            volume = {7},
            number = {2},
            pages = {1-1},
            year = {2007},
            issn = {1534-7362},
            doi = {10.1167/7.2.1},
            url = {https://doi.org/10.1167/7.2.1}
        }"""


data_registry['Malania2007.equal2'] = lambda: load_assembly_from_s3(
    identifier='Malania2007_equal-2',
    version_id="null",
    sha1="277b2fbffed00e16b6a69b488f73eeda5abaaf10",
    bucket="brainscore-storage/brainio-brainscore",
    cls=PropertyAssembly,
    stimulus_set_loader=None,
)
data_registry['Malania2007.equal16'] = lambda: load_assembly_from_s3(
    identifier='Malania2007_equal-16',
    version_id="null",
    sha1="ef49506238e8d2554918b113fbc60c133077186e",
    bucket="brainscore-storage/brainio-brainscore",
    cls=PropertyAssembly,
    stimulus_set_loader=None,
)
data_registry['Malania2007.long2'] = lambda: load_assembly_from_s3(
    identifier='Malania2007_long-2',
    version_id="null",
    sha1="9076a5b693948c4992b6c8e753f04a7acd2014a1",
    bucket="brainscore-storage/brainio-brainscore",
    cls=PropertyAssembly,
    stimulus_set_loader=None,
)
data_registry['Malania2007.long16'] = lambda: load_assembly_from_s3(
    identifier='Malania2007_long-16',
    version_id="null",
    sha1="3106cf1f2fa9e66617ebf231df05d29077fc478f",
    bucket="brainscore-storage/brainio-brainscore",
    cls=PropertyAssembly,
    stimulus_set_loader=None,
)
data_registry['Malania2007.short2'] = lambda: load_assembly_from_s3(
    identifier='Malania2007_short-2',
    version_id="null",
    sha1="85fb65ad76de48033c704b9c5689771e1ea0457d",
    bucket="brainscore-storage/brainio-brainscore",
    cls=PropertyAssembly,
    stimulus_set_loader=None,
)
data_registry['Malania2007.short4'] = lambda: load_assembly_from_s3(
    identifier='Malania2007_short-4',
    version_id="null",
    sha1="75506be9a26ec38a223e41510f1a8cb32d5b0bc9",
    bucket="brainscore-storage/brainio-brainscore",
    cls=PropertyAssembly,
    stimulus_set_loader=None,
)
data_registry['Malania2007.short6'] = lambda: load_assembly_from_s3(
    identifier='Malania2007_short-6',
    version_id="null",
    sha1="2901be6b352e67550da040d79d744819365b8626",
    bucket="brainscore-storage/brainio-brainscore",
    cls=PropertyAssembly,
    stimulus_set_loader=None,
)
data_registry['Malania2007.short8'] = lambda: load_assembly_from_s3(
    identifier='Malania2007_short-8',
    version_id="null",
    sha1="6daf47b086cb969d75222e320f49453ed8437885",
    bucket="brainscore-storage/brainio-brainscore",
    cls=PropertyAssembly,
    stimulus_set_loader=None,
)
data_registry['Malania2007.short16'] = lambda: load_assembly_from_s3(
    identifier='Malania2007_short-16',
    version_id="null",
    sha1="8ae0898caad718b747f85fce5888416affc3a569",
    bucket="brainscore-storage/brainio-brainscore",
    cls=PropertyAssembly,
    stimulus_set_loader=None,
)
data_registry['Malania2007.vernier_only'] = lambda: load_assembly_from_s3(
    identifier='Malania2007_vernier-only',
    version_id="null",
    sha1="1cf83e8b6141f8b0d67ea46994f342325f62001f",
    bucket="brainscore-storage/brainio-brainscore",
    cls=PropertyAssembly,
    stimulus_set_loader=None,
)


stimulus_set_registry['Malania2007.equal2'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_equal-2',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="36f3c92a1335895b10c4150f5c25a68ab4576d4a",
    zip_sha1="80be52e8701ecb8e7fbb81c0bff9c148ddc2b401",
    csv_version_id="null",
    zip_version_id="null"
)
stimulus_set_registry['Malania2007.equal2_fit'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_equal-2_fit',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="b7105f44d5d781f5e06159008a3f63c9f774c2d1",
    zip_sha1="ba5c1bacbb4afe40c5a19eddb07fc9f98312ec69",
    csv_version_id="null",
    zip_version_id="null"
)
stimulus_set_registry['Malania2007.equal16'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_equal-16',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="14f9f7098831691811abf9953766951edc952203",
    zip_sha1="5127e88eaed1ef64247c7cb4262868533fb4ebae",
    csv_version_id="null",
    zip_version_id="null"
)
stimulus_set_registry['Malania2007.equal16_fit'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_equal-16_fit',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="2ff7f2f97250b9bcce3d2753be6e5b98e083892b",
    zip_sha1="db07ef4862fd9cb65c1e726cacc5914821296a5b",
    csv_version_id="null",
    zip_version_id="null"
)
stimulus_set_registry['Malania2007.long2'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_long-2',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="153b987c4c6b8a22efb88be26aaa46bd36912c9b",
    zip_sha1="07bb413d56ac77fc71823559780cdb16e4df563d",
    csv_version_id="null",
    zip_version_id="null"
)
stimulus_set_registry['Malania2007.long2_fit'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_long-2_fit',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="8b6d1557879e6271554c0fcb67bf6b2941dad2c8",
    zip_sha1="66205529af748ffd88579caef86b107838c0b0da",
    csv_version_id="null",
    zip_version_id="null"
)
stimulus_set_registry['Malania2007.long16'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_long-16',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="6c5d45b489bc290e41013d211d18570368012c9b",
    zip_sha1="10944e5b65e8da9e52087d3cbbdc6575538c2847",
    csv_version_id="null",
    zip_version_id="null"
)
stimulus_set_registry['Malania2007.long16_fit'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_long-16_fit',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="603dc8edb169f39e322f8980972eda1930c300ed",
    zip_sha1="a67330e18f1e5d9ad3829d8d8c000487fe3e4d48",
    csv_version_id="null",
    zip_version_id="null"
)
stimulus_set_registry['Malania2007.short2'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_short-2',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="c8bb84c5468a43348149afe24d5c0ebda233d54e",
    zip_sha1="1739226e7e32f60a7bb060fecc14c4a6353ca2ad",
    csv_version_id="null",
    zip_version_id="null"
)
stimulus_set_registry['Malania2007.short2_fit'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_short-2_fit',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="600c754811aa27d80a155c8ac643a81f2347ce3a",
    zip_sha1="a1a121dbbbf761caea0a086c2a74ab511f296ed5",
    csv_version_id="null",
    zip_version_id="null"
)
stimulus_set_registry['Malania2007.short4'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_short-4',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="181c912c03fdb3e4f89a737584a3a0816859f816",
    zip_sha1="820019a7f68db60fac11a5c5f3e42037cf205248",
    csv_version_id="null",
    zip_version_id="null"
)
stimulus_set_registry['Malania2007.short4_fit'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_short-4_fit',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="65af7b5d3845a7ea284aefba21734e1d298742c8",
    zip_sha1="5234b449c05a43e726543029137fe26930157b09",
    csv_version_id="null",
    zip_version_id="null"
)
stimulus_set_registry['Malania2007.short6'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_short-6',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="57813230e337a09c4c423da927c1f33e62054547",
    zip_sha1="dab58e8e996f91af643a0b61247e7ef87f35338d",
    csv_version_id="null",
    zip_version_id="null"
)
stimulus_set_registry['Malania2007.short6_fit'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_short-6_fit',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="ea7eb26b42fe9e4fc1ac2ed6e9bad439e8077ce1",
    zip_sha1="895e69c835b22b07ee66a0f5f53e7a108ac8287c",
    csv_version_id="null",
    zip_version_id="null"
)
stimulus_set_registry['Malania2007.short8'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_short-8',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="3df9a38605a4590eac8a1151779ba68c3cd54dc1",
    zip_sha1="7626364e0776b2809ae36d9cb713c6ff9a0d0c05",
    csv_version_id="null",
    zip_version_id="null"
)
stimulus_set_registry['Malania2007.short8_fit'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_short-8_fit',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="2782c818056b374e86195cbdb0ab1a52ef0d01da",
    zip_sha1="ec2fa2a261d965455ffa81acdb0fddef447ad4ff",
    csv_version_id="null",
    zip_version_id="null"
)
stimulus_set_registry['Malania2007.short16'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_short-16',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="9f5f4e3597006c50530017ce769c8689d43b06f5",
    zip_sha1="b67b1e70e8ba698907c95614bcb16eea6ff2f090",
    csv_version_id="null",
    zip_version_id="null"
)
stimulus_set_registry['Malania2007.short16_fit'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_short-16_fit',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="5bc0314a6c16095a70049fa5e8df5e9b94345f30",
    zip_sha1="0ca3930831ca926ea8b8c3600695b639ff05ddb5",
    csv_version_id="null",
    zip_version_id="null"
)
stimulus_set_registry['Malania2007.vernier_only'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_vernier-only',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="c71f654fccf1265a95dd0585c186232a2519e944",
    zip_sha1="eadff359975c3ba250224ce1942544b091415095",
    csv_version_id="null",
    zip_version_id="null"
)