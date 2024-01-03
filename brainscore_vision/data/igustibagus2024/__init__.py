import brainscore_vision
from brainio.assemblies import NeuronRecordingAssembly
from brainscore_vision import stimulus_set_registry, data_registry
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3, load_stimulus_set_from_s3

stimulus_set_registry['Igustibagus2024'] = lambda: load_stimulus_set_from_s3(
    identifier="Igustibagus2024",
    bucket="brainio-brainscore",
    csv_version_id="leBGJAufzrG6C7Dw.0QKAmzAI3QD93_a",
    csv_sha1="df68b579e564eadee486212a0528dce0957828d9",
    zip_version_id="_WwQpLoSo4WfwAxvoMEpUcxQEEadz3Nr",
    zip_sha1="9aa13d94da49fb7535ef14115633fc58a4810a61",
    filename_prefix='stimulus_',
)

data_registry['Igustibagus2024'] = lambda: load_assembly_from_s3(
    identifier="Igustibagus2024",
    version_id="B8dCcfaL3Pkt0EDp8fPYY5XKxgkp.TUk",
    sha1="eb180ddd1ae0144b5927fab00fd62ecd1aa98003",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: brainscore_vision.load_stimulus_set('Igustibagus2024'),
)
#