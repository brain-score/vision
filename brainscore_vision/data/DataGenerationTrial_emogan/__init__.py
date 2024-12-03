from brainio.assemblies import NeuronRecordingAssembly
from brainscore_vision import load_stimulus_set
from brainscore_vision import stimulus_set_registry, data_registry
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3, load_stimulus_set_from_s3

stimulus_set_registry["DataGenerationTrial_emogan"] = lambda: load_stimulus_set_from_s3(
    identifier="DataGenerationTrial_emogan",
    bucket="brainio-brainscore",
    csv_version_id="1xandU4vs7JoiBkk7x6SqWfzGvxLkqHY",
    csv_sha1="676c3e865b3f42fcd0325362258096c39441d63d",
    zip_version_id="wOY97c.2x.rM8DFmDhp0lwJinMOD25_W",
    zip_sha1="17fa99659bfee762cc2b5128003cb02d86475cb0",
    filename_prefix='stimulus_',
)

data_registry["DataGenerationTrial_emogan"] = lambda: load_assembly_from_s3(
    identifier="DataGenerationTrial_emogan",
    version_id="0joVv8WNIUyQDZ_OmlXpB6y86gsB9FtV",
    sha1="9acfb2412c32e23b326838b0c6ad13f639536ecf",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('DataGenerationTrial_emogan'),
)

    