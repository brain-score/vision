from brainio.assemblies import NeuronRecordingAssembly
from brainscore_vision import load_stimulus_set
from brainscore_vision import stimulus_set_registry, data_registry
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3, load_stimulus_set_from_s3

stimulus_set_registry["DataGenerationTrial_emogan"] = lambda: load_stimulus_set_from_s3(
    identifier="DataGenerationTrial_emogan",
    bucket="brainio-brainscore",
    csv_version_id="xOZ3RdZsWxf2wiaN8mCMtWON2hufE9ta",
    csv_sha1="c3dbd5934190a33098f6f7e93589c910550a1fe2",
    zip_version_id="lzIEgzvJ0ZOhfOs5lo1LHPgJ4tgkWCfj",
    zip_sha1="9bfd9bca01c282c8c80621441e63529f43a7a05c",
    filename_prefix='stimulus_',
)

data_registry["DataGenerationTrial_emogan"] = lambda: load_assembly_from_s3(
    identifier="DataGenerationTrial_emogan",
    version_id="vo5913rE6aM3eON72lVA7zlzGu5rwzWR",
    sha1="7375f37b1318c31243d310e45e085f7daa1a7ff1",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('DataGenerationTrial_emogan'),
)

    