from brainio.assemblies import NeuronRecordingAssembly
from brainscore_vision import load_stimulus_set
from brainscore_vision import stimulus_set_registry, data_registry
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3, load_stimulus_set_from_s3

stimulus_set_registry["DataGenerationTrial_emogan"] = lambda: load_stimulus_set_from_s3(
    identifier="DataGenerationTrial_emogan",
    bucket="brainio-brainscore",
    csv_version_id="P6Mghve_6LmJ94TJMnaQDaTjVBMcYvKA",
    csv_sha1="4a1a644b8218fb963f7442af97a26384807e5ee5",
    zip_version_id="gRQdEw8XH744lB_u04eBhzXXIgFpgCYk",
    zip_sha1="731d4a5d97c507f115e99ad27b555cbd56066df3",
    filename_prefix='stimulus_',
)

data_registry["DataGenerationTrial_emogan"] = lambda: load_assembly_from_s3(
    identifier="DataGenerationTrial_emogan",
    version_id="7WxsklTrW8SYqneQc5oyp0jhbN4r5u98",
    sha1="f994d1d11d96b813f2e0ea44c8d7cf116cbcfb1b",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('DataGenerationTrial_emogan'),
)

    