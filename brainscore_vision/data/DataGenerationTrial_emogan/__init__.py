from brainio.assemblies import NeuronRecordingAssembly
from brainscore_vision import load_stimulus_set
from brainscore_vision import stimulus_set_registry, data_registry
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3, load_stimulus_set_from_s3

stimulus_set_registry["DataGenerationTrial_emogan"] = lambda: load_stimulus_set_from_s3(
    identifier="DataGenerationTrial_emogan",
    bucket="brainio-brainscore",
    csv_version_id="Qq7dNhtGWMyVlLllhmsMam7qSjzxDsSR",
    csv_sha1="3cb3dd389fc6397d75af0445564cf7ce10ce33ce",
    zip_version_id="f6BQWLIMj9dr6g4ehV1dzVdSvUeYpMql",
    zip_sha1="4e9bc6f15993ae652445ed53b0ec235448a932fd",
    filename_prefix='stimulus_',
)

data_registry["DataGenerationTrial_emogan"] = lambda: load_assembly_from_s3(
    identifier="DataGenerationTrial_emogan",
    version_id="mmrwu.hRTxkCLZAGxQlI1QWgVZxcKCMY",
    sha1="a962aa6975e9bad1032136a135167d2832f0db13",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('DataGenerationTrial_emogan'),
)

    