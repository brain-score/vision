from brainio.assemblies import NeuronRecordingAssembly
from brainscore_vision import load_stimulus_set
from brainscore_vision import stimulus_set_registry, data_registry
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3, load_stimulus_set_from_s3

stimulus_set_registry["DataGenerationTrial_emogan"] = lambda: load_stimulus_set_from_s3(
    identifier="DataGenerationTrial_emogan",
    bucket="brainio-brainscore",
    csv_version_id="T1Vo2jzM8ykidRVWEWS3E8xgwpCnkcQA",
    csv_sha1="c02122b1715b20318958d4ded3b6504e15bf098f",
    zip_version_id="j72kF0ZPJqPcfa3N1nBk18nq7yWvivTG",
    zip_sha1="d90dd5eec980645598c6185ca673a5d0edf45b09",
    filename_prefix='stimulus_',
)

data_registry["DataGenerationTrial_emogan"] = lambda: load_assembly_from_s3(
    identifier="DataGenerationTrial_emogan",
    version_id="XmeT8syk97w6m7mZe8NlhOso428zl0qN",
    sha1="97a03a6b74dd1b8a68d2235e33136e65686aaf89",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('DataGenerationTrial_emogan'),
)

    