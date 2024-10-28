from brainio.assemblies import NeuronRecordingAssembly
from brainscore_vision import load_stimulus_set
from brainscore_vision import stimulus_set_registry, data_registry
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3, load_stimulus_set_from_s3

stimulus_set_registry["DataGenerationTrial_emogan"] = lambda: load_stimulus_set_from_s3(
    identifier="DataGenerationTrial_emogan",
    bucket="brainio-brainscore",
    csv_version_id="eSkkdLTpUBKFdag.Xz_6Uq_HUo3CiN8c",
    csv_sha1="087229269f5cfc9018009b8020065567ee366b54",
    zip_version_id="eof2xRz5l9IEmnnjmBT2VH1Y1ruShiDJ",
    zip_sha1="530b472fa9dd2008a8911364b1725df56f8c45bc",
    filename_prefix='stimulus_',
)

data_registry["DataGenerationTrial_emogan"] = lambda: load_assembly_from_s3(
    identifier="DataGenerationTrial_emogan",
    version_id="_mJCWVxrROgzjRSlkH7kFgKW0PmvLbwx",
    sha1="0337d87832a0b38960a909fc53188a9f93e1b459",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('DataGenerationTrial_emogan'),
)

    