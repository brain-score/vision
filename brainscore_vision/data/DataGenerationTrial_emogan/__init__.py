from brainio.assemblies import NeuronRecordingAssembly
from brainscore_vision import load_stimulus_set
from brainscore_vision import stimulus_set_registry, data_registry
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3, load_stimulus_set_from_s3

stimulus_set_registry["DataGenerationTrial_emogan"] = lambda: load_stimulus_set_from_s3(
    identifier="DataGenerationTrial_emogan",
    bucket="brainio-brainscore",
    csv_version_id="kxwsgksq9NlcbFRS.Y3Dce_OOqbf8T6s",
    csv_sha1="cf2c791439cbb8124ce5a3037f7e220c46fd7984",
    zip_version_id="X4ZKRgNz7Zkzp1JeewpdTNtH_RP72y4h",
    zip_sha1="b1413686cfa58889be09d34ceac7252f9558985f",
    filename_prefix='stimulus_',
)

data_registry["DataGenerationTrial_emogan"] = lambda: load_assembly_from_s3(
    identifier="DataGenerationTrial_emogan",
    version_id="QZOSZhyoTz5Nl1fzi7jwdbM_vgjqsHGa",
    sha1="ee44af87b60c1228cc0ec2e0c41d624a5af9369e",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('DataGenerationTrial_emogan'),
)

    