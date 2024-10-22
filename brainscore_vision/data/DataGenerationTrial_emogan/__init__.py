from brainio.assemblies import NeuronRecordingAssembly
from brainscore_vision import load_stimulus_set
from brainscore_vision import stimulus_set_registry, data_registry
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3, load_stimulus_set_from_s3

stimulus_set_registry["DataGenerationTrial_emogan"] = lambda: load_stimulus_set_from_s3(
    identifier="DataGenerationTrial_emogan",
    bucket="brainio-brainscore",
    csv_version_id=".jW9YD.GEKkh.E1PHDUR6DXM.cFrRpb7",
    csv_sha1="1cd591c3b55fb8304f78f122732a6988d0eb01d3",
    zip_version_id="isvd_2j9RNTIf0dsflctQuDcjcu2UMf1",
    zip_sha1="5f1d7f660c6e15e11f0ea722bcf7446b65ef31bf",
    filename_prefix='stimulus_',
)

data_registry["DataGenerationTrial_emogan"] = lambda: load_assembly_from_s3(
    identifier="DataGenerationTrial_emogan",
    version_id="1QV9tP8wujcaUGFkalvfM12M9qf05BNy",
    sha1="829e7474ee8c8f5d7601b1a61ba27d9360130269",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('DataGenerationTrial_emogan'),
)

    