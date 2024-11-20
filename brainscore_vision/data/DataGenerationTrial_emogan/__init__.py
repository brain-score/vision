from brainio.assemblies import NeuronRecordingAssembly
from brainscore_vision import load_stimulus_set
from brainscore_vision import stimulus_set_registry, data_registry
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3, load_stimulus_set_from_s3

stimulus_set_registry["DataGenerationTrial_emogan"] = lambda: load_stimulus_set_from_s3(
    identifier="DataGenerationTrial_emogan",
    bucket="brainio-brainscore",
    csv_version_id="RBaEMjyI2SQMjKzCQQjkfcNhZZ1V772E",
    csv_sha1="650f0bdc062a28ad88ebaa8701bae20cab9ce6c0",
    zip_version_id="ucQy0k0GpwjSk9uELPntBrC4JnM9waG3",
    zip_sha1="a27a7f53e5ec6143a7fd2ac5555dda61abce88d9",
    filename_prefix='stimulus_',
)

data_registry["DataGenerationTrial_emogan"] = lambda: load_assembly_from_s3(
    identifier="DataGenerationTrial_emogan",
    version_id="dv4icML9fYv3.H7WsOGlomXdsEjdI2Rv",
    sha1="054e6f699c2d495adf9dca3d6eb0c7bda67d0eac",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('DataGenerationTrial_emogan'),
)

    