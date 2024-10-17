from brainio.assemblies import NeuronRecordingAssembly
from brainscore_vision import load_stimulus_set
from brainscore_vision import stimulus_set_registry, data_registry
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3, load_stimulus_set_from_s3

stimulus_set_registry["DataGenerationTrial_emogan"] = lambda: load_stimulus_set_from_s3(
    identifier="DataGenerationTrial_emogan",
    bucket="brainio-brainscore",
    csv_version_id="IQuBOlnXIrf4R2i168W73GB4vo7ZOdCl",
    csv_sha1="c1b9d500f67cf8ab1afadc173de97095f7bf381a",
    zip_version_id="Ro2DVDP0.DWYY_bBiKRRnG5_.4K_3YUX",
    zip_sha1="2af590c23b65d07f1d45c03ce0c4d66cb26b5f84",
    filename_prefix='stimulus_',
)

data_registry["DataGenerationTrial_emogan"] = lambda: load_assembly_from_s3(
    identifier="DataGenerationTrial_emogan",
    version_id="5.4XH2YcAXxhe_Hys6kmSDue1sXLIXxb",
    sha1="62039806447acb8fcde45f2bf0debc6d8cb52010",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('DataGenerationTrial_emogan'),
)

    