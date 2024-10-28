from brainio.assemblies import NeuronRecordingAssembly
from brainscore_vision import load_stimulus_set
from brainscore_vision import stimulus_set_registry, data_registry
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3, load_stimulus_set_from_s3

stimulus_set_registry["DataGenerationTrial_emogan"] = lambda: load_stimulus_set_from_s3(
    identifier="DataGenerationTrial_emogan",
    bucket="brainio-brainscore",
    csv_version_id="AbwXndwqg6MwkkKGR5v8gKGz0noCcNRb",
    csv_sha1="9a8623eb0fe33889758146b0272754b9a5222125",
    zip_version_id="oqd8tpdsr0y6g3JrBPv_j3R6NwOOEx7n",
    zip_sha1="a69ddd88cf130b91a4ffaa48926fa98b67efbca6",
    filename_prefix='stimulus_',
)

data_registry["DataGenerationTrial_emogan"] = lambda: load_assembly_from_s3(
    identifier="DataGenerationTrial_emogan",
    version_id="6a7NvJssAUUzoj3suKZVHdpNvHOI4Cr.",
    sha1="7e053617cecf99b0565c738db6db81fef8d71f1c",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('DataGenerationTrial_emogan'),
)

    