from brainio.assemblies import NeuronRecordingAssembly
from brainscore_vision import load_stimulus_set
from brainscore_vision import stimulus_set_registry, data_registry
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3, load_stimulus_set_from_s3

stimulus_set_registry["DataGenerationTrial_emogan"] = lambda: load_stimulus_set_from_s3(
    identifier="DataGenerationTrial_emogan",
    bucket="brainio-brainscore",
    csv_version_id="8XR3kofwmN91uNffyGWU32mER3JfFfbl",
    csv_sha1="b5c180e21dcd898bf4cf8c037b3f48b6b398f7f6",
    zip_version_id="l8i_0JL3LoAxZbMJlnAgpSywRf39WLi.",
    zip_sha1="a86a0379f2ac6ea998fc37a71cf22fcb85c0ed47",
    filename_prefix='stimulus_',
)

data_registry["DataGenerationTrial_emogan"] = lambda: load_assembly_from_s3(
    identifier="DataGenerationTrial_emogan",
    version_id="XSwc5VJ88v32_OrTQcc4ndIZZjLH0R7n",
    sha1="008960c01accc72212dcb40d48faedee819014a2",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('DataGenerationTrial_emogan'),
)

    