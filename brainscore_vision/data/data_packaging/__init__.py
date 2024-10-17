from brainio.assemblies import NeuronRecordingAssembly
from brainscore_vision import load_stimulus_set
from brainscore_vision import stimulus_set_registry, data_registry
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3, load_stimulus_set_from_s3

stimulus_set_registry["DataGenerationTrial_emogan"] = lambda: load_stimulus_set_from_s3(
    identifier="DataGenerationTrial_emogan",
    bucket="brainio-brainscore",
    csv_version_id="GjSyD2fjkm.5aWrr0vyuUvCFZ.VMDS9P",
    csv_sha1="82111accffb561c614bdccd436aa921b60634822",
    zip_version_id="VQkg9F0VkYNDwVI0V9bL1EvEvYyqDJcX",
    zip_sha1="2af590c23b65d07f1d45c03ce0c4d66cb26b5f84",
    filename_prefix='stimulus_',
)

data_registry["DataGenerationTrial_emogan"] = lambda: load_assembly_from_s3(
    identifier="DataGenerationTrial_emogan",
    version_id="1d6qakjzMjjPENOUHinslBUs2YTTC.o7",
    sha1="ae4c5f8a28b432b6ec021aabd784caf0c2c69a38",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('DataGenerationTrial_emogan'),
)

    