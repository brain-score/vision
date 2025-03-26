from brainio.assemblies import NeuronRecordingAssembly
from brainscore_vision import load_stimulus_set
from brainscore_vision import stimulus_set_registry, data_registry
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3, load_stimulus_set_from_s3

stimulus_set_registry["DataGenerationTrial_IAPS"] = lambda: load_stimulus_set_from_s3(
    identifier="DataGenerationTrial_IAPS",
    bucket="brainio-brainscore",
    csv_version_id="VSoXKN2pWMj38wBMqDtm0AcXjsZUERqq",
    csv_sha1="69b21751264a3d7b91ac636823c69c8db5ca484c",
    zip_version_id="MBuo5WbqLLLBESKnxUo7GJ0Q4zawwsA_",
    zip_sha1="f3540467e5b9226977f201d355927c924d0c5c6b",
    filename_prefix='stimulus_',
)

data_registry["DataGenerationTrial_IAPS"] = lambda: load_assembly_from_s3(
    identifier="DataGenerationTrial_IAPS",
    version_id="Fl9xa1bevPQJvofxpM_LzGGtRLK0Swyn",
    sha1="3f882458bfdc7b2e1e77d8bb854561000d650a8c",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('DataGenerationTrial_IAPS'),
)

    