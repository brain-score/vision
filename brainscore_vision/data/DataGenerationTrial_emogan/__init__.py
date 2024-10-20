from brainio.assemblies import NeuronRecordingAssembly
from brainscore_vision import load_stimulus_set
from brainscore_vision import stimulus_set_registry, data_registry
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3, load_stimulus_set_from_s3

stimulus_set_registry["DataGenerationTrial_emogan"] = lambda: load_stimulus_set_from_s3(
    identifier="DataGenerationTrial_emogan",
    bucket="brainio-brainscore",
    csv_version_id="uaMuF5sFgQzj2WTDHiCDuX5Zzb13M3as",
    csv_sha1="a26c9089672837dd7f5186c7c42c84d9805a3965",
    zip_version_id="flAOeJG4aLNcdEErZ5XhI1uW3ucwqVcS",
    zip_sha1="dd39ddb1fd053151793e3f9f6e85b664cc4636bc",
    filename_prefix='stimulus_',
)

data_registry["DataGenerationTrial_emogan"] = lambda: load_assembly_from_s3(
    identifier="DataGenerationTrial_emogan",
    version_id="TJ4EIBPvnpxITkljkkjNNsAxdRDvRPqr",
    sha1="14a331c49da73ee1838e9b5e85c76073288f601e",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('DataGenerationTrial_emogan'),
)

    