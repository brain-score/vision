from brainio.assemblies import NeuronRecordingAssembly
from brainscore_vision import load_stimulus_set
from brainscore_vision import stimulus_set_registry, data_registry
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3, load_stimulus_set_from_s3

stimulus_set_registry["DataGenerationTrial_emogan"] = lambda: load_stimulus_set_from_s3(
    identifier="DataGenerationTrial_emogan",
    bucket="brainio-brainscore",
    csv_version_id="_EHGPGEdM43l8TGrRKeRFJYzj.YegrGg",
    csv_sha1="0a42f0a37cf4371afe3876201805872866befea4",
    zip_version_id="cFH2RwBwDNQjApAos5B2XaUFEPoJI4fU",
    zip_sha1="70c0bc261c799eff8c8662f20f3da9b8e3b2ece6",
    filename_prefix='stimulus_',
)

data_registry["DataGenerationTrial_emogan"] = lambda: load_assembly_from_s3(
    identifier="DataGenerationTrial_emogan",
    version_id="VBsgPnh5Lr0F9jo4iE_7VDJfdiUSHYR1",
    sha1="c18d8237b20590fbf0f1175059755645a94db259",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('DataGenerationTrial_emogan'),
)

    