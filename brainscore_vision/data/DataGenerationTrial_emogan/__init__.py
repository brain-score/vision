from brainio.assemblies import NeuronRecordingAssembly
from brainscore_vision import load_stimulus_set
from brainscore_vision import stimulus_set_registry, data_registry
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3, load_stimulus_set_from_s3

stimulus_set_registry["DataGenerationTrial_emogan"] = lambda: load_stimulus_set_from_s3(
    identifier="DataGenerationTrial_emogan",
    bucket="brainio-brainscore",
    csv_version_id="cg4XIjcWGiON0wXxBMksQN_J4LpZx2Ww",
    csv_sha1="06c40eeaba263dce468ac8c34e1c12de211347d7",
    zip_version_id="zCCGfouNrS7kXhZZs.HmDyO4axAqVeum",
    zip_sha1="375056e59531943d0a9a066786b4def366d90147",
    filename_prefix='stimulus_',
)

data_registry["DataGenerationTrial_emogan"] = lambda: load_assembly_from_s3(
    identifier="DataGenerationTrial_emogan",
    version_id="28bqnbHXCI_d.x2NV6L3pA_bK6AAuBJg",
    sha1="0c9f188bb730afc6e1ebe1eeef8f6aefe70b291b",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('DataGenerationTrial_emogan'),
)

    