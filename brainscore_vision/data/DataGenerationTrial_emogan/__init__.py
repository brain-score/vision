from brainio.assemblies import NeuronRecordingAssembly
from brainscore_vision import load_stimulus_set
from brainscore_vision import stimulus_set_registry, data_registry
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3, load_stimulus_set_from_s3

stimulus_set_registry["DataGenerationTrial_emogan"] = lambda: load_stimulus_set_from_s3(
    identifier="DataGenerationTrial_emogan",
    bucket="brainio-brainscore",
    csv_version_id="brt18aIOtgDZb9QVrna_BuKyDdlbamqe",
    csv_sha1="d9d91569770ff69e23fa870bd8325f50d3ce4944",
    zip_version_id="MV_msFUe86Asm1Dv2pTM8A_QujAwuKY5",
    zip_sha1="37e6eb1ec48439630d10607c6a1253792a41293b",
    filename_prefix='stimulus_',
)

data_registry["DataGenerationTrial_emogan"] = lambda: load_assembly_from_s3(
    identifier="DataGenerationTrial_emogan",
    version_id="C3ni9mcCCVDWpaM.gqcUMOFLuUfMOc6l",
    sha1="071b8d0d2c7f9adf242442aed9ceb4d2dc34cf92",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('DataGenerationTrial_emogan'),
)

    