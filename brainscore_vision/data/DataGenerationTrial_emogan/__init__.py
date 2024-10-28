from brainio.assemblies import NeuronRecordingAssembly
from brainscore_vision import load_stimulus_set
from brainscore_vision import stimulus_set_registry, data_registry
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3, load_stimulus_set_from_s3

stimulus_set_registry["DataGenerationTrial_emogan"] = lambda: load_stimulus_set_from_s3(
    identifier="DataGenerationTrial_emogan",
    bucket="brainio-brainscore",
    csv_version_id="RI6zBB1xCvxVS4DP5PbA2_abypKw81Wf",
    csv_sha1="ba6b9b922ecd7127256022fe8dae064fbc4c1446",
    zip_version_id="HJG1qUL7XuKWbHwkP5_v4Ep.tPmNt8iW",
    zip_sha1="ea38855add7d48850520c5e9de9cd5806d6e0c65",
    filename_prefix='stimulus_',
)

data_registry["DataGenerationTrial_emogan"] = lambda: load_assembly_from_s3(
    identifier="DataGenerationTrial_emogan",
    version_id="lAzVjn4lmMPfTjQrd2_sw8_blW_1EWQF",
    sha1="4f6c5e16cf652c2005f15c9361a6c2ee30510f4b",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('DataGenerationTrial_emogan'),
)

    