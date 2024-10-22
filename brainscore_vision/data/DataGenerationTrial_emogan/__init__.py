from brainio.assemblies import NeuronRecordingAssembly
from brainscore_vision import load_stimulus_set
from brainscore_vision import stimulus_set_registry, data_registry
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3, load_stimulus_set_from_s3

stimulus_set_registry["DataGenerationTrial_emogan"] = lambda: load_stimulus_set_from_s3(
    identifier="DataGenerationTrial_emogan",
    bucket="brainio-brainscore",
    csv_version_id="FLFycHAOMQn_z4hGgfjLOw1lrEa6rojz",
    csv_sha1="7eb497b3d55221e6606f2de996b4af802274e124",
    zip_version_id="t9ozrBNgsSuMMKAiQeeOwiVnQ7xKOXG1",
    zip_sha1="6a773a737fad71198caf28b6981cbee43bfc4287",
    filename_prefix='stimulus_',
)

data_registry["DataGenerationTrial_emogan"] = lambda: load_assembly_from_s3(
    identifier="DataGenerationTrial_emogan",
    version_id="B7xHJknb158lS.wzFV24dE8TlJ_Cxhnn",
    sha1="9f8b1e84345f97785491ac0c5ef1957fb8350f5f",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('DataGenerationTrial_emogan'),
)

    