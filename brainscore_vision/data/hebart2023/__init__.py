from brainio.assemblies import NeuronRecordingAssembly
from brainscore_vision import load_stimulus_set
from brainscore_vision import stimulus_set_registry, data_registry
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3, load_stimulus_set_from_s3

stimulus_set_registry['Hebart2023'] = lambda: load_stimulus_set_from_s3(
    identifier="Hebart2023",
    bucket="brainio-brainscore",
    csv_version_id="rHfHstbIZesJuXguqTpO3kAsDdRJ8FY_",
    csv_sha1="8fd5a8d4d68cc206000878dd835829fd14a426e8",
    zip_version_id="Vo4af.yuEzGR8d.tHtKzeQNAigThiiyE",
    zip_sha1="f244ebfe07c98470885026188c801857ba2ec0ea",
    filename_prefix='stimulus_',
)

data_registry['Hebart2023'] = lambda: load_assembly_from_s3(
    identifier="Hebart2023",
    version_id="nO4rlBtVj1agBGyrguZfLsAxJ9qDIzfy",
    sha1="90f66d37c202dcd1c74d82854efd4a1e8c5fe82e",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Hebart2023'),
)
