from brainio.assemblies import NeuronRecordingAssembly

from brainscore_vision import data_registry
from brainscore_vision.utils.s3 import load_assembly_from_s3, load_stimulus_set_from_s3


# TODO: add correct version ids
# assembly
data_registry['gallant.David2004'] = lambda: load_assembly_from_s3(
    identifier="gallant.David2004",
    version_id="",
    sha1="d2ed9834c054da2333f5d894285c9841a1f27313",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly)

# stimulus set
data_registry['gallant.David2004'] = lambda: load_stimulus_set_from_s3(
    identifier="gallant.David2004",
    bucket="brainio-brainscore",
    csv_sha1="8ec76338b998cadcdf1e57edd2dd992e2ab2355b",
    zip_sha1="0200421d66a0613946d39cab64c00b561160016e",
    csv_version_id="",
    zip_version_id="")
