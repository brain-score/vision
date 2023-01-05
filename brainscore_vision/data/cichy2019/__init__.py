from brainio.assemblies import DataAssembly
from brainscore_vision import data_registry
from brainscore_vision.utils.s3 import load_assembly_from_s3, load_stimulus_set_from_s3

# TODO: add correct version id
# assembly
data_registry['aru.Cichy2019'] = lambda: load_assembly_from_s3(
    identifier="aru.Cichy2019",
    version_id="",
    sha1="701e63be62b642082d476244d0d91d510b3ff05d",
    bucket="brainio.contrib",
    cls=DataAssembly)

# stimulus set
data_registry['aru.Cichy2019'] = lambda: load_stimulus_set_from_s3(
    identifier="aru.Cichy2019",
    bucket="brainio.contrib",
    csv_sha1="281c4d9d0dd91a2916674638098fe94afb87d29a",
    zip_sha1="d2166dd9c2720cb24bc520f5041e6830779c0240",
    csv_version_id="",
    zip_version_id="")
