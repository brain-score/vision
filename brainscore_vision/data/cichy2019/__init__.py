from brainio.assemblies import DataAssembly
from brainscore_vision import data_registry, stimulus_set_registry
from brainscore_vision.utils.s3 import load_assembly_from_s3, load_stimulus_set_from_s3
from brainscore_vision.data.data_helpers.helper import version_id_df, build_filename

# extract version ids from version_ids csv
assembly_version = version_id_df.at[build_filename('aru.Cichy2019', '.nc'), 'version_id']
csv_version = version_id_df.at[build_filename('aru.Cichy2019', '.csv'), 'version_id']
zip_version = version_id_df.at[build_filename('aru.Cichy2019', '.zip'), 'version_id']

# assembly
data_registry['aru.Cichy2019'] = lambda: load_assembly_from_s3(
    identifier="aru.Cichy2019",
    version_id=assembly_version,
    sha1="701e63be62b642082d476244d0d91d510b3ff05d",
    bucket="brainio-brainscore",
    cls=DataAssembly)

# stimulus set
stimulus_set_registry['aru.Cichy2019'] = lambda: load_stimulus_set_from_s3(
    identifier="aru.Cichy2019",
    bucket="brainio-brainscore",
    csv_sha1="281c4d9d0dd91a2916674638098fe94afb87d29a",
    zip_sha1="d2166dd9c2720cb24bc520f5041e6830779c0240",
    csv_version_id=csv_version,
    zip_version_id=zip_version)
