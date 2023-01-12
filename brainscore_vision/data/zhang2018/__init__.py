from brainio.assemblies import BehavioralAssembly

from brainscore_vision import data_registry, stimulus_set_registry
from brainscore_vision.utils.s3 import load_assembly_from_s3, load_stimulus_set_from_s3
from brainscore_vision.data.data_helpers.helper import version_id_df, build_filename

BIBTEX = """"""

# extract version ids from version_ids csv
assembly_version = version_id_df.at[build_filename('klab.Zhang2018search_obj_array', '.nc'), 'version_id']
csv_version = version_id_df.at[build_filename('klab.Zhang2018search_obj_array', '.csv'), 'version_id']
zip_version = version_id_df.at[build_filename('klab.Zhang2018search_obj_array', '.zip'), 'version_id']

# assembly: klab.Zhang2018search_obj_array
data_registry['klab.Zhang2018search_obj_array'] = lambda: load_assembly_from_s3(
    identifier="klab.Zhang2018search_obj_array",
    version_id=assembly_version,
    sha1="84400dc814b79df8d8eca99a557df5f741cce4b9",
    bucket="brainio-brainscore",
    cls=BehavioralAssembly)

# stimulus set: klab.Zhang2018.search_obj_array
stimulus_set_registry['klab.Zhang2018.search_obj_array'] = lambda: load_stimulus_set_from_s3(
    identifier="klab.Zhang2018.search_obj_array",
    bucket="brainio-brainscore",
    csv_sha1="e9c2f6b35b84256242d257e8d36261aa26d3ed4a",
    zip_sha1="d92dfb20a87bb9a015c77c2862a215e8fa4f2cc3",
    csv_version_id=csv_version,
    zip_version_id=zip_version)
