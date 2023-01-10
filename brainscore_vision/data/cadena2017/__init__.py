import logging

from brainio.assemblies import NeuronRecordingAssembly
from brainscore_vision import data_registry
from brainscore_vision.utils.s3 import load_assembly_from_s3, load_stimulus_set_from_s3
from brainscore_vision.data.data_helpers.helper import version_id_df, build_filename

# extract version ids from version_ids csv
assembly_version = version_id_df.at[build_filename('tolias.Cadena2017', '.nc'), 'version_id']
csv_version = version_id_df.at[build_filename('tolias.Cadena2017', '.csv'), 'version_id']
zip_version = version_id_df.at[build_filename('tolias.Cadena2017', '.zip'), 'version_id']

_logger = logging.getLogger(__name__)

BIBTEX = """"""

# assembly
data_registry['tolias.Cadena2017'] = lambda: load_assembly_from_s3(
    identifier="tolias.Cadena2017",
    version_id=assembly_version,
    sha1="69bcaaa9370dceb0027beaa06235ef418c3d7063",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly)

# stimulus set
data_registry['tolias.Cadena2017'] = lambda: load_stimulus_set_from_s3(
    identifier="tolias.Cadena2017",
    bucket="brainio-brainscore",
    csv_sha1="f55b174cc4540e5612cfba5e695324328064b051",
    zip_sha1="88cc2ce3ef5e197ffd1477144a2e6a68d424ef6c",
    csv_version_id=csv_version,
    zip_version_id=zip_version)
