import logging

from brainio.assemblies import NeuronRecordingAssembly

from brainscore_vision import data_registry, stimulus_set_registry
from brainscore_vision.utils.s3 import load_assembly_from_s3, load_stimulus_set_from_s3
from brainscore_vision.data.data_helpers.helper import version_id_df, build_filename

_logger = logging.getLogger(__name__)

BIBTEX = """"""


# assembly: uses below stimulus set
data_registry['dicarlo.Rajalingham2020'] = lambda: load_assembly_from_s3(
    identifier="dicarlo.Rajalingham2020",
    version_id=version_id_df.at[build_filename('dicarlo.Rajalingham2020', '.nc'), 'version_id'],
    sha1="ab95ae6c9907438f87b9b13b238244049f588680",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly)

# stimulus set
stimulus_set_registry['dicarlo.Rajalingham2020'] = lambda: load_stimulus_set_from_s3(
    identifier="dicarlo.Rajalingham2020",
    bucket="brainio-brainscore",
    csv_sha1="9a9a6b3115d2d8ce5d54ec2522093d8a87ed13a0",
    zip_sha1="6097086901032e20f8ae764e9cc06e0a891a3e18",
    csv_version_id=version_id_df.at[build_filename('dicarlo.Rajalingham2020', '.csv'), 'version_id'],
    zip_version_id=version_id_df.at[build_filename('dicarlo.Rajalingham2020', '.zip'), 'version_id'])

# stimulus set: orthographic_IT
stimulus_set_registry['dicarlo.Rajalingham2020orthographic_IT'] = lambda: load_stimulus_set_from_s3(
    identifier="dicarlo.Rajalingham2020orthographic_IT",
    bucket="brainio-brainscore",
    csv_sha1="3ac9ab73b653ac9cf839f9bfde131354a3766ccd",
    zip_sha1="0e025f6f8b06e803a6d8d1a17bd25a41af3e81db",
    csv_version_id=version_id_df.at[build_filename('dicarlo.Rajalingham2020orthographic_IT', '.csv'), 'version_id'],
    zip_version_id=version_id_df.at[build_filename('dicarlo.Rajalingham2020orthographic_IT', '.zip'), 'version_id'])
