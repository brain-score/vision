from brainio.assemblies import NeuronRecordingAssembly

from brainscore_vision import data_registry
from brainscore_vision.utils.s3 import load_assembly_from_s3
from brainscore_vision.data.data_helpers.helper import version_id_df, build_filename


# assembly: uses dicarlo.hvm
data_registry['dicarlo.Seibert2019'] = lambda: load_assembly_from_s3(
    identifier="dicarlo.Seibert2019",
    version_id=version_id_df.at[build_filename('dicarlo.Seibert2019', '.nc'), 'version_id'],
    sha1="eef41bb1f3d83c0e60ebf0e91511ce71ef5fee32",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly)
