from brainio.assemblies import NeuronRecordingAssembly

from brainscore_vision import data_registry, stimulus_set_registry
from brainscore_vision.utils.s3 import load_assembly_from_s3, load_stimulus_set_from_s3
from brainscore_vision.data.data_helpers.helper import version_id_df, build_filename

BIBTEX = """@article{david2004evaluation,
  title={Evaluation of different measures of functional connectivity using a neural mass model},
  author={David, Olivier and Cosmelli, Diego and Friston, Karl J},
  journal={Neuroimage},
  volume={21},
  number={2},
  pages={659--673},
  year={2004},
  publisher={Elsevier}
}"""

# extract version ids from version_ids csv
assembly_version = version_id_df.at[build_filename('gallant.David2004', '.nc'), 'version_id']
csv_version = version_id_df.at[build_filename('gallant.David2004', '.csv'), 'version_id']
zip_version = version_id_df.at[build_filename('gallant.David2004', '.zip'), 'version_id']

# assembly
data_registry['gallant.David2004'] = lambda: load_assembly_from_s3(
    identifier="gallant.David2004",
    version_id=assembly_version,
    sha1="d2ed9834c054da2333f5d894285c9841a1f27313",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly)

# stimulus set
stimulus_set_registry['gallant.David2004'] = lambda: load_stimulus_set_from_s3(
    identifier="gallant.David2004",
    bucket="brainio-brainscore",
    csv_sha1="8ec76338b998cadcdf1e57edd2dd992e2ab2355b",
    zip_sha1="0200421d66a0613946d39cab64c00b561160016e",
    csv_version_id=csv_version,
    zip_version_id=zip_version)
