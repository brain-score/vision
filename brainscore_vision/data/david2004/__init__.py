from brainio.assemblies import NeuronRecordingAssembly

from brainscore_vision import data_registry, stimulus_set_registry
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3, load_stimulus_set_from_s3

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

# assembly
data_registry['gallant.David2004'] = lambda: load_assembly_from_s3(
    identifier="gallant.David2004",
    version_id="8getDVrrr1iT0DA385T8ZdSzCcuM3_m0",
    sha1="d2ed9834c054da2333f5d894285c9841a1f27313",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly)

# stimulus set
stimulus_set_registry['gallant.David2004'] = lambda: load_stimulus_set_from_s3(
    identifier="gallant.David2004",
    bucket="brainio-brainscore",
    csv_sha1="8ec76338b998cadcdf1e57edd2dd992e2ab2355b",
    zip_sha1="0200421d66a0613946d39cab64c00b561160016e",
    csv_version_id="0Ks2opc6t_IpZdB.kylVCbYgXV6ADB4O",
    zip_version_id="hFTILhFSGqL95wp1li1aBzzbqCyuAlb_")
