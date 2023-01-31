from brainio.assemblies import NeuronRecordingAssembly

from brainscore_vision import data_registry
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3

BIBTEX = """@article{lee2019sensitivity,
  title={Sensitivity of inferotemporal cortex to naturalistic image statistics in developing macaques},
  author={Lee, Gerick M and Seibert, Darren A and Majaj, Najib J and Movshon, J Anthony and Kiorpes, Lynne},
  journal={Journal of Vision},
  volume={19},
  number={10},
  pages={124--124},
  year={2019},
  publisher={The Association for Research in Vision and Ophthalmology}
}"""

# assembly: uses dicarlo.hvm
data_registry['dicarlo.Seibert2019'] = lambda: load_assembly_from_s3(
    identifier="dicarlo.Seibert2019",
    version_id="vSUte2bpVhGxPRKocfk_MP0s3LAPElBi",
    sha1="eef41bb1f3d83c0e60ebf0e91511ce71ef5fee32",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly)
