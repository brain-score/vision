import logging

from brainio.assemblies import NeuronRecordingAssembly

from brainscore_vision import data_registry, stimulus_set_registry, load_stimulus_set
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3, load_stimulus_set_from_s3

_logger = logging.getLogger(__name__)

BIBTEX = """@article{rajalingham2020inferior,
  title={The inferior temporal cortex is a potential cortical precursor of orthographic processing in untrained monkeys},
  author={Rajalingham, Rishi and Kar, Kohitij and Sanghavi, Sachi and Dehaene, Stanislas and DiCarlo, James J},
  journal={Nature communications},
  volume={11},
  number={1},
  pages={1--13},
  year={2020},
  publisher={Nature Publishing Group}
}"""

# assembly: uses below stimulus set
data_registry['Rajalingham2020'] = lambda: load_assembly_from_s3(
    identifier="dicarlo.Rajalingham2020",
    version_id="L4YlA5o2gToDj4sbXE0Utn362sPyy_GW",
    sha1="ab95ae6c9907438f87b9b13b238244049f588680",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Rajalingham2020'),
)

# stimulus set
stimulus_set_registry['Rajalingham2020'] = lambda: load_stimulus_set_from_s3(
    identifier="dicarlo.Rajalingham2020",
    bucket="brainio-brainscore",
    csv_sha1="9a9a6b3115d2d8ce5d54ec2522093d8a87ed13a0",
    zip_sha1="6097086901032e20f8ae764e9cc06e0a891a3e18",
    csv_version_id="kPZEA.xmFrtYesjID0KbTZxuuD_LCu4M",
    zip_version_id="eTXlzNCSlwK.Lm6Smh.YJe2KmiaW_QNY")

