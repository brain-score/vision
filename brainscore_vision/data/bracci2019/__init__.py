from brainio.assemblies import NeuroidAssembly
from brainscore_vision import stimulus_set_registry, data_registry
from brainscore_vision.data_helpers.s3 import (
    load_assembly_from_s3,
    load_stimulus_set_from_s3,
)

BIBTEX = """"@article{bracci2019ventral,
  title={The ventral visual pathway represents animal appearance over animacy, unlike human behavior and deep neural networks},
  author={Bracci, Stefania and Ritchie, J Brendan and Kalfas, Ioannis and de Beeck, Hans P Op},
  journal={Journal of Neuroscience},
  volume={39},
  number={33},
  pages={6513--6525},
  year={2019},
  publisher={Soc Neuroscience}
}"""

# Human Stimulus Set
stimulus_set_registry["Bracci2019"] = lambda: load_stimulus_set_from_s3(
    identifier="Bracci2019",
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="05b1af9b8e6ed478ea49339e11b0024c2da8c35f",
    zip_sha1="a79b249e758421f46ec781301cd4b498f64853ce",
    csv_version_id="null",
    zip_version_id="null",
)

# Human Data Assembly (brain)
data_registry["Bracci2019"] = lambda: load_assembly_from_s3(
    identifier="Bracci2019",
    version_id="null",
    sha1="cbec165bb20f09d0527fddba7cfbf115a396a2f3",
    bucket="brainscore-storage/brainio-brainscore",
    cls=NeuroidAssembly,
)
