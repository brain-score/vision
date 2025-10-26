from brainio.assemblies import NeuronRecordingAssembly

import brainscore_vision
from brainscore_vision import data_registry, stimulus_set_registry
from brainscore_vision.data_helpers.s3 import load_stimulus_set_from_s3, load_assembly_from_s3

BIBTEX = """@article{keles2024multimodal,
  title={Multimodal single-neuron, intracranial EEG, and fMRI brain responses during movie watching in human patients},
  author={Keles, Umit and Dubois, Julien and Le, Kevin JM and Tyszka, J Michael and Kahn, David A and Reed, Chrystal M and Chung, Jeffrey M and Mamelak, Adam N and Adolphs, Ralph and Rutishauser, Ueli},
  journal={Scientific data},
  volume={11},
  number={1},
  pages={214},
  year={2024},
  publisher={Nature Publishing Group UK London}
}"""

# assembly: Keles2024-fMRI
data_registry['Keles2024-fMRI'] = lambda: load_assembly_from_s3(
    identifier="Keles2024-fMRI",
    version_id="qbmUcZGjShAjAifkTzF.cuAQcgvUy2Nl",
    sha1="929288d0ec99a7991f157e7787a10452ee453021",
    bucket="brainscore-storage/brainscore-vision/benchmarks/Keles2024-fMRI",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: brainscore_vision.load_stimulus_set('Keles2024'),
)

# stimulus set: Keles2024
stimulus_set_registry['Keles2024'] = lambda: load_stimulus_set_from_s3(
    identifier="Keles2024",
    bucket="brainscore-storage/brainscore-vision/benchmarks/Keles2024-fMRI",
    csv_sha1="1fcf6d65a823b4e9ec4b02d6610190f92674b932",
    zip_sha1="350536296800fc2f46172f231da2266a0fbac361",
    csv_version_id="UKUxrOBp3eO4vbyGeqdhXGhfKhQ4Wrne",
    zip_version_id="AB.1FCauYVAItaW3pYDYGq2AAd3VpacQ")

