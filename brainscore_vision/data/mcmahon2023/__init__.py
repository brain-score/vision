from brainio.assemblies import NeuronRecordingAssembly

import brainscore_vision
from brainscore_vision import data_registry, stimulus_set_registry
from brainscore_vision.data_helpers.s3 import load_stimulus_set_from_s3, load_assembly_from_s3

BIBTEX = """@article{mcmahon2023hierarchical,
  title={Hierarchical organization of social action features along the lateral visual pathway},
  author={McMahon, Emalie and Bonner, Michael F and Isik, Leyla},
  journal={Current Biology},
  volume={33},
  number={23},
  pages={5035--5047},
  year={2023},
  publisher={Elsevier}
}"""

# assembly: McMahon2023-fMRI
data_registry['McMahon2023-fMRI'] = lambda: load_assembly_from_s3(
    identifier="McMahon2023-fMRI",
    version_id="0k8B44WRZaQyRc1eynbi_RBrjRT1mKL8",
    sha1="c8f1ae9981c11f10135c20eca225b6aade039d23",
    bucket="brainscore-storage/brainscore-vision/benchmarks/McMahon2023-fMRI",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: brainscore_vision.load_stimulus_set('McMahon2023'),
)

# stimulus set: McMahon2023
stimulus_set_registry['McMahon2023'] = lambda: load_stimulus_set_from_s3(
    identifier="McMahon2023",
    bucket="brainscore-storage/brainscore-vision/benchmarks/McMahon2023-fMRI",
    csv_sha1="fe617e7bce845c22eccf46242c95c0a74bcb501e",
    zip_sha1="f136c855a457b6b40d0b91f621d9fbb68f1c6c48",
    csv_version_id="zbknzf08aV.CLb1pUkt1ZMRk1nWheSPp",
    zip_version_id="EVMJwIgUmE_NlLMCLTvMOpBQC__aiLXz")
