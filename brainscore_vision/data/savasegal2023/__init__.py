from brainscore_core.supported_data_standards.brainio.assemblies import NeuronRecordingAssembly

import brainscore_vision
from brainscore_vision import data_registry, stimulus_set_registry
from brainscore_core.supported_data_standards.brainio.s3 import load_stimulus_set_from_s3, load_assembly_from_s3

BIBTEX = """@article{sava2023individual,
  title={Individual differences in neural event segmentation of continuous experiences},
  author={Sava-Segal, Clara and Richards, Chandler and Leung, Megan and Finn, Emily S},
  journal={Cerebral Cortex},
  volume={33},
  number={13},
  pages={8164--8178},
  year={2023},
  publisher={Oxford University Press}
}"""

# assembly: Savasegal2023-fMRI-Defeat
data_registry['Savasegal2023-fMRI-Defeat'] = lambda: load_assembly_from_s3(
    identifier="Savasegal2023-fMRI-Defeat",
    version_id="nmpiAvdqfNI8WuAOI5nRGjK8wz2UCVPn",
    sha1="62eca09c366d2ed9a6fec829a51980a7783c0bf5",
    bucket="brainscore-storage/brainscore-vision/benchmarks/Savageal2023-fMRI-Defeat",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: brainscore_vision.load_stimulus_set('Savasegal2023-Defeat'),
)

# stimulus set: Savasegal2023-Defeat
stimulus_set_registry['Savasegal2023-Defeat'] = lambda: load_stimulus_set_from_s3(
    identifier="Savasegal2023-Defeat",
    bucket="brainscore-storage/brainscore-vision/benchmarks/Savageal2023-fMRI-Defeat",
    csv_sha1="02e84de95d419eccf36c52bbb06f92497123a410",
    zip_sha1="5de346354a1fc2a74a7a013e74583a57b83dfe01",
    csv_version_id="V9v1GZMZfoBPf8ZILsxr.7cBJlkDV.Kx",
    zip_version_id="Lu0fhHWzz0KDVr9T2337.GSOTDOvubCP")


# assembly: Savasegal2023-fMRI-Growth
data_registry['Savasegal2023-fMRI-Growth'] = lambda: load_assembly_from_s3(
    identifier="Savasegal2023-fMRI-Growth",
    version_id=".0Z_clfUa1Hm2dA6f6IAcyR9BXQcSpAI",
    sha1="f78852f8aadb81226d614f76935f4b708a01e3a1",
    bucket="brainscore-storage/brainscore-vision/benchmarks/Savageal2023-fMRI-Growth",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: brainscore_vision.load_stimulus_set('Savasegal2023-Growth'),
)

# stimulus set: Savasegal2023-Growth
stimulus_set_registry['Savasegal2023-Growth'] = lambda: load_stimulus_set_from_s3(
    identifier="Savasegal2023-Growth",
    bucket="brainscore-storage/brainscore-vision/benchmarks/Savageal2023-fMRI-Growth",
    csv_sha1="f5e226985307162b57b639c6e951d3876b6887ed",
    zip_sha1="7dbdbb36920017b39793c0cc51c96393e33d8a99",
    csv_version_id="F6O66kC2SEMf4yOHYdzGku5eKQXEORQh",
    zip_version_id="vOW_mUks2qfdNtM4ELjR0EoA8DxzPrj5")


# assembly: Savasegal2023-fMRI-Iteration
data_registry['Savasegal2023-fMRI-Iteration'] = lambda: load_assembly_from_s3(
    identifier="Savasegal2023-fMRI-Iteration",
    version_id="MsButHpOJxMhZKauoHM4v9GINzUGHnF9",
    sha1="76b04bf6219b0947c8908b4f00802c34c1e63c91",
    bucket="brainscore-storage/brainscore-vision/benchmarks/Savageal2023-fMRI-Iteration",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: brainscore_vision.load_stimulus_set('Savasegal2023-Iteration'),
)

# stimulus set: Savasegal2023-Iteration
stimulus_set_registry['Savasegal2023-Iteration'] = lambda: load_stimulus_set_from_s3(
    identifier="Savasegal2023-Iteration",
    bucket="brainscore-storage/brainscore-vision/benchmarks/Savageal2023-fMRI-Iteration",
    csv_sha1="5861e2b1928f2e7de00376fd086c1bf0919a1d04",
    zip_sha1="0f5cca320e6eaf814cad162cac4f40dd9dc9dcbe",
    csv_version_id="B27QJf53Xe_KfcCi0stgJBYfVNbUoGef",
    zip_version_id="o4JWk2r6PZtVUmuGuZrJbdtNwCzpFAdb")


# assembly: Savasegal2023-fMRI-Lemonade
data_registry['Savasegal2023-fMRI-Lemonade'] = lambda: load_assembly_from_s3(
    identifier="Savasegal2023-fMRI-Lemonade",
    version_id="KBZ6g4MTZWamPmROn3bnCRO9gXMjoXGk",
    sha1="8320075db5e0a71c7e19c38536795f6fa7953d36",
    bucket="brainscore-storage/brainscore-vision/benchmarks/Savageal2023-fMRI-Lemonade",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: brainscore_vision.load_stimulus_set('Savasegal2023-Lemonade'),
)

# stimulus set: Savasegal2023-Lemonade
stimulus_set_registry['Savasegal2023-Lemonade'] = lambda: load_stimulus_set_from_s3(
    identifier="Savasegal2023-Lemonade",
    bucket="brainscore-storage/brainscore-vision/benchmarks/Savageal2023-fMRI-Lemonade",
    csv_sha1="c10badbedc8443849b923b39bc74ddcbf8ad1178",
    zip_sha1="79edea7316c9f5dde7a8682dce7000ed506d2026",
    csv_version_id="nxry4CmhpIWw0wQhoLSLG_fezjODvzgC",
    zip_version_id="Uj6OnRLswF9z6bB.HzIuaFx.1hy_eqMD")