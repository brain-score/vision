from brainio.assemblies import NeuronRecordingAssembly

import brainscore_vision
from brainscore_vision import data_registry, stimulus_set_registry
from brainscore_vision.data_helpers.s3 import load_stimulus_set_from_s3, load_assembly_from_s3

BIBTEX = """@article{berezutskaya2022open,
  title={Open multimodal iEEG-fMRI dataset from naturalistic stimulation with a short audiovisual film},
  author={Berezutskaya, Julia and Vansteensel, Mariska J and Aarnoutse, Erik J and Freudenburg, Zachary V and Piantoni, Giovanni and Branco, Mariana P and Ramsey, Nick F},
  journal={Scientific Data},
  volume={9},
  number={1},
  pages={91},
  year={2022},
  publisher={Nature Publishing Group UK London}
}"""

# assembly: Berezutskaya2021-fMRI
data_registry['Berezutskaya2021-fMRI'] = lambda: load_assembly_from_s3(
    identifier="Berezutskaya2021-fMRI",
    version_id="AY03ldTEnzdocCBXmJPVnb9yVL7Slx_S",
    sha1="936b2751a9d1eaa182a0646faa3ce1d974dd0d92",
    bucket="brainscore-storage/brainscore-vision/benchmarks/Berezutskaya2021-fMRI",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: brainscore_vision.load_stimulus_set('Berezutskaya2021'),
)

# stimulus set: Berezutskaya2021
stimulus_set_registry['Berezutskaya2021'] = lambda: load_stimulus_set_from_s3(
    identifier="Berezutskaya2021",
    bucket="brainscore-storage/brainscore-vision/benchmarks/Berezutskaya2021-fMRI",
    csv_sha1="42b06621fb5e3d4dadbc3975ddb31ed2df75aab4",
    zip_sha1="3b05a9d055918da9536d615589bef6644e28d8a9",
    csv_version_id="2BGwIC9sxZvNOW_bQr98Csk9hvbW.9lX",
    zip_version_id="9QPVG.DYvWZcD_Q0wWrqUJrCsLDIvO8T")
