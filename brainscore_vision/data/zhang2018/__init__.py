from brainio.assemblies import BehavioralAssembly

from brainscore_vision import data_registry, stimulus_set_registry, load_stimulus_set
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3, load_stimulus_set_from_s3

BIBTEX = """@article{zhang2018finding,
  title={Finding any Waldo with zero-shot invariant and efficient visual search},
  author={Zhang, Mengmi and Feng, Jiashi and Ma, Keng Teck and Lim, Joo Hwee and Zhao, Qi and Kreiman, Gabriel},
  journal={Nature communications},
  volume={9},
  number={1},
  pages={3730},
  year={2018},
  publisher={Nature Publishing Group UK London}
}"""

PACKAGING_FIXED = False  # There is an issue with the packaging where the zip filenames do not match the csv filenames

if PACKAGING_FIXED:
    # assembly: klab.Zhang2018search_obj_array
    data_registry['Zhang2018search_obj_array'] = lambda: load_assembly_from_s3(
        identifier="klab.Zhang2018search_obj_array",
        version_id="EDRblSpQTmoaKRdiY4JEoQ.1UO8mo0Tk",
        sha1="9357581c4082de3ed5031c914468fd24c57ac9cf",
        bucket="brainio-brainscore",
        cls=BehavioralAssembly,
        stimulus_set_loader=lambda: load_stimulus_set('Zhang2018.search_obj_array'),
    )

    # stimulus set: Zhang2018.search_obj_array
    stimulus_set_registry['Zhang2018.search_obj_array'] = lambda: load_stimulus_set_from_s3(
        identifier="Zhang2018.search_obj_array",
        bucket="brainio-brainscore",
        csv_version_id="HC_2UVxrTqbKtBtMny2YqYjm6cjdx6JI",
        csv_sha1="e9c2f6b35b84256242d257e8d36261aa26d3ed4a",
        zip_version_id="B2uxHg7zx9Zx4dnAD2Lk_TxaQZPB6p_P",
        zip_sha1="eab5e52730cc236e71e583a2c401208380dc8628",
    )
