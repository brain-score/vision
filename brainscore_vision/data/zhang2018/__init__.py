from brainio.assemblies import BehavioralAssembly

from brainscore_vision import data_registry, stimulus_set_registry, load_stimulus_set
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3, load_stimulus_set_from_s3

BIBTEX = """"""

# assembly: klab.Zhang2018search_obj_array
data_registry['klab.Zhang2018search_obj_array'] = lambda: load_assembly_from_s3(
    identifier="klab.Zhang2018search_obj_array",
    version_id="EDRblSpQTmoaKRdiY4JEoQ.1UO8mo0Tk",
    sha1="9357581c4082de3ed5031c914468fd24c57ac9cf",
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('klab.Zhang2018.search_obj_array'),
)

# stimulus set: klab.Zhang2018.search_obj_array
stimulus_set_registry['klab.Zhang2018.search_obj_array'] = lambda: load_stimulus_set_from_s3(
    identifier="klab.Zhang2018.search_obj_array",
    bucket="brainio-brainscore",
    csv_sha1="4808db57a59c44ee4900e2cf914e76bf42c61bcd",
    zip_sha1="eab5e52730cc236e71e583a2c401208380dc8628",
    csv_version_id="HC_2UVxrTqbKtBtMny2YqYjm6cjdx6JI",
    zip_version_id="B2uxHg7zx9Zx4dnAD2Lk_TxaQZPB6p_P")
