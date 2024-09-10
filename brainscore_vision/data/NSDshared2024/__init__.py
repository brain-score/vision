from brainio.assemblies import NeuronRecordingAssembly

import brainscore_vision
from brainscore_vision import data_registry, stimulus_set_registry
from brainscore_vision.data_helpers.s3 import load_stimulus_set_from_s3, load_assembly_from_s3

BIBTEX = """@article{allen2022massive,
              title={A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial intelligence},
              author={Allen, Emily J and St-Yves, Ghislain and Wu, Yihan and Breedlove, Jesse L and Prince, Jacob S and Dowdle, Logan T and Nau, Matthias and Caron, Brad and Pestilli, Franco and Charest, Ian and others},
              journal={Nature neuroscience},
              volume={25},
              number={1},
              pages={116--126},
              year={2022},
              publisher={Nature Publishing Group US New York}
        }"""

# assembly: dicarlo.MajajHong2015
data_registry['NSD.V1.SharedCombinedSubs.2024'] = lambda: load_assembly_from_s3(
    identifier="NSD.V1.SharedCombinedSubs.2024",
    version_id="Ylfj.ICYXD_80ytOLu5AVkaleoxapxlA",
    sha1="362409ea6384becba7e6bb9d92d3525301d50e24",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: brainscore_vision.load_stimulus_set('NSDimagesShared2024'),
)

# stimulus set: hvm - majajhong2015
stimulus_set_registry['NSDimagesShared2024'] = lambda: load_stimulus_set_from_s3(
    identifier="NSDimagesShared2024",
    bucket="brainio-brainscore",
    csv_sha1="f2ce5837edbe576dcb5a81c3af490e9242ac35e0",
    zip_sha1="1cf9fd7459db180faff9f23a1a59aca22d0c74b5",
    csv_version_id="ZF.Yd027bVIbmFw.wQNbwaTo86xMkRTZ",
    zip_version_id="bEbf6eiTgEjpU201HM.0Avh4PH4UG4OJ",
    filename_prefix="stimulus_")

