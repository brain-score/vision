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

data_registry['NSD.V2.SharedCombinedSubs.2024'] = lambda: load_assembly_from_s3(
    identifier="NSD.V2.SharedCombinedSubs.2024",
    version_id="wbnz3q3TDO7fcFffc0EzbUPbRTc.C8De",
    sha1="b1eb7e93e897032b3ae2bc1ced5428cf9eb7d0c3",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: brainscore_vision.load_stimulus_set('NSDimagesShared2024'),
)

data_registry['NSD.V3.SharedCombinedSubs.2024'] = lambda: load_assembly_from_s3(
    identifier="NSD.V3.SharedCombinedSubs.2024",
    version_id="y32ovKi46Xdq3s1kY7h0W0yXzLZNej8j",
    sha1="3f3b86459a30527d7eb10a8b379e044664a362dc",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: brainscore_vision.load_stimulus_set('NSDimagesShared2024'),
)

data_registry['NSD.V4.SharedCombinedSubs.2024'] = lambda: load_assembly_from_s3(
    identifier="NSD.V4.SharedCombinedSubs.2024",
    version_id="ILIY_4neYgOIz4C_MeimwL9iJ13nuUzw",
    sha1="5f65c823c67fbef76e0b84ec8131c8b8ac39d549",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: brainscore_vision.load_stimulus_set('NSDimagesShared2024'),
)

data_registry['NSD.early.SharedCombinedSubs.2024'] = lambda: load_assembly_from_s3(
    identifier="NSD.early.SharedCombinedSubs.2024",
    version_id="S1A33n.60v95tZc3afWAGXlsIF7hV3lB",
    sha1="01fee20e70f954332d182c1b2973804f3e81c1c9",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: brainscore_vision.load_stimulus_set('NSDimagesShared2024'),
)

data_registry['NSD.lateral.SharedCombinedSubs.2024'] = lambda: load_assembly_from_s3(
    identifier="NSD.lateral.SharedCombinedSubs.2024",
    version_id="D3yHFL32AcZ5I9ypSrtC7FaQQAby0yII",
    sha1="a6f4685450c2e6760cb7b38495900c2cae22d49f",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: brainscore_vision.load_stimulus_set('NSDimagesShared2024'),
)

data_registry['NSD.parietal.SharedCombinedSubs.2024'] = lambda: load_assembly_from_s3(
    identifier="NSD.parietal.SharedCombinedSubs.2024",
    version_id="fthglixlxShRsETly2SBFbyFxNOvxXQS",
    sha1="7e913dc69bf0e5cd21327aa63fc54c7e0eb73072",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: brainscore_vision.load_stimulus_set('NSDimagesShared2024'),
)

data_registry['NSD.ventral.SharedCombinedSubs.2024'] = lambda: load_assembly_from_s3(
    identifier="NSD.ventral.SharedCombinedSubs.2024",
    version_id="OuQk.I7pg558fsbqwws1CXDRTFypslFz",
    sha1="83a81602493c4b1533066a7ddab5e0853c9430bf",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: brainscore_vision.load_stimulus_set('NSDimagesShared2024'),
)

# stimulus set
stimulus_set_registry['NSDimagesShared2024'] = lambda: load_stimulus_set_from_s3(
    identifier="NSDimagesShared2024",
    bucket="brainio-brainscore",
    csv_sha1="f2ce5837edbe576dcb5a81c3af490e9242ac35e0",
    zip_sha1="1cf9fd7459db180faff9f23a1a59aca22d0c74b5",
    csv_version_id="ZF.Yd027bVIbmFw.wQNbwaTo86xMkRTZ",
    zip_version_id="bEbf6eiTgEjpU201HM.0Avh4PH4UG4OJ",
    filename_prefix="stimulus_")

