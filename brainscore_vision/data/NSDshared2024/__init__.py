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
    version_id="Z3ITwtE8h93U_EmyuIDmp1wQ7OB83Qhs",
    sha1="7d655b73fff7b2177e667c90b218be1556fcbd4c",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: brainscore_vision.load_stimulus_set('NSDimagesShared2024'),
)

# stimulus set: hvm - majajhong2015
stimulus_set_registry['NSDimagesShared2024'] = lambda: load_stimulus_set_from_s3(
    identifier="NSDimagesShared2024",
    bucket="brainio-brainscore",
    csv_sha1="09cfeb9f47ebe3e62ca7ed6d19802c04fa32a4c0",
    zip_sha1="1cf9fd7459db180faff9f23a1a59aca22d0c74b5",
    csv_version_id="G.JOqtyOJaIMovcSnDVuejN6h8iu4hkK",
    zip_version_id="W_VVoE0v7Wf2cCH3u.i35p3MgfOfzDoE",
    filename_prefix="stimulus_")
