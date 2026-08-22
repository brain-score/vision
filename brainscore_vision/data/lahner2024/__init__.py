from brainscore_core.supported_data_standards.brainio.assemblies import NeuronRecordingAssembly

import brainscore_vision
from brainscore_vision import data_registry, stimulus_set_registry
from brainscore_core.supported_data_standards.brainio.s3 import load_stimulus_set_from_s3, load_assembly_from_s3

BIBTEX = """@misc{lahner2024modeling,
  title={Modeling short visual events through the BOLD moments video fMRI dataset and metadata. Nat. Commun. 15, 6241},
  author={Lahner, B and Dwivedi, K and Iamshchinina, P and Graumann, M and Lascelles, A and Roig, G and Gifford, AT and Pan, B and Jin, SY and Ratan Murty, NA and others},
  year={2024}
}"""

# assembly: Lahner2024-fMRI
data_registry['Lahner2024-fMRI'] = lambda: load_assembly_from_s3(
    identifier="Lahner2024-fMRI",
    version_id="zr_i3T9Saww44rPNJwLaxo0hgp8rYjPO",
    sha1="2c7f1d2e5724b8cc3c5cf47986e956c4f13001e4",
    bucket="brainscore-storage/brainscore-vision/benchmarks/Lahner2024-fMRI",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: brainscore_vision.load_stimulus_set('BOLDMoments'),
)

# stimulus set: BOLDMoments - Lahner2024-fMRI
stimulus_set_registry['BOLDMoments'] = lambda: load_stimulus_set_from_s3(
    identifier="BOLDMoments",
    bucket="brainscore-storage/brainscore-vision/benchmarks/Lahner2024-fMRI",
    csv_sha1="0b27388f5898c908f58cd1f21f8f5cb3eda8536e",
    zip_sha1="dc9c3bf631632cd433d02f2f1847fd33c01ae0b3",
    csv_version_id="WaGkWh59b1drhy1MmAVVSxh7_VT_eTay",
    zip_version_id="OxpOYy_3bveay9NFFFxNCVyghyAbqyIt")
