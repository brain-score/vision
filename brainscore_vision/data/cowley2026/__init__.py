from brainscore_vision import data_registry, stimulus_set_registry, load_stimulus_set
from brainscore_core.supported_data_standards.brainio.s3 import load_stimulus_set_from_s3, load_assembly_from_s3
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly

BIBTEX = """@article{cowley2026compact,
  title={Compact deep neural network models of the visual cortex},
  author={Cowley, Benjamin R and Stan, Patricia L and Pillow, Jonathan W and Smith, Matthew A},
  journal={Nature},
  volume={652},
  number={8111},
  pages={947--954},
  year={2026},
  publisher={Nature Publishing Group}}"""

BUCKET = "brainscore-storage/brainscore-vision/data/user_718/"


# keep literal `*_registry['<id>'] =` lines below: plugin discovery greps for that substring
def stimulus_set(identifier, csv_sha1, zip_sha1, csv_version_id, zip_version_id):
    return lambda: load_stimulus_set_from_s3(
        identifier=identifier, bucket=BUCKET,
        csv_sha1=csv_sha1, zip_sha1=zip_sha1,
        csv_version_id=csv_version_id, zip_version_id=zip_version_id)


def assembly(identifier, sha1, version_id):
    return lambda: load_assembly_from_s3(
        identifier=identifier, bucket=BUCKET, sha1=sha1, version_id=version_id,
        cls=NeuroidAssembly, stimulus_set_loader=lambda: load_stimulus_set(identifier))


# session 190923
stimulus_set_registry['Cowley2026.190923'] = stimulus_set(
    'Cowley2026.190923',
    csv_sha1="7752f43fc809c193334dd97171867e733291b8fd",
    zip_sha1="a14f9d4cfc98cb253f23d4eaa159c60666903668",
    csv_version_id="4ZuvTJxZptY8V04ayk2CRLb209BihWis",
    zip_version_id="tWyQAXN_fM4Y2fLnQtITbi0QMLawr4Nd")
data_registry['Cowley2026.190923'] = assembly(
    'Cowley2026.190923',
    sha1="2ac7f60f21ccc5137074633c0614f52566acff6a",
    version_id="kpX10KvUti_Vg4WAMHsiayllYlStMcgI")
