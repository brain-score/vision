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

# session 201025
stimulus_set_registry['Cowley2026.201025'] = stimulus_set(
    'Cowley2026.201025',
    csv_sha1="9ed6f9711ed0966ab2461ce1fa72fe3d10077304",
    zip_sha1="a845c479d2c118ef0308394d52e12c4679691cee",
    csv_version_id="vWd2MMs8i_7WkA1BRnISdeT003..N533",
    zip_version_id="0svmbYxEvtL9iUhL8_cHcFXvehqQ1oVp")
data_registry['Cowley2026.201025'] = assembly(
    'Cowley2026.201025',
    sha1="78c6bad4c761b6a0ed969c88946f9a4238b26341",
    version_id="1eGVyGwBqxNN_zYjCNNk9Gl5ZZp0k8Du")

# session 210225
stimulus_set_registry['Cowley2026.210225'] = stimulus_set(
    'Cowley2026.210225',
    csv_sha1="9ed6f9711ed0966ab2461ce1fa72fe3d10077304",
    zip_sha1="75aa2ed9c77406b6bb74c432a6c86abb3d5faa2c",
    csv_version_id="gfZ7QtQQhthTjqiQAy9kGKPNroEoLSPN",
    zip_version_id="x6KRTVDpLrDHIgSEABSHCkNooJcJnkQI")
data_registry['Cowley2026.210225'] = assembly(
    'Cowley2026.210225',
    sha1="9923f48042e3609fa90aa53310f92dd06d77bf39",
    version_id="uZY5jdVhXNbc8_Qq6iZMuifBt5RsSOTM")

# session 211022
stimulus_set_registry['Cowley2026.211022'] = stimulus_set(
    'Cowley2026.211022',
    csv_sha1="d9d9c00d600c008bf45512961bd86124ced39a3d",
    zip_sha1="32967e0cd1cac33c3ffa892b5f244617d2cbc3b9",
    csv_version_id="a5GHFYMYIJwpyph51sDxeL6xmDvk8uII",
    zip_version_id="cycg7NDoRSzDz_Y_qOPczLjgg13bNtUb")
data_registry['Cowley2026.211022'] = assembly(
    'Cowley2026.211022',
    sha1="64cbb1b23208d29d7e6908f931e4adea8550ef80",
    version_id="We7wLKhFc2ssl9hBeSzEBEHTcWvpGHwr")
