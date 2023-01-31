from brainio.assemblies import NeuronRecordingAssembly
from brainscore_vision import data_registry, stimulus_set_registry
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3, load_stimulus_set_from_s3
from brainscore_vision.data_helpers.lookup_legacy import version_id_df, build_filename

BIBTEX = """@article{bashivan2019neural,
  title={Neural population control via deep image synthesis},
  author={Bashivan, Pouya and Kar, Kohitij and DiCarlo, James J},
  journal={Science},
  volume={364},
  number={6439},
  pages={eaav9436},
  year={2019},
  publisher={American Association for the Advancement of Science}
}"""

# assemblies
data_registry['dicarlo.BashivanKar2019.naturalistic'] = lambda: load_assembly_from_s3(
    identifier="dicarlo.BashivanKar2019.naturalistic",
    version_id="jDlQnpuhORCtzITHfOFcJiogR7jg8ihJ",
    sha1="1ec2f32ef800f0c6e15879d883be1d55b51b8b67",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly)

data_registry['dicarlo.BashivanKar2019.synthetic'] = lambda: load_assembly_from_s3(
    identifier="dicarlo.BashivanKar2019.synthetic",
    version_id="30CNHSBYS_HN5tEuM1aa5hDFlFKq6gbI",
    sha1="f687c8d26f8943dc379dbcbe94d3feb148400c6b",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly)

# stimulus sets
# naturalistic
stimulus_set_registry['dicarlo.BashivanKar2019.naturalistic'] = lambda: load_stimulus_set_from_s3(
    identifier="dicarlo.BashivanKar2019.naturalistic",
    bucket="brainio-brainscore",
    csv_sha1="48ef84282552b8796142ffe7d0d2c632f8ef061a",
    zip_sha1="d7b71b431cf23d435395205f1e38036a9e10acca",
    csv_version_id="DVRPJT0fKdeRTu6FIjOWClzB9kyyg63R",
    zip_version_id="ZxlD_lHE23VzgSCnYypZAHRzO7oblr4h")

# synthetic
stimulus_set_registry['dicarlo.BashivanKar2019.synthetic'] = lambda: load_stimulus_set_from_s3(
    identifier="dicarlo.BashivanKar2019.synthetic",
    bucket="brainio-brainscore",
    csv_sha1="81da195e9b2a128b228fc4867e23ae6b21bd7abd",
    zip_sha1="e2de33f25c5c19bcfb400055c1db399d553487e5",
    csv_version_id="tV0YP8QZ5DahblYrItWsURtUxwMnPXOV",
    zip_version_id="5YI9FrlUvtVMJETKmThsLJV2IOQlfzvm")
