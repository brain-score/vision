from brainio.assemblies import NeuronRecordingAssembly

from brainscore_vision import data_registry
from brainscore_vision.utils.s3 import load_stimulus_set_from_s3, load_assembly_from_s3


# TODO: add correct version ids
# dicarlo.Rust2012.single assembly
data_registry['dicarlo.Rust2012.single'] = lambda: load_assembly_from_s3(
    identifier="dicarlo.Rust2012.single",
    version_id="",
    sha1="4ef420e70fbd0de3745df5be7c83dfc0a8f2e528",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly)

# dicarlo.Rust2012.array assembly
data_registry['dicarlo.Rust2012.array'] = lambda: load_assembly_from_s3(
    identifier="dicarlo.Rust2012.array",
    version_id="",
    sha1="6709b641751370acfccd9567e3d75b71865a71ab",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly)


# stimulus set
data_registry['dicarlo.Rust2012'] = lambda: load_stimulus_set_from_s3(
    identifier="dicarlo.Rust2012",
    bucket="brainio-brainscore",
    csv_sha1="482da1f9f4a0ab5433c3b7b57073ad30e45c2bf1",
    zip_sha1="7cbf5dcec235f7705eaad1cfae202eda77e261a2",
    csv_version_id="",
    zip_version_id="")
