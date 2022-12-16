from brainscore_vision import data_registry
from brainscore_vision.utils.s3 import load_from_s3


# TODO: add correct version id
# assemblies
data_registry['dicarlo.Rust2012.single'] = lambda: load_from_s3(
    identifier="dicarlo.Rust2012.single",
    version_id="",
    sha1="4ef420e70fbd0de3745df5be7c83dfc0a8f2e528")

data_registry['dicarlo.Rust2012.array'] = lambda: load_from_s3(
    identifier="dicarlo.Rust2012.single",
    version_id="",
    sha1="6709b641751370acfccd9567e3d75b71865a71ab")


# stimulus set csv
data_registry['dicarlo.Rust2012'] = lambda: load_from_s3(
    identifier="dicarlo.Rust2012",
    version_id="",
    sha1="482da1f9f4a0ab5433c3b7b57073ad30e45c2bf1")

# stimulus set zip
data_registry['dicarlo.Rust2012'] = lambda: load_from_s3(
    identifier="dicarlo.Rust2012",
    version_id="",
    sha1="7cbf5dcec235f7705eaad1cfae202eda77e261a2")
