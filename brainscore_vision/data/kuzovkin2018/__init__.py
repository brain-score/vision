from brainio.assemblies import NeuronRecordingAssembly

from brainscore_vision import data_registry
from brainscore_vision.utils.s3 import load_stimulus_set_from_s3, load_assembly_from_s3


# TODO: add correct version ids
# assembly
data_registry['aru.Kuzovkin2018'] = lambda: load_assembly_from_s3(
    identifier="aru.Kuzovkin2018",
    version_id="",
    sha1="5fae8b283a043562ce9925d48ad99db151f39067",
    bucket="brainio.contrib",
    cls=NeuronRecordingAssembly)

# stimulus set
data_registry['aru.Kuzovkin2018'] = lambda: load_stimulus_set_from_s3(
    identifier="aru.Kuzovkin2018",
    bucket="brainio.contrib",
    csv_sha1="a5990b24aea3e453756141cbe69a83304db72d0b",
    zip_sha1="cca4d819d7743bdd4bf65c1cb2439fd0ec97543a",
    csv_version_id="",
    zip_version_id="")
