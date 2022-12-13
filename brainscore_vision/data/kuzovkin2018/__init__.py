from brainscore_vision import data_registry
from brainscore_vision.utils.s3 import load_from_s3


# TODO: add correct version id and sha1
data_registry['aru.Kuzovkin2018'] = lambda: load_from_s3(
    identifier="aru.Kuzovkin2018",
    version_id="",
    sha1="5fae8b283a043562ce9925d48ad99db151f39067")
