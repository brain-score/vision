from brainscore_vision import data_registry
from brainscore_vision.utils.s3 import load_stimulus_set_from_s3


# TODO: add correct version id
# stimulus set
data_registry['katz.BarbuMayo2019'] = lambda: load_stimulus_set_from_s3(
    identifier="katz.BarbuMayo2019",
    bucket="brainio-brainscore",
    csv_sha1="e4d8888ccb6beca28636e6698e7beb130e278e12",
    zip_sha1="1365eb2a7231516806127a7d2a908343a7ac9464",
    csv_version_id="",
    zip_version_id="")


