from brainscore_vision import data_registry
from brainscore_vision.utils.s3 import load_from_s3


# TODO: add correct version id
data_registry['dicarlo.ImageNetSlim15000'] = lambda: load_from_s3(
    identifier="dicarlo.ImageNetSlim15000",
    version_id="",
    sha1="56f79de36060147bc43ae89778e26d50f73c3a63")
