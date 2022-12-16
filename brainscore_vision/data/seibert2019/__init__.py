from brainscore_vision import data_registry
from brainscore_vision.utils.s3 import load_from_s3


# TODO: add correct version id
data_registry['dicarlo.Seibert2019'] = lambda: load_from_s3(
    identifier="dicarlo.Seibert2019",
    version_id="",
    sha1="eef41bb1f3d83c0e60ebf0e91511ce71ef5fee32")
