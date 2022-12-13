from brainscore_vision import data_registry
from brainscore_vision.utils.s3 import load_from_s3


# TODO: add correct version ids
data_registry['dicarlo.BashivanKar2019.naturalistic'] = lambda: load_from_s3(
    identifier="dicarlo.BashivanKar2019.naturalistic",
    version_id="",
    sha1="1ec2f32ef800f0c6e15879d883be1d55b51b8b67")

data_registry['dicarlo.BashivanKar2019.synthetic'] = lambda: load_from_s3(
    identifier="dicarlo.BashivanKar2019.synthetic",
    version_id="",
    sha1="f687c8d26f8943dc379dbcbe94d3feb148400c6b")
