from brainscore_vision import data_registry
from brainscore_vision.metrics.ceiling import InternalConsistency
from brainscore_vision.metrics.transformations import CrossValidation
from brainscore_vision.utils.s3 import load_from_s3

# TODO: add correct version id
data_registry['dicarlo.Kar2018hvm'] = lambda: load_from_s3(
    identifier="dicarlo.Kar2018hvm",
    version_id="",
    sha1="96ccacc76c5fa30ee68bdc8736d1d43ace93f3e7")

data_registry['dicarlo.Kar2018cocogray'] = lambda: load_from_s3(
    identifier="dicarlo.Kar2018cocogray",
    version_id="",
    sha1="4202cb3992a5d71f71a7ca9e28ba3f8b27937b43")


def filter_neuroids(assembly, threshold):
    ceiler = InternalConsistency()
    ceiling = ceiler(assembly)
    ceiling = ceiling.raw
    ceiling = CrossValidation().aggregate(ceiling)
    ceiling = ceiling.sel(aggregation='center')
    pass_threshold = ceiling >= threshold
    assembly = assembly[{'neuroid': pass_threshold}]
    return assembly
