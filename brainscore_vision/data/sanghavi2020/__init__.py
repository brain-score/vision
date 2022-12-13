from brainscore_vision.metrics.ceiling import InternalConsistency
from brainscore_vision.metrics.transformations import CrossValidation
from brainscore_vision import data_registry
from brainscore_vision.utils.s3 import load_from_s3


BIBTEX = """"""

data_registry['dicarlo.Sanghavi2020'] = lambda: load_from_s3(
    identifier="dicarlo.Sanghavi2020",
    version_id="",
    sha1="12e94e9dcda797c851021dfe818b64615c785866")

data_registry['dicarlo.SanghaviJozwik2020'] = lambda: load_from_s3(
    identifier="dicarlo.SanghaviJozwik2020",
    version_id="",
    sha1="c5841f1e7d2cf0544a6ee010e56e4e2eb0994ee0")

data_registry['dicarlo.SanghaviMurty2020'] = lambda: load_from_s3(
    identifier="dicarlo.SanghaviMurty2020",
    version_id="",
    sha1="6cb8e054688066d1d86d4944e1385efc6a69ebd4")


def filter_neuroids(assembly, threshold):
    ceiler = InternalConsistency()
    ceiling = ceiler(assembly)
    ceiling = ceiling.raw
    ceiling = CrossValidation().aggregate(ceiling)
    ceiling = ceiling.sel(aggregation='center')
    pass_threshold = ceiling >= threshold
    assembly = assembly[{'neuroid': pass_threshold}]
    return assembly
