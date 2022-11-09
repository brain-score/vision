from brainscore_vision.metrics.ceiling import InternalConsistency
from brainscore_vision.metrics.transformations import CrossValidation


def filter_neuroids(assembly, threshold):
    ceiler = InternalConsistency()
    ceiling = ceiler(assembly)
    ceiling = ceiling.raw
    ceiling = CrossValidation().aggregate(ceiling)
    ceiling = ceiling.sel(aggregation='center')
    pass_threshold = ceiling >= threshold
    assembly = assembly[{'neuroid': pass_threshold}]
    return assembly
