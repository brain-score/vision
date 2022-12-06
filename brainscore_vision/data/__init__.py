from brainscore_vision.metrics.ceiling import InternalConsistency
from brainscore_vision.metrics.transformations import CrossValidation

from brainio import get_assembly as brainio_get_assembly
from brainscore_vision import data_registry


def get_assembly(identifier):
    assembly = brainio_get_assembly(identifier)
    assert hasattr(assembly.stimulus_set, 'identifier')
    assert assembly.stimulus_set.identifier == assembly.stimulus_set_identifier
    return assembly


for identifier in ['']:
    # TODO: I think we should use `load_from_s3` here with version ids, and get rid of the `lookup.csv`
    data_registry[identifier] = lambda identifier=identifier: get_assembly(identifier)


def filter_neuroids(assembly, threshold):
    ceiler = InternalConsistency()
    ceiling = ceiler(assembly)
    ceiling = ceiling.raw
    ceiling = CrossValidation().aggregate(ceiling)
    ceiling = ceiling.sel(aggregation='center')
    pass_threshold = ceiling >= threshold
    assembly = assembly[{'neuroid': pass_threshold}]
    return assembly
