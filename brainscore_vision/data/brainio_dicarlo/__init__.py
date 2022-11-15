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
