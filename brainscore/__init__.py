from brainio_collection.fetch import get_assembly as brainio_get_assembly


def get_assembly(name):
    assembly = brainio_get_assembly(name)
    if not hasattr(assembly.stimulus_set, 'name'):
        assembly.stimulus_set.name = assembly.stimulus_set_name

    stimulus_set_degrees = {'dicarlo.hvm': 8, 'movshon.FreemanZiemba2013': 4}
    if assembly.stimulus_set.name in stimulus_set_degrees:
        assembly.stimulus_set['degrees'] = stimulus_set_degrees[assembly.stimulus_set.name]
    return assembly
