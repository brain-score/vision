from brainio_collection.fetch import get_assembly as brainio_get_assembly, get_stimulus_set
from brainio_collection import list_stimulus_sets, list_assemblies


def get_assembly(name):
    assembly = brainio_get_assembly(name)
    if not hasattr(assembly.stimulus_set, 'name'):
        assembly.stimulus_set.name = assembly.stimulus_set_name

    stimulus_set_degrees = {'dicarlo.hvm': 8, 'movshon.FreemanZiemba2013': 4, 'tolias.Cadena2017': 2,
                            'dicarlo.objectome.private': 6, 'dicarlo.objectome.public': 6}
    if assembly.stimulus_set.name in stimulus_set_degrees:
        assembly.stimulus_set['degrees'] = stimulus_set_degrees[assembly.stimulus_set.name]
    return assembly
