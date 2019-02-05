from brainio_collection.fetch import get_assembly as brainio_get_assembly, get_stimulus_set


# from brainscore.contrib import benchmarks as contrib_benchmarks
#
# contrib_benchmarks.inject()
def get_assembly(name):
    assembly = brainio_get_assembly(name)
    assembly.stimulus_set.name = assembly.stimulus_set_name
    return assembly
