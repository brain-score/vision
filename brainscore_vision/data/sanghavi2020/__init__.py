from brainio.assemblies import NeuronRecordingAssembly

from brainscore_vision.metrics.ceiling import InternalConsistency
from brainscore_vision.metrics.transformations import CrossValidation
from brainscore_vision import data_registry
from brainscore_vision.utils.s3 import load_stimulus_set_from_s3, load_assembly_from_s3


BIBTEX = """"""

# TODO: update version_ids
# assemblies: dicarlo.Sanghavi2020 uses dicarlo.hvm
data_registry['dicarlo.Sanghavi2020'] = lambda: load_assembly_from_s3(
    identifier="dicarlo.Sanghavi2020",
    version_id="",
    sha1="12e94e9dcda797c851021dfe818b64615c785866",
    bucket="brainio.dicarlo",
    cls=NeuronRecordingAssembly)

# assemblies: dicarlo.SanghaviJozwik2020 uses dicarlo.BOLD5000
data_registry['dicarlo.SanghaviJozwik2020'] = lambda: load_assembly_from_s3(
    identifier="dicarlo.SanghaviJozwik2020",
    version_id="",
    sha1="c5841f1e7d2cf0544a6ee010e56e4e2eb0994ee0",
    bucket="brainio.dicarlo",
    cls=NeuronRecordingAssembly)

# assemblies: dicarlo.SanghaviMurty2020 uses dicarlo.Rust2012
data_registry['dicarlo.SanghaviMurty2020'] = lambda: load_assembly_from_s3(
    identifier="dicarlo.SanghaviMurty2020",
    version_id="",
    sha1="6cb8e054688066d1d86d4944e1385efc6a69ebd4",
    bucket="brainio.dicarlo",
    cls=NeuronRecordingAssembly)

# assemblies: dicarlo.SanghaviMurty2020THINGS1 uses dicarlo.THINGS1
data_registry['dicarlo.SanghaviMurty2020THINGS1'] = lambda: load_assembly_from_s3(
    identifier="dicarlo.SanghaviMurty2020THINGS1",
    version_id="",
    sha1="718def227d38c8425f449512e47a2df81c04de62",
    bucket="brainio.dicarlo",
    cls=NeuronRecordingAssembly)

# assemblies: dicarlo.SanghaviMurty2020THINGS2 uses dicarlo.THINGS2
data_registry['dicarlo.SanghaviMurty2020THINGS2'] = lambda: load_assembly_from_s3(
    identifier="dicarlo.SanghaviMurty2020THINGS2",
    version_id="",
    sha1="80962139823cb145e2385c344e3945e99ed97fa2",
    bucket="brainio.dicarlo",
    cls=NeuronRecordingAssembly)


def filter_neuroids(assembly, threshold):
    ceiler = InternalConsistency()
    ceiling = ceiler(assembly)
    ceiling = ceiling.raw
    ceiling = CrossValidation().aggregate(ceiling)
    ceiling = ceiling.sel(aggregation='center')
    pass_threshold = ceiling >= threshold
    assembly = assembly[{'neuroid': pass_threshold}]
    return assembly
