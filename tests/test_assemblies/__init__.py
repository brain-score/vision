import numpy as np
from brainio_base.assemblies import NeuroidAssembly


def check_standard_format(assembly):
    assert isinstance(assembly, NeuroidAssembly)
    assert set(assembly.dims).issuperset({'presentation', 'neuroid'})
    assert hasattr(assembly, 'image_id')
    assert hasattr(assembly, 'neuroid_id')
    assert not np.isnan(assembly).any()
    assert 'stimulus_set_name' in assembly.attrs
