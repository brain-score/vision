from brainscore_vision.benchmark_helpers.neural_common import NeuralBenchmark, average_repetition
from brainscore_vision import load_metric, load_ceiling, load_dataset, load_stimulus_set

import numpy as np
import pandas as pd

import xarray as xr
from brainscore_vision import load_stimulus_set
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly


BIBTEX = """@article{cowley2026compact,
  title={Compact deep neural network models of the visual cortex},
  author={Cowley, Benjamin R and Stan, Patricia L and Pillow, Jonathan W and Smith, Matthew A},
  journal={Nature},
  volume={652},
  number={8111},
  pages={947--954},
  year={2026},
  publisher={Nature Publishing Group}}"""



def benchmark_Cowley2026_190923():

    assembly = load_assembly(average_repetitions=True)
    assembly_repetition = load_assembly(average_repetitions=False)

    return NeuralBenchmark(
        identifier='Cowley2026_190923.V4-pls',
        version=1,
        assembly=assembly,
        similarity_metric=load_metric('pls'),
        visual_degrees=11.2,
        number_of_trials=14,  # mode of num repeats --- different images had different repeats
        ceiling_func=lambda: load_ceiling('internal_consistency')(assembly_repetition),
        parent='V4',
        bibtex=BIBTEX
    )


def load_assembly(average_repetitions, access='private'):
    # 1. Load the pristine dataset from your disk
    assembly = load_dataset('Cowley2026.190923')
    assembly = assembly.sel(region='V4')
    assembly = assembly.stack(neuroid=['neuroid_id'])  # work around xarray multiindex issues
    assembly['region'] = 'neuroid', ['V4'] * len(assembly['neuroid'])
    assembly.load()

    # 2. Squeeze out the physical time_bin axis to align shapes with static models
    if 'time_bin' in assembly.dims:
        assembly = assembly.squeeze('time_bin')

    if average_repetitions:
        # 3. Collapse repetitions down to unique stimulus_ids
        # Your clean packaging ensures object_name and category survive natively!
        assembly = average_repetition(assembly)
        
    return assembly


#   # copying FreemanZiemba2013???
#     assembly = load_dataset('Cowley2026.190923')
#     assembly = assembly.sel(region='V4')
#     assembly = assembly.stack(neuroid=['neuroid_id'])  # work around xarray multiindex issues
#     assembly['region'] = 'neuroid', ['V4'] * len(assembly['neuroid'])
#     assembly.load()
#     time_window = (50, 150)
#     assembly = assembly.sel(time_bin=[(t, t + 1) for t in range(*time_window)])
#     assembly = assembly.mean(dim='time_bin', keep_attrs=True)
#     assembly = assembly.expand_dims('time_bin_start').expand_dims('time_bin_end')
#     assembly['time_bin_start'], assembly['time_bin_end'] = [time_window[0]], [time_window[1]]
#     assembly = assembly.stack(time_bin=['time_bin_start', 'time_bin_end'])
#     assembly = assembly.squeeze('time_bin')
#     assembly = assembly.transpose('presentation', 'neuroid')
#     if average_repetitions:
#         assembly = average_repetition(assembly)
#     return assembly



