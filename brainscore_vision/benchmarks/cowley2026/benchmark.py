from brainscore_vision import load_metric, load_ceiling, load_dataset
from brainscore_vision.benchmark_helpers.neural_common import NeuralBenchmark, average_repetition

VISUAL_DEGREES = 11.2
NUMBER_OF_TRIALS = 14  # mode of the per-image repeat counts
BIBTEX = """@article{cowley2026compact,
  title={Compact deep neural network models of the visual cortex},
  author={Cowley, Benjamin R and Stan, Patricia L and Pillow, Jonathan W and Smith, Matthew A},
  journal={Nature},
  volume={652},
  number={8111},
  pages={947--954},
  year={2026},
  publisher={Nature Publishing Group}}"""

# no object categories -> plain random CV splits, not object_name stratification
pls_metric = lambda: load_metric('pls', crossvalidation_kwargs=dict(stratification_coord=None))


def _Cowley2026V4PLS(session: str):
    identifier = f'Cowley2026.{session}'
    assembly_repetition = alternate_repetition_halves(load_assembly(identifier, average_repetitions=False))
    assembly = load_assembly(identifier, average_repetitions=True)
    return NeuralBenchmark(
        identifier=f'{identifier}.V4-pls', version=1,
        assembly=assembly, similarity_metric=pls_metric(),
        visual_degrees=VISUAL_DEGREES, number_of_trials=NUMBER_OF_TRIALS,
        ceiling_func=lambda: load_ceiling('internal_consistency')(assembly_repetition),
        parent='V4', bibtex=BIBTEX)


def alternate_repetition_halves(assembly):
    """Relabel repetitions to even/odd halves so the split-half ceiling balances per image."""
    names = list(assembly.indexes['presentation'].names)
    half = (assembly['repetition'].values % 2).astype(int)
    assembly = assembly.reset_index('presentation')
    assembly['repetition'] = 'presentation', half
    return assembly.set_index(presentation=names)


def load_assembly(identifier: str, average_repetitions: bool):
    assembly = load_dataset(identifier)
    assembly = assembly.sel(region='V4')
    assembly = assembly.stack(neuroid=['neuroid_id'])  # work around xarray multiindex issues
    assembly['region'] = 'neuroid', ['V4'] * len(assembly['neuroid'])
    assembly.load()
    if 'time_bin' in assembly.dims:  # single static window (50, 150) ms
        assembly = assembly.squeeze('time_bin')
    if average_repetitions:
        assembly = average_repetition(assembly)
    return assembly


def Cowley2026_190923_V4PLS():
    return _Cowley2026V4PLS('190923')
