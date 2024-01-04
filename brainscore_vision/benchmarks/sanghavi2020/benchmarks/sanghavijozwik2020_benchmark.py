import numpy as np

from brainscore_vision import load_ceiling, load_metric, load_dataset
from brainscore_vision.benchmark_helpers.neural_common import NeuralBenchmark, average_repetition
from brainscore_vision.data.sanghavi2020 import BIBTEX_SANGHAVIJOZWIK
from brainscore_vision.utils import LazyLoad

VISUAL_DEGREES = 8
NUMBER_OF_TRIALS = 37


def _DicarloSanghaviJozwik2020Region(region):
    assembly_repetition = LazyLoad(lambda region=region: load_assembly(average_repetitions=False, region=region))
    assembly = LazyLoad(lambda region=region: load_assembly(average_repetitions=True, region=region))
    metric = load_metric('pls', crossvalidation_kwargs=dict(stratification_coord=None))
    ceiler = load_ceiling('internal_consistency')
    return NeuralBenchmark(identifier=f'dicarlo.SanghaviJozwik2020.{region}-pls', version=1,
                           assembly=assembly, similarity_metric=metric,
                           visual_degrees=VISUAL_DEGREES, number_of_trials=NUMBER_OF_TRIALS,
                           ceiling_func=lambda: ceiler(assembly_repetition),
                           parent=region,
                           bibtex=BIBTEX_SANGHAVIJOZWIK)


def DicarloSanghaviJozwik2020V4PLS():
    return _DicarloSanghaviJozwik2020Region('V4')


def DicarloSanghaviJozwik2020ITPLS():
    return _DicarloSanghaviJozwik2020Region('IT')


def load_assembly(average_repetitions, region):
    assembly = load_dataset(f'SanghaviJozwik2020')
    assembly = assembly.sel(region=region)
    assembly['region'] = 'neuroid', [region] * len(assembly['neuroid'])
    assembly.load()
    assembly = assembly.sel(time_bin_id=0)  # 70-170ms
    assembly = assembly.squeeze('time_bin')
    assert NUMBER_OF_TRIALS == len(np.unique(assembly.coords['repetition']))
    assert VISUAL_DEGREES == assembly.attrs['image_size_degree']
    if average_repetitions:
        assembly = average_repetition(assembly)
    return assembly
