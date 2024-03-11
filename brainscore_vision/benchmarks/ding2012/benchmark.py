from result_caching import store
from brainscore_vision import load_dataset, load_metric, load_ceiling
from .registry import load_dataset
from brainscore_vision.benchmark_helpers.neural_common import NeuralBenchmark, average_repetition
from brainscore_vision.utils import LazyLoad

VISUAL_DEGREES = 4
NUMBER_OF_TRIALS = None
BIBTEX = """@article{ding2012neural,
  title={Neural correlates of perceptual decision making before, during, and after decision commitment in monkey frontal eye field},
  author={Ding, Long and Gold, Joshua I},
  journal={Cerebral cortex},
  volume={22},
  number={5},
  pages={1052--1067},
  year={2012},
  publisher={Oxford University Press}
}
"""


def Ding2011(region, identifier_metric_suffix, similarity_metric, ceiler):
    assembly_repetition = LazyLoad(lambda region=region: load_assembly(False, region=region))
    assembly = LazyLoad(lambda region=region: load_assembly(True, region=region))
    return NeuralBenchmark(identifier=f'FreemanZiemba2013.{region}-{identifier_metric_suffix}', version=2,
                           assembly=assembly, similarity_metric=similarity_metric, parent=region,
                           ceiling_func=lambda: ceiler(assembly_repetition),
                           visual_degrees=VISUAL_DEGREES, number_of_trials=NUMBER_OF_TRIALS,
                           bibtex=BIBTEX)


@store()
def load_assembly(average_repetitions, region, access='private'):
    assembly = load_dataset(f'FreemanZiemba2013.{access}')
    assembly = assembly.sel(region=region)
    assembly = assembly.stack(neuroid=['neuroid_id'])  # work around xarray multiindex issues
    assembly['region'] = 'neuroid', [region] * len(assembly['neuroid'])
    assembly.load()
    time_window = (50, 200)
    assembly = assembly.sel(time_bin=[(t, t + 1) for t in range(*time_window)])
    assembly = assembly.mean(dim='time_bin', keep_attrs=True)
    assembly = assembly.expand_dims('time_bin_start').expand_dims('time_bin_end')
    assembly['time_bin_start'], assembly['time_bin_end'] = [time_window[0]], [time_window[1]]
    assembly = assembly.stack(time_bin=['time_bin_start', 'time_bin_end'])
    assembly = assembly.squeeze('time_bin')
    assembly = assembly.transpose('presentation', 'neuroid')
    if average_repetitions:
        assembly = average_repetition(assembly)
    return assembly