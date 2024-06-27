# Created by David Coggan on 2024 06 26

from pytest import approx
from brainscore_vision import benchmark_registry, load_benchmark
from brainscore_vision import load_model


def test_benchmark_registry():
    for region in ['V1', 'V2', 'V4', 'IT']:
        assert f'Coggan2024_fMRI.{region}' in benchmark_registry


def test_benchmarks():
    regions = ['V1', 'V2', 'V4', 'IT']
    expected_results = [0.0182585, 0.33471652, 0.30045866, 0.44864318]
    model = load_model('alexnet')
    for region, expected in zip(regions, expected_results):
        benchmark = load_benchmark(f'Coggan2024_fMRI.{region}')
        result = benchmark(model)
        assert result.values == approx(expected)

"""
from brainscore_vision.benchmark_helpers import PrecomputedFeatures
from brainio.packaging import write_netcdf
precomputed_test = PrecomputedTests()
features_path = 'alexnet_features.nc'
if not op.isfile(features_path):
    model_commitment = load_model('alexnet')
    layers = [model_commitment.layers[l] for l in [0, 1, 2, 4]]
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    model = model_commitment.activations_model
    stimuli = sorted(glob.glob('../data/stimuli/*.png'))
    activations = model(stimuli=stimuli, layers=layers)
    write_netcdf(activations, features_path)
    
def test_Coggan2024_fMRI():
    precomputed_test.run_test(benchmark='Coggan2024_fMRI.V1',
                              precomputed_features_filepath=features_path,
                              expected=0.0182585)
    precomputed_test.run_test(benchmark='Coggan2024_fMRI.V2',
                              precomputed_features_filepath=features_path,
                              expected=0.33471652)
    precomputed_test.run_test(benchmark='Coggan2024_fMRI.V4',
                              precomputed_features_filepath=features_path,
                              expected=0.30045866)
    precomputed_test.run_test(benchmark='Coggan2024_fMRI.IT',
                              precomputed_features_filepath=features_path,
                              expected=0.44864318)
"""

