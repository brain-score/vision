import numpy as np

from brainio_base.assemblies import NeuroidAssembly
from brainio_base.stimuli import StimulusSet
from brainscore.benchmarks.trials import repeat_trials, average_trials


def _dummy_stimulus_set():
    stimulus_set = StimulusSet([
        {'image_id': 'a'},
        {'image_id': 'b'},
        {'image_id': 'c'},
    ])
    stimulus_set.image_paths = {
        'a': 'a.png',
        'b': 'b.png',
        'c': 'c.png',
    }
    stimulus_set.identifier = 'dummy'
    return stimulus_set


def test_integer_repeat():
    stimulus_set = _dummy_stimulus_set()
    repeat_stimulus_set = repeat_trials(stimulus_set, number_of_trials=5)
    assert len(repeat_stimulus_set) == len(stimulus_set) * 5
    original_image_paths = [stimulus_set.get_image(image_id) for image_id in stimulus_set['image_id']]
    repeat_image_paths = [repeat_stimulus_set.get_image(image_id) for image_id in repeat_stimulus_set['image_id']]
    assert set(repeat_image_paths) == set(original_image_paths)
    assert all(len(group) == 5 and set(group['repetition']) == {0, 1, 2, 3, 4}
               for name, group in repeat_stimulus_set.groupby('image_id'))
    assert repeat_stimulus_set.identifier == 'dummy-5trials'


def test_average_neural_trials():
    assembly = NeuroidAssembly([[1, 2, 3],
                                [2, 3, 4],
                                [3, 4, 5],
                                [4, 5, 6],
                                [5, 6, 7],
                                [6, 7, 8],
                                [7, 8, 9],
                                [8, 9, 10]],
                               coords={'image_id': ('presentation', ['a', 'a', 'b', 'b', 'c', 'c', 'd', 'd']),
                                       'repetition': ('presentation', [0, 1, 0, 1, 0, 1, 0, 1]),
                                       'presentation_dummy': ('presentation', ['x'] * 8),
                                       'neuroid_id': ('neuroid', [0, 1, 2]),
                                       'region': ('neuroid', ['IT', 'IT', 'IT'])},
                               dims=['presentation', 'neuroid'])
    averaged_assembly = average_trials(assembly)
    assert len(averaged_assembly['neuroid']) == 3, "messed up neuroids"
    assert len(averaged_assembly['presentation']) == 4
    assert set(averaged_assembly['image_id'].values) == {'a', 'b', 'c', 'd'}
    np.testing.assert_array_equal(averaged_assembly['neuroid_id'].values, assembly['neuroid_id'].values)
    np.testing.assert_array_equal(averaged_assembly.sel(image_id='a').values, [[1.5, 2.5, 3.5]])
    np.testing.assert_array_equal(averaged_assembly.sel(image_id='b').values, [[3.5, 4.5, 5.5]])
    np.testing.assert_array_equal(averaged_assembly.sel(image_id='c').values, [[5.5, 6.5, 7.5]])
    np.testing.assert_array_equal(averaged_assembly.sel(image_id='d').values, [[7.5, 8.5, 9.5]])


def test_average_label_trials():
    assembly = NeuroidAssembly([['a'],
                                ['a'],
                                ['a'],
                                ['b'],
                                ['b'],
                                ['a'],
                                ],
                               coords={'image_id': ('presentation', ['a', 'a', 'a', 'b', 'b', 'b']),
                                       'repetition': ('presentation', [0, 1, 2, 0, 1, 2]),
                                       'presentation_dummy': ('presentation', ['x'] * 6),
                                       'choice': ('choice', ['dummy'])},
                               dims=['presentation', 'choice'])
    averaged_assembly = average_trials(assembly)
    assert len(averaged_assembly['choice']) == 1, "messed up dimension"
    assert len(averaged_assembly['presentation']) == 2
    assert set(averaged_assembly['image_id'].values) == {'a', 'b'}
    np.testing.assert_array_equal(averaged_assembly.sel(image_id='a').values, [['a']])
    np.testing.assert_array_equal(averaged_assembly.sel(image_id='b').values, [['b']])
