"""Old vision answers in the response formats, with the real old classes.

The label task (``LabelBehavior`` over a real ``PytorchWrapper`` on a tiny network with a
1000-unit ``logits`` layer) and the probabilities task (``ProbabilitiesMapping``) answer
through the vision adapter. No weights, no downloads; core and vision only.
"""
import functools

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from brainscore_core.contract import TaskContext
from brainscore_core.responses import Choice
from brainscore_core.streaming_helpers import stimulus_session
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from brainscore_vision.compat.unified_adapter import VisionModelAdapter
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper, load_preprocess_images
from brainscore_vision.model_helpers.brain_transformation.behavior import (
    LabelBehavior, ProbabilitiesMapping)

LABELS = ['car', 'cat', 'dog']
TRUTH = ['dog', 'cat', 'car', 'dog', 'cat', 'car', 'cat', 'dog']
CONDITION = ['a', 'a', 'b', 'b', 'c', 'c', 'a', 'b']


def stimuli(folder, identifier='legacy-formats'):
    rng = np.random.default_rng(0)
    ids = [f's{i}' for i in range(len(TRUTH))]
    paths = []
    for sid in ids:
        path = folder / f'{sid}.png'
        Image.fromarray(rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)).save(path)
        paths.append(str(path))
    table = StimulusSet(pd.DataFrame({'stimulus_id': ids, 'truth': TRUTH, 'condition': CONDITION}))
    table.identifier = identifier
    table.stimulus_paths = dict(zip(ids, paths))
    return table


def label_model(name='tiny-label'):
    """The real label path: LabelBehavior over a real PytorchWrapper (1000 logits)."""
    torch.manual_seed(0)
    net = torch.nn.Sequential()
    net.add_module('flat', torch.nn.Flatten())
    net.add_module('logits', torch.nn.Linear(3 * 8 * 8, 1000))
    wrapper = PytorchWrapper(identifier=name, model=net, batch_size=4,
                             preprocessing=functools.partial(load_preprocess_images, image_size=8))
    model = LabelBehavior(name, wrapper)
    model.visual_degrees = lambda: 8
    return model


def _features(stimuli, layers, number_of_trials=1, require_variance=False):
    values = np.stack([[float(int(sid.strip('sf'))), float(len(label))]
                       for sid, label in zip(stimuli['stimulus_id'], stimuli['image_label'])])
    return NeuroidAssembly(values, dims=['presentation', 'neuroid'], coords={
        'stimulus_id': ('presentation', list(stimuli['stimulus_id'])),
        'image_label': ('presentation', list(stimuli['image_label'])),
        'neuroid_id': ('neuroid', [0, 1]), 'layer': ('neuroid', ['L', 'L'])})


def fitting():
    table = StimulusSet(pd.DataFrame({'stimulus_id': [f'f{i}' for i in range(9)],
                                      'image_label': ['dog', 'cat', 'car'] * 3}))
    table.identifier = 'legacy-formats-fit'
    return table


def test_the_label_task_stores_the_old_labels_in_order_with_every_coordinate(tmp_path):
    table = stimuli(tmp_path)
    old = label_model('tiny-label-old')
    old.start_task('label', LABELS)
    expected = old.look_at(table)
    session = stimulus_session(table, ['behavior:choice'], TaskContext(response=Choice(LABELS)))
    VisionModelAdapter(label_model('tiny-label-new')).participate(session)
    stored = session.collect('behavior:choice')
    labels = list(stored['choice'].values)
    assert [labels[i] for i in np.argmax(stored.values, axis=1)] == list(expected.values[0])
    assert list(stored['raw_answer'].values) == list(expected.values[0])
    for name in expected.indexes['presentation'].names:          # every coordinate an old metric reads
        assert list(stored[name].values) == list(expected[name].values), name
    assert list(stored['response_state'].values) == ['valid'] * len(TRUTH)


def test_the_probabilities_task_stores_the_old_rows_with_their_digits():
    table = fitting().copy()
    table.identifier = 'legacy-formats-test'
    old = ProbabilitiesMapping('old', _features, 'L')
    old.start_task('probabilities', fitting())
    expected = old.look_at(table)
    session = stimulus_session(table, ['behavior:probabilities'],
                               TaskContext(response=Choice(LABELS), fitting_stimuli=fitting()))
    VisionModelAdapter(ProbabilitiesMapping('new', _features, 'L')).participate(session)
    stored = session.collect('behavior:probabilities')
    columns = list(expected['choice'].values)
    reordered = expected.values[:, [columns.index(label) for label in LABELS]]
    assert stored.values.dtype == expected.values.dtype == np.float64
    assert stored.values.tobytes() == np.ascontiguousarray(reordered).tobytes()
    assert stored.attrs['response']['parameters']['option_order'] == 'task'


def _old_and_new_label_answers(folder):
    """The label answer of the plain old model, and the one the facade gives an old benchmark
    now (a choice among the labels, through the vision adapter and a core session)."""
    from brainscore_core.legacy_facade import LegacyBenchmarkFacade
    table = stimuli(folder)
    old = label_model('tiny-label-plain')
    old.start_task('label', LABELS)
    expected = old.look_at(table)
    facade = LegacyBenchmarkFacade(VisionModelAdapter(label_model('tiny-label-facade')), visual_degrees=8)
    facade.start_task('label', LABELS)
    return expected, facade.look_at(table), facade


def test_the_facade_gives_an_old_label_benchmark_the_old_answer_exactly(tmp_path):
    import xarray as xr
    expected, received, facade = _old_and_new_label_answers(tmp_path)
    assert facade._task.response == Choice(LABELS)                 # asked as a choice
    assert received.dims == expected.dims == ('choice', 'presentation')
    assert received.values.dtype == expected.values.dtype
    np.testing.assert_array_equal(received.values, expected.values)
    assert list(received.indexes['presentation'].names) == list(expected.indexes['presentation'].names)
    for name in expected.indexes['presentation'].names:
        np.testing.assert_array_equal(received[name].values, expected[name].values, err_msg=name)
    xr.testing.assert_identical(received, expected)


def test_the_old_label_metrics_give_the_identical_number(tmp_path):
    """accuracy (Geirhos engineering, Hermann, Scialom) and error consistency (Geirhos,
    Scialom) on the old answer and on the facade's answer."""
    from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
    from brainscore_vision.metrics.accuracy.metric import Accuracy
    from brainscore_vision.metrics.error_consistency.metric import ErrorConsistency
    expected, received, _ = _old_and_new_label_answers(tmp_path)
    truth = np.array(TRUTH)
    assert float(Accuracy()(received, truth)) == float(Accuracy()(expected, truth))
    ids = [f's{i}' for i in range(len(TRUTH))]
    human = {'h1': ['dog', 'cat', 'cat', 'dog', 'car', 'car', 'cat', 'cat'],
             'h2': ['dog', 'dog', 'car', 'cat', 'cat', 'car', 'cat', 'dog']}
    target = BehavioralAssembly(
        [label for subject in human for label in human[subject]],
        coords={'stimulus_id': ('presentation', ids * 2), 'truth': ('presentation', TRUTH * 2),
                'condition': ('presentation', CONDITION * 2),
                'subject': ('presentation', [s for s in human for _ in ids])},
        dims=['presentation'])
    old_score = ErrorConsistency()(expected.squeeze('choice'), target)
    new_score = ErrorConsistency()(received.squeeze('choice'), target)
    assert float(new_score) == float(old_score)
    np.testing.assert_array_equal(new_score.attrs['raw'].values, old_score.attrs['raw'].values)


def test_a_subject_without_a_choice_output_is_refused_on_a_label_task(tmp_path):
    from brainscore_core.compatibility import ExecutionNotFeasibleError
    from brainscore_core.legacy_facade import LegacyBenchmarkFacade

    class _BareOnly(VisionModelAdapter):
        @property
        def out_channels(self):
            return {'behavior'}

    facade = LegacyBenchmarkFacade(_BareOnly(label_model()), visual_degrees=8)
    facade.start_task('label', LABELS)
    with pytest.raises(ExecutionNotFeasibleError, match="declares no 'behavior:choice' output"):
        facade.look_at(stimuli(tmp_path))


def test_the_imagenet_label_task_and_the_probabilities_task_stay_on_bare_behavior():
    from brainscore_core.legacy_facade import LegacyBenchmarkFacade
    facade = LegacyBenchmarkFacade(VisionModelAdapter(label_model()), visual_degrees=8)
    facade.start_task('label', 'imagenet')
    assert facade._task.response is None and facade._task.task_type == 'label'
    facade.start_task('probabilities', fitting())
    assert facade._task.response is None and facade._task.task_type == 'probabilities'
    facade.start_task('odd_one_out')
    assert facade._task.response is None


def test_the_behavior_arbiter_of_a_model_commitment_gives_the_same_label_answer(tmp_path):
    """The arbiter a ModelCommitment builds dispatches the label task to LabelBehavior."""
    import xarray as xr
    from brainscore_core.legacy_facade import LegacyBenchmarkFacade
    from brainscore_vision.model_helpers.brain_transformation.behavior import BehaviorArbiter
    table = stimuli(tmp_path)
    old = label_model('tiny-arbiter-plain')
    old.start_task('label', LABELS)
    expected = old.look_at(table)
    class _Arbiter(BehaviorArbiter):      # a ModelCommitment names it; the arbiter alone has no name
        identifier = 'tiny-arbiter'

    arbiter = _Arbiter({'label': label_model('tiny-arbiter'),
                        'probabilities': ProbabilitiesMapping('tiny-arbiter', _features, 'L')})
    facade = LegacyBenchmarkFacade(VisionModelAdapter(arbiter), visual_degrees=8)
    facade.start_task('label', LABELS)
    xr.testing.assert_identical(facade.look_at(table), expected)
