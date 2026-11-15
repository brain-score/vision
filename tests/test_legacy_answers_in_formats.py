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


def test_the_probabilities_task_stores_the_old_rows_cast_to_float32():
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
    np.testing.assert_array_equal(stored.values, reordered.astype(np.float32))
    assert expected.values.dtype == np.float64 and stored.values.dtype == np.float32
    assert stored.attrs['response']['parameters'] == {'cast': 'float64 to float32', 'option_order': 'task'}
