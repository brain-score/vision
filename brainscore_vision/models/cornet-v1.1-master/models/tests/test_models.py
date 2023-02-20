import os

import numpy as np
from brainscore.benchmarks._neural_common import average_repetition, timebins_from_assembly
from brainscore.metrics.ceiling import InternalConsistency
from brainscore.metrics.transformations import Split

from brainscore.metrics.regression import CrossRegressedCorrelation, pls_regression, pearsonr_correlation

from brainio_base.assemblies import NeuroidAssembly
from brainio_base.stimuli import StimulusSet
from brainscore.benchmarks import BenchmarkBase, ceil_score
from brainscore.metrics import Score
from brainscore.model_interface import BrainModel
from model_tools.brain_transformation import ModelCommitment


def test_brain_model(module):
    module = __import__(module)
    for model in module.get_model_list():
        layers = module.get_layers(model)
        assert layers is not None
        assert isinstance(layers, list)
        assert len(layers) > 0
        model = module.get_model(model)
        assert model is not None
        assert isinstance(model, BrainModel)
        test_brain_model_processing(model, module)
    print('Test successful, you are ready to submit!')


def test_brain_model_processing(model, module):
    # to be done
    return


def test_base_models(module):
    module = __import__(module)
    for model in module.get_model_list():
        layers = module.get_layers(model)
        assert layers is not None
        assert isinstance(layers, list)
        assert len(layers) > 0
        assert module.get_model(model) is not None
        test_processing(model, module)
        print('Test successful, you are ready to submit!')


def test_processing(model, module):
    os.environ['RESULTCACHING_DISABLE'] = '1'
    model_instance = module.get_model(model)
    layers = module.get_layers(model)
    brain_model = ModelCommitment(identifier=model, activations_model=model_instance,
                                  layers=layers, region_benchmarks={'IT': _MockBenchmark()})
    brain_model.commit_region('IT')
    benchmark = _MockBenchmark()
    score = benchmark(brain_model, do_behavior=True)
    assert score is not None
    assert score.sel(aggregation='center')


class _MockBenchmark(BenchmarkBase):
    def __init__(self):
        ceiling = Score([1, np.nan], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
        assembly_repetition = get_assembly()
        assert len(np.unique(assembly_repetition['region'])) == 1
        assert hasattr(assembly_repetition, 'repetition')
        self.region = 'IT'
        self.assembly = average_repetition(assembly_repetition)
        self._assembly=self.assembly
        self.timebins = timebins_from_assembly(self.assembly)

        self._similarity_metric = CrossRegressedCorrelation(
            regression=pls_regression(), correlation=pearsonr_correlation(),
            crossvalidation_kwargs=dict(stratification_coord=Split.Defaults.stratification_coord
            if hasattr(self.assembly, Split.Defaults.stratification_coord) else None))
        identifier = f'{assembly_repetition.name}-layer_selection'
        ceiler = InternalConsistency()
        super(_MockBenchmark, self).__init__(identifier=identifier, ceiling_func=lambda: ceiler(assembly_repetition), version='1.0')

    def __call__(self, candidate: BrainModel, do_behavior=False):
        # Do brain region task
        candidate.start_recording(self.region, time_bins=self.timebins)
        source_assembly = candidate.look_at(self.assembly.stimulus_set)
        # Do behavior task
        if do_behavior:
            candidate.start_task(BrainModel.Task.probabilities, self.assembly.stimulus_set)
            candidate.look_at(self.assembly.stimulus_set)
        raw_score = self._similarity_metric(source_assembly, self.assembly)
        return ceil_score(raw_score, self.ceiling)


def get_assembly():
    image_names = []
    for i in range(1, 21):
        image_names.append(f'images/{i}.png')
    assembly = NeuroidAssembly((np.arange(40 * 5) + np.random.standard_normal(40 * 5)).reshape((5, 40, 1)),
                               coords={'image_id': (
                                   'presentation',
                                   image_names * 2),
                                   'object_name': ('presentation', ['a'] * 40),
                                   'repetition': ('presentation', ([1] * 20 + [2] * 20)),
                                   'neuroid_id': ('neuroid', np.arange(5)),
                                   'region': ('neuroid', ['IT'] * 5),
                                   'time_bin_start': ('time_bin', [70]),
                                   'time_bin_end': ('time_bin', [170])
                               },
                               dims=['neuroid', 'presentation', 'time_bin'])
    labels = ['a'] * 10 + ['b'] * 10
    stimulus_set = StimulusSet([{'image_id': image_names[i], 'object_name': 'a', 'image_label': labels[i]}
                                for i in range(20)])
    stimulus_set.image_paths = {image_name: os.path.join(os.path.dirname(__file__), image_name)
                                for image_name in image_names}
    stimulus_set.name = 'test'
    assembly.attrs['stimulus_set'] = stimulus_set
    assembly.attrs['stimulus_set_name'] = stimulus_set.name
    assembly = assembly.squeeze("time_bin")
    return assembly.transpose('presentation', 'neuroid')
