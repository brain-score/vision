import numpy as np
from brainscore_core.supported_data_standards.brainio.assemblies import walk_coords
from brainscore_vision import load_dataset, load_metric
from brainscore_vision.benchmark_helpers.screen import place_on_screen
from brainscore_vision.benchmarks import BenchmarkBase
from brainscore_vision.metrics import Score
from brainscore_vision.model_interface import BrainModel
from brainscore_vision.utils import LazyLoad
from brainscore_vision.benchmark_helpers import bound_score

BIBTEX = """@article{Kato2026,
    title = {Systematic image perturbations reveal persistent gaps between human and machine vision},
    volume = {},
    doi = {10.1016/j.isci.2026.117370},
    journal = {iScience},
    author = {Kato, Mugihiko, and He, Biyu},
    year = {2026},
    }"""

DATASETS = ['Ori', 'Fil', 'Lin', 'SilTex', 'Sil', 'Tex', 'LinFil', 'Sparse', 'Dense']

# create functions so that users can import individual benchmarks as e.g. Geirhos2021sketchErrorConsistency
for dataset in DATASETS:
    # behavioral benchmark
    identifier = f"Kato2026{dataset.replace('-', '')}ErrorConsistency"
    globals()[identifier] = lambda dataset=dataset: _Kato2026ErrorConsistency(dataset)
    # engineering benchmark
    identifier = f"Kato2026{dataset.replace('-', '')}AccuracyDistance"
    globals()[identifier] = lambda dataset=dataset: _Kato2026AccuracyDistance(dataset)

class _Kato2026ErrorConsistency(BenchmarkBase):
    # behavioral benchmark
    def __init__(self, dataset):
        self._metric = load_metric('kato_error_consistency')
        self._assembly = LazyLoad(lambda: load_assembly(dataset))
        if dataset in ['Sparse', 'Dense']:
            self._assembly_base = LazyLoad(lambda: load_assembly('Sil'))
        elif dataset != 'Ori':
            self._assembly_base = LazyLoad(lambda: load_assembly('Ori'))
        self._visual_degrees = 10
        self._number_of_trials = 1
        
        dataset_nodashes = dataset.replace("-", "")
        super(_Kato2026ErrorConsistency, self).__init__(
            identifier=f'Kato2026{dataset_nodashes}-error_consistency', version=1,
            ceiling_func=lambda: self._metric.ceiling(self._assembly),
            parent='Kato2026',
            bibtex=BIBTEX)

    def __call__(self, candidate: BrainModel):
        choice_labels = set(self._assembly['truth'].values)
        choice_labels = list(sorted(choice_labels))
        # add one elements to choice_labels "others"
        choice_labels.append("kato2026_others")
        candidate.start_task(BrainModel.Task.label, choice_labels)
        stimulus_set = place_on_screen(self._assembly.stimulus_set, target_visual_degrees=candidate.visual_degrees(),
                                    source_visual_degrees=self._visual_degrees)
        labels = candidate.look_at(stimulus_set, number_of_trials=self._number_of_trials)
        
        # only consider images for which the model is correct in the base condition
        if hasattr(self, '_assembly_base'):
            candidate.start_task(BrainModel.Task.label, choice_labels)
            stimulus_set_base = place_on_screen(self._assembly_base.stimulus_set, target_visual_degrees=candidate.visual_degrees(),
                                        source_visual_degrees=self._visual_degrees)
            labels_base = candidate.look_at(stimulus_set_base, number_of_trials=self._number_of_trials)
            # compare labels_base['stimulus_id'].values and labels['stimulus_id'].values and if they are not the same, remove those that only exist in one of them
            base_key = np.array([s.rsplit('_', 1)[0] for s in labels_base['stimulus_id'].values])
            key = np.array([s.rsplit('_', 1)[0] for s in labels['stimulus_id'].values])
            common_stimuli = np.intersect1d(base_key, key)
            mask        = np.isin(base_key, common_stimuli)
            labels_base = labels_base.isel(presentation=mask)
            base_correct = np.asarray(labels_base.values) == np.asarray(labels_base['truth'].values)
            valid  = base_correct.ravel()
            labels = labels.where(valid, other='NaN')
        
        raw_score = self._metric(labels, self._assembly)
        ceiling = self.ceiling
        score = raw_score / ceiling
        bound_score(score)
        score.attrs['raw'] = raw_score
        score.attrs['ceiling'] = ceiling
        return score

class _Kato2026AccuracyDistance(BenchmarkBase):
    # behavioral benchmark
    def __init__(self, dataset):
        self._metric = load_metric('kato_accuracy_distance')
        self._assembly = LazyLoad(lambda: load_assembly(dataset))
        if dataset in ['Sparse', 'Dense']:
            self._assembly_base = LazyLoad(lambda: load_assembly('Sil'))
        elif dataset != 'Ori':
            self._assembly_base = LazyLoad(lambda: load_assembly('Ori'))
        self._visual_degrees = 10
        self._number_of_trials = 1
        
        dataset_nodashes = dataset.replace("-", "")
        super(_Kato2026AccuracyDistance, self).__init__(
            identifier=f'Kato2026{dataset_nodashes}-accuracy_distance', version=1,
            ceiling_func=lambda: self._metric.ceiling(self._assembly),
            parent='Kato2026',
            bibtex=BIBTEX)

    def __call__(self, candidate: BrainModel):
        choice_labels = set(self._assembly['truth'].values)
        choice_labels = list(sorted(choice_labels))
        # add one elements to choice_labels "others"
        choice_labels.append("kato2026_others")
        candidate.start_task(BrainModel.Task.label, choice_labels)
        stimulus_set = place_on_screen(self._assembly.stimulus_set, target_visual_degrees=candidate.visual_degrees(),
                                    source_visual_degrees=self._visual_degrees)
        labels = candidate.look_at(stimulus_set, number_of_trials=self._number_of_trials)
        
        # only consider images for which the model is correct in the base condition
        if hasattr(self, '_assembly_base'):
            candidate.start_task(BrainModel.Task.label, choice_labels)
            stimulus_set_base = place_on_screen(self._assembly_base.stimulus_set, target_visual_degrees=candidate.visual_degrees(),
                                        source_visual_degrees=self._visual_degrees)
            labels_base = candidate.look_at(stimulus_set_base, number_of_trials=self._number_of_trials)
            # compare labels_base['stimulus_id'].values and labels['stimulus_id'].values and if they are not the same, remove those that only exist in one of them
            base_key = np.array([s.rsplit('_', 1)[0] for s in labels_base['stimulus_id'].values])
            key = np.array([s.rsplit('_', 1)[0] for s in labels['stimulus_id'].values])
            common_stimuli = np.intersect1d(base_key, key)
            mask        = np.isin(base_key, common_stimuli)
            labels_base = labels_base.isel(presentation=mask)
            base_correct = np.asarray(labels_base.values) == np.asarray(labels_base['truth'].values)
            valid  = base_correct.ravel()
            labels = labels.where(valid, other='NaN')
            
        raw_score = self._metric(labels, self._assembly)
        ceiling = self.ceiling
        score = raw_score / ceiling
        bound_score(score)
        score.attrs['raw'] = raw_score
        score.attrs['ceiling'] = ceiling
        return score

def load_assembly(dataset):
    assembly = load_dataset(f'Kato2026.{dataset}')
    return assembly

