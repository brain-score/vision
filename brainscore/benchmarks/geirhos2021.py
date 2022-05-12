import numpy as np

import brainscore
from brainio.assemblies import walk_coords
from brainscore.benchmarks import BenchmarkBase
from brainscore.benchmarks.screen import place_on_screen
from brainscore.metrics.cohen_kappa import CohensKappa
from brainscore.model_interface import BrainModel
from brainscore.utils import LazyLoad

BIBTEX = """@article{geirhos2021partial,
              title={Partial success in closing the gap between human and machine vision},
              author={Geirhos, Robert and Narayanappa, Kantharaju and Mitzkus, Benjamin and Thieringer, Tizian and Bethge, Matthias and Wichmann, Felix A and Brendel, Wieland},
              journal={Advances in Neural Information Processing Systems},
              volume={34},
              year={2021}
        }"""

DATASETS = ['colour', 'contrast', 'edge',  # FIXME 'colour', 'contrast', 'cue-conflict', 'edge',
            'eidolonI', 'eidolonII', 'eidolonIII',
            'false-colour', 'high-pass', 'low-pass', 'phase-scrambling', 'power-equalisation',
            'rotation', 'silhouette', 'sketch', 'stylized', 'uniform-noise']

# exclusion criteria: (from https://github.com/bethgelab/model-vs-human/blob/1a2dc996349cc6560bb8a98734d2882b3b308585/modelvshuman/plotting/plot.py#L43)
# - not OOD: control condition without manipulation (e.g. 100% contrast)
# - mean human accuracy < 0.2 (error consistency etc. not meaningful)
EXCLUDE_CONDITIONS = {
    "colour": ["cr"],
    "contrast": ["c100", "c03", "c01"],
    "high-pass": [np.inf, 0.55, 0.45, 0.4],
    "low-pass": [0, 15, 40],
    "phase-scrambling": [0, 150, 180],
    "power-equalisation": ["0"],
    "false-colour": ["True"],
    "rotation": [0],
    "eidolonI": ["1-10-10", "64-10-10", "128-10-10"],
    "eidolonII": ["1-3-10", "32-3-10", "64-3-10", "128-3-10"],
    "eidolonIII": ["1-0-10", "16-0-10", "32-0-10", "64-0-10", "128-0-10"],
    "uniform-noise": [0.0, 0.6, 0.9]
}

# create functions so that users can import individual benchmarks as e.g. Geirhos2021sketchCohenKappa
for dataset in DATASETS:
    identifier = f"Geirhos2021{dataset.replace('-', '')}CohenKappa"
    globals()[identifier] = lambda dataset=dataset: _Geirhos2021CohenKappa(dataset)


class _Geirhos2021CohenKappa(BenchmarkBase):
    def __init__(self, dataset):
        self._metric = CohensKappa()
        self._assembly = LazyLoad(lambda: load_assembly(dataset))
        self._visual_degrees = 8  # FIXME: 3

        self._number_of_trials = 1

        super(_Geirhos2021CohenKappa, self).__init__(
            identifier=f'brendel.Geirhos2021{dataset}-cohen_kappa', version=1,
            ceiling_func=lambda: self._metric.ceiling(self._assembly),
            parent='behavior',
            bibtex=BIBTEX)

    def __call__(self, candidate: BrainModel):
        choice_labels = set(self._assembly['truth'].values)
        choice_labels = list(sorted(choice_labels))
        candidate.start_task(BrainModel.Task.label, choice_labels)
        stimulus_set = place_on_screen(self._assembly.stimulus_set, target_visual_degrees=candidate.visual_degrees(),
                                       source_visual_degrees=self._visual_degrees)
        labels = candidate.look_at(stimulus_set, number_of_trials=self._number_of_trials)
        raw_score = self._metric(labels, self._assembly)
        ceiling = self.ceiling
        score = raw_score / ceiling
        score.attrs['raw'] = raw_score
        score.attrs['ceiling'] = ceiling
        return score


def load_assembly(dataset):
    assembly = brainscore.get_assembly(f'brendel.Geirhos2021_{dataset}')
    # FIXME
    stimulus_set = assembly.stimulus_set
    # fix: use unique image_id referencing between assembly + stimulus_set
    assembly = type(assembly)(assembly.values, coords={
        coord: (dims, values) for coord, dims, values in walk_coords(assembly) if coord != 'image_id'},
                              dims=assembly.dims)
    assembly['image_id'] = 'presentation', assembly['image_lookup_id'].values
    image_id_to_lookup = dict(zip(stimulus_set['image_id'], stimulus_set['image_lookup_id']))
    stimulus_set['image_id'] = stimulus_set['image_lookup_id']
    stimulus_set.image_paths = {image_id_to_lookup[image_id]: path
                                for image_id, path in stimulus_set.image_paths.items()}
    # fix: add truth
    stimulus_set['truth'] = stimulus_set[
        'image_category' if 'image_category' in stimulus_set.columns else 'category_ground_truth']
    # fix: add condition
    image_id_to_condition = dict(zip(assembly['image_id'].values, assembly['condition'].values))
    stimulus_set['condition'] = [image_id_to_condition[image_id] for image_id in stimulus_set['image_id']]
    assembly.attrs['stimulus_set'] = stimulus_set

    # exclude conditions following the paper
    if dataset in EXCLUDE_CONDITIONS:
        excluded = EXCLUDE_CONDITIONS[dataset]
        assert all(condition in assembly['condition'].values for condition in excluded)
        assembly = assembly[{'presentation': ~np.isin(assembly['condition'], excluded)}]
        stimulus_set = assembly.attrs['stimulus_set']
        stimulus_set = stimulus_set[stimulus_set['image_id'].isin(set(assembly['image_id'].values))]
        assembly.attrs['stimulus_set'] = stimulus_set
    return assembly
