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

# create functions so that users can import individual benchmarks as e.g. Geirhos2021sketchCohenKappa
for dataset in DATASETS:
    identifier = f"Geirhos2021{dataset.replace('-', '')}CohenKappa"
    globals()[identifier] = lambda dataset=dataset: _Geirhos2021CohenKappa(dataset)


class _Geirhos2021CohenKappa(BenchmarkBase):
    def __init__(self, dataset):
        self._metric = CohensKappa()
        self._assembly = LazyLoad(lambda: load_assembly(dataset))
        self._visual_degrees = 3

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
    assembly = type(assembly)(assembly.values, coords={
        coord: (dims, values) for coord, dims, values in walk_coords(assembly) if coord != 'image_id'},
                              dims=assembly.dims)
    assembly['image_id'] = 'presentation', assembly['image_lookup_id'].values
    image_id_to_lookup = dict(zip(stimulus_set['image_id'], stimulus_set['image_lookup_id']))
    stimulus_set['image_id'] = stimulus_set['image_lookup_id']
    stimulus_set.image_paths = {image_id_to_lookup[image_id]: path
                                for image_id, path in stimulus_set.image_paths.items()}
    stimulus_set['truth'] = stimulus_set[
        'image_category' if 'image_category' in stimulus_set.columns else 'category_ground_truth']
    assembly.attrs['stimulus_set'] = stimulus_set
    # drop the 40 rows with "na" as subject response -> cannot use for correlations, etc.
    assembly = assembly.where(assembly.choice != "na", drop=True)
    return assembly
