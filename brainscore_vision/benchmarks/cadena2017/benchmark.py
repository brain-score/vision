import brainscore_vision
from brainscore_vision import load_metric, load_ceiling
from brainio.assemblies import walk_coords, array_is_element
from brainscore_vision.benchmark_helpers.neural_common import NeuralBenchmark

VISUAL_DEGREES = 2
NUMBER_OF_TRIALS = 2
BIBTEX = """@article{cadena2019deep,
  title={Deep convolutional models improve predictions of macaque V1 responses to natural images},
  author={Cadena, Santiago A and Denfield, George H and Walker, Edgar Y and Gatys, Leon A and Tolias, Andreas S and Bethge, Matthias and Ecker, Alexander S},
  journal={PLoS computational biology},
  volume={15},
  number={4},
  pages={e1006897},
  year={2019},
  url={https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006897},
  publisher={Public Library of Science San Francisco, CA USA}
}"""


def ToliasCadena2017PLS():
    loader = AssemblyLoader()
    assembly_repetition = loader(average_repetition=False)
    assembly = loader(average_repetition=True)
    assembly.stimulus_set.identifier = assembly.stimulus_set_identifier

    similarity_metric = load_metric('pls', crossvalidation_kwargs={'stratification_coord': None})
    ceiler = load_ceiling('internal_consistency', split_coord='repetition_id')

    def ceiling():
        # This assembly has many stimuli that are only shown to a subset of the neurons.
        # When the loader runs with `average_repetition=True`, it automatically drops nan cells,
        # but for the `assembly_repetition`, it keeps all the rows.
        # If we now use the usual ceiling approach, the two halves will end up with NaN values
        # due to stimuli not being shown to neurons, which doesn't let us correlate.
        # Instead, we here drop all NaN cells and their corresponding stimuli,
        # which keeps only 43% of the original presentation rows, but lets us correlate again.
        assembly_nonan, stimuli = loader.dropna(assembly_repetition, assembly_repetition.attrs['stimulus_set'])
        return ceiler(assembly_nonan)

    return NeuralBenchmark(identifier='tolias.Cadena2017-pls', version=1,
                           assembly=assembly, similarity_metric=similarity_metric,
                           visual_degrees=VISUAL_DEGREES, number_of_trials=NUMBER_OF_TRIALS,
                           parent='V1', bibtex=BIBTEX,
                           ceiling_func=ceiling)


def ToliasCadena2017Mask():
    loader = AssemblyLoader()
    assembly_repetition = loader(average_repetition=False)
    assembly = loader(average_repetition=True)
    assembly.stimulus_set.identifier = assembly.stimulus_set_identifier

    similarity_metric = load_metric('mask_regression',
                                    crossvalidation_kwargs={'splits': 4, 'stratification_coord': None})
    ceiler = load_ceiling('internal_consistency', split_coord='repetition_id')
    return NeuralBenchmark(identifier='tolias.Cadena2017-mask', version=1,
                           assembly=assembly, similarity_metric=similarity_metric,
                           visual_degrees=VISUAL_DEGREES, number_of_trials=NUMBER_OF_TRIALS,
                           parent='V1', bibtex=BIBTEX,
                           ceiling_func=lambda: ceiler(assembly_repetition))


class AssemblyLoader:
    def __call__(self, average_repetition=True):
        assembly = brainscore_vision.load_dataset(identifier='Cadena2017')
        assembly = assembly.squeeze("time_bin")
        assembly = assembly.transpose('presentation', 'neuroid')
        if average_repetition:
            assembly = self.average_repetition(assembly)
        return assembly

    def _align_stimuli(self, stimulus_set, image_ids):
        stimulus_set = stimulus_set.loc[stimulus_set['stimulus_id'].isin(image_ids)]
        return stimulus_set

    def average_repetition(self, assembly):
        attrs = assembly.attrs  # workaround to keeping attrs
        presentation_coords = [coord for coord, dims, values in walk_coords(assembly)
                               if array_is_element(dims, 'presentation')]
        presentation_coords = set(presentation_coords) - {'repetition_id', 'id'}
        assembly = assembly.multi_groupby(presentation_coords).mean(dim='presentation', skipna=True)
        assembly, stimulus_set = self.dropna(assembly, stimulus_set=attrs['stimulus_set'])
        attrs['stimulus_set'] = stimulus_set
        assembly.attrs = attrs
        return assembly

    def dropna(self, assembly, stimulus_set):
        assembly = assembly.dropna('presentation')  # discard any images with NaNs (~14%)
        stimulus_set = self._align_stimuli(stimulus_set, assembly.stimulus_id.values)
        return assembly, stimulus_set
