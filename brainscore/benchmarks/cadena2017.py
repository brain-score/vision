import brainscore
from brainio_base.assemblies import walk_coords, array_is_element
from brainscore.benchmarks._neural_common import NeuralBenchmark
from brainscore.metrics.ceiling import InternalConsistency
from brainscore.metrics.regression import CrossRegressedCorrelation, mask_regression, pls_regression, \
    pearsonr_correlation

VISUAL_DEGREES = 2


def ToliasCadena2017PLS():
    loader = AssemblyLoader()
    assembly_repetition = loader(average_repetition=False)
    assembly = loader(average_repetition=True)
    assembly.stimulus_set.name = assembly.stimulus_set_name

    similarity_metric = CrossRegressedCorrelation(
        regression=pls_regression(),
        correlation=pearsonr_correlation(),
        crossvalidation_kwargs={'stratification_coord': None})
    identifier = f'tolias.Cadena2017-pls'
    ceiler = InternalConsistency(split_coord='repetition_id')

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

    return NeuralBenchmark(identifier=identifier, parent='V1', version=1,
                           assembly=assembly, similarity_metric=similarity_metric, visual_degrees=VISUAL_DEGREES,
                           ceiling_func=ceiling,
                           bibtex=
                           """@article{10.1371/journal.pcbi.1006897,
                               author = {Cadena, Santiago A. AND Denfield, George H. AND Walker, Edgar Y. AND Gatys, Leon A. AND Tolias, Andreas S. AND Bethge, Matthias AND Ecker, Alexander S.},
                               journal = {PLOS Computational Biology},
                               publisher = {Public Library of Science},
                               title = {Deep convolutional models improve predictions of macaque V1 responses to natural images},
                               year = {2019},
                               month = {04},
                               volume = {15},
                               url = {https://doi.org/10.1371/journal.pcbi.1006897},
                               pages = {1-27},
                               abstract = {Predicting the responses of sensory neurons to arbitrary natural stimuli is of major importance for understanding their function. Arguably the most studied cortical area is primary visual cortex (V1), where many models have been developed to explain its function. However, the most successful models built on neurophysiologists’ intuitions still fail to account for spiking responses to natural images. Here, we model spiking activity in primary visual cortex (V1) of monkeys using deep convolutional neural networks (CNNs), which have been successful in computer vision. We both trained CNNs directly to fit the data, and used CNNs trained to solve a high-level task (object categorization). With these approaches, we are able to outperform previous models and improve the state of the art in predicting the responses of early visual neurons to natural images. Our results have two important implications. First, since V1 is the result of several nonlinear stages, it should be modeled as such. Second, functional models of entire visual pathways, of which V1 is an early stage, do not only account for higher areas of such pathways, but also provide useful representations for V1 predictions.},
                               number = {4},
                               doi = {10.1371/journal.pcbi.1006897}
                           }""")


def ToliasCadena2017Mask():
    loader = AssemblyLoader()
    assembly_repetition = loader(average_repetition=False)
    assembly = loader(average_repetition=True)
    assembly.stimulus_set.name = assembly.stimulus_set_name

    similarity_metric = CrossRegressedCorrelation(
        regression=mask_regression(),
        correlation=pearsonr_correlation(),
        crossvalidation_kwargs={'splits': 4, 'stratification_coord': None})
    identifier = f'tolias.Cadena2017-mask'
    ceiler = InternalConsistency(split_coord='repetition_id')
    return NeuralBenchmark(identifier=identifier, version=1,
                           assembly=assembly, similarity_metric=similarity_metric, visual_degrees=VISUAL_DEGREES,
                           ceiling_func=lambda: ceiler(assembly_repetition))


class AssemblyLoader:
    name = 'tolias.Cadena2017'

    def __call__(self, average_repetition=True):
        assembly = brainscore.get_assembly(name='tolias.Cadena2017')
        assembly = assembly.rename({'neuroid': 'neuroid_id'}).stack(neuroid=['neuroid_id'])
        assembly.load()
        assembly['region'] = 'neuroid', ['V1'] * len(assembly['neuroid'])
        assembly = assembly.squeeze("time_bin")
        assembly = assembly.transpose('presentation', 'neuroid')
        if average_repetition:
            assembly = self.average_repetition(assembly)
        return assembly

    def _align_stimuli(self, stimulus_set, image_ids):
        stimulus_set = stimulus_set.loc[stimulus_set['image_id'].isin(image_ids)]
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
        stimulus_set = self._align_stimuli(stimulus_set, assembly.image_id.values)
        return assembly, stimulus_set
