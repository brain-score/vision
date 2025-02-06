import logging

from brainio.assemblies import NeuronRecordingAssembly, walk_coords, array_is_element, DataAssembly
from brainscore_vision import data_registry, load_stimulus_set, stimulus_set_registry
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3, load_stimulus_set_from_s3

_logger = logging.getLogger(__name__)

BIBTEX = """@article {Cadena201764,
    author = {Cadena, Santiago A. and Denfield, George H. and Walker, Edgar Y. and Gatys, Leon A. and Tolias, Andreas S. and Bethge, Matthias and Ecker, Alexander S.},
    title = {Deep convolutional models improve predictions of macaque V1 responses to natural images},
    elocation-id = {201764},
    year = {2017},
    doi = {10.1101/201764},
    publisher = {Cold Spring Harbor Laboratory},
    abstract = {Despite great efforts over several decades, our best models of primary visual cortex (V1) still predict neural responses quite poorly when probed with natural stimuli, highlighting our limited understanding of the nonlinear computations in V1. At the same time, recent advances in machine learning have shown that deep neural networks can learn highly nonlinear functions for visual information processing. Two approaches based on deep learning have recently been successfully applied to neural data: transfer learning for predicting neural activity in higher areas of the primate ventral stream and data-driven models to predict retina and V1 neural activity of mice. However, so far there exists no comparison between the two approaches and neither of them has been used to model the early primate visual system. Here, we test the ability of both approaches to predict neural responses to natural images in V1 of awake monkeys. We found that both deep learning approaches outperformed classical linear-nonlinear and wavelet-based feature representations building on existing V1 encoding theories. On our dataset, transfer learning and data-driven models performed similarly, while the data-driven model employed a much simpler architecture. Thus, multi-layer CNNs set the new state of the art for predicting neural responses to natural images in primate V1. Having such good predictive in-silico models opens the door for quantitative studies of yet unknown nonlinear computations in V1 without being limited by the available experimental time.},
    URL = {https://www.biorxiv.org/content/early/2017/10/11/201764},
    eprint = {https://www.biorxiv.org/content/early/2017/10/11/201764.full.pdf},
    journal = {bioRxiv}
}
"""

# assembly
data_registry['Cadena2017'] = lambda: reindex(load_assembly_from_s3(
    identifier="tolias.Cadena2017",
    version_id="null",
    sha1="69bcaaa9370dceb0027beaa06235ef418c3d7063",
    bucket="brainscore-storage/brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Cadena2017'),
))


def reindex(assembly: DataAssembly) -> DataAssembly:  # make sure we have the expected coordinate names
    attrs = assembly.attrs
    coords = {coord: (dims, values) for coord, dims, values in walk_coords(assembly)
              if not array_is_element(dims, 'neuroid')}  # all non-neuroid dims
    coords['neuroid_id'] = ('neuroid', assembly['neuroid'].values)
    coords['region'] = ('neuroid', ['V1'] * len(assembly['neuroid']))
    assembly = type(assembly)(assembly.values, coords=coords, dims=assembly.dims)
    assembly.attrs = attrs
    return assembly


# stimulus set
stimulus_set_registry['Cadena2017'] = lambda: load_stimulus_set_from_s3(
    identifier="Cadena2017",
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="f55b174cc4540e5612cfba5e695324328064b051",
    zip_sha1="88cc2ce3ef5e197ffd1477144a2e6a68d424ef6c",
    csv_version_id="null",
    zip_version_id="null")
