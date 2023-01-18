import logging

from brainio.assemblies import NeuronRecordingAssembly
from brainscore_vision import data_registry
from brainscore_vision.utils.s3 import load_assembly_from_s3, load_stimulus_set_from_s3
from brainscore_vision.data_helpers.helper import version_id_df, build_filename

# extract version ids from version_ids csv
assembly_version = version_id_df.at[build_filename('tolias.Cadena2017', '.nc'), 'version_id']
csv_version = version_id_df.at[build_filename('tolias.Cadena2017', '.csv'), 'version_id']
zip_version = version_id_df.at[build_filename('tolias.Cadena2017', '.zip'), 'version_id']

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
data_registry['tolias.Cadena2017'] = lambda: load_assembly_from_s3(
    identifier="tolias.Cadena2017",
    version_id=assembly_version,
    sha1="69bcaaa9370dceb0027beaa06235ef418c3d7063",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly)

# stimulus set
data_registry['tolias.Cadena2017'] = lambda: load_stimulus_set_from_s3(
    identifier="tolias.Cadena2017",
    bucket="brainio-brainscore",
    csv_sha1="f55b174cc4540e5612cfba5e695324328064b051",
    zip_sha1="88cc2ce3ef5e197ffd1477144a2e6a68d424ef6c",
    csv_version_id=csv_version,
    zip_version_id=zip_version)
