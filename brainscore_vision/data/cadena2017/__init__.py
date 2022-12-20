import logging

from brainio.assemblies import NeuronRecordingAssembly
from brainscore_vision import data_registry
from brainscore_vision.utils.s3 import load_assembly_from_s3, load_stimulus_set_from_s3

_logger = logging.getLogger(__name__)

BIBTEX = """"""

# TODO: add correct version ids
# assembly
data_registry['tolias.Cadena2017'] = lambda: load_assembly_from_s3(
    identifier="tolias.Cadena2017",
    version_id="",
    sha1="69bcaaa9370dceb0027beaa06235ef418c3d7063",
    bucket="brainio.contrib",
    cls=NeuronRecordingAssembly)

# stimulus set
data_registry['tolias.Cadena2017'] = lambda: load_stimulus_set_from_s3(
    identifier="tolias.Cadena2017",
    bucket="brainio.contrib",
    csv_sha1="f55b174cc4540e5612cfba5e695324328064b051",
    zip_sha1="88cc2ce3ef5e197ffd1477144a2e6a68d424ef6c",
    csv_version_id="",
    zip_version_id="")
