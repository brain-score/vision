import logging

from brainio.assemblies import NeuronRecordingAssembly

from brainscore_vision import data_registry
from brainscore_vision.utils.s3 import load_assembly_from_s3, load_stimulus_set_from_s3

_logger = logging.getLogger(__name__)

BIBTEX = """"""

# TODO: add correct version ids
# assembly: uses below stimulus set
data_registry['dicarlo.Rajalingham2020'] = lambda: load_assembly_from_s3(
    identifier="dicarlo.Rajalingham2020",
    version_id="",
    sha1="ab95ae6c9907438f87b9b13b238244049f588680",
    bucket="brainio.dicarlo",
    cls=NeuronRecordingAssembly)

# stimulus set
data_registry['dicarlo.Rajalingham2020'] = lambda: load_stimulus_set_from_s3(
    identifier="dicarlo.Rajalingham2020",
    bucket="brainio.dicarlo",
    csv_sha1="9a9a6b3115d2d8ce5d54ec2522093d8a87ed13a0",
    zip_sha1="6097086901032e20f8ae764e9cc06e0a891a3e18",
    csv_version_id="",
    zip_version_id="")

# stimulus set: orthographic_IT
data_registry['dicarlo.Rajalingham2020orthographic_IT'] = lambda: load_stimulus_set_from_s3(
    identifier="dicarlo.Rajalingham2020orthographic_IT",
    bucket="brainio.dicarlo",
    csv_sha1="3ac9ab73b653ac9cf839f9bfde131354a3766ccd",
    zip_sha1="0e025f6f8b06e803a6d8d1a17bd25a41af3e81db",
    csv_version_id="",
    zip_version_id="")
