import logging

from brainio.assemblies import BehavioralAssembly

from brainscore_vision import data_registry
from brainscore_vision.utils.s3 import load_assembly_from_s3

_logger = logging.getLogger(__name__)

BIBTEX = """"""


# TODO: add correct version ids, discuss public vs. private
# public assembly: uses dicarlo.objectome.public stimuli
data_registry['dicarlo.Rajalingham2018.public'] = lambda: load_assembly_from_s3(
    identifier="dicarlo.Rajalingham2018.public",
    version_id="",
    sha1="34c6a8b6f7c523589c1861e4123232e5f7c7df4c",
    bucket="brainio.dicarlo",
    cls=BehavioralAssembly)

# private assembly: uses dicarlo.objectome.private stimuli
data_registry['dicarlo.Rajalingham2018.private'] = lambda: load_assembly_from_s3(
    identifier="dicarlo.Rajalingham2018.private",
    version_id="",
    sha1="516f13793d1c5b72bb445bb4008448ce97a02d23",
    bucket="brainio.dicarlo",
    cls=BehavioralAssembly)
