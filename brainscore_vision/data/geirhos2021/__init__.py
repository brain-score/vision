import logging

from brainscore_vision import data_registry
from brainscore_vision.utils.s3 import load_from_s3

_logger = logging.getLogger(__name__)


DATASETS = ['colour', 'contrast', 'cue-conflict', 'edge',
            'eidolonI', 'eidolonII', 'eidolonIII',
            'false-colour', 'high-pass', 'low-pass', 'phase-scrambling', 'power-equalisation',
            'rotation', 'silhouette', 'sketch', 'stylized', 'uniform-noise']

# TODO: create dict to hold version ids.. can get sha1 for each from lookup csv, is this ok
for dataset in DATASETS:
    assembly_identifier = f'brendel.Geirhos2021_{dataset}'
    data_registry[assembly_identifier] = lambda: load_from_s3(
        identifier=assembly_identifier,
        version_id="",
        sha1="")




