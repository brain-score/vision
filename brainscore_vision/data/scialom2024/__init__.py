from brainio.assemblies import BehavioralAssembly

from brainscore_vision import data_registry, stimulus_set_registry, load_stimulus_set
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3, load_stimulus_set_from_s3

BIBTEX = ""  # to appear in a future article

DATASETS = ['rgb', 'contours', 'phosphenes-12', 'phosphenes-16', 'phosphenes-21', 'phosphenes-27', 'phosphenes-35',
            'phosphenes-46', 'phosphenes-59', 'phosphenes-77', 'phosphenes-100', 'segments-12', 'segments-16',
            'segments-21', 'segments-27', 'segments-35', 'segments-46', 'segments-59', 'segments-77', 'segments-100',
            'phosphenes-all', 'segments-all']


for dataset in DATASETS:
    # assembly
    data_registry[f'Scialom2024_{dataset}'] = lambda: load_assembly_from_s3(
        identifier=f'Scialom2024_{dataset}',
        version_id="",
        sha1="",
        bucket="brainio-brainscore",
        cls=BehavioralAssembly,
        stimulus_set_loader=lambda: load_stimulus_set(f'Scialom2024_{dataset}'),
    )

    # stimulus set
    stimulus_set_registry[f'Scialom2024_{dataset}'] = lambda: load_stimulus_set_from_s3(
        identifier=f'Scialom2024_{dataset}',
        bucket="brainio-brainscore",
        csv_sha1="",
        zip_sha1="",
        csv_version_id="",
        zip_version_id="")