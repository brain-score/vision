import logging
import pandas as pd
from brainio.assemblies import BehavioralAssembly

from brainscore_vision import data_registry
from brainscore_vision.utils.s3 import load_assembly_from_s3, load_stimulus_set_from_s3

_logger = logging.getLogger(__name__)

DATASETS = ['colour', 'contrast', 'cue-conflict', 'edge',
            'eidolonI', 'eidolonII', 'eidolonIII',
            'false-colour', 'high-pass', 'low-pass', 'phase-scrambling', 'power-equalisation',
            'rotation', 'silhouette', 'sketch', 'stylized', 'uniform-noise']

# create dictionary of hashes for each dataset
hash_dict = {}
geirhos_df = pd.read_csv('geirhos_hashes.csv')
geirhos_df[['general', 'dataset_id']] = geirhos_df['identifier'].str.split('_', expand=True)
for dataset_id in DATASETS:
    dataset_subset = geirhos_df[geirhos_df['dataset_id'] == dataset_id]

    # use classes (Assembly vs. Stim Set vs. Nan) to determine file type
    assembly_hash = dataset_subset[dataset_subset['class'] == 'BehavioralAssembly']['sha1'].item()
    stim_csv_hash = dataset_subset[dataset_subset['class'] == 'StimulusSet']['sha1'].item()
    stim_zip_hash = dataset_subset[dataset_subset['class'].isnull()]['sha1'].item()
    hash_dict[dataset_id] = (assembly_hash, stim_csv_hash, stim_zip_hash)

for dataset, (assembly_hash, stim_csv_hash, stim_zip_hash) in hash_dict.items():
    assembly_identifier = f'brendel.Geirhos2021_{dataset}'

    # assembly
    data_registry[assembly_identifier] = lambda: load_assembly_from_s3(
        identifier=assembly_identifier,
        version_id="",
        sha1=assembly_hash,
        bucket="brainio.dicarlo",
        cls=BehavioralAssembly)

    # stimulus set
    data_registry[assembly_identifier] = lambda: load_stimulus_set_from_s3(
        identifier=assembly_identifier,
        bucket="brainio.dicarlo",
        csv_sha1=stim_csv_hash,
        zip_sha1=stim_zip_hash,
        csv_version_id="",
        zip_version_id="")






