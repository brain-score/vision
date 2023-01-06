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


# stores the necessary sha1's and other info corresponding to each dataset
class S3Info:
    def __init__(self, dataset_identifier, assembly_sha1, stim_csv_sha1, stim_zip_sha1):
        self.dataset_identifier = dataset_identifier
        self.assembly_sha1 = assembly_sha1
        self.stim_csv_sha1 = stim_csv_sha1
        self.stim_zip_sha1 = stim_zip_sha1


# create dictionary of info for each dataset
s3_dict = {}
geirhos_df = pd.read_csv('geirhos_hashes.csv')
geirhos_df[['general', 'dataset_id']] = geirhos_df['identifier'].str.split('_', expand=True)
for dataset_id in DATASETS:
    dataset_subset = geirhos_df[geirhos_df['dataset_id'] == dataset_id]

    # use classes (Assembly vs. Stim Set vs. Nan) to determine file type
    assembly_hash = dataset_subset[dataset_subset['class'] == 'BehavioralAssembly']['sha1'].item()
    stim_csv_hash = dataset_subset[dataset_subset['class'] == 'StimulusSet']['sha1'].item()
    stim_zip_hash = dataset_subset[dataset_subset['class'].isnull()]['sha1'].item()
    s3_dict[dataset_id] = S3Info(dataset_id, assembly_hash, stim_csv_hash, stim_zip_hash)


# 'colour'
# assembly
data_registry['brendel.Geirhos2021_colour'] = lambda: load_assembly_from_s3(
    identifier='brendel.Geirhos2021_colour',
    version_id="",
    sha1=s3_dict['colour'].assembly_sha1,
    bucket="brainio-brainscore",
    cls=BehavioralAssembly)

# stimulus set
data_registry['brendel.Geirhos2021_colour'] = lambda: load_stimulus_set_from_s3(
    identifier='brendel.Geirhos2021_colour',
    bucket="brainio-brainscore",
    csv_sha1=s3_dict['colour'].stim_csv_sha1,
    zip_sha1=s3_dict['colour'].stim_zip_sha1,
    csv_version_id="",
    zip_version_id="")

# 'contrast'
# assembly
data_registry['brendel.Geirhos2021_contrast'] = lambda: load_assembly_from_s3(
    identifier='brendel.Geirhos2021_contrast',
    version_id="",
    sha1=s3_dict['contrast'].assembly_sha1,
    bucket="brainio-brainscore",
    cls=BehavioralAssembly)

# stimulus set
data_registry['brendel.Geirhos2021_contrast'] = lambda: load_stimulus_set_from_s3(
    identifier='brendel.Geirhos2021_contrast',
    bucket="brainio-brainscore",
    csv_sha1=s3_dict['contrast'].stim_csv_sha1,
    zip_sha1=s3_dict['contrast'].stim_zip_sha1,
    csv_version_id="",
    zip_version_id="")

# 'cue-conflict'
# assembly
data_registry['brendel.Geirhos2021_cue-conflict'] = lambda: load_assembly_from_s3(
    identifier='brendel.Geirhos2021_cue-conflict',
    version_id="",
    sha1=s3_dict['cue-conflict'].assembly_sha1,
    bucket="brainio-brainscore",
    cls=BehavioralAssembly)

# stimulus set
data_registry['brendel.Geirhos2021_cue-conflict'] = lambda: load_stimulus_set_from_s3(
    identifier='brendel.Geirhos2021_cue-conflict',
    bucket="brainio-brainscore",
    csv_sha1=s3_dict['cue-conflict'].stim_csv_sha1,
    zip_sha1=s3_dict['cue-conflict'].stim_zip_sha1,
    csv_version_id="",
    zip_version_id="")

# 'edge'
# assembly
data_registry['brendel.Geirhos2021_edge'] = lambda: load_assembly_from_s3(
    identifier='brendel.Geirhos2021_edge',
    version_id="",
    sha1=s3_dict['edge'].assembly_sha1,
    bucket="brainio-brainscore",
    cls=BehavioralAssembly)

# stimulus set
data_registry['brendel.Geirhos2021_edge'] = lambda: load_stimulus_set_from_s3(
    identifier='brendel.Geirhos2021_edge',
    bucket="brainio-brainscore",
    csv_sha1=s3_dict['edge'].stim_csv_sha1,
    zip_sha1=s3_dict['edge'].stim_zip_sha1,
    csv_version_id="",
    zip_version_id="")

# 'eidolonI'
# assembly
data_registry['brendel.Geirhos2021_eidolonI'] = lambda: load_assembly_from_s3(
    identifier='brendel.Geirhos2021_eidolonI',
    version_id="",
    sha1=s3_dict['eidolonI'].assembly_sha1,
    bucket="brainio-brainscore",
    cls=BehavioralAssembly)

# stimulus set
data_registry['brendel.Geirhos2021_eidolonI'] = lambda: load_stimulus_set_from_s3(
    identifier='brendel.Geirhos2021_eidolonI',
    bucket="brainio-brainscore",
    csv_sha1=s3_dict['eidolonI'].stim_csv_sha1,
    zip_sha1=s3_dict['eidolonI'].stim_zip_sha1,
    csv_version_id="",
    zip_version_id="")

# 'eidolonII'
# assembly
data_registry['brendel.Geirhos2021_eidolonII'] = lambda: load_assembly_from_s3(
    identifier='brendel.Geirhos2021_eidolonII',
    version_id="",
    sha1=s3_dict['eidolonII'].assembly_sha1,
    bucket="brainio-brainscore",
    cls=BehavioralAssembly)

# stimulus set
data_registry['brendel.Geirhos2021_eidolonII'] = lambda: load_stimulus_set_from_s3(
    identifier='brendel.Geirhos2021_eidolonII',
    bucket="brainio-brainscore",
    csv_sha1=s3_dict['eidolonII'].stim_csv_sha1,
    zip_sha1=s3_dict['eidolonII'].stim_zip_sha1,
    csv_version_id="",
    zip_version_id="")

# 'eidolonIII'
# assembly
data_registry['brendel.Geirhos2021_eidolonIII'] = lambda: load_assembly_from_s3(
    identifier='brendel.Geirhos2021_eidolonIII',
    version_id="",
    sha1=s3_dict['eidolonIII'].assembly_sha1,
    bucket="brainio-brainscore",
    cls=BehavioralAssembly)

# stimulus set
data_registry['brendel.Geirhos2021_eidolonIII'] = lambda: load_stimulus_set_from_s3(
    identifier='brendel.Geirhos2021_eidolonIII',
    bucket="brainio-brainscore",
    csv_sha1=s3_dict['eidolonIII'].stim_csv_sha1,
    zip_sha1=s3_dict['eidolonIII'].stim_zip_sha1,
    csv_version_id="",
    zip_version_id="")

# 'false-colour'
# assembly
data_registry['brendel.Geirhos2021_false-colour'] = lambda: load_assembly_from_s3(
    identifier='brendel.Geirhos2021_false-colour',
    version_id="",
    sha1=s3_dict['false-colour'].assembly_sha1,
    bucket="brainio-brainscore",
    cls=BehavioralAssembly)

# stimulus set
data_registry['brendel.Geirhos2021_false-colour'] = lambda: load_stimulus_set_from_s3(
    identifier='brendel.Geirhos2021_false-colour',
    bucket="brainio-brainscore",
    csv_sha1=s3_dict['false-colour'].stim_csv_sha1,
    zip_sha1=s3_dict['false-colour'].stim_zip_sha1,
    csv_version_id="",
    zip_version_id="")

# 'high-pass'
# assembly
data_registry['brendel.Geirhos2021_high-pass'] = lambda: load_assembly_from_s3(
    identifier='brendel.Geirhos2021_high-pass',
    version_id="",
    sha1=s3_dict['high-pass'].assembly_sha1,
    bucket="brainio-brainscore",
    cls=BehavioralAssembly)

# stimulus set
data_registry['brendel.Geirhos2021_high-pass'] = lambda: load_stimulus_set_from_s3(
    identifier='brendel.Geirhos2021_high-pass',
    bucket="brainio-brainscore",
    csv_sha1=s3_dict['high-pass'].stim_csv_sha1,
    zip_sha1=s3_dict['high-pass'].stim_zip_sha1,
    csv_version_id="",
    zip_version_id="")

# 'low-pass'
# assembly
data_registry['brendel.Geirhos2021_low-pass'] = lambda: load_assembly_from_s3(
    identifier='brendel.Geirhos2021_low-pass',
    version_id="",
    sha1=s3_dict['low-pass'].assembly_sha1,
    bucket="brainio-brainscore",
    cls=BehavioralAssembly)

# stimulus set
data_registry['brendel.Geirhos2021_low-pass'] = lambda: load_stimulus_set_from_s3(
    identifier='brendel.Geirhos2021_low-pass',
    bucket="brainio-brainscore",
    csv_sha1=s3_dict['low-pass'].stim_csv_sha1,
    zip_sha1=s3_dict['low-pass'].stim_zip_sha1,
    csv_version_id="",
    zip_version_id="")

# 'phase-scrambling'
# assembly
data_registry['brendel.Geirhos2021_phase-scrambling'] = lambda: load_assembly_from_s3(
    identifier='brendel.Geirhos2021_phase-scrambling',
    version_id="",
    sha1=s3_dict['phase-scrambling'].assembly_sha1,
    bucket="brainio-brainscore",
    cls=BehavioralAssembly)

# stimulus set
data_registry['brendel.Geirhos2021_phase-scrambling'] = lambda: load_stimulus_set_from_s3(
    identifier='brendel.Geirhos2021_phase-scrambling',
    bucket="brainio-brainscore",
    csv_sha1=s3_dict['phase-scrambling'].stim_csv_sha1,
    zip_sha1=s3_dict['phase-scrambling'].stim_zip_sha1,
    csv_version_id="",
    zip_version_id="")

# 'power-equalisation'
# assembly
data_registry['brendel.Geirhos2021_power-equalisation'] = lambda: load_assembly_from_s3(
    identifier='brendel.Geirhos2021_power-equalisation',
    version_id="",
    sha1=s3_dict['power-equalisation'].assembly_sha1,
    bucket="brainio-brainscore",
    cls=BehavioralAssembly)

# stimulus set
data_registry['brendel.Geirhos2021_power-equalisation'] = lambda: load_stimulus_set_from_s3(
    identifier='brendel.Geirhos2021_power-equalisation',
    bucket="brainio-brainscore",
    csv_sha1=s3_dict['power-equalisation'].stim_csv_sha1,
    zip_sha1=s3_dict['power-equalisation'].stim_zip_sha1,
    csv_version_id="",
    zip_version_id="")

# 'rotation'
# assembly
data_registry['brendel.Geirhos2021_rotation'] = lambda: load_assembly_from_s3(
    identifier='brendel.Geirhos2021_rotation',
    version_id="",
    sha1=s3_dict['rotation'].assembly_sha1,
    bucket="brainio-brainscore",
    cls=BehavioralAssembly)

# stimulus set
data_registry['brendel.Geirhos2021_rotation'] = lambda: load_stimulus_set_from_s3(
    identifier='brendel.Geirhos2021_rotation',
    bucket="brainio-brainscore",
    csv_sha1=s3_dict['rotation'].stim_csv_sha1,
    zip_sha1=s3_dict['rotation'].stim_zip_sha1,
    csv_version_id="",
    zip_version_id="")

# 'silhouette'
# assembly
data_registry['brendel.Geirhos2021_silhouette'] = lambda: load_assembly_from_s3(
    identifier='brendel.Geirhos2021_silhouette',
    version_id="",
    sha1=s3_dict['silhouette'].assembly_sha1,
    bucket="brainio-brainscore",
    cls=BehavioralAssembly)

# stimulus set
data_registry['brendel.Geirhos2021_silhouette'] = lambda: load_stimulus_set_from_s3(
    identifier='brendel.Geirhos2021_silhouette',
    bucket="brainio-brainscore",
    csv_sha1=s3_dict['silhouette'].stim_csv_sha1,
    zip_sha1=s3_dict['silhouette'].stim_zip_sha1,
    csv_version_id="",
    zip_version_id="")

# 'sketch'
# assembly
data_registry['brendel.Geirhos2021_sketch'] = lambda: load_assembly_from_s3(
    identifier='brendel.Geirhos2021_sketch',
    version_id="",
    sha1=s3_dict['sketch'].assembly_sha1,
    bucket="brainio-brainscore",
    cls=BehavioralAssembly)

# stimulus set
data_registry['brendel.Geirhos2021_sketch'] = lambda: load_stimulus_set_from_s3(
    identifier='brendel.Geirhos2021_sketch',
    bucket="brainio-brainscore",
    csv_sha1=s3_dict['sketch'].stim_csv_sha1,
    zip_sha1=s3_dict['sketch'].stim_zip_sha1,
    csv_version_id="",
    zip_version_id="")

# 'stylized'
# assembly
data_registry['brendel.Geirhos2021_stylized'] = lambda: load_assembly_from_s3(
    identifier='brendel.Geirhos2021_stylized',
    version_id="",
    sha1=s3_dict['stylized'].assembly_sha1,
    bucket="brainio-brainscore",
    cls=BehavioralAssembly)

# stimulus set
data_registry['brendel.Geirhos2021_stylized'] = lambda: load_stimulus_set_from_s3(
    identifier='brendel.Geirhos2021_stylized',
    bucket="brainio-brainscore",
    csv_sha1=s3_dict['stylized'].stim_csv_sha1,
    zip_sha1=s3_dict['stylized'].stim_zip_sha1,
    csv_version_id="",
    zip_version_id="")

# 'uniform-noise'
# assembly
data_registry['brendel.Geirhos2021_uniform-noise'] = lambda: load_assembly_from_s3(
    identifier='brendel.Geirhos2021_uniform-noise',
    version_id="",
    sha1=s3_dict['uniform-noise'].assembly_sha1,
    bucket="brainio-brainscore",
    cls=BehavioralAssembly)

# stimulus set
data_registry['brendel.Geirhos2021_uniform-noise'] = lambda: load_stimulus_set_from_s3(
    identifier='brendel.Geirhos2021_uniform-noise',
    bucket="brainio-brainscore",
    csv_sha1=s3_dict['uniform-noise'].stim_csv_sha1,
    zip_sha1=s3_dict['uniform-noise'].stim_zip_sha1,
    csv_version_id="",
    zip_version_id="")

