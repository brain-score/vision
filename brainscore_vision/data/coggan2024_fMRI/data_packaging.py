# Created by David Coggan on 2024 06 23

from brainio.stimuli import StimulusSet
from brainio.packaging import package_stimulus_set
import glob
import os.path as op
from brainio.assemblies import DataAssembly, NeuroidAssembly
from brainio.packaging import package_data_assembly
import pickle as pkl
from itertools import product as itp
import numpy as np


# imagenet metadata
imagenet_metadata = dict(
    bear=dict(
        class_index_1K=294,
        class_name='n02132136',
        class_description='brown bear, bruin, Ursus arctos',
        path='val/n02132136/ILSVRC2012_val_00049345.jpg'),
    bison=dict(
        class_index_1K=347,
        class_name='n02410509',
        class_description='bison',
        path='val/n02410509/ILSVRC2012_val_00048511.jpg'),
    elephant=dict(
        class_index_1K=386,
        class_name='n02504458',
        class_description='African elephant, Loxodonta africana',
        path='val/n02504458/ILSVRC2012_val_00030840.jpg'),
    hare=dict(
        class_index_1K=331,
        class_name='n02326432',
        class_description='hare',
        path='val/n02326432/ILSVRC2012_val_00004064.jpg'),
    jeep=dict(
        class_index_1K=609,
        class_name='n03594945',
        class_description='jeep, landrover',
        path='val/n03594945/ILSVRC2012_val_00036304.jpg'),
    lamp=dict(
        class_index_1K=846,
        class_name='n04380533',
        class_description='table lamp',
        path='val/n04380533/ILSVRC2012_val_00001055.jpg'),
    sportsCar=dict(
        class_index_1K=817,
        class_name='n04285008',
        class_description='sports car, sport car',
        path='val/n04285008/ILSVRC2012_val_00001247.jpg'),
    teapot=dict(
        class_index_1K=849,
        class_name='n04398044',
        class_description='teapot',
        path='val/n04398044/ILSVRC2012_val_00033663.jpg')
)

# stimuli
stimuli = []  # collect meta
stimulus_paths = {}  # collect mapping of stimulus_id to filepath
for f, filepath in enumerate(sorted(glob.glob('stimuli/*.png'))):
    stimulus_id = op.basename(filepath).split('.')[0]
    object_name, occlusion_condition = stimulus_id.split('_')
    occlusion_condition = occlusion_condition.split('.')[0]
    stimulus_paths[stimulus_id] = filepath
    object_data = imagenet_metadata[object_name]
    stimuli.append({
        'stimulus_id': stimulus_id,
        'object_name': object_name,
        'occlusion_condition': occlusion_condition,
        'imagenet_class_index_1K': object_data['class_index_1K'],
        'imagenet_class_name': object_data['class_name'],
        'imagenet_class_description': object_data['class_description'],
        'imagenet_path': object_data['path'],
    })
stimuli = StimulusSet(stimuli)
stimuli.stimulus_paths = stimulus_paths
stimuli.name = 'coggan2024_fMRI'
"""
packaged_stimulus_metadata = package_stimulus_set(
    catalog_name=None,
    proto_stimulus_set=stimuli,
    stimulus_set_identifier=stimuli.name,
    bucket_name="brainscore-storage/brainio-brainscore")
pkl.dump(packaged_stimulus_metadata, open('packaged_stimulus_metadata.pkl', 'wb'))
print(packaged_stimulus_metadata)
"""


# fMRI data
dataset = pkl.load(open('dataset.pkl', 'rb'))
regions = ['V1', 'V2', 'V4', 'IT']
subjects = list(dataset['V1'].keys())
n_subs = len(subjects)
off_diag_indices = np.array(1 - np.eye(24).flatten(), dtype=bool)
data_all, splits_all, sides_all, subjects_all, regions_all = [], [], [], [], []
for region, subject in itp(regions, subjects):
    data_all.append(dataset[region][subject])
    subjects_all.append(subject)
    regions_all.append(region)
data_all = np.stack(data_all, axis=2)
assembly = NeuroidAssembly(
    data_all, dims=['presentation', 'presentation', 'neuroid'],
    coords={
        'stimulus_id': ('presentation', stimuli.stimulus_id),
        'object_name': ('presentation', stimuli.object_name),
        'occlusion_condition': ('presentation', stimuli.occlusion_condition),
        'subject': ('neuroid', subjects_all),
        'region': ('neuroid', regions_all),
})
assembly.name = 'coggan2024_fMRI'

packaged_neural_metadata = package_data_assembly(
    proto_data_assembly=assembly,
    assembly_identifier=assembly.name,
    stimulus_set_identifier=stimuli.name,
    assembly_class_name="NeuroidAssembly",
    bucket_name="brainscore-storage/brainio-brainscore",
    catalog_identifier=None)

# save the packaged metadata
pkl.dump(packaged_neural_metadata, open('packaged_neural_metadata.pkl', 'wb'))
print(packaged_neural_metadata)