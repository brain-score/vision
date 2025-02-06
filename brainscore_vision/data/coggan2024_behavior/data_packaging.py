# Created by David Coggan on 2024 06 23

# package stimuli
from brainio.stimuli import StimulusSet
from brainio.packaging import package_stimulus_set, package_data_assembly
from brainio.assemblies import BehavioralAssembly
import os
import os.path as op
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from PIL import Image


imagenet_metadata = dict(
    bear=dict(
        class_index_1K=294,
        class_name='n02132136',
        class_description='brown bear, bruin, Ursus arctos'),
    bison=dict(
        class_index_1K=347,
        class_name='n02410509',
        class_description='bison'),
    elephant=dict(
        class_index_1K=386,
        class_name='n02504458',
        class_description='African elephant, Loxodonta africana'),
    hare=dict(
        class_index_1K=331,
        class_name='n02326432',
        class_description='hare'),
    jeep=dict(
        class_index_1K=609,
        class_name='n03594945',
        class_description='jeep, landrover'),
    lamp=dict(
        class_index_1K=846,
        class_name='n04380533',
        class_description='table lamp'),
    sportsCar=dict(
        class_index_1K=817,
        class_name='n04285008',
        class_description='sports car, sport car'),
    teapot=dict(
        class_index_1K=849,
        class_name='n04398044',
        class_description='teapot')
)


"""
Stimuli used in human behavioral experiment. Each subject was shown a unique 
set of 753 occluded images so the entire stimulus set is 22560 images.
"""
trials = pd.read_parquet('/home/tonglab/david/projects/p022_occlusion/in_vivo'
                         '/behavioral/exp1/analysis/trials.parquet')
trials.object_class.replace('car', 'sportsCar', inplace=True)
trials.prediction.replace('car', 'sportsCar', inplace=True)
stimuli = []  # collect meta
stimulus_paths = {}
exp_dir = (
    '/home/tonglab/david/projects/p022_occlusion/in_vivo/behavioral/exp1/data')
subjects = sorted(trials.subject.unique())
for t, trial in trials.iterrows():
    subj = subjects.index(trial.subject)  # 0-indexed
    stim_path = exp_dir + trial.occluded_object_path.split('logFiles')[1]
    stimulus_id = f'{t:05}_sub-{subj:02}_trial-{trial.trial-1:03}'
    object_class, occluder_type, coverage, occluder_color, rep = (
        op.basename(stim_path).split('.png')[0].split('_'))
    visibility = np.round(1 - float(coverage), 1)
    object_data = imagenet_metadata[object_class]
    stimulus_paths[stimulus_id] = stim_path
    stimuli.append({
        'stimulus_id': stimulus_id,
        'subject': f'sub-{subj:02}',
        'trial': trial.trial-1,
        'object_class': object_class,
        'imagenet_class_index_1K': object_data['class_index_1K'],
        'imagenet_class_name': object_data['class_name'],
        'imagenet_class_description': object_data['class_description'],
        'occluder_type': occluder_type,
        'occluder_color': occluder_color,
        'visibility': visibility,
        'repetition': int(rep)})
stimuli = StimulusSet(stimuli)
stimuli.stimulus_paths = stimulus_paths
stimuli.name = "tong.Coggan2024_behavior"

assert len(stimuli) == 22560

packaged_stimulus_metadata = package_stimulus_set(
    catalog_name=None,
    proto_stimulus_set=stimuli,
    stimulus_set_identifier=stimuli.name,
    bucket_name="brainio-brainscore")
print(packaged_stimulus_metadata)


"""
Stimuli used to fit the models.
These are an independent set of imagenet images from the same 8 classes as the 
behavioral experiment.
"""
# get stimuli pngs
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToPILImage()])
svc_dataset = pd.read_csv('/home/tonglab/david/projects/p022_occlusion/'
    'in_silico/analysis/scripts/utils/SVC_images.csv')
svc_dataset['class'].replace('car', 'sportsCar', inplace=True)
image_dir = ('/home/tonglab/david/projects/p022_occlusion/in_silico/images'
             '/behavior_svc')
#os.makedirs(image_dir, exist_ok=True)
#for image_path in svc_dataset.filepath.values:
#    image = Image.open(image_path)
#    image = transform(image)
#    image.save(f'{image_dir}/{op.basename(image_path).split(".")[0]}.png')

# package stimuli
stimuli = []
stimulus_paths = {}
for i, row in svc_dataset.iterrows():
    stimulus_id = f'{i:04}_{op.basename(row.filepath).split(".")[0]}'
    stimulus_paths[stimulus_id] = \
        f'{image_dir}/{op.basename(row.filepath).split(".")[0]}.png'
    stimuli.append({
        'stimulus_id': stimulus_id,
        'image_label': row['class'],
        'imagenet_path': row.filepath.split('ILSVRC2012/')[-1]})
stimuli = StimulusSet(stimuli)
stimuli.stimulus_paths = stimulus_paths
stimuli.name = "tong.Coggan2024_behavior_fitting"
assert len(stimuli) == 2048
packaged_stimulus_metadata = package_stimulus_set(
    catalog_name=None,
    proto_stimulus_set=stimuli,
    stimulus_set_identifier=stimuli.name,
    bucket_name="brainscore-storage/brainio-brainscore")
print(packaged_stimulus_metadata)


"""
Human behavioral responses to occluded images.
"""
# package data assembly
predictions = trials.prediction.to_list()
stimulus_ids = stimuli.stimulus_id.to_list()
assembly = BehavioralAssembly(
    predictions,
    dims=['presentation'],
    coords={'stimulus_id': ('presentation', stimulus_ids),
            'human_accuracy': ('presentation', trials.accuracy)})
assembly.name = "tong.Coggan2024_behavior"

# upload to S3
packaged_behavioral_metadata = package_data_assembly(
    proto_data_assembly=assembly,
    assembly_identifier=assembly.name,
    stimulus_set_identifier=stimuli.name,
    assembly_class_name="BehavioralAssembly",
    bucket_name="brainscore-storage/brainio-brainscore",
    catalog_identifier=None)
print(packaged_behavioral_metadata)
