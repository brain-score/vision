import h5py
import numpy as np
import os
import pandas as pd
from PIL import Image
from matplotlib import pyplot
from numpy.random import RandomState
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

from brainio.assemblies import BehavioralAssembly
from brainio.packaging import package_stimulus_set, package_data_assembly
from brainio.stimuli import StimulusSet


def collect_stimuli(data_path, stimuli_dir):
    stimuli = []
    image_paths = {}
    with h5py.File(data_path, 'r') as f:
        meta = f['meta']
        os.makedirs(stimuli_dir, exist_ok=True)
        for image_id, filename, reference, obj, sample_number, object_number, phase in tqdm(zip(
                *[meta[key][0] for key in ['id', 'filename', 'img', 'obj', 'sampleNr', 'objNr', 'phase']]),
                desc='stimuli', total=len(meta['img'][0])):
            image = np.array(f[reference])
            image = image.transpose([1, 2, 0])
            image_id = ''.join([chr(i) for i in f[image_id]])
            object_name = ''.join(chr(i) for i in f[f[obj][0][0]])
            sample_number = int(f[sample_number][0][0])
            object_number = int(f[object_number][0][0])
            filename = ''.join([chr(i) for i in f[filename]])
            phase = ''.join([chr(i) for i in f[phase]])  # only used for monkey training
            target_path = os.path.join(stimuli_dir, filename)
            image = np.rot90(np.fliplr(image))  # images are mis-arranged in original dataset for some reason
            pil_image = Image.fromarray(image)
            pil_image.save(target_path)
            image_paths[image_id] = target_path
            stimuli.append({
                'image_path_within_store': filename,
                'image_id': image_id,
                'object_name': object_name,
                'image_label': object_name,
                'sample_number': sample_number,
                'object_number': object_number,
                'phase': phase,
            })
    stimuli = StimulusSet(stimuli)
    stimuli.image_paths = image_paths
    return stimuli


def collect_data(data_path, stimuli):
    responses = []
    with h5py.File(data_path, 'r') as f:
        data = f['data']
        keys = ['sample_number', 'sample_object_number', 'distractor_number', 'correct']
        for trial in tqdm(np.array(data['dataTurk'], dtype=int).transpose(), desc='trials'):
            response = {key: value for key, value in zip(keys, trial)}
            meta = stimuli[stimuli['sample_number'] == response['sample_number']]
            assert len(meta) == 1
            meta = meta.iloc[0]
            response['sample_object'] = response['truth'] = meta['obj']
            response['image_id'] = meta['image_id']
            distractor_meta = stimuli[stimuli['object_number'] == response['distractor_number']]
            distractor_obj = set(distractor_meta['obj'])
            assert len(distractor_obj) == 1
            response['distractor_object'] = list(distractor_obj)[0]
            response['choice'] = response['sample_object'] if response['correct'] else response['distractor_object']
            responses.append(response)
    responses = pd.DataFrame(responses)
    # convert to xarray
    responses = BehavioralAssembly(responses['choice'],
                                   coords={column: ('presentation', responses[column]) for column in responses.columns},
                                   dims=['presentation'])
    return responses


def main():
    # collect
    data_dir = Path(__file__).parent
    data_path = data_dir / 'coco_data.mat'
    stimuli = collect_stimuli(data_path, stimuli_dir=data_dir / 'stimuli')
    assembly = collect_data(data_path, stimuli)
    assert len(stimuli) == 1_600
    assert len(assembly) == 179_660
    assert len(set(assembly['image_id'].values)) == 1_600
    assert set(assembly['image_id'].values) == set(stimuli['image_id'])
    assert len(set(assembly['choice'].values)) == len(set(assembly['sample_object'].values)) \
           == len(set(assembly['distractor_object'].values)) == 10

    assembly.name = 'dicarlo.Kar2018coco_behavior'
    stimuli.name = 'dicarlo.Kar2018coco_color'

    # visualize
    random_state = RandomState(0)
    side_length = 5
    image_ids = random_state.choice(stimuli.image_id, size=side_length * side_length, replace=False)
    fig, axes = pyplot.subplots(nrows=side_length, ncols=side_length, figsize=(side_length * 2, side_length * 2))
    for image_id, ax in zip(image_ids, axes.flatten()):
        image_path = stimuli.get_image(image_id)
        image = Image.open(image_path)
        ax.imshow(image)
        ax.axis('off')
    fig.show()

    # split into private / public (fitting)
    print("Splitting into public/private")
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=15)
    public_idx, private_idx = next(split.split(stimuli['image_id'].values, stimuli['object_name'].values))
    public_stimuli, private_stimuli = stimuli.iloc[public_idx], stimuli.iloc[private_idx]
    assert len(public_stimuli) == 1_280 and len(private_stimuli) == 320
    public_stimuli.name, private_stimuli.name = stimuli.name + '.public', stimuli.name + '.private'
    public_assembly = assembly[{'presentation': [image_id in public_stimuli['image_id'].values
                                                 for image_id in assembly['image_id'].values]}]
    private_assembly = assembly[{'presentation': [image_id in private_stimuli['image_id'].values
                                                  for image_id in assembly['image_id'].values]}]
    assert len(public_assembly) == 143_492 and len(private_assembly) == 36_168
    public_assembly.name, private_assembly.name = assembly.name + '.public', assembly.name + '.private'

    # package (upload into shared S3 repository)
    print("Packaging")
    for stimuli, assembly in zip([public_stimuli, private_stimuli], [public_assembly, private_assembly]):
        package_stimulus_set(stimuli, stimulus_set_identifier=stimuli.name)
        package_data_assembly(assembly, assembly_identifier=assembly.name, stimulus_set_identifier=stimuli.name,
                              assembly_class='BehavioralAssembly')


if __name__ == '__main__':
    main()
