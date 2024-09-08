import glob, os, re
import pandas as pd

from brainio.stimuli import StimulusSet
from dandi.dandiapi import DandiAPIClient
from extract_nwb_data import validate_nwb_file, old_validate_nwb_file
from IPython.display import display
from pathlib import Path
from pynwb import NWBHDF5IO
from pynwb.base import Images
from PIL import Image
from tqdm import tqdm

# hardcoded based on dandiset
DANDISET_NUMBER = '000812'
IMAGE_SET = 'IAPS'

def extract_number(filename):
    # Extract the number from the filename and return it as an integer
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else 0

# root_dir  = '/braintree/home/aliya277/dandi_folder_test'
root_dir = '/braintree/home/aliya277/dandi_folder_train'
test_train = 'Train'

def get_video_stimulus_set(dandiset_id, exp_path):
    video_paths = []
    with DandiAPIClient() as client:
        client.dandi_authenticate()

        filepath = os.path.join(exp_path, 'videos')

        dandiset = client.get_dandiset(dandiset_id, 'draft')
        assets_dirpath = 'VideoStimulusSet/'
        if assets_dirpath and not assets_dirpath.endswith("/"):
            assets_dirpath += "/"
        assets = list(dandiset.get_assets_with_path_prefix(assets_dirpath))

        try: os.mkdir(os.path.join(exp_path, 'videos'))
        except: pass

        # print('h', os.path.join(exp_path, 'videos'))
        # return

        for a in assets:
            filepath = Path(os.path.join(exp_path, 'videos'), a.path[len(assets_dirpath) :])
            video_paths.append(filepath)
            # filepath.parent.mkdir(parents=True, exist_ok=True)
            # a.download(filepath, chunk_size=1024 * 1024 * 8)

    return video_paths

def get_stimuli(dandiset_id, nwb_file, experiment_path, exp_name):
    VideoStimulusSet = 'StimulusSet' not in nwb_file.stimulus_template
    stimuli          = []
    stimulus_id      = 0

    # VideoStimulusSet = False
    # if "VideoStimulusSet" in os.listdir(experiment_path): 
    #     VideoStimulusSet = True
    #     list_videos = sorted(os.listdir(os.path.join(experiment_path, 'VideoStimulusSet')),key = extract_number)

    if not VideoStimulusSet:
        image_paths = []
        image_ids   = [int(x.split('_')[-1].split('.png')[0]) for x in sorted(list(nwb_file.stimulus_template[f'StimulusSet'].images), key = extract_number)]
        
        try: os.mkdir(os.path.join(experiment_path, 'images'))
        except: pass 

        print("Iterating over the images ...")
        for i in tqdm(image_ids):
            try:
                image = nwb_file.stimulus_template[f'StimulusSet'][f'exp_{exp_name}_{i}.png'][:]
                im = Image.fromarray(image)
                if not os.path.isfile(os.path.join( experiment_path, 'images', f'exp_{exp_name}_{i}.png')):
                    im.save(os.path.join( experiment_path, 'images', f'exp_{exp_name}_{i}.png'))
                image_paths.append(os.path.join( experiment_path, 'images', f'exp_{exp_name}_{i}.png'))
            
                stimuli.append({
                    'stimulus_id': stimulus_id, # ask: uploading to s3 looks for 'stimulus_path_within_store' (commented out) else 'stimulus_id' (needs to be str)
                    'image_id': stimulus_id,
                    'id': stimulus_id,
                    'stimulus_path_within_store': f"exp_{exp_name}_{i}", # uncomment and cast stimulus_id to str
                    # 'stimulus_set': exp_name,
                    'image_number': i,
                    # 'stimulus_nwb_file_path': f"{os.path.join(*experiment_path.split('/')[:-1])}/stimulus_template/StimulusSet/exp_{exp_name}_{i}.png",
                    'image_file_name': f'exp_{exp_name}_{i}.png',
                    # 'filename': f'exp_{exp_name}_{i}.png',
                    'background_id':'',
                    's':'',	
                    'rxy':'',
                    'tz':'',	
                    'category_name':'',
                    'rxz_semantic':'',	
                    'ty':'',
                    'ryz':'',
                    'object_name':'',	
                    'variation':'',	
                    'size':'',	
                    'rxy_semantic':'',
                    'ryz_semantic':'',
                    'rxz':''	                
                })
                stimulus_id += 1
            except Exception as e: 
                print(e)
    else:
        print("Iterating over the videos ...")
        video_paths = get_video_stimulus_set(dandiset_id, experiment_path)
        for i in range(len(video_paths)):
            stimuli.append({
                'stimulus_id': stimulus_id,
                'image_number': f'{i}',
                'image_file_name': f'exp_{exp_name}_{i}.mp4'
            })
            stimulus_id += 1
        # for video, i in zip(list_videos, range(len(list_videos))): 
        #     video_paths.append(os.path.join(experiment_path, 'VideoStimulusSet', video))
        #     stimuli.append({
        #             'stimulus_id': stimulus_id,
        #             # 'stimulus_path_within_store': f"{i}",
        #             # 'stimulus_set': ImageSet,
        #             'video_number': f'{i}',
        #             # 'stimulus_nwb_file_path': f"{nwb_file_name}/stimulus_template/StimulusSetTrain/external_file"
        #         })
        #     stimulus_id += 1

    # stimuli = pd.DataFrame(stimuli)   
    stimuli = StimulusSet(stimuli)
    stimulus_paths = image_paths if not VideoStimulusSet else video_paths
    stimuli.stimulus_paths = stimulus_paths
    stimuli.name = f"DataGenerationTrial_{exp_name}"
    stimuli.identifier = f"DataGenerationTrial_{exp_name}"
    return stimuli, stimulus_paths

def convert_to_stimulus_set():
    if VideoStimulusSet == False:
        image_paths = []
        stimuli     = []
        i = 0
        iterating_over_images = True
        try: os.mkdir(os.path.join(experiment_path, 'images'))
        except: pass 

        print("Iterating over the images ...")
        while iterating_over_images == True:
            try:
                # 000812 key is just StimulusSet, ask
                key = f'StimulusSet{test_train}' if f'StimulusSet{test_train}' in nwb_file.stimulus_template.keys() else 'StimulusSet'
                # image = nwb_file.stimulus_template[f'StimulusSet{test_train}'][f'exp_{ImageSet}_{i}.png'][:]
                image = nwb_file.stimulus_template[key][f'exp_{ImageSet}_{i}.png'][:]
                im = Image.fromarray(image)
                im.save(os.path.join(os.path.join(experiment_path, 'images', f'exp_{ImageSet}_{i}.png')))
                image_paths.append(os.path.join(os.path.join(experiment_path, 'images', f'exp_{ImageSet}_{i}.png')))

                stimuli.append({
                    'stimulus_id': i,
                    'stimulus_path_within_store': f"{i}",
                    'stimulus_set': ImageSet,
                    'image_number': f"{i}",
                    'stimulus_nwb_file_path': f"{nwb_file_name}/stimulus_template/StimulusSet{test_train}/exp_{ImageSet}_{i}.png"
                })
                i += 1
            except Exception as e: 
                print(e)
                iterating_over_images = False
        print(i)
        stimuli = StimulusSet(stimuli)
        stimuli.stimulus_paths = image_paths
        stimuli.name = f"{ImageSet}"
        
    else:
        print("Iterating over the videos ...")
        video_paths = []
        stimuli     = []
        for video, i in zip(list_videos, range(len(list_videos))): 
            video_paths.append(os.path.join(experiment_path, 'VideoStimulusSet', video))
            stimuli.append({
                    'stimulus_id': i,
                    'stimulus_path_within_store': f"{i}",
                    'stimulus_set': ImageSet,
                    'video_number': f"{i}",
                    'stimulus_nwb_file_path': f"{nwb_file_name}/stimulus_template/StimulusSetTrain/external_file"
                })


        stimuli = StimulusSet(stimuli)
        stimuli.stimulus_paths = video_paths
        stimuli.name = f"{ImageSet}"
        
        io.close()

    return stimuli


if __name__ == '__main__':
    experiment_path = f"/Users/caroljiang/Downloads/vision/brainscore_vision/data/data_generation_trial/"
    dandiset_id = '000812'
    # filepath = 'sub-pico/sub-pico_ecephys+image.nwb'
    filepath = 'sub-pico/sub-pico_ecephys.nwb'
    nwb_file = validate_nwb_file(dandiset_id, filepath)
    exp_name = 'IAPS'

    stimuli = get_stimuli(dandiset_id, nwb_file, experiment_path, exp_name)[0]
    display(stimuli)
    print(stimuli.name)
