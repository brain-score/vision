import os, re

from brainio.stimuli import StimulusSet
from dandi.dandiapi import DandiAPIClient
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def extract_number(filename):
    # Extract the number from the filename and return it as an integer
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else 0

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

        for a in assets:
            filepath = Path(os.path.join(exp_path, 'videos'), a.path[len(assets_dirpath) :])
            video_paths.append(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            a.download(filepath, chunk_size=1024 * 1024 * 8)

    return video_paths

def get_stimuli(dandiset_id, nwb_file, experiment_path, exp_name):
    stimuli          = []
    stimulus_id      = 0

    try:
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
                    'stimulus_id': stimulus_id, 
                    'image_id': stimulus_id,
                    'id': stimulus_id,
                    'stimulus_path_within_store': f"exp_{exp_name}_{i}",
                    'image_number': i,
                    'image_file_name': f'exp_{exp_name}_{i}.png',
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
    except Exception as e:
        print(e)
        print('no images found')

    try:
        print("Iterating over the videos ...")
        video_paths = get_video_stimulus_set(dandiset_id, experiment_path)
        for i in tqdm(range(len(video_paths))):
            stimuli.append({
                'stimulus_id': stimulus_id,
                'stimulus_path_within_store': f"{i}",
                'image_number': f'{i}',
                'image_file_name': f'exp_{exp_name}_{i}.mp4'
            })
            stimulus_id += 1
    except Exception as e:
        print(e)
        print('no videos found')

    stimuli = StimulusSet(stimuli)
    stimulus_paths = image_paths if image_paths else video_paths
    stimuli.stimulus_paths = stimulus_paths
    stimuli.name = f"DataGenerationTrial_{exp_name}"
    stimuli.identifier = f"DataGenerationTrial_{exp_name}"
    return stimuli, stimulus_paths
