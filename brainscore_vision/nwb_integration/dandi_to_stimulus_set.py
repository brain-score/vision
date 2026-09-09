import os, re
import logging
from brainio.stimuli import StimulusSet
from dandi.dandiapi import DandiAPIClient
from pathlib import Path
from PIL import Image
from pynwb.file import NWBFile
from tqdm import tqdm
from typing import Tuple


logger = logging.getLogger(__name__)

def extract_number(filename: str) -> int:
    #-----------------------------------------------------------------------------------------------------------------------------
    # Extract the number from the filename and return it as an integer
    #-----------------------------------------------------------------------------------------------------------------------------
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else 0

def get_video_stimulus_set(dandiset_id: int, exp_path: str) -> list:
    #-----------------------------------------------------------------------------------------------------------------------------
    # Parses through video stimuli and returns list of paths. Attempts iterating through 
    # local file then if unable to do so, attempts streaming from DANDI.
    #-----------------------------------------------------------------------------------------------------------------------------
    video_paths = []
    try:
        list_videos = sorted(os.listdir(os.path.join(exp_path, 'VideoStimulusSet')),key = extract_number)
        video_paths = [os.path.join(exp_path, 'VideoStimulusSet', video) for video in list_videos]
    except:
        try:
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
        except Exception as e:
            logger.error(e)

    return video_paths

def get_stimuli(dandiset_id: str, nwb_file: NWBFile, experiment_path: str, exp_name: str) -> Tuple[StimulusSet, list]:
    #-----------------------------------------------------------------------------------------------------------------------------
    # Iterates over the stimuli and packages a StimulusSet with the corresponding information
    #-----------------------------------------------------------------------------------------------------------------------------
    stimuli          = []
    stimulus_id      = 0

    try:
        image_paths = []
        image_ids   = [int(x.split('_')[-1].split('.png')[0]) for x in sorted(list(nwb_file.stimulus_template[f'StimulusSet'].images), key = extract_number)]
        
        try: os.mkdir(os.path.join(experiment_path, 'images'))
        except: pass 

        logger.info("Iterating over the images ...")
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
                logger.error(e)
    except Exception as e:
        logger.error(e)
        logger.info('No images found')

    try:
        logger.info("Iterating over the videos ...")
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
        logger.error(e)
        logger.info('No videos found')

    stimuli = StimulusSet(stimuli)
    stimulus_paths = image_paths if image_paths else video_paths
    if not stimulus_paths:
        raise ValueError('No stimuli found')
    
    stimuli.stimulus_paths = stimulus_paths
    stimuli.name = f"DataGenerationTrial_{exp_name}"
    stimuli.identifier = f"DataGenerationTrial_{exp_name}"
    return stimuli, stimulus_paths
