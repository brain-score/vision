import glob, os, re

from brainio.stimuli import StimulusSet
from IPython.display import display
from pynwb import NWBHDF5IO
from pynwb.base import Images
from PIL import Image

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

# experiment_file_paths = glob.glob(os.path.join(root_dir, '*'))
# print('h', experiment_file_paths)
# for experiment_path in sorted(experiment_file_paths)[0:1]: 
#     print(os.path.basename(experiment_path))
experiment_path = f"/Users/caroljiang/Downloads/vision/brainscore_vision/data/data_generation_trial/{DANDISET_NUMBER}/"
VideoStimulusSet = False
# if "VideoStimulusSet" in os.listdir(experiment_path): 
#     VideoStimulusSet = True
#     list_videos = sorted(os.listdir(os.path.join(experiment_path, 'VideoStimulusSet')),key = extract_number)
if "VideoStimulusSet" in os.listdir(experiment_path): 
    VideoStimulusSet = True
    list_videos = sorted(os.listdir(os.path.join(experiment_path, 'VideoStimulusSet')),key = extract_number)
    
# ImageSet        = os.path.basename(experiment_path)
# nwb_file_name   = os.listdir(os.path.join(experiment_path, f"{ImageSet}.sub_pico"))[0]
# nwb_file_path   = os.path.join(os.path.join(experiment_path, f"{ImageSet}.sub_pico", nwb_file_name))
ImageSet        = IMAGE_SET
nwb_file_name   = os.listdir(os.path.join(experiment_path, "sub-pico"))[0]
nwb_file_path   = os.path.join(os.path.join(experiment_path, "sub-pico", nwb_file_name))

experiment_path = ''

print("Loading the NWB file ...")
io = NWBHDF5IO(nwb_file_path, "r") 
nwb_file = io.read()    

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
    stimuli = convert_to_stimulus_set()
    display(stimuli)
    print(stimuli.name)
