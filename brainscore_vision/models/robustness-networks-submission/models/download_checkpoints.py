import requests
import tarfile
import sys
import os

SCRIPT_DIR = os.path.abspath(__file__)
VISUAL_CHECKPOINTS_LOCATION = os.path.join(os.path.dirname(SCRIPT_DIR), 'model_metamers_pytorch/model_analysis_folders/visual_networks/pytorch_checkpoints/'

def download_extract_remove(url, extract_location):
    temp_file_location = os.path.join(extract_location, 'temp.tar')
    print('Downloading %s to %s'%(url, temp_file_location))
    with open(temp_file_location, 'wb') as f:
        r = requests.get(url, stream=True)
        for chunk in r.raw.stream(1024, decode_content=False):
            if chunk:
                f.write(chunk)
                f.flush()
    print('Extracting %s'%temp_file_location)
    tar = tarfile.open(temp_file_location)
    tar.extractall(path=extract_location) # untar file into same directory
    tar.close()

    print('Removing temp file %s'%temp_file_location)
    os.remove(temp_file_location)

# Download the visual checkpoints (~5.5GB)
url_visual_checkpoints = 'https://mcdermottlab.mit.edu//jfeather/model_metamers_assets/pytorch_metamers_visual_model_checkpoints.tar'
download_extract_remove(url_visual_checkpoints, VISUAL_CHECKPOINTS_LOCATION)
