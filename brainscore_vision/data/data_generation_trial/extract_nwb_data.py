import json
import numpy as np
# import fsspec
import h5py
import remfile
import requests

from brainio.stimuli import StimulusSet
from dandi.dandiapi import DandiAPIClient
from IPython.display import display
from pynwb import NWBHDF5IO
from pynwb.base import Images
from PIL import Image

exp_name    = 'emogan' 
nwb_name    = 'ecephys'
subject     = 'pico'

nwb_file_path   = "/Users/caroljiang/Downloads/vision/brainscore_vision/data/data_generation_trial/000788/sub-{}/sub-{}_{}.nwb".format(subject, subject, nwb_name)
# experiment_path   = "/Users/caroljiang/Downloads/vision/brainscore_vision/data/data_generation_trial/000788/sub-{}".format(subject)

### Load nwb file
def load_nwb_file(nwb_file_path):
    print("Loading the NWB file ...")
    io = NWBHDF5IO(nwb_file_path, "r") 
    nwb_file = io.read()    
    return nwb_file

def old_validate_nwb_file(nwb_file_path):
    nwb_file = load_nwb_file(nwb_file_path)
    if nwb_file.electrodes and 'PSTHs_QualityApproved_ZScored_SessionMerged' in nwb_file.scratch.keys():
        return nwb_file
    else:
        raise ValueError('Type of NWB file not accepted, needs to contain electrode data.')

def validate_nwb_file(dandiset_id, nwb_file_path):
    nwb_file = stream_from_dandi(dandiset_id, nwb_file_path)
    if nwb_file.electrodes and 'PSTHs_QualityApproved_ZScored_SessionMerged' in nwb_file.scratch.keys():
        return nwb_file
    else:
        raise ValueError('Type of NWB file not accepted, needs to contain electrode data.')
    
def get_meta(nwb_file):
    s = (nwb_file.scratch['PSTHs_QualityApproved_ZScored_SessionMerged'].description.split('[start_time_ms, stop_time_ms, tb_ms]: ')[-1])
    array = s.strip('[]').split()
    # numbers = s.strip('[]').split()
    # array = np.array(numbers, dtype=int)

    return array

def generate_json_file(nwb_file, json_file_path):
    array = get_meta(nwb_file)
    # s = (nwb_file.scratch['PSTHs_QualityApproved_ZScored_SessionMerged'].description.split('[start_time_ms, stop_time_ms, tb_ms]: ')[-1])
    # array = s.strip('[]').split()
    nwb_metadata = {'start_time_ms': array[0],
                    'stop_time_ms': array[1],
                    'tb_ms': array[2],
                    'subject': nwb_file.subject.subject_id,
                    'exp_name': nwb_file.session_id
    }
    json_str = json.dumps(nwb_metadata, indent=4)

    with open(json_file_path, "w") as f:
        f.write(json_str)

def stream_from_dandi(dandiset_id, filepath):
    with DandiAPIClient() as client:
        client.dandi_authenticate()
        asset = client.get_dandiset(dandiset_id, 'draft').get_asset_by_path(filepath)
        s3_url = asset.get_content_url(follow_redirects=1, strip_query=False)

        # with NWBHDF5IO(s3_url, mode='r', driver='ros3') as io:
        #     nwb_file = io.read()

    rem_file = remfile.File(s3_url)
    h5py_file = h5py.File(rem_file, "r")
    # with h5py.File(rem_file, "r") as h5py_file:
    io = NWBHDF5IO(file=h5py_file, load_namespaces=True)
    nwb_file = io.read()
        # with NWBHDF5IO(file=h5py_file, load_namespaces=True) as io:
        #     nwb_file = io.read()

    return nwb_file


# def download_file(url, local_path):
#   with requests.get(url, stream=True) as r:
#     r.raise_for_status()
#     with open(local_path, 'wb') as f:
#       for chunk in r.iter_content(chunk_size=8192):
#         f.write(chunk)
# def stream_from_dandi(dandiset_id, filepath, local_path):
#   # Authenticate with DANDI
#     # dandiset_id = '000000'  # Replace with your dandiset ID

#     with DandiAPIClient() as client:
#         client.dandi_authenticate()
#         dandiset = client.get_dandiset(dandiset_id, 'draft')
#         assets = list(dandiset.get_assets())
#         for asset in assets:
#             print(asset.path)

#     with DandiAPIClient() as client:
#         client.dandi_authenticate()
#         asset = client.get_dandiset(dandiset_id, 'draft').get_asset_by_path(filepath)
#         # print('Asset:', asset)
#         # Generate a presigned URL
#         s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)
#         print('Presigned S3 URL:', s3_url)
#         # Test if we can access the URL directly
#         response = requests.get(s3_url)
#         print('HTTP Status Code:', response.status_code)
#         if response.status_code != 200:
#             raise Exception(f"Failed to access the S3 URL: {response.status_code}")
#         # Download the file locally
#         download_file(s3_url, local_path)
#     # Open the local file using NWBHDF5IO
#     with NWBHDF5IO(local_path, mode='r') as io:
#         nwb_file = io.read()
#     return nwb_file


if __name__ == '__main__':
    # nwb_file_path   = "/Users/caroljiang/Downloads/vision/brainscore_vision/data/data_generation_trial/000788/sub-pico/sub-pico_ecephys.nwb"
    # nwb_file = validate_nwb_file(nwb_file_path)
    dandiset_id = '000788'
    filepath = 'sub-pico/sub-pico_ecephys.nwb'
    # filepath = 'sub-491604967/sub-491604967_ses-496908818-StimB_behavior+image+ophys.nwb'
    nwb_file = stream_from_dandi(dandiset_id, filepath)

    json_file_path = "/Users/caroljiang/Downloads/vision/brainscore_vision/data/data_generation_trial/nwb_metadata.json"
    generate_json_file(nwb_file, json_file_path)
