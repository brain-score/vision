import json
import h5py
import remfile

from dandi.dandiapi import DandiAPIClient
from pynwb import NWBHDF5IO


### Load nwb file
def load_nwb_file(nwb_file_path):
    print("Loading the NWB file ...")
    io = NWBHDF5IO(nwb_file_path, "r") 
    nwb_file = io.read()    
    return nwb_file
    
def validate_nwb_file(nwb_file_path, dandiset_id=None):
    try:
        nwb_file = load_nwb_file(nwb_file_path)
    except:
        try:
            nwb_file = stream_from_dandi(dandiset_id, nwb_file_path)
        except:
            raise ValueError('Unable to load NWB file')

    if nwb_file.electrodes and 'PSTHs_QualityApproved_ZScored_SessionMerged' in nwb_file.scratch.keys():
        return nwb_file
    else:
        raise ValueError('Type of NWB file not accepted, needs to contain electrode data.')

def generate_json_file(nwb_file, json_file_path):
    scratch = nwb_file.scratch['PSTHs_QualityApproved_ZScored_SessionMerged'].description.split('[start_time_ms, stop_time_ms, tb_ms]: ')[-1]
    array = scratch.strip('[]').split()
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
    print('Streaming the NWB file from DANDI ...')
    with DandiAPIClient() as client:
        client.dandi_authenticate()
        asset = client.get_dandiset(dandiset_id, 'draft').get_asset_by_path(filepath)
        s3_url = asset.get_content_url(follow_redirects=1, strip_query=False)

    rem_file = remfile.File(s3_url)
    h5py_file = h5py.File(rem_file, "r")
    io = NWBHDF5IO(file=h5py_file, load_namespaces=True)
    nwb_file = io.read()

    return nwb_file
