import h5py
import logging
import remfile
from dandi.dandiapi import DandiAPIClient
from pynwb import NWBHDF5IO
from pynwb.file import NWBFile


logger = logging.getLogger(__name__)

def load_nwb_file(nwb_file_path: str) -> NWBFile:
    #-----------------------------------------------------------------------------------------------------------------------------
    # Load nwb file
    #-----------------------------------------------------------------------------------------------------------------------------
    logger.info("Loading the NWB file ...")
    io = NWBHDF5IO(nwb_file_path, "r") 
    nwb_file = io.read()    
    return nwb_file
    
def validate_nwb_file(nwb_file_path: str, dandiset_id: str = None) -> NWBFile:
    #-----------------------------------------------------------------------------------------------------------------------------
    # Check if the provided NWB file contains PSTH electrode data
    #-----------------------------------------------------------------------------------------------------------------------------
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

def stream_from_dandi(dandiset_id: str, filepath: str) -> NWBFile:
    #-----------------------------------------------------------------------------------------------------------------------------
    # Streams NWB file from DANDI using the DandiAPIClient
    #-----------------------------------------------------------------------------------------------------------------------------
    logger.info('Streaming the NWB file from DANDI ...')
    with DandiAPIClient() as client:
        client.dandi_authenticate()
        asset = client.get_dandiset(dandiset_id, 'draft').get_asset_by_path(filepath)
        s3_url = asset.get_content_url(follow_redirects=1, strip_query=False)

    rem_file = remfile.File(s3_url)
    h5py_file = h5py.File(rem_file, "r")
    io = NWBHDF5IO(file=h5py_file, load_namespaces=True)
    nwb_file = io.read()

    return nwb_file
