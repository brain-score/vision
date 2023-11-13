"""
Loading data assemblies and stimuli sets from s3.
"""
import functools
import logging
import os
from pathlib import Path
from typing import Callable, Union

import boto3
from botocore import UNSIGNED
from botocore.config import Config

from brainio.assemblies import DataAssembly, AssemblyLoader, StimulusMergeAssemblyLoader, \
    StimulusReferenceAssemblyLoader
from brainio.fetch import fetch_file, unzip, resolve_stimulus_set_class
from brainio.stimuli import StimulusSetLoader, StimulusSet

_logger = logging.getLogger(__name__)


def get_path(identifier: str, file_type: str, bucket: str, version_id: str, sha1: str):
    """
    Finds path of desired file (for .csvs, .zips, and .ncs).
    """
    # for stimuli sets
    if file_type == 'csv' or file_type == 'zip':
        assembly_prefix = 'image_'
    # for data assemblies
    else:
        assembly_prefix = 'assy_'

    filename = f"{assembly_prefix}{identifier.replace('.', '_')}.{file_type}"
    file_path = fetch_file(location_type="S3",
                           location=f"https://{bucket}.s3.amazonaws.com/{filename}",
                           version_id=version_id,
                           sha1=sha1)
    return file_path


def load_assembly_from_s3(identifier: str, version_id: str, sha1: str, bucket: str, cls: type,
                          stimulus_set_loader: Callable[[], StimulusSet] = None,
                          merge_stimulus_set_meta: bool = True) -> DataAssembly:
    file_path = get_path(identifier, 'nc', bucket, version_id, sha1)
    if stimulus_set_loader:  # merge stimulus set meta into assembly if `stimulus_set_loader` is passed
        stimulus_set = stimulus_set_loader()
        loader_base_class = StimulusMergeAssemblyLoader if merge_stimulus_set_meta else StimulusReferenceAssemblyLoader
        loader_class = functools.partial(loader_base_class,
                                         stimulus_set_identifier=stimulus_set.identifier, stimulus_set=stimulus_set)
    else:  # if no `stimulus_set_loader` passed, just load assembly
        loader_class = AssemblyLoader
    loader = loader_class(cls=cls, file_path=file_path)
    assembly = loader.load()
    assembly.attrs['identifier'] = identifier
    return assembly


def load_stimulus_set_from_s3(identifier: str, bucket: str, csv_sha1: str, zip_sha1: str,
                              csv_version_id: str, zip_version_id: str):
    csv_path = get_path(identifier, 'csv', bucket, csv_version_id, csv_sha1)
    zip_path = get_path(identifier, 'zip', bucket, zip_version_id, zip_sha1)
    stimuli_directory = unzip(zip_path)
    loader = StimulusSetLoader(
        csv_path=csv_path,
        stimuli_directory=stimuli_directory,
        cls=resolve_stimulus_set_class('StimulusSet')
    )
    stimulus_set = loader.load()
    stimulus_set.identifier = identifier
    # ensure perfect overlap
    stimuli_paths = [Path(stimuli_directory) / local_path for local_path in os.listdir(stimuli_directory)
                     if not local_path.endswith('.zip') and not local_path.endswith('.csv')]
    assert set(stimulus_set.stimulus_paths.values()) == set(stimuli_paths), \
        "Inconsistency: unzipped stimuli paths do not match csv paths"
    return stimulus_set


def download_file_if_not_exists(local_path: Union[str, Path], bucket: str, remote_filepath: str):
    if local_path.is_file():
        return  # nothing to do, file already exists
    unsigned_config = Config(signature_version=UNSIGNED)  # do not attempt to look up credentials
    s3 = boto3.client('s3', config=unsigned_config)
    s3.download_file(bucket, remote_filepath, str(local_path))
