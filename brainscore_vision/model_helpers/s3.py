"""
Contain functions for loading model contents such as weights from s3.
"""
import os
from pathlib import Path
from os.path import expanduser
import logging
from typing import Tuple, List, Union

from brainio.fetch import BotoFetcher, verify_sha1
from brainio.lookup import sha1_hash  # so that users can easily import and get the hash of their file(s)

_logger = logging.getLogger(__name__)


def load_folder_from_s3(bucket: str, folder_path: str, filename_version_sha: List[Tuple[str, str, str]],
                        save_directory: Union[Path, str]):
    for filename, version_id, sha1 in filename_version_sha:
        load_file_from_s3(bucket=bucket, path=f"{folder_path}/{filename}", version_id=version_id, sha1=sha1,
                          local_filepath=Path(save_directory) / filename)


def load_file_from_s3(bucket: str, path: str, local_filepath: Union[Path, str],
                      sha1: str, version_id: Union[str, None] = None):
    """
    Load a file from AWS S3 and validate its contents.
    :param bucket: The name of the S3 bucket
    :param path: The path of the file inside the S3 bucket
    :param local_filepath: The local path of the file to be saved to
    :param sha1: The SHA1 hash of the file. If you are not sure of this, use the `sha1_hash` function in this same file
    :param version_id: Which version of the file on S3 to use.
        Optional but strongly encouraged to avoid accidental overwrites.
        If you use Brain-Score functionality to upload files to S3, the version id will be printed to the console.
        You can also find this on the S3 user interface by opening the file and then clicking on the versions tab.
    """
    fetcher = BotoFetcher(location=f"https://{bucket}.s3.amazonaws.com/{path}", version_id=version_id,
                          # this is a bit hacky: don't tell BotoFetcher the full path because it will make a directory
                          # where there should be a file
                          local_filename=Path(local_filepath).parent)
    fetcher.output_filename = str(local_filepath)  # force using this local path instead of folder structure
    fetcher.fetch()
    verify_sha1(local_filepath, sha1)


def load_weight_file(bucket: str, relative_path: str, version_id: str, sha1: str, folder_name: str = None) -> Path:
    """
    :param bucket: main bucket to add file to. Usually is 'brainscore_vision' for vision model weights.
    :param relative_path: The path of the file inside the S3 bucket, relative to the `{folder_name}/` directory.
        The local path will be the same, relative to the local cache's `{folder_name}/` directory (inside `BRAINSCORE_HOME`).
        Example: `alexnet/weights.pth` will download from brain-score S3:models/alexnet/weights.pth and
        and store into local ~/.brain-score/models/alexnet/weights.pth.
    :param version_id: version_id of the object to download, found in AWS under object properties
    :param sha1: sha1 hash of the object
    :param folder_name: name of the folder inside the bucket to download from, i.e. 'models'
    """
    brainscore_cache = os.getenv("BRAINSCORE_HOME", expanduser("~/.brain-score"))
    s3_weight_folder = folder_name if folder_name is not None else os.getenv("BRAINSCORE_S3_WEIGHT_FOLDER", "models")
    local_path = Path(brainscore_cache) / s3_weight_folder / relative_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    load_file_from_s3(bucket=bucket, path=f"{s3_weight_folder}/{relative_path}", version_id=version_id, sha1=sha1,
                          local_filepath=local_path)
    return local_path
