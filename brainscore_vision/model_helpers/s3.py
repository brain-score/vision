"""
Contain functions for loading model contents such as weights from s3.
"""
import os
from os.path import expanduser
from pathlib import Path

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
<<<<<<< HEAD
    verify_sha1(local_filename, sha1)


def load_weight(bucket: str, relative_path: str, version_id: str, sha1: str):
    s3_weight_folder = os.getenv("BRAINSCORE_S3_WEIGHT_FOLDER", "models-to-integrate-for-2.0")
    brainscore_cache = os.getenv("BRAINSCORE_HOME", expanduser("~/.brain-score"))
    local_path = Path(brainscore_cache) / "models" / relative_path
    if not os.path.exists(local_path):
        os.makedirs(local_path.parent, exist_ok=True)
        load_file_from_s3(bucket=bucket, path=f"{s3_weight_folder}/{relative_path}", version_id=version_id, sha1=sha1,
                        local_filename=local_path)
    else:
        _logger.info(f"Weight already exists at {local_path}.")
    return local_path


if __name__ == "__main__":
    # Example usage
    pth = load_weight(
        bucket="brainscore-vision", 
        relative_path="temporal_models/mae_st/mae_pretrain_vit_large_k400.pth", 
        version_id="HuKboFIWw6Tl3fZIY4aVNEqvcS4Yag66",
        sha1="c7fb91864a4ddf8b99309440121a3abe66b846bb"
    )
    
    print(pth)
=======
    verify_sha1(local_filepath, sha1)
>>>>>>> upstream/master
