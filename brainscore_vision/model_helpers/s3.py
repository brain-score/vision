"""
Contain functions for loading model contents such as weights from s3.
"""
from pathlib import Path

import logging
from typing import Tuple, List, Union

from brainio.fetch import BotoFetcher, verify_sha1

_logger = logging.getLogger(__name__)


def load_folder_from_s3(bucket: str, folder_path: str, filename_version_sha: List[Tuple[str, str, str]],
                        save_directory: Union[Path, str]):
    for filename, version_id, sha1 in filename_version_sha:
        load_file_from_s3(bucket=bucket, path=f"{folder_path}/{filename}", version_id=version_id, sha1=sha1,
                          local_filename=Path(save_directory) / filename)


def load_file_from_s3(bucket: str, path: str, version_id: str, sha1: str, local_filename: Union[Path, str]):
    fetcher = BotoFetcher(location=f"https://{bucket}.s3.amazonaws.com/{path}", version_id=version_id,
                          # this is a bit hacky: don't tell BotoFetcher the full path because it will make a directory
                          # where there should be a file
                          local_filename=Path(local_filename).parent)
    fetcher.output_filename = str(local_filename)  # force using this local path instead of folder structure
    fetcher.fetch()
    verify_sha1(local_filename, sha1)
