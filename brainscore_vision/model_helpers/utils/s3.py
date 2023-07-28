import logging
import os
import sys

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm

_logger = logging.getLogger(__name__)

_DEFAULT_BUCKET = 'brain-score-models'
_DEFAULT_REGION = 'us-east-1'
_NO_SIGNATURE = Config(signature_version=UNSIGNED)


def download_folder(folder_key, target_directory, bucket=_DEFAULT_BUCKET, region=_DEFAULT_REGION):
    if not folder_key.endswith('/'):
        folder_key = folder_key + '/'
    s3 = boto3.resource('s3', region_name=region, config=_NO_SIGNATURE)
    bucket = s3.Bucket(bucket)
    bucket_contents = list(bucket.objects.all())
    files = [obj.key for obj in bucket_contents if obj.key.startswith(folder_key)]
    _logger.debug(f"Found {len(files)} files")
    for file in tqdm(files):
        # get part of file after given folder_key
        filename = file[len(folder_key):]
        target_path = os.path.join(target_directory, filename)
        temp_path = target_path + '.filepart'
        bucket.download_file(file, temp_path)
        os.rename(temp_path, target_path)


def download_file(key, target_path, bucket=_DEFAULT_BUCKET, region=_DEFAULT_REGION):
    s3 = boto3.resource('s3', region_name=region, config=_NO_SIGNATURE)
    obj = s3.Object(bucket, key)
    # show progress. see https://gist.github.com/wy193777/e7607d12fad13459e8992d4f69b53586
    with tqdm(total=obj.content_length, unit='B', unit_scale=True, desc=key, file=sys.stdout) as progress_bar:
        def progress_hook(bytes_amount):
            progress_bar.update(bytes_amount)

        obj.download_file(target_path, Callback=progress_hook)
