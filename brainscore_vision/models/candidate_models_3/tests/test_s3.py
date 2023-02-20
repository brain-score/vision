import os
import tempfile

import logging

from candidate_models import s3

for disable_logger in ['boto3', 'botocore', 's3transfer', 'urllib3']:
    logging.getLogger(disable_logger).setLevel(logging.WARNING)


class TestDownloadFolder:
    def test_mobilenet_weights(self):
        model = 'mobilenet_v1_0.25_128'
        with tempfile.TemporaryDirectory() as target_path:
            s3.download_folder(f"slim/{model}", target_path)
            downloaded_files = os.listdir(target_path)
            expected_suffixes = ['.ckpt.data-00000-of-00001', '.ckpt.index', '.ckpt.meta', '.tflite',
                                 '_eval.pbtxt', '_frozen.pb', '_info.txt']
            assert set(downloaded_files) == set([model + suffix for suffix in expected_suffixes])


class TestDownloadFile:
    def test_mobilenet_weight_info(self):
        key = 'slim/mobilenet_v1_0.25_128/mobilenet_v1_0.25_128_info.txt'
        with tempfile.TemporaryDirectory() as target_dir:
            target_path = os.path.join(target_dir, 'info.txt')
            s3.download_file(key, target_path)
            assert os.path.isfile(target_path)
