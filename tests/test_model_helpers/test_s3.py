import shutil
from pathlib import Path

import pytest

from brainscore_vision.model_helpers import load_folder_from_s3, load_weight_file


class TestLoadFolderFromS3:
    save_directory = Path(__file__).parent / 'test_loader_folder_from_s3'        

    @pytest.mark.private_access
    def test_model_weights(self):
        try:
            filename_version_sha = [
                ('.data-00000-of-00001', 'yYBXN7uf57Y70EdQLchC_dMU5KO6GkIi', 'fef2a64f8c591f5d7562677272c91dcf88989d53'),
                ('.index', '7SJtRE0pdSahjajwjW5c_nVPVGBKn3q2', 'ba9e531243cb87de8562152ab8a22cdb3d218c3b'),
                ('checkpoint', '7rAEuKAwsWAmjT9OvXMsPOaQkogaxMMC', 'a39e7f800a473d18781931c26f84eb3399bdb484')]
            load_folder_from_s3(
                bucket='brainscore-vision', folder_path='models/4o_model_submission/rcnn_cand_4o_weights_10ep',
                filename_version_sha=filename_version_sha,
                save_directory=str(self.save_directory))
            assert self.save_directory.is_dir()
            assert len(list(self.save_directory.glob('*'))) == 3
            for filename, _, _ in filename_version_sha:
                assert (self.save_directory / filename).is_file()
        finally:
            shutil.rmtree(self.save_directory)

    @pytest.mark.private_access
    def test_load_weight_file(self):
        relative_path = 'resnet18_lr0.05_epoch60/model_best.pth.tar'
        local_path = load_weight_file(
            bucket='brainscore-vision', relative_path=relative_path,
            version_id='5GeSHx6pAADhBtoRbAqj3JPAssZ4osMW', sha1='55a627678dd5670c945ae49dcbf83924c7772cd6'
        )
        assert local_path.is_file()
        assert str(local_path).endswith(str(relative_path))