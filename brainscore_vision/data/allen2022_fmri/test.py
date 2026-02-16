import pytest
import numpy as np

from brainscore_vision import load_dataset, load_stimulus_set


@pytest.mark.private_access
class TestAllen2022fmriAssemblies:
    def test_train_assembly_shape(self):
        assembly = load_dataset("Allen2022_fmri_train")
        assert assembly.dims == ("presentation", "neuroid", "time_bin")
        assert assembly.sizes["presentation"] == 412
        assert assembly.sizes["time_bin"] == 1

    def test_test_assembly_shape(self):
        assembly = load_dataset("Allen2022_fmri_test")
        assert assembly.dims == ("presentation", "neuroid", "time_bin")
        assert assembly.sizes["presentation"] == 309  # 103 images x 3 reps
        assert assembly.sizes["time_bin"] == 1

    def test_test_assembly_has_repetitions(self):
        assembly = load_dataset("Allen2022_fmri_test")
        assert "repetition" in assembly.coords
        reps = sorted(set(assembly["repetition"].values))
        assert reps == [0, 1, 2]
        n_unique = len(set(assembly["stimulus_id"].values))
        assert n_unique == 103

    def test_neuroid_coords(self):
        assembly = load_dataset("Allen2022_fmri_train")
        required_coords = ["neuroid_id", "subject", "region", "nc_testset",
                           "voxel_x", "voxel_y", "voxel_z"]
        for coord in required_coords:
            assert coord in assembly.coords, f"Missing coord: {coord}"

    def test_regions_present(self):
        assembly = load_dataset("Allen2022_fmri_train")
        regions = set(assembly["region"].values)
        assert regions == {"V1", "V2", "V4", "IT"}

    def test_subjects_present(self):
        assembly = load_dataset("Allen2022_fmri_train")
        subjects = set(assembly["subject"].values)
        expected = {f"subj{i:02d}" for i in range(1, 9)}
        assert subjects == expected

    def test_neuroid_consistency(self):
        train = load_dataset("Allen2022_fmri_train")
        test = load_dataset("Allen2022_fmri_test")
        assert train.sizes["neuroid"] == test.sizes["neuroid"]
        assert np.array_equal(
            train["neuroid_id"].values, test["neuroid_id"].values
        )

    def test_no_nans(self):
        for split in ["train", "test"]:
            assembly = load_dataset(f"Allen2022_fmri_{split}")
            assert not np.any(np.isnan(assembly.values))


@pytest.mark.private_access
class TestAllen2022fmriStimulusSets:
    def test_train_stimulus_set(self):
        stimulus_set = load_stimulus_set("Allen2022_fmri_stim_train")
        assert len(stimulus_set) == 412
        assert "stimulus_id" in stimulus_set.columns

    def test_test_stimulus_set(self):
        stimulus_set = load_stimulus_set("Allen2022_fmri_stim_test")
        assert len(stimulus_set) == 103
        assert "stimulus_id" in stimulus_set.columns

    def test_no_overlap(self):
        train = load_stimulus_set("Allen2022_fmri_stim_train")
        test = load_stimulus_set("Allen2022_fmri_stim_test")
        train_ids = set(train["stimulus_id"].values)
        test_ids = set(test["stimulus_id"].values)
        assert len(train_ids & test_ids) == 0
