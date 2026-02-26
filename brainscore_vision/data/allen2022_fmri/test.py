import pytest
import numpy as np

from brainscore_vision import load_dataset, load_stimulus_set


# --- 8-subject volumetric assemblies ---

@pytest.mark.private_access
class TestAllen2022fmriAssemblies8Subj:
    def test_train_assembly_shape(self):
        assembly = load_dataset("Allen2022_fmri_train")
        assert assembly.dims == ("presentation", "neuroid", "time_bin")
        assert assembly.sizes["presentation"] == 412
        assert assembly.sizes["neuroid"] == 84564
        assert assembly.sizes["time_bin"] == 1

    def test_test_assembly_shape(self):
        assembly = load_dataset("Allen2022_fmri_test")
        assert assembly.dims == ("presentation", "neuroid", "time_bin")
        assert assembly.sizes["presentation"] == 309  # 103 images x 3 reps
        assert assembly.sizes["neuroid"] == 84564
        assert assembly.sizes["time_bin"] == 1

    def test_test_assembly_has_repetitions(self):
        assembly = load_dataset("Allen2022_fmri_test")
        pres_levels = list(assembly.indexes["presentation"].names)
        assert "repetition" in pres_levels
        reps = sorted(set(assembly["repetition"].values))
        assert reps == [0, 1, 2]
        n_unique = len(set(assembly["stimulus_id"].values))
        assert n_unique == 103

    def test_neuroid_coords(self):
        assembly = load_dataset("Allen2022_fmri_train")
        neuroid_levels = list(assembly.indexes["neuroid"].names)
        required_coords = ["neuroid_id", "subject", "region", "nc_testset",
                           "voxel_x", "voxel_y", "voxel_z"]
        for coord in required_coords:
            assert coord in neuroid_levels, f"Missing coord: {coord}"

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

    def test_no_train_test_overlap(self):
        train = load_dataset("Allen2022_fmri_train")
        test = load_dataset("Allen2022_fmri_test")
        train_ids = set(train["stimulus_id"].values)
        test_ids = set(test["stimulus_id"].values)
        assert len(train_ids & test_ids) == 0

    def test_per_region_neuroid_counts(self):
        assembly = load_dataset("Allen2022_fmri_train")
        regions = assembly["region"].values
        assert (regions == "V1").sum() == 9039
        assert (regions == "V2").sum() == 8792
        assert (regions == "V4").sum() == 3982
        assert (regions == "IT").sum() == 62751


# --- 4-subject volumetric assemblies ---

@pytest.mark.private_access
class TestAllen2022fmriAssemblies4Subj:
    def test_train_assembly_shape(self):
        assembly = load_dataset("Allen2022_fmri_4subj_train")
        assert assembly.dims == ("presentation", "neuroid", "time_bin")
        assert assembly.sizes["presentation"] == 800
        assert assembly.sizes["neuroid"] == 40314
        assert assembly.sizes["time_bin"] == 1

    def test_test_assembly_shape(self):
        assembly = load_dataset("Allen2022_fmri_4subj_test")
        assert assembly.dims == ("presentation", "neuroid", "time_bin")
        assert assembly.sizes["presentation"] == 600  # 200 images x 3 reps
        assert assembly.sizes["neuroid"] == 40314
        assert assembly.sizes["time_bin"] == 1

    def test_subjects_present(self):
        assembly = load_dataset("Allen2022_fmri_4subj_train")
        subjects = set(assembly["subject"].values)
        expected = {"subj01", "subj02", "subj05", "subj07"}
        assert subjects == expected

    def test_test_assembly_has_repetitions(self):
        assembly = load_dataset("Allen2022_fmri_4subj_test")
        pres_levels = list(assembly.indexes["presentation"].names)
        assert "repetition" in pres_levels
        reps = sorted(set(assembly["repetition"].values))
        assert reps == [0, 1, 2]
        n_unique = len(set(assembly["stimulus_id"].values))
        assert n_unique == 200

    def test_neuroid_coords(self):
        assembly = load_dataset("Allen2022_fmri_4subj_train")
        neuroid_levels = list(assembly.indexes["neuroid"].names)
        required_coords = ["neuroid_id", "subject", "region", "nc_testset",
                           "voxel_x", "voxel_y", "voxel_z"]
        for coord in required_coords:
            assert coord in neuroid_levels, f"Missing coord: {coord}"

    def test_no_nans(self):
        for split in ["train", "test"]:
            assembly = load_dataset(f"Allen2022_fmri_4subj_{split}")
            assert not np.any(np.isnan(assembly.values))

    def test_no_train_test_overlap(self):
        train = load_dataset("Allen2022_fmri_4subj_train")
        test = load_dataset("Allen2022_fmri_4subj_test")
        train_ids = set(train["stimulus_id"].values)
        test_ids = set(test["stimulus_id"].values)
        assert len(train_ids & test_ids) == 0


# --- Stimulus sets ---

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
