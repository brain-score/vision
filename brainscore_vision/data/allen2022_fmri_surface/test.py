import pytest
import numpy as np

from brainscore_vision import load_dataset


# --- 8-subject surface assemblies ---

@pytest.mark.private_access
class TestAllen2022fmriSurfaceAssemblies8Subj:
    def test_train_assembly_shape(self):
        assembly = load_dataset("Allen2022_fmri_surface_train")
        assert assembly.dims == ("presentation", "neuroid", "time_bin")
        assert assembly.sizes["presentation"] == 412
        assert assembly.sizes["neuroid"] == 221168
        assert assembly.sizes["time_bin"] == 1

    def test_test_assembly_shape(self):
        assembly = load_dataset("Allen2022_fmri_surface_test")
        assert assembly.dims == ("presentation", "neuroid", "time_bin")
        assert assembly.sizes["presentation"] == 309  # 103 images x 3 reps
        assert assembly.sizes["neuroid"] == 221168
        assert assembly.sizes["time_bin"] == 1

    def test_test_assembly_has_repetitions(self):
        assembly = load_dataset("Allen2022_fmri_surface_test")
        pres_levels = list(assembly.indexes["presentation"].names)
        assert "repetition" in pres_levels
        reps = sorted(set(assembly["repetition"].values))
        assert reps == [0, 1, 2]
        n_unique = len(set(assembly["stimulus_id"].values))
        assert n_unique == 103

    def test_neuroid_coords(self):
        assembly = load_dataset("Allen2022_fmri_surface_train")
        neuroid_levels = list(assembly.indexes["neuroid"].names)
        required_coords = ["neuroid_id", "subject", "hemisphere",
                           "vertex_index", "region", "nc_testset"]
        for coord in required_coords:
            assert coord in neuroid_levels, f"Missing coord: {coord}"

    def test_regions_present(self):
        assembly = load_dataset("Allen2022_fmri_surface_train")
        regions = set(assembly["region"].values)
        assert regions == {"V1", "V2", "V4", "IT"}

    def test_subjects_present(self):
        assembly = load_dataset("Allen2022_fmri_surface_train")
        subjects = set(assembly["subject"].values)
        expected = {f"subj{i:02d}" for i in range(1, 9)}
        assert subjects == expected

    def test_hemispheres_present(self):
        assembly = load_dataset("Allen2022_fmri_surface_train")
        hemispheres = set(assembly["hemisphere"].values)
        assert hemispheres == {"lh", "rh"}

    def test_neuroid_consistency(self):
        train = load_dataset("Allen2022_fmri_surface_train")
        test = load_dataset("Allen2022_fmri_surface_test")
        assert train.sizes["neuroid"] == test.sizes["neuroid"]
        assert np.array_equal(
            train["neuroid_id"].values, test["neuroid_id"].values
        )

    def test_no_nans(self):
        for split in ["train", "test"]:
            assembly = load_dataset(f"Allen2022_fmri_surface_{split}")
            assert not np.any(np.isnan(assembly.values))

    def test_no_train_test_overlap(self):
        train = load_dataset("Allen2022_fmri_surface_train")
        test = load_dataset("Allen2022_fmri_surface_test")
        train_ids = set(train["stimulus_id"].values)
        test_ids = set(test["stimulus_id"].values)
        assert len(train_ids & test_ids) == 0

    def test_per_region_neuroid_counts(self):
        assembly = load_dataset("Allen2022_fmri_surface_train")
        regions = assembly["region"].values
        assert (regions == "V1").sum() == 34208
        assert (regions == "V2").sum() == 27128
        assert (regions == "V4").sum() == 7312
        assert (regions == "IT").sum() == 152520


# --- 4-subject surface assemblies ---

@pytest.mark.private_access
class TestAllen2022fmriSurfaceAssemblies4Subj:
    def test_train_assembly_shape(self):
        assembly = load_dataset("Allen2022_fmri_surface_4subj_train")
        assert assembly.dims == ("presentation", "neuroid", "time_bin")
        assert assembly.sizes["presentation"] == 800
        assert assembly.sizes["neuroid"] == 110584
        assert assembly.sizes["time_bin"] == 1

    def test_test_assembly_shape(self):
        assembly = load_dataset("Allen2022_fmri_surface_4subj_test")
        assert assembly.dims == ("presentation", "neuroid", "time_bin")
        assert assembly.sizes["presentation"] == 600  # 200 images x 3 reps
        assert assembly.sizes["neuroid"] == 110584
        assert assembly.sizes["time_bin"] == 1

    def test_subjects_present(self):
        assembly = load_dataset("Allen2022_fmri_surface_4subj_train")
        subjects = set(assembly["subject"].values)
        expected = {"subj01", "subj02", "subj05", "subj07"}
        assert subjects == expected

    def test_test_assembly_has_repetitions(self):
        assembly = load_dataset("Allen2022_fmri_surface_4subj_test")
        pres_levels = list(assembly.indexes["presentation"].names)
        assert "repetition" in pres_levels
        reps = sorted(set(assembly["repetition"].values))
        assert reps == [0, 1, 2]
        n_unique = len(set(assembly["stimulus_id"].values))
        assert n_unique == 200

    def test_neuroid_coords(self):
        assembly = load_dataset("Allen2022_fmri_surface_4subj_train")
        neuroid_levels = list(assembly.indexes["neuroid"].names)
        required_coords = ["neuroid_id", "subject", "hemisphere",
                           "vertex_index", "region", "nc_testset"]
        for coord in required_coords:
            assert coord in neuroid_levels, f"Missing coord: {coord}"

    def test_no_nans(self):
        for split in ["train", "test"]:
            assembly = load_dataset(f"Allen2022_fmri_surface_4subj_{split}")
            assert not np.any(np.isnan(assembly.values))

    def test_no_train_test_overlap(self):
        train = load_dataset("Allen2022_fmri_surface_4subj_train")
        test = load_dataset("Allen2022_fmri_surface_4subj_test")
        train_ids = set(train["stimulus_id"].values)
        test_ids = set(test["stimulus_id"].values)
        assert len(train_ids & test_ids) == 0
