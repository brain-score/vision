import json
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import xarray as xr

from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from brainscore_core.supported_data_standards.brainio.packaging import (
    package_data_assembly, package_stimulus_set,
)

DEFAULT_BUCKET_NAME = "brainscore-storage/brainscore-vision/benchmarks/Allen2022_fmri"
DEFAULT_OUTPUT_PATH = "./uploaded_data_info/"

# Default paths to notebook 03 output
NSD_BRAINSCORE_DIR = Path("/Volumes/Hagibis/nsd/brainscore")

REPS = {"train": 1, "test": 3}


def load_assembly(brainscore_dir: Path, split: str) -> NeuroidAssembly:
    """Load a pre-built assembly from notebook 03 output."""
    path = brainscore_dir / f"Allen2022_fmri_{split}.nc"
    da = xr.open_dataarray(str(path))
    da.load()

    assembly = NeuroidAssembly(da)
    assembly.name = f"Allen2022_fMRI_{split}_Assembly"
    return assembly


def build_stimulus_set(brainscore_dir: Path, split: str) -> StimulusSet:
    """Build a StimulusSet from extracted images + metadata CSV."""
    meta_csv = brainscore_dir / f"stimulus_metadata_{split}.csv"
    image_dir = brainscore_dir / f"stimuli_{split}"
    meta = pd.read_csv(meta_csv)

    stimuli = []
    stimulus_paths = {}
    for _, row in meta.iterrows():
        sid = row["stimulus_id"]
        stimuli.append({"stimulus_id": sid, "nsd_id": int(row["nsd_id"])})
        stimulus_paths[sid] = str(image_dir / row["image_file_name"])

    stimulus_set = StimulusSet(stimuli)
    stimulus_set.stimulus_paths = stimulus_paths
    stimulus_set.name = f"Allen2022_fMRI_{split}_Stimuli"
    stimulus_set.identifier = f"Allen2022_fMRI_{split}_Stimuli"
    return stimulus_set


def package_data(
    brainscore_dir: Path,
    bucket_name: str,
    output_path: Path,
) -> None:
    """Package pre-built assemblies and stimulus sets for S3 upload."""
    output_path.mkdir(parents=True, exist_ok=True)

    for split in ["train", "test"]:
        stimulus_set = build_stimulus_set(brainscore_dir, split)
        assembly = load_assembly(brainscore_dir, split)
        assembly.attrs["stimulus_set"] = stimulus_set

        n_pres = assembly.sizes["presentation"]
        n_neur = assembly.sizes["neuroid"]
        n_tb = assembly.sizes["time_bin"]
        print(f"\n{split}: {n_pres} presentations x {n_neur} neuroids x {n_tb} time_bin")
        print(f"  Regions: {np.unique(assembly.coords['region'].values)}")

        # For test: verify repetition structure
        if split == "test":
            n_unique = len(set(assembly.coords["stimulus_id"].values))
            print(f"  Unique images: {n_unique}, Reps: {REPS[split]}")
            assert n_pres == n_unique * REPS[split]

        # Verify stimulus_id alignment (for train, 1:1; for test, stimulus_ids are repeated)
        stim_ids_assembly = assembly.coords["stimulus_id"].values
        if split == "train":
            assert np.array_equal(stim_ids_assembly, stimulus_set["stimulus_id"].values)

        print(f"  Packaging {split} stimulus set...")
        stim_info = package_stimulus_set(
            catalog_name=None,
            proto_stimulus_set=stimulus_set,
            stimulus_set_identifier=stimulus_set.name,
            bucket_name=bucket_name,
        )

        print(f"  Packaging {split} assembly...")
        assy_info = package_data_assembly(
            catalog_identifier=None,
            proto_data_assembly=assembly,
            assembly_identifier=assembly.name,
            stimulus_set_identifier=stimulus_set.name,
            assembly_class_name="NeuroidAssembly",
            bucket_name=bucket_name,
        )

        with open(output_path / f"stim_{split}_info.json", "w") as f:
            json.dump(stim_info, f, indent=2)
        with open(output_path / f"assy_{split}_info.json", "w") as f:
            json.dump(assy_info, f, indent=2)

        print(f"  Stimulus info: {stim_info}")
        print(f"  Assembly info: {assy_info}")

    print("\nDone!")


if __name__ == "__main__":
    parser = ArgumentParser(description="Package Allen2022 NSD fMRI data for Brain-Score")
    parser.add_argument(
        "--brainscore-dir",
        type=str,
        default=str(NSD_BRAINSCORE_DIR),
        help="Path to notebook 03 output directory",
    )
    parser.add_argument(
        "--bucket-name",
        type=str,
        default=DEFAULT_BUCKET_NAME,
        help="S3 bucket for upload",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="Directory to save packaging info JSONs",
    )
    args = parser.parse_args()

    package_data(
        brainscore_dir=Path(args.brainscore_dir),
        bucket_name=args.bucket_name,
        output_path=Path(args.output_path),
    )
