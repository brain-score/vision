import json
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import xarray as xr

from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
import brainscore_core.supported_data_standards.brainio.packaging as _brainio_pkg
from brainscore_core.supported_data_standards.brainio.packaging import (
    package_data_assembly, package_stimulus_set,
)

# Monkey-patch upload_to_s3 to handle bucket_name with path prefix.
# brainio's upload_to_s3 passes bucket_name directly to boto3, which rejects
# slashes. We split "bucket/prefix" into the actual bucket and prepend the
# prefix to the S3 key so files land in the correct subdirectory.
_original_upload = _brainio_pkg.upload_to_s3


def _upload_with_prefix(source_file_path, bucket_name, target_s3_key):
    if "/" in bucket_name:
        actual_bucket, prefix = bucket_name.split("/", 1)
        target_s3_key = f"{prefix}/{target_s3_key}"
        bucket_name = actual_bucket
    return _original_upload(source_file_path, bucket_name, target_s3_key)


_brainio_pkg.upload_to_s3 = _upload_with_prefix

DEFAULT_BUCKET_NAME = "brainscore-storage/brainscore-vision/benchmarks/Allen2022/Allen2022_fmri"
DEFAULT_OUTPUT_PATH = "./uploaded_data_info/"

NSD_BRAINSCORE_DIR = Path("/Volumes/Hagibis/nsd/brainscore")

VARIANTS = {
    "_8subj": {"reps": {"train": 1, "test": 3}, "id_infix": ""},
    "_4subj": {"reps": {"train": 1, "test": 3}, "id_infix": "_4subj"},
}


def load_assembly(brainscore_dir: Path, split: str, variant: str) -> NeuroidAssembly:
    """Load a pre-built assembly from notebook 03 output."""
    path = brainscore_dir / f"Allen2022_fmri_{split}{variant}.nc"
    da = xr.open_dataarray(str(path))
    da.load()

    infix = VARIANTS[variant]["id_infix"]
    assembly = NeuroidAssembly(da)
    assembly.name = f"Allen2022_fMRI{infix}_{split}_Assembly"
    return assembly


def build_stimulus_set(
    brainscore_dir: Path, split: str, variant: str,
) -> StimulusSet:
    """Build a StimulusSet from extracted images + metadata CSV.

    Images may reside in either stimuli_train/ or stimuli_test/ depending
    on which variant's split they belong to, so both directories are searched.
    """
    meta_csv = brainscore_dir / f"stimulus_metadata_{split}{variant}.csv"
    meta = pd.read_csv(meta_csv)

    search_dirs = [
        brainscore_dir / "stimuli_train",
        brainscore_dir / "stimuli_test",
    ]

    stimuli = []
    stimulus_paths = {}
    for _, row in meta.iterrows():
        sid = row["stimulus_id"]
        stimuli.append({"stimulus_id": sid, "nsd_id": int(row["nsd_id"])})
        for d in search_dirs:
            p = d / row["image_file_name"]
            if p.exists():
                stimulus_paths[sid] = str(p)
                break
        else:
            raise FileNotFoundError(
                f"Image {row['image_file_name']} not found in {search_dirs}"
            )

    infix = VARIANTS[variant]["id_infix"]
    stimulus_set = StimulusSet(stimuli)
    stimulus_set.stimulus_paths = stimulus_paths
    stimulus_set.name = f"Allen2022_fMRI{infix}_{split}_Stimuli"
    stimulus_set.identifier = stimulus_set.name
    return stimulus_set


def package_data(
    brainscore_dir: Path,
    bucket_name: str,
    output_path: Path,
) -> None:
    """Package all variant assemblies and stimulus sets for S3 upload."""
    output_path.mkdir(parents=True, exist_ok=True)

    for variant, cfg in VARIANTS.items():
        infix = cfg["id_infix"]
        tag = variant.lstrip("_")
        print(f"\n{'='*60}")
        print(f"Variant: {tag}")
        print(f"{'='*60}")

        for split in ["train", "test"]:
            stimulus_set = build_stimulus_set(brainscore_dir, split, variant)
            assembly = load_assembly(brainscore_dir, split, variant)
            assembly.attrs["stimulus_set"] = stimulus_set

            n_pres = assembly.sizes["presentation"]
            n_neur = assembly.sizes["neuroid"]
            n_tb = assembly.sizes["time_bin"]
            print(f"\n{tag}/{split}: {n_pres} pres x {n_neur} neuroids x {n_tb} time_bin")
            print(f"  Regions: {np.unique(assembly.coords['region'].values)}")

            if split == "test":
                n_unique = len(set(assembly.coords["stimulus_id"].values))
                reps = cfg["reps"][split]
                print(f"  Unique images: {n_unique}, Reps: {reps}")
                assert n_pres == n_unique * reps

            if split == "train":
                assert np.array_equal(
                    assembly.coords["stimulus_id"].values,
                    stimulus_set["stimulus_id"].values,
                )

            print(f"  Packaging stimulus set: {stimulus_set.name}")
            stim_info = package_stimulus_set(
                catalog_name=None,
                proto_stimulus_set=stimulus_set,
                stimulus_set_identifier=stimulus_set.name,
                bucket_name=bucket_name,
            )

            print(f"  Packaging assembly: {assembly.name}")
            assy_info = package_data_assembly(
                catalog_identifier=None,
                proto_data_assembly=assembly,
                assembly_identifier=assembly.name,
                stimulus_set_identifier=stimulus_set.name,
                assembly_class_name="NeuroidAssembly",
                bucket_name=bucket_name,
            )

            prefix = f"{tag}_{split}"
            with open(output_path / f"stim_{prefix}_info.json", "w") as f:
                json.dump(stim_info, f, indent=2)
            with open(output_path / f"assy_{prefix}_info.json", "w") as f:
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
    )
    parser.add_argument(
        "--bucket-name",
        type=str,
        default=DEFAULT_BUCKET_NAME,
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
    )
    args = parser.parse_args()

    package_data(
        brainscore_dir=Path(args.brainscore_dir),
        bucket_name=args.bucket_name,
        output_path=Path(args.output_path),
    )
