import json
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import xarray as xr

from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
import brainscore_core.supported_data_standards.brainio.packaging as _brainio_pkg
from brainscore_core.supported_data_standards.brainio.packaging import (
    package_data_assembly,
)

# Monkey-patch upload_to_s3 to handle bucket_name with path prefix.
_original_upload = _brainio_pkg.upload_to_s3


def _upload_with_prefix(source_file_path, bucket_name, target_s3_key):
    if "/" in bucket_name:
        actual_bucket, prefix = bucket_name.split("/", 1)
        target_s3_key = f"{prefix}/{target_s3_key}"
        bucket_name = actual_bucket
    return _original_upload(source_file_path, bucket_name, target_s3_key)


_brainio_pkg.upload_to_s3 = _upload_with_prefix

DEFAULT_BUCKET_NAME = (
    "brainscore-storage/brainscore-vision/benchmarks/Allen2022/Allen2022_fmri_surface"
)
DEFAULT_OUTPUT_PATH = "./uploaded_data_info_surface/"

NSD_BRAINSCORE_DIR = Path("/Volumes/Hagibis/nsd/brainscore_surface")

VARIANTS = {
    "_8subj": {"reps": {"train": 1, "test": 3}, "id_infix": ""},
    "_4subj": {"reps": {"train": 1, "test": 3}, "id_infix": "_4subj"},
}

# Surface assemblies reuse the volumetric stimulus sets (identical COCO images).
# Stimulus set identifiers must match those uploaded by the volumetric packaging script.
STIM_ID_TEMPLATE = "Allen2022_fMRI{infix}_{split}_Stimuli"


def load_assembly(brainscore_dir: Path, split: str, variant: str) -> NeuroidAssembly:
    """Load a pre-built surface assembly from notebooks/surface/03_surface_packaging_prep.ipynb output."""
    path = brainscore_dir / f"Allen2022_fmri_surface_{split}{variant}.nc"
    da = xr.open_dataarray(str(path))
    da.load()

    infix = VARIANTS[variant]["id_infix"]
    assembly = NeuroidAssembly(da)
    assembly.name = f"Allen2022_fMRI_surface{infix}_{split}_Assembly"
    return assembly


def package_data(
    brainscore_dir: Path,
    bucket_name: str,
    output_path: Path,
) -> None:
    """Package surface assemblies for S3 upload (no stimulus sets)."""
    output_path.mkdir(parents=True, exist_ok=True)

    for variant, cfg in VARIANTS.items():
        infix = cfg["id_infix"]
        tag = variant.lstrip("_")
        print(f"\n{'='*60}")
        print(f"Variant: {tag}")
        print(f"{'='*60}")

        for split in ["train", "test"]:
            assembly = load_assembly(brainscore_dir, split, variant)
            stim_identifier = STIM_ID_TEMPLATE.format(infix=infix, split=split)

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

            print(f"  Packaging assembly: {assembly.name}")
            print(f"  Linked stimulus set: {stim_identifier}")
            assy_info = package_data_assembly(
                catalog_identifier=None,
                proto_data_assembly=assembly,
                assembly_identifier=assembly.name,
                stimulus_set_identifier=stim_identifier,
                assembly_class_name="NeuroidAssembly",
                bucket_name=bucket_name,
            )

            prefix = f"{tag}_{split}"
            with open(output_path / f"assy_{prefix}_info.json", "w") as f:
                json.dump(assy_info, f, indent=2)

            print(f"  Assembly info: {assy_info}")

    print("\nDone!")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Package Allen2022 NSD surface fMRI data for Brain-Score"
    )
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
