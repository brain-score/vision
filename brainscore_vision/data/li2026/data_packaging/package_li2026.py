"""Package the Triple-N (Li et al. 2026) macaque Neuropixels static assembly + stimulus
set for Brain-Score S3 storage.

Source artifacts (built + validated locally, see tasks/todo.md):
  /Volumes/Hagibis/triple-n/build/Li2026_static_assembly.nc   presentation x neuroid x time_bin
  /Volumes/Hagibis/triple-n/build/Li2026_stimulus_set.csv     Allen2022-aligned stimulus_id
  /Volumes/Hagibis/triple-n/build/stimuli/<stimulus_id>.png   1000 full-res NSD images
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from brainscore_core.supported_data_standards.brainio.assemblies import NeuronRecordingAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
import brainscore_core.supported_data_standards.brainio.packaging as _brainio_pkg
from brainscore_core.supported_data_standards.brainio.packaging import (
    package_data_assembly, package_stimulus_set,
)

# brainio's upload_to_s3 hands bucket_name straight to boto3, which rejects slashes.
# Split "bucket/prefix" so files land under the prefix (same patch as Allen2022).
_original_upload = _brainio_pkg.upload_to_s3


def _upload_with_prefix(source_file_path, bucket_name, target_s3_key):
    if "/" in bucket_name:
        actual_bucket, prefix = bucket_name.split("/", 1)
        target_s3_key = f"{prefix}/{target_s3_key}"
        bucket_name = actual_bucket
    return _original_upload(source_file_path, bucket_name, target_s3_key)


_brainio_pkg.upload_to_s3 = _upload_with_prefix

BUILD_DIR = Path("/Volumes/Hagibis/triple-n/build")
BUCKET_NAME = "brainscore-storage/brainscore-vision/benchmarks/Li2026"
OUTPUT_PATH = Path(__file__).parent / "uploaded_data_info"

STIMULUS_IDENTIFIER = "Li2026_Stimuli"
ASSEMBLY_IDENTIFIER = "Li2026_Assembly"


def build_stimulus_set() -> StimulusSet:
    meta = pd.read_csv(BUILD_DIR / "Li2026_stimulus_set.csv")
    stim_dir = BUILD_DIR / "stimuli"
    stimuli, paths = [], {}
    for _, row in meta.iterrows():
        sid = row["stimulus_id"]
        stimuli.append({
            "stimulus_id": sid,
            "nsd_id": int(row["nsd_id"]),                      # Allen2022-aligned (0-indexed 73k)
            "nsd_id_1indexed": int(row["nsd_id_1indexed"]),
            "tn_index": int(row["tn_index"]),
            "coco_id": int(row["coco_id"]),
        })
        p = stim_dir / f"{sid}.png"
        if not p.exists():
            raise FileNotFoundError(p)
        paths[sid] = str(p)
    ss = StimulusSet(stimuli)
    ss.stimulus_paths = paths
    ss.name = STIMULUS_IDENTIFIER
    ss.identifier = STIMULUS_IDENTIFIER
    return ss


def build_assembly() -> NeuronRecordingAssembly:
    da = xr.open_dataarray(str(BUILD_DIR / "Li2026_static_assembly.nc"))
    da.load()
    assembly = NeuronRecordingAssembly(da)
    assembly.name = ASSEMBLY_IDENTIFIER
    return assembly


def main():
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    stimulus_set = build_stimulus_set()
    assembly = build_assembly()
    assembly.attrs["stimulus_set"] = stimulus_set

    print(f"stimulus_set: {len(stimulus_set)} images")
    print(f"assembly: {dict(assembly.sizes)}  regions={np.unique(assembly['region'].values)}")
    # integrity: presentation order == stimulus_set order
    assert np.array_equal(assembly["stimulus_id"].values, stimulus_set["stimulus_id"].values), \
        "assembly presentation order must match stimulus_set row order"

    print(f"\nPackaging stimulus set -> {BUCKET_NAME}")
    stim_info = package_stimulus_set(
        catalog_name=None, proto_stimulus_set=stimulus_set,
        stimulus_set_identifier=STIMULUS_IDENTIFIER, bucket_name=BUCKET_NAME,
    )
    print("stim_info:", stim_info)

    print(f"\nPackaging assembly -> {BUCKET_NAME}")
    assy_info = package_data_assembly(
        catalog_identifier=None, proto_data_assembly=assembly,
        assembly_identifier=ASSEMBLY_IDENTIFIER, stimulus_set_identifier=STIMULUS_IDENTIFIER,
        assembly_class_name="NeuroidAssembly", bucket_name=BUCKET_NAME,
    )
    print("assy_info:", assy_info)

    (OUTPUT_PATH / "stim_info.json").write_text(json.dumps(stim_info, indent=2))
    (OUTPUT_PATH / "assy_info.json").write_text(json.dumps(assy_info, indent=2))
    print(f"\nsaved upload info -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
