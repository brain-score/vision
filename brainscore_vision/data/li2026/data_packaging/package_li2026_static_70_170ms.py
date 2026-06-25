"""Upload the fixed-window (70-170 ms) Li2026 static assembly to S3.

Companion to ``build_li2026_static_70_170ms.py``. Reuses the existing
``Li2026_Stimuli`` upload (stimulus set is unchanged) and replaces the
``Li2026_Assembly`` upload with the new fixed-window static so consumers
of ``brainscore_vision.data.li2026`` get the canonical, MajajHong-aligned
response matrix without any code change on their side.

After running this, paste the printed ``version_id`` and ``sha1`` into
``../__init__.py``'s ``data_registry['Li2026']`` entry, replacing the
previous best-window-derived values. The README's "v2" sequencing note
also gets updated to reflect that v2 is now shipped.
"""
import json
from pathlib import Path

import xarray as xr

from brainscore_core.supported_data_standards.brainio.assemblies import NeuronRecordingAssembly
import brainscore_core.supported_data_standards.brainio.packaging as _brainio_pkg
from brainscore_core.supported_data_standards.brainio.packaging import package_data_assembly

# Same brainio bucket/prefix patch as package_li2026.py — brainio's
# upload_to_s3 hands bucket_name straight to boto3 which rejects slashes.
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

ASSEMBLY_IDENTIFIER = "Li2026_Assembly"
STIMULUS_IDENTIFIER = "Li2026_Stimuli"


def build_assembly() -> NeuronRecordingAssembly:
    da = xr.open_dataarray(str(BUILD_DIR / "Li2026_static_70_170ms_assembly.nc"))
    da.load()
    assembly = NeuronRecordingAssembly(da)
    assembly.name = ASSEMBLY_IDENTIFIER
    return assembly


def main() -> None:
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    assembly = build_assembly()
    print(f"assembly: {dict(assembly.sizes)}")
    print(f"window: {int(assembly.time_bin_start.values[0])}-{int(assembly.time_bin_end.values[0])} ms")

    print(f"\nPackaging assembly -> {BUCKET_NAME}")
    assy_info = package_data_assembly(
        catalog_identifier=None, proto_data_assembly=assembly,
        assembly_identifier=ASSEMBLY_IDENTIFIER,
        stimulus_set_identifier=STIMULUS_IDENTIFIER,
        assembly_class_name="NeuroidAssembly",
        bucket_name=BUCKET_NAME,
    )
    print("assy_info:", assy_info)

    (OUTPUT_PATH / "assy_info.json").write_text(json.dumps(assy_info, indent=2))
    print(f"\nsaved upload info -> {OUTPUT_PATH / 'assy_info.json'}")
    print("\nNext step: paste version_id + sha1 into data/li2026/__init__.py")


if __name__ == "__main__":
    main()
