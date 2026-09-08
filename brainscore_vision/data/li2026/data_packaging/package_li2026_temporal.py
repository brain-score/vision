"""Package the Li2026 v3 TEMPORAL assembly (built on EC2 from GoodUnit PSTHs) to S3.
Reuses the already-uploaded Li2026_Stimuli stimulus set.
Source: /Volumes/Hagibis/triple-n/build/Li2026_temporal_assembly.nc
"""
import json
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr

from brainscore_core.supported_data_standards.brainio.assemblies import NeuronRecordingAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
import brainscore_core.supported_data_standards.brainio.packaging as _brainio_pkg
from brainscore_core.supported_data_standards.brainio.packaging import package_data_assembly

_orig = _brainio_pkg.upload_to_s3
def _upload_with_prefix(src, bucket, key):
    if "/" in bucket:
        b, p = bucket.split("/", 1); key = f"{p}/{key}"; bucket = b
    return _orig(src, bucket, key)
_brainio_pkg.upload_to_s3 = _upload_with_prefix

BUILD = Path("/Volumes/Hagibis/triple-n/build")
BUCKET = "brainscore-storage/brainscore-vision/benchmarks/Li2026"
OUT = Path(__file__).parent / "uploaded_data_info"
STIM_ID = "Li2026_Stimuli"
ASM_ID = "Li2026.temporal_Assembly"


def build_stimulus_set() -> StimulusSet:
    meta = pd.read_csv(BUILD / "Li2026_stimulus_set.csv")
    ss = StimulusSet([{"stimulus_id": r.stimulus_id, "nsd_id": int(r.nsd_id),
                       "nsd_id_1indexed": int(r.nsd_id_1indexed), "tn_index": int(r.tn_index),
                       "coco_id": int(r.coco_id)} for r in meta.itertuples()])
    ss.stimulus_paths = {r.stimulus_id: str(BUILD / "stimuli" / f"{r.stimulus_id}.png") for r in meta.itertuples()}
    ss.name = ss.identifier = STIM_ID
    return ss


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    da = xr.open_dataarray(str(BUILD / "Li2026_temporal_assembly.nc")); da.load()
    asm = NeuronRecordingAssembly(da)
    asm.name = ASM_ID
    asm.attrs["stimulus_set"] = build_stimulus_set()
    print(f"temporal assembly: {dict(asm.sizes)} regions={np.unique(asm['region'].values)}")
    assert set(np.unique(asm["stimulus_id"].values)) <= set(asm.attrs["stimulus_set"]["stimulus_id"].values)
    info = package_data_assembly(
        catalog_identifier=None, proto_data_assembly=asm,
        assembly_identifier=ASM_ID, stimulus_set_identifier=STIM_ID,
        assembly_class_name="NeuroidAssembly", bucket_name=BUCKET)
    print("assy_info:", info)
    (OUT / "assy_temporal_info.json").write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
