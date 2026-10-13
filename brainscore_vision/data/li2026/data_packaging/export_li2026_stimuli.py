"""Export the 1000 Li2026 stimuli as PNGs from NSD's ``nsd_stimuli.hdf5``.

Run after build_li2026_static.py (which writes the stimulus CSV). NSD's bucket is
public, so no AWS credentials are needed::

  aws s3 cp --no-sign-request \\
      s3://natural-scenes-dataset/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5 .   # 39.6 GB

Output: ``<out-dir>/stimuli/<stimulus_id>.png``, the layout load_local_temporal reads.
"""
import argparse
from pathlib import Path

import h5py
import pandas as pd
from PIL import Image


def export_stimuli(h5: h5py.File, stim: pd.DataFrame, out_dir: Path) -> None:
    """Write one PNG per stimulus from the NSD ``imgBrick`` (73000 x 425 x 425 x 3).

    :param h5: open ``nsd_stimuli.hdf5``.
    :param stim: stimulus table with ``stimulus_id`` and 0-indexed ``nsd_id``.
    :param out_dir: build directory; images go to ``out_dir/stimuli``.
    """
    stimuli_dir = out_dir / "stimuli"
    stimuli_dir.mkdir(parents=True, exist_ok=True)
    brick = h5["imgBrick"]
    for row in stim.itertuples():
        Image.fromarray(brick[int(row.nsd_id)]).save(stimuli_dir / f"{row.stimulus_id}.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nsd-stimuli", required=True, help="path to nsd_stimuli.hdf5")
    ap.add_argument("--out-dir", default="/Volumes/Hagibis/triple-n/build",
                    help="build directory holding Li2026_stimulus_set.csv")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    stim = pd.read_csv(out_dir / "Li2026_stimulus_set.csv")
    with h5py.File(args.nsd_stimuli, "r") as h5:
        export_stimuli(h5, stim, out_dir)
    print(f"wrote {len(stim)} PNGs -> {out_dir / 'stimuli'}")


if __name__ == "__main__":
    main()
