import os
from pathlib import Path

import pandas as pd

from brainscore_vision import data_registry, stimulus_set_registry, load_stimulus_set
from brainscore_core.supported_data_standards.brainio.s3 import (
    load_stimulus_set_from_s3, load_assembly_from_s3,
)
from brainscore_core.supported_data_standards.brainio.assemblies import (
    NeuronRecordingAssembly, StimulusMergeAssemblyLoader,
)
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

BIBTEX = """@article{li2026triplen,
    title = {Triple-N dataset: large-scale fMRI-guided dense recordings of nonhuman
             primate neural responses to natural scenes},
    author = {Li, Yipeng and Liu, Xieyi and Li, Wanru and Yang, Jia and Gong, Baoqi
              and Jin, Wei and Gong, Zhengxin and Wang, Kesheng and Luo, Jingqiu
              and Zhao, Zishuo and Bao, Pinglei},
    journal = {Nature Neuroscience},
    year = {2026},
    doi = {10.1038/s41593-026-02322-z},
    url = {https://doi.org/10.1038/s41593-026-02322-z},
}"""

# Macaque Neuropixels responses (5 monkeys, 90 sessions) to the 1000 NSD Shared1000
# images. stimulus_id is aligned to Allen2022 (nsd_<0-indexed 73k id>).
_BUCKET = "brainscore-storage/brainscore-vision/benchmarks/Li2026"

stimulus_set_registry['Li2026'] = lambda: load_stimulus_set_from_s3(
    identifier="Li2026_Stimuli",
    bucket=_BUCKET,
    csv_sha1="3e3603f9080dcc1c5519a3ac66895d55a3c486a1",
    zip_sha1="76cef574cbb09e53cc18573690558a85f9470231",
    csv_version_id="gRAM7LbVvIAgPeDCgvzX8sPuLsbsBbTN",
    zip_version_id="jIe9AbMniKL7UXUJShW8ypdXwucfBg5N",
    filename_prefix="stimulus_")

data_registry['Li2026'] = lambda: load_assembly_from_s3(
    # Static response matrix: mean firing rate in a fixed 70-170 ms post-onset window
    # applied uniformly to every unit (MajajHong convention), derived from the temporal
    # assembly. Carries two reliability coords: `reliability` (paper-canonical best-window
    # split-half SB, kept for provenance) and `reliability_window` (split-half SB at
    # 70-170 ms) -- the benchmark selects and ceils on `reliability_window` so the noise
    # ceiling matches the scored response. See data_packaging/build_li2026_static_70_170ms.py
    # and build_li2026_reliability_70_170ms.py.
    identifier="Li2026_Assembly",
    version_id="jJWOONfWTN3pbp6clmE3Ea8eQ9ZZb_D6",
    sha1="bb59217628132183a57ef10bb9017d857816ed53",
    bucket=_BUCKET,
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Li2026'))

# Temporal variant: per-image PSTHs rebinned to 10ms bins (0-300ms), all 90 sessions.
# The Li2026 S3 prefix is not public, so a user without AWS credentials sets
# BRAINSCORE_LI2026_DIR to the output directory of the data_packaging build scripts
# (see README "Building Li2026.temporal locally").
LOCAL_DIR_ENV = 'BRAINSCORE_LI2026_DIR'


def load_local_temporal(build_dir: str) -> NeuronRecordingAssembly:
    """Load ``Li2026.temporal`` from a local build directory, without S3.

    :param build_dir: directory holding ``Li2026_stimulus_set.csv``, ``stimuli/<stimulus_id>.png``
        and ``Li2026_temporal_assembly.nc``.
    :return: the temporal assembly with the local stimulus set attached.
    """
    build_dir = Path(build_dir)
    meta = pd.read_csv(build_dir / 'Li2026_stimulus_set.csv')
    stimulus_set = StimulusSet(meta)
    stimulus_set.stimulus_paths = {sid: str(build_dir / 'stimuli' / f'{sid}.png') for sid in meta['stimulus_id']}
    stimulus_set.identifier = 'Li2026_Stimuli'
    loader = StimulusMergeAssemblyLoader(cls=NeuronRecordingAssembly,
                                         file_path=str(build_dir / 'Li2026_temporal_assembly.nc'),
                                         stimulus_set_identifier=stimulus_set.identifier,
                                         stimulus_set=stimulus_set)
    assembly = loader.load()
    assembly.attrs['identifier'] = 'Li2026.temporal_Assembly'
    return assembly


def _load_temporal():
    if os.getenv(LOCAL_DIR_ENV):
        return load_local_temporal(os.environ[LOCAL_DIR_ENV])
    return load_assembly_from_s3(
        identifier="Li2026.temporal_Assembly",
        version_id="JmOzjXLy0K192AvgcpUxkN_X3B7z4i74",
        sha1="c27623909299dd4617672596c114fbe370ed271a",
        bucket=_BUCKET,
        cls=NeuronRecordingAssembly,
        stimulus_set_loader=lambda: load_stimulus_set('Li2026'))


data_registry['Li2026.temporal'] = _load_temporal
