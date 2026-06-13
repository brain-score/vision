from brainscore_vision import data_registry, stimulus_set_registry, load_stimulus_set
from brainscore_core.supported_data_standards.brainio.s3 import (
    load_stimulus_set_from_s3, load_assembly_from_s3,
)
from brainscore_core.supported_data_standards.brainio.assemblies import NeuronRecordingAssembly

BIBTEX = """@article{li2026triplen,
    title = {Triple-N dataset: large-scale fMRI-guided dense recordings of nonhuman
             primate neural responses to natural scenes},
    author = {Li, Yipeng and Liu, Xieyi and Li, Wanru and Yang, Jia and Gong, Baoqi
              and Jin, Wei and Gong, Zhengxin and Wang, Kesheng and Luo, Jingqiu
              and Zhao, Zishuo and Bao, Pinglei},
    journal = {Nature Neuroscience},
    year = {2026},
    doi = {10.1038/s41593-026-02322-z},
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
    identifier="Li2026_Assembly",
    version_id="bdLlHwPAmkVr4aptJTI.CECHRghlOS_k",
    sha1="830eae557ad8ed59ca0fd57b9ae243f72aa09ccf",
    bucket=_BUCKET,
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Li2026'))
