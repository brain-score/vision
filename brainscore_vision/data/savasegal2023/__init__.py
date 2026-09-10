from brainscore_core.supported_data_standards.brainio.assemblies import NeuronRecordingAssembly

import brainscore_vision
from brainscore_vision import data_registry, stimulus_set_registry
from brainscore_core.supported_data_standards.brainio.s3 import load_stimulus_set_from_s3, load_assembly_from_s3

BIBTEX = """@article{sava2023individual,
  title={Individual differences in neural event segmentation of continuous experiences},
  author={Sava-Segal, Clara and Richards, Chandler and Leung, Megan and Finn, Emily S},
  journal={Cerebral Cortex},
  volume={33},
  number={13},
  pages={8164--8178},
  year={2023},
  publisher={Oxford University Press}
}"""

# assembly: Savasegal2023-fMRI-Defeat
data_registry['Savasegal2023-fMRI-Defeat'] = lambda: load_assembly_from_s3(
    identifier="Savasegal2023-fMRI-Defeat",
    version_id="eb..6dECXftcmMTZX2DXWPVKz2luoUn9",
    sha1="b6ea38ad12718122f7fef78d2ec5e809cce7c977",
    bucket="brainscore-storage/brainscore-vision/benchmarks/Savasegal2023-fMRI-Defeat",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: brainscore_vision.load_stimulus_set('Savasegal2023-Defeat'),
)

# stimulus set: Savasegal2023-Defeat
stimulus_set_registry['Savasegal2023-Defeat'] = lambda: load_stimulus_set_from_s3(
    identifier="Savasegal2023-Defeat",
    bucket="brainscore-storage/brainscore-vision/benchmarks/Savasegal2023-fMRI-Defeat",
    csv_sha1="ef5de41cc775b12583cb2f41b25ebbcd157bd87a",
    zip_sha1="5de346354a1fc2a74a7a013e74583a57b83dfe01",
    csv_version_id="FKYjSSKOMZEpQo3h6x6MflH3NzNqP4Lg",
    zip_version_id="Ll2ENeVR7E_.AUB9chchamR3MFNGqZgk")


# assembly: Savasegal2023-fMRI-Growth
data_registry['Savasegal2023-fMRI-Growth'] = lambda: load_assembly_from_s3(
    identifier="Savasegal2023-fMRI-Growth",
    version_id="fKUd3lqYkmVuhk7tQhWgEgjYUuYxzHr2",
    sha1="93b6b0db811dce340e709dfb57156156e78d437d",
    bucket="brainscore-storage/brainscore-vision/benchmarks/Savasegal2023-fMRI-Growth",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: brainscore_vision.load_stimulus_set('Savasegal2023-Growth'),
)

# stimulus set: Savasegal2023-Growth
stimulus_set_registry['Savasegal2023-Growth'] = lambda: load_stimulus_set_from_s3(
    identifier="Savasegal2023-Growth",
    bucket="brainscore-storage/brainscore-vision/benchmarks/Savasegal2023-fMRI-Growth",
    csv_sha1="029e2d68ad952d2929aa8cfb836443f7d3714fea",
    zip_sha1="7dbdbb36920017b39793c0cc51c96393e33d8a99",
    csv_version_id="DSmsK1BISRn5j2lLjWINJdjmuTx1MuYk",
    zip_version_id="E.EBCeeQYY8MM8cAJvL261n4ox7Z.huW")


# assembly: Savasegal2023-fMRI-Iteration
data_registry['Savasegal2023-fMRI-Iteration'] = lambda: load_assembly_from_s3(
    identifier="Savasegal2023-fMRI-Iteration",
    version_id="VI_dZOWapYT.C2i1o_BkmYJpjVAQQQB9",
    sha1="1595605baf4b461f11164b768337a10986b77618",
    bucket="brainscore-storage/brainscore-vision/benchmarks/Savasegal2023-fMRI-Iteration",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: brainscore_vision.load_stimulus_set('Savasegal2023-Iteration'),
)

# stimulus set: Savasegal2023-Iteration
stimulus_set_registry['Savasegal2023-Iteration'] = lambda: load_stimulus_set_from_s3(
    identifier="Savasegal2023-Iteration",
    bucket="brainscore-storage/brainscore-vision/benchmarks/Savasegal2023-fMRI-Iteration",
    csv_sha1="5634119759f04886dc741f41243bc84a66d5d569",
    zip_sha1="0f5cca320e6eaf814cad162cac4f40dd9dc9dcbe",
    csv_version_id="jSz5.kOpWrwUTHjMhUGRCbHYke1o_NzW",
    zip_version_id="qJoYn1uhNbWisP.K6ix4Npszh7sTYcn7")


# assembly: Savasegal2023-fMRI-Lemonade
data_registry['Savasegal2023-fMRI-Lemonade'] = lambda: load_assembly_from_s3(
    identifier="Savasegal2023-fMRI-Lemonade",
    version_id="BL3._yZ.5QnpmqfnZUQo8w6uJskg_uj7",
    sha1="3c5849b7c9682a879059ff0e8c4b20656807bb4f",
    bucket="brainscore-storage/brainscore-vision/benchmarks/Savasegal2023-fMRI-Lemonade",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: brainscore_vision.load_stimulus_set('Savasegal2023-Lemonade'),
)

# stimulus set: Savasegal2023-Lemonade
stimulus_set_registry['Savasegal2023-Lemonade'] = lambda: load_stimulus_set_from_s3(
    identifier="Savasegal2023-Lemonade",
    bucket="brainscore-storage/brainscore-vision/benchmarks/Savasegal2023-fMRI-Lemonade",
    csv_sha1="4537ca9955473b6688cd9f990b9a16182cbbb50e",
    zip_sha1="79edea7316c9f5dde7a8682dce7000ed506d2026",
    csv_version_id="EeZlMuTe2gsAmqbyeSMAOXMqovDGDkrK",
    zip_version_id="ZBnxvkOf8WZVAIoMOieNhVbEXTxfixnF")