from brainscore_vision import stimulus_set_registry, data_registry
from brainscore_vision.data_helpers.s3 import load_stimulus_set_from_s3, load_assembly_from_s3

from brainio.assemblies import BehavioralAssembly

BIBTEX = """@article{bear2021physion,
title={Physion: Evaluating physical prediction from vision in humans and machines},
author={Bear, Daniel M and Wang, Elias and Mrowca, Damian and Binder, Felix J and Tung, Hsiao-Yu Fish and Pramod, RT and Holdaway, Cameron and Tao, Sirui and Smith, Kevin and Sun, Fan-Yun and others},
journal={arXiv preprint arXiv:2106.08261},
year={2021}
}"""


data_registry['PhysionHumanPrediction2024'] = lambda: load_assembly_from_s3(
    identifier='PhysionHumanPrediction2024',
    version_id="rmohBN6G.0ZiDLflcbD6tk_nBKpsL3eC",
    sha1="936ccef1b4dc272c0eaa3c8b1679a2196cb51c6a",
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
)

data_registry['PhysionHumanDetection2024'] = lambda: load_assembly_from_s3(
    identifier='PhysionHumanDetection2024',
    version_id="OkBlsTu3rWVBtGgWVJsDrdqVt8.PsFAi",
    sha1="48633473c8ddb985abe98b61feac65c5acbfe5b9",
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
)

