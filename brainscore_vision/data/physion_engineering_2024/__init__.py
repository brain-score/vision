from brainio.assemblies import NeuronRecordingAssembly

from brainscore_vision import data_registry, load_stimulus_set
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3

BIBTEX = """@article{bear2021physion,
title={Physion: Evaluating physical prediction from vision in humans and machines},
author={Bear, Daniel M and Wang, Elias and Mrowca, Damian and Binder, Felix J and Tung, Hsiao-Yu Fish and Pramod, RT and Holdaway, Cameron and Tao, Sirui and Smith, Kevin and Sun, Fan-Yun and others},
journal={arXiv preprint arXiv:2106.08261},
year={2021}
}"""

stimulus_set_registry['Physionv1.5-ocd'] = lambda: load_stimulus_set_from_s3(
    identifier='PhysionOCPSmall',
    bucket="brainio-brainscore",
    csv_sha1="030a29fd427105aa27ca8549b2e855262f814e17",
    zip_sha1="3c60816ca2f06fa91caa5afaffb257b5ea5dc939",
    csv_version_id="VgOO_7iVA5ECa.Jl0RwJwSwQv0KC_z76",
    zip_version_id="d4BkiwhR3g6f9IvyjFEN18C31cmO295R")