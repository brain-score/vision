from brainscore_vision import data_registry
from brainscore_vision.data_helpers.s3 import load_stimulus_set_from_s3

BIBTEX = """@article{bear2021physion,
title={Physion: Evaluating physical prediction from vision in humans and machines},
author={Bear, Daniel M and Wang, Elias and Mrowca, Damian and Binder, Felix J and Tung, Hsiao-Yu Fish and Pramod, RT and Holdaway, Cameron and Tao, Sirui and Smith, Kevin and Sun, Fan-Yun and others},
journal={arXiv preprint arXiv:2106.08261},
year={2021}
}"""

data_registry['PhysionOCPSmall'] = lambda: load_stimulus_set_from_s3(
    identifier='PhysionOCPSmall',
    bucket="brainio-brainscore",
    csv_sha1="5eef5bdd9a8c6f372b0c45fdc262507586376241",
    zip_sha1="3c60816ca2f06fa91caa5afaffb257b5ea5dc939",
    csv_version_id="4dpsWUIWJywoyUl809QSRwxx0vcqSUcZ",
    zip_version_id="fvY8ZKAFBszDgkDZHY1jJ24ieHfEE9X3")
