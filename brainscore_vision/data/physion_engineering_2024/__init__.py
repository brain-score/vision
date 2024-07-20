from brainscore_vision import stimulus_set_registry
from brainscore_vision.data_helpers.s3 import load_stimulus_set_from_s3

BIBTEX = """@article{bear2021physion,
title={Physion: Evaluating physical prediction from vision in humans and machines},
author={Bear, Daniel M and Wang, Elias and Mrowca, Damian and Binder, Felix J and Tung, Hsiao-Yu Fish and Pramod, RT and Holdaway, Cameron and Tao, Sirui and Smith, Kevin and Sun, Fan-Yun and others},
journal={arXiv preprint arXiv:2106.08261},
year={2021}
}"""

stimulus_set_registry['PhysionOCPSmall'] = lambda: load_stimulus_set_from_s3(
    identifier='PhysionOCPSmall',
    bucket="brainio-brainscore",
    csv_sha1="5eef5bdd9a8c6f372b0c45fdc262507586376241",
    zip_sha1="3c60816ca2f06fa91caa5afaffb257b5ea5dc939",
    csv_version_id="4dpsWUIWJywoyUl809QSRwxx0vcqSUcZ",
    zip_version_id="fvY7ZKAFBszDgkDZHY1jJ24ieHfEE9X3",
    filename_prefix="stimulus_")

stimulus_set_registry['PhysionGlobalPrediction2024'] = lambda: load_stimulus_set_from_s3(
    identifier='PhysionGlobalPrediction2024',
    bucket="brainio-brainscore",
    csv_sha1="8beaca2935925a6d6f0df425c1bcddfd0e7fa2e9",
    zip_sha1="b25ad000246b2d6f3f2de9aec1eab2ba42f8d819",
    csv_version_id="N8XB8CB8XYGuRAft9.b44HnBZyRd5FJV",
    zip_version_id="NPwRAqmyEOEeeDDUo0jEXK5oQc6afga7",
    filename_prefix="stimulus_")

stimulus_set_registry['PhysionGlobalDetection2024'] = lambda: load_stimulus_set_from_s3(
    identifier='PhysionGlobalDetection2024',
    bucket="brainio-brainscore",
    csv_sha1="1a2800360740f59db3f18751c30146ba5282cf84",
    zip_sha1="3688e6baa146c3162e24be7796099db066c77e19",
    csv_version_id="A7XWOQNyzoAVHYFzmliZxYNE2RDE.IsK",
    zip_version_id=".2wTFnRlyLV.mpVDvyPAcp68AIeMU.7a",
    filename_prefix="stimulus_")

stimulus_set_registry['PhysionSnippetDetection2024'] = lambda: load_stimulus_set_from_s3(
    identifier='PhysionSnippetDetection2024',
    bucket="brainio-brainscore",
    csv_sha1="df7efe1c99f4614d386e26358e3542f3fdffe35b",
    zip_sha1="5223a070970311f67ce1c3a021da79d020942249",
    csv_version_id="iC7zUXRikmWePpFEYi1CdSBxM.A5XjlA",
    zip_version_id="KCWFS49CUYVpxsZ9NS0p8.9pIR3VRTvm",
    filename_prefix="stimulus_")

stimulus_set_registry['PhysionSnippetDebug'] = lambda: load_stimulus_set_from_s3(
    identifier='PhysionSnippetDebug',
    bucket="brainio-brainscore",
    csv_sha1="bde4a2ffdb7c8aa75900ddd797dd0f156387092f",
    zip_sha1="50debb292e722bf8fad9a2170c560f7e5b1a7235",
    csv_version_id="Ach6l51PwhcTmfqZVN3d3VaXhUd9X_F4",
    zip_version_id="VOmW6TrR6uuosOTUlwxWpYvBaFUwyji8",
    filename_prefix="stimulus_")
