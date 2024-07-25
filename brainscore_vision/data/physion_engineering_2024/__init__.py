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
    csv_sha1="ef0cf49957aa033ae5eff5a6609b132c9f5e2ac6",
    zip_sha1="b25ad000246b2d6f3f2de9aec1eab2ba42f8d819",
    csv_version_id="g_ZhMFBDbXc_1416SJ4YSno6VeBPzliQ",
    zip_version_id="9uSzmhSA_Fz5ADY_02G79oVTR3tio3wL",
    filename_prefix="stimulus_")

stimulus_set_registry['PhysionGlobalDetection2024'] = lambda: load_stimulus_set_from_s3(
    identifier='PhysionGlobalDetection2024',
    bucket="brainio-brainscore",
    csv_sha1="e499ae251ff5afc67fbc09fefe766aa1abe8861b",
    zip_sha1="3688e6baa146c3162e24be7796099db066c77e19",
    csv_version_id="YqWtqOj3WUkoqvW..vTzuCZRI1IzKkmH",
    zip_version_id="M7h9NmJU.._6KbWH_.pQuKjYwWN3qLvQ",
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

stimulus_set_registry['PhysionPlacementDebug'] = lambda: load_stimulus_set_from_s3(
    identifier='PhysionPlacementDebug',
    bucket="brainio-brainscore",
    csv_sha1="27f8894d46cc49429162a19b80c64c5d2f38556c",
    zip_sha1="654ddef8be234b04e3b4d8d1347b75cd685d028f",
    csv_version_id="EqNl3xvVL2iyK3AWOtKI4TrkNNT23ePS",
    zip_version_id="0.jkDmGETaBkT3DINCy6CVksb8eW4v2d",
    filename_prefix="stimulus_")
