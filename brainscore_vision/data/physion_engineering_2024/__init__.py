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
    csv_sha1="be7bc565b477bc1d1b07bf5045bb6094c83914ad",
    zip_sha1="b25ad000246b2d6f3f2de9aec1eab2ba42f8d819",
    csv_version_id="_dnEmzApXMGxd0yJexEiXWv._vNBsVIe",
    zip_version_id="HtG9K169GBBWdWHy7EOEd5Iqs3KsyFaU",
    filename_prefix="stimulus_")

stimulus_set_registry['PhysionGlobalDetection2024'] = lambda: load_stimulus_set_from_s3(
    identifier='PhysionGlobalDetection2024',
    bucket="brainio-brainscore",
    csv_sha1="f9c8f92a0fcbd76121fe0982b13df9a01ebb5c4d",
    zip_sha1="3688e6baa146c3162e24be7796099db066c77e19",
    csv_version_id="5RBvTpGHVYFYBymYRf3xH4_ubSEr6w.7",
    zip_version_id="C.i2VnbEWAD.5H3egw.9s8MOgDEB0Lek",
    filename_prefix="stimulus_")

stimulus_set_registry['PhysionSnippetDetection2024'] = lambda: load_stimulus_set_from_s3(
    identifier='PhysionSnippetDetection2024',
    bucket="brainio-brainscore",
    csv_sha1="96e3f5e1bd8f0b59d35274ffbeac1db18fb75a85",
    zip_sha1="5223a070970311f67ce1c3a021da79d020942249",
    csv_version_id="qz8Za0xC2Uh49oAYiuie1AXd2oZqoRyZ",
    zip_version_id="ZZNcBz2oAumcAWwAu5azGHWtAzGK4ppx",
    filename_prefix="stimulus_")

stimulus_set_registry['PhysionSnippetDebug'] = lambda: load_stimulus_set_from_s3(
    identifier='PhysionSnippetDebug',
    bucket="brainio-brainscore",
    csv_sha1="bde4a2ffdb7c8aa75900ddd797dd0f156387092f",
    zip_sha1="50debb292e722bf8fad9a2170c560f7e5b1a7235",
    csv_version_id="Ach6l51PwhcTmfqZVN3d3VaXhUd9X_F4",
    zip_version_id="VOmW6TrR6uuosOTUlwxWpYvBaFUwyji8",
    filename_prefix="stimulus_")
