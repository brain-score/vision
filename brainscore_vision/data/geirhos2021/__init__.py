from brainio.assemblies import BehavioralAssembly

from brainscore_vision import data_registry, stimulus_set_registry, load_stimulus_set
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3, load_stimulus_set_from_s3

BIBTEX = """@article{geirhos2021partial,
  title={Partial success in closing the gap between human and machine vision},
  author={Geirhos, Robert and Narayanappa, Kantharaju and Mitzkus, Benjamin and Thieringer, Tizian and Bethge, Matthias and Wichmann, Felix A and Brendel, Wieland},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={23885--23899},
  year={2021}
}"""

DATASETS = ['colour', 'contrast', 'cue-conflict', 'edge',
            'eidolonI', 'eidolonII', 'eidolonIII',
            'false-colour', 'high-pass', 'low-pass', 'phase-scrambling', 'power-equalisation',
            'rotation', 'silhouette', 'sketch', 'stylized', 'uniform-noise']


# 'colour'
# assembly
data_registry['brendel.Geirhos2021_colour'] = lambda: load_assembly_from_s3(
    identifier='brendel.Geirhos2021_colour',
    version_id="RDjCFAFt_J5mMwFBN9Ifo0OyNPKlToqf",
    sha1="258862d82467614e45cc1e488a5ac909eb6e122d",
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('brendel.Geirhos2021_colour'),
)

# stimulus set
stimulus_set_registry['brendel.Geirhos2021_colour'] = lambda: load_stimulus_set_from_s3(
    identifier='brendel.Geirhos2021_colour',
    bucket="brainio-brainscore",
    csv_sha1="9c97c155fd6039a95978be89eb604c6894c5fa16",
    zip_sha1="d166f1d3dc3d00c4f51a489e6fcf96dbbe778d2c",
    csv_version_id="1ZaFYwHPBkDOrgdrwGHYqMfJJBCWei21",
    zip_version_id="X62ivk_UuHgh7Sd7VwDxgnB8tWPK06gt")

# 'contrast'
# assembly
data_registry['brendel.Geirhos2021_contrast'] = lambda: load_assembly_from_s3(
    identifier='brendel.Geirhos2021_contrast',
    version_id="JMyi4jtdirYkdIQEIZRJ2024NQ1McQXc",
    sha1="1e114c987dc035ccca43781ff8cee9689acd3c3f",
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('brendel.Geirhos2021_contrast'),
)

# stimulus set
stimulus_set_registry['brendel.Geirhos2021_contrast'] = lambda: load_stimulus_set_from_s3(
    identifier='brendel.Geirhos2021_contrast',
    bucket="brainio-brainscore",
    csv_sha1="f6ef69a2e8937e1d0d83c8a21b325b4273494cb5",
    zip_sha1="ebeeef2f9c6a7282e20ef2026dc77eefa026957b",
    csv_version_id="OsunVV3gPXsgB3lp_kUirl6t.Qy9Xzu9",
    zip_version_id="Gezu9w0T8jL0BHIACTy3VBoR2nleOTK0")

# 'cue-conflict'
# assembly
data_registry['brendel.Geirhos2021_cue-conflict'] = lambda: load_assembly_from_s3(
    identifier='brendel.Geirhos2021_cue-conflict',
    version_id="WCGAQqS86x2z5CXGlegLPHu3XsH.tEk3",
    sha1="cc214e3595d34565b13963c5f56049769a39a5c9",
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('brendel.Geirhos2021_cue-conflict'),
)

# stimulus set
stimulus_set_registry['brendel.Geirhos2021_cue-conflict'] = lambda: load_stimulus_set_from_s3(
    identifier='brendel.Geirhos2021_cue-conflict',
    bucket="brainio-brainscore",
    csv_sha1="8d3ae89d8870fb7d7c5d5ff387085b1f0116e2b7",
    zip_sha1="7e601186b181102939cd0b43a4e8a3ca95c18259",
    csv_version_id="fNwBooRV2kLU9HenaHhr7RSV_9RRt.DF",
    zip_version_id="Zox6nvmZim8DBEx2uLmu6KQBIvv5wLis")

# 'edge'
# assembly
data_registry['brendel.Geirhos2021_edge'] = lambda: load_assembly_from_s3(
    identifier='brendel.Geirhos2021_edge',
    version_id="gTmCANJ797ZNcfKzecCnwlnUYcgl1.xa",
    sha1="ab1dc9e188e248da07215b375eb3dbcc58fde7fb",
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('brendel.Geirhos2021_edge'),
)

# stimulus set
stimulus_set_registry['brendel.Geirhos2021_edge'] = lambda: load_stimulus_set_from_s3(
    identifier='brendel.Geirhos2021_edge',
    bucket="brainio-brainscore",
    csv_sha1="fb57005ecb80e4e37b01e084f1f176fe7f59ff7f",
    zip_sha1="6c1199d90836a26be454aa799864a63c5efacaa1",
    csv_version_id="3zqzUnK1hHlYh9tRNMHpuAP9bnDLtxTp",
    zip_version_id="6OQpha9nx0GBrLYaYCJ89S2vKaLALKeY")

# 'eidolonI'
# assembly
data_registry['brendel.Geirhos2021_eidolonI'] = lambda: load_assembly_from_s3(
    identifier='brendel.Geirhos2021_eidolonI',
    version_id="rVmuuTXAi6XqVI3hdyhe7WqSJNFOsFA4",
    sha1="0f01f351ae19eafc2cb5e504d98e5cd01b4c07b4",
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('brendel.Geirhos2021_eidolonI'),
)

# stimulus set
stimulus_set_registry['brendel.Geirhos2021_eidolonI'] = lambda: load_stimulus_set_from_s3(
    identifier='brendel.Geirhos2021_eidolonI',
    bucket="brainio-brainscore",
    csv_sha1="049adbed36fed52c609d38d5230e6084336df6b9",
    zip_sha1="abaa45a225628dd66e424ec7d8e2b10a0c88bc0d",
    csv_version_id="q4QRehm2cIAi0E1erEQCBfyxMbqbNItt",
    zip_version_id="3AGu.Q8jBbJh5iSiH5Mpq9QZDu3itfiI")

# 'eidolonII'
# assembly
data_registry['brendel.Geirhos2021_eidolonII'] = lambda: load_assembly_from_s3(
    identifier='brendel.Geirhos2021_eidolonII',
    version_id="yOXPugtdV6fx0_BAEY4NeYlzwfCChIaR",
    sha1="499eea0f0c0817d02b5b97d2ebab89bc4c40a153",
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('brendel.Geirhos2021_eidolonII'),
)

# stimulus set
stimulus_set_registry['brendel.Geirhos2021_eidolonII'] = lambda: load_stimulus_set_from_s3(
    identifier='brendel.Geirhos2021_eidolonII',
    bucket="brainio-brainscore",
    csv_sha1="1806ada128c260ab54db570d2e73aea71d679754",
    zip_sha1="2654ba55291f8ab972f18b36565f9ead80a45339",
    csv_version_id="4kXnYMC1rTdUZ8jy1ga29dTbhWuVuRCb",
    zip_version_id="okHEb16F_gxXl9EgMRZpUy87.zziTA90")

# 'eidolonIII'
# assembly
data_registry['brendel.Geirhos2021_eidolonIII'] = lambda: load_assembly_from_s3(
    identifier='brendel.Geirhos2021_eidolonIII',
    version_id="zkbswpduS0z51_OrUF_XA7UaKb_4bYkN",
    sha1="e7c9a49e729f8666f8aedc6e47c746fbbe2ebe36",
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('brendel.Geirhos2021_eidolonIII'),
)

# stimulus set
stimulus_set_registry['brendel.Geirhos2021_eidolonIII'] = lambda: load_stimulus_set_from_s3(
    identifier='brendel.Geirhos2021_eidolonIII',
    bucket="brainio-brainscore",
    csv_sha1="ba0173b315f02df16d418dc3ff1df7dc498b4893",
    zip_sha1="d0304c0c0024d0f493ea9c0c47ae0221da391016",
    csv_version_id="JVa2jLVBG9_EoFe3dV8kQmBAvXQxb4AA",
    zip_version_id="SJRD2F8CsfHwrpppz.FGLClBAAhOV2wT")

# 'false-colour'
# assembly
data_registry['brendel.Geirhos2021_false-colour'] = lambda: load_assembly_from_s3(
    identifier='brendel.Geirhos2021_false-colour',
    version_id="GmYh1SBFUiEBLzkf1eRvA_GdXs1kCqtO",
    sha1="4dc072264651c81575564ba4818a12b8e8039c65",
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('brendel.Geirhos2021_false-colour'),
)

# stimulus set
stimulus_set_registry['brendel.Geirhos2021_false-colour'] = lambda: load_stimulus_set_from_s3(
    identifier='brendel.Geirhos2021_false-colour',
    bucket="brainio-brainscore",
    csv_sha1="8a09a7af8ec44339bcae5500ae5900d9c4309042",
    zip_sha1="ec0ba347fc14d0c0587d38bfa96e4ab5d2f7979a",
    csv_version_id="jOj2gCxQDPkWk8.CcpILCHAdyxDVgGog",
    zip_version_id="H61i.jgHll7KOh8CJUKPMI3hyp.0C9o1")

# 'high-pass'
# assembly
data_registry['brendel.Geirhos2021_high-pass'] = lambda: load_assembly_from_s3(
    identifier='brendel.Geirhos2021_high-pass',
    version_id="FB_uDoX3nMZR1Qb6vcR4Y.PdEKKa1eKo",
    sha1="5df45c69127758f1ba3391671c521711050e3b4d",
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('brendel.Geirhos2021_high-pass'),
)

# stimulus set
stimulus_set_registry['brendel.Geirhos2021_high-pass'] = lambda: load_stimulus_set_from_s3(
    identifier='brendel.Geirhos2021_high-pass',
    bucket="brainio-brainscore",
    csv_sha1="ddf523dcf43398cc15894c7b51c436d526e6c992",
    zip_sha1="12322bb17270a5dde206314fcdc125c4bb235e3b",
    csv_version_id="J9v3TDGWx0dqdNYh2Vz8BctY2EpQ7ndd",
    zip_version_id="L.6uThKdjIE3U0WxH.7IehrjiBLdbkQ6")

# 'low-pass'
# assembly
data_registry['brendel.Geirhos2021_low-pass'] = lambda: load_assembly_from_s3(
    identifier='brendel.Geirhos2021_low-pass',
    version_id="qUX5BRBV1rWK75m4QKH5aL1CVKs8x7u7",
    sha1="75ab628d9e6d0d634290567b1cb261d7f8dc61e2",
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('brendel.Geirhos2021_low-pass'),
)

# stimulus set
stimulus_set_registry['brendel.Geirhos2021_low-pass'] = lambda: load_stimulus_set_from_s3(
    identifier='brendel.Geirhos2021_low-pass',
    bucket="brainio-brainscore",
    csv_sha1="1264f9be407c7d428cf3d62a7bb1b1bb45a821bc",
    zip_sha1="ad087676e04e51adadea7c7c5c1fa28e4dd6360c",
    csv_version_id="yI1wYlc4y3DvtXGFb81saMOHgDP_t75r",
    zip_version_id="WUd5KNK.lV05AObHabGXQYKdrJ3F9YL4")

# 'phase-scrambling'
# assembly
data_registry['brendel.Geirhos2021_phase-scrambling'] = lambda: load_assembly_from_s3(
    identifier='brendel.Geirhos2021_phase-scrambling',
    version_id="Dg4Je4uHy4pYe7_CIN27fWgeVNonHlZP",
    sha1="4124f9f5b86fb6ed82c98197d292eef50b608aba",
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('brendel.Geirhos2021_phase-scrambling'),
)

# stimulus set
stimulus_set_registry['brendel.Geirhos2021_phase-scrambling'] = lambda: load_stimulus_set_from_s3(
    identifier='brendel.Geirhos2021_phase-scrambling',
    bucket="brainio-brainscore",
    csv_sha1="0cc87f7ac42c2266f98d3a08783f7173499ec2fc",
    zip_sha1="462e77ab9533072b7036118f1f697e8c9bf30ae4",
    csv_version_id="VWItIr6fMdSSP1yfNRSkMDZ4uuxMxrgb",
    zip_version_id="xguK90WVWgOHUmgYFcoe13qo8BJ6z6MW")

# 'power-equalisation'
# assembly
data_registry['brendel.Geirhos2021_power-equalisation'] = lambda: load_assembly_from_s3(
    identifier='brendel.Geirhos2021_power-equalisation',
    version_id="XKaUgVEpoBaTNL0qpFFRNyTq3WRcPQJY",
    sha1="0aba1b50a7e0802d76c41d332a121a3f96ef4f7d",
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('brendel.Geirhos2021_power-equalisation'),
)

# stimulus set
stimulus_set_registry['brendel.Geirhos2021_power-equalisation'] = lambda: load_stimulus_set_from_s3(
    identifier='brendel.Geirhos2021_power-equalisation',
    bucket="brainio-brainscore",
    csv_sha1="743935476b1fe4b7bd4d01a4eed24cd9ed5b3a22",
    zip_sha1="a0d5307525bccf8da8f3d293e7e324b9b20248c6",
    csv_version_id="cJ82v9s6U1CF9ZlQwlYy3f8cUqP3oA3W",
    zip_version_id="gzdXAFvWg777viFJhkE_9h1lxSda.SRH")

# 'rotation'
# assembly
data_registry['brendel.Geirhos2021_rotation'] = lambda: load_assembly_from_s3(
    identifier='brendel.Geirhos2021_rotation',
    version_id="zkIC2EQlN5oDwhtLDbnl_THME74mtFfc",
    sha1="e51a5c3bc95ade159e71aa602232063730bcd57b",
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('brendel.Geirhos2021_rotation'),
)

# stimulus set
stimulus_set_registry['brendel.Geirhos2021_rotation'] = lambda: load_stimulus_set_from_s3(
    identifier='brendel.Geirhos2021_rotation',
    bucket="brainio-brainscore",
    csv_sha1="2577831e9ead905669613fa17cb2651d0c48a455",
    zip_sha1="8f5d9cb217807e96ace61337144e429d0d4ba04c",
    csv_version_id="gWUg4SS.OtrtbV14RXemhMgjQeGw5M64",
    zip_version_id="VoHUYi_3g0awBpC2k8WV.2lCuxBQvdoT")

# 'silhouette'
# assembly
data_registry['brendel.Geirhos2021_silhouette'] = lambda: load_assembly_from_s3(
    identifier='brendel.Geirhos2021_silhouette',
    version_id="xlpXfo7iy8_nssKR4OwM4199E6wCA0nA",
    sha1="7dc94991465fe8009244e0d6fb8283419a1f9885",
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('brendel.Geirhos2021_silhouette'),
)

# stimulus set
stimulus_set_registry['brendel.Geirhos2021_silhouette'] = lambda: load_stimulus_set_from_s3(
    identifier='brendel.Geirhos2021_silhouette',
    bucket="brainio-brainscore",
    csv_sha1="fb57005ecb80e4e37b01e084f1f176fe7f59ff7f",
    zip_sha1="36c8a481f7876a2af2ad4fe80890b302fe3ae91e",
    csv_version_id="7zg2Ex3miPaGvktK7kjIfB5e9DNhVS6P",
    zip_version_id="RFDrz2.PR7A5DfjeQpmejx3R4orIZM4q")

# 'sketch'
# assembly
data_registry['brendel.Geirhos2021_sketch'] = lambda: load_assembly_from_s3(
    identifier='brendel.Geirhos2021_sketch',
    version_id="w7_3YJYTmVMYWbNY6SQvzRvYb45mRqte",
    sha1="6709850864cea16d99a29fb31ae3c4a489983562",
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('brendel.Geirhos2021_sketch'),
)

# stimulus set
stimulus_set_registry['brendel.Geirhos2021_sketch'] = lambda: load_stimulus_set_from_s3(
    identifier='brendel.Geirhos2021_sketch',
    bucket="brainio-brainscore",
    csv_sha1="a5741b2f7bd08541a0dbefd7fb2d6a3845ca800b",
    zip_sha1="8e8712f08a5ad3655ea2bd8cd675db8cdf65129a",
    csv_version_id="Z.EJlugAmsHoy1kwR9TSI6uCmldgn747",
    zip_version_id="8pw3EY5H4oLJVhygvrUcAQc05lj4JllD")

# 'stylized'
# assembly
data_registry['brendel.Geirhos2021_stylized'] = lambda: load_assembly_from_s3(
    identifier='brendel.Geirhos2021_stylized',
    version_id="0ESwWyMc5wODloonsSkzURncuYYJMxql",
    sha1="dcf15f292e787a88e1e0f271e6b2838d6bdadfd3",
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('brendel.Geirhos2021_stylized'),
)

# stimulus set
stimulus_set_registry['brendel.Geirhos2021_stylized'] = lambda: load_stimulus_set_from_s3(
    identifier='brendel.Geirhos2021_stylized',
    bucket="brainio-brainscore",
    csv_sha1="2265a540dee6915150bf7c61143eaf788f603866",
    zip_sha1="75d273e8de643b0d814fbe60cd237c31ebe19c44",
    csv_version_id="r5ESVPnObUmVhhHbHCIXs4gKSvT2Ezu5",
    zip_version_id="p4lxCUU.4WG.xEPGx3gsILhjhcapZiBS")

# 'uniform-noise'
# assembly
data_registry['brendel.Geirhos2021_uniform-noise'] = lambda: load_assembly_from_s3(
    identifier='brendel.Geirhos2021_uniform-noise',
    version_id="89aJ7jLIB68QCNCEcQsBjchKbv.31YWH",
    sha1="f5e8b2636738f978c71591b8df6f8a21a66b72d1",
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('brendel.Geirhos2021_uniform-noise'),
)

# stimulus set
stimulus_set_registry['brendel.Geirhos2021_uniform-noise'] = lambda: load_stimulus_set_from_s3(
    identifier='brendel.Geirhos2021_uniform-noise',
    bucket="brainio-brainscore",
    csv_sha1="89b62a6af878974d388278ed0e23e8ed1c2fd855",
    zip_sha1="ff4566542d65056028660293e2409b532e887714",
    csv_version_id="8edSpj0OAnnBdG3IgRNVTff6x3v6i7aS",
    zip_version_id="XRXqDJpQHZLZKERvNOevdDsqRESnNuSU")

