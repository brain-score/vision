from brainio.assemblies import BehavioralAssembly
from brainscore_vision import data_registry, stimulus_set_registry, load_stimulus_set
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3, load_stimulus_set_from_s3

BIBTEX = """"""

# circle_line:
stimulus_set_registry['Ferguson2024_circle_line'] = lambda: load_stimulus_set_from_s3(
    identifier='Ferguson2024_circle_line',
    bucket="brainio-brainscore",
    csv_sha1="bc351933e1f21eee9704985c1b8231be6955d816",
    zip_sha1="f5bb4e944cce202d85ec8f81dda823d7e68b84e6",
    csv_version_id="g5f5gSTlGM2ozd8LdVtRc6tIDuyFdLNW",
    zip_version_id="vi1_ZhwjbAwzZP7Q3f8MKFI0VUAbtHFI")

data_registry['Ferguson2024_circle_line'] = lambda: load_assembly_from_s3(
    identifier='Ferguson2024_circle_line',
    version_id="IdbRmABMnxs8f6n3zPB1SZ_XAsEdHgsz",
    sha1="eb6f05b53b0863ba180d3ab0e571a498b3050c2e",
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Ferguson2024_circle_line'),
)


# color:
stimulus_set_registry['Ferguson2024_color'] = lambda: load_stimulus_set_from_s3(
    identifier='Ferguson2024_color',
    bucket="brainio-brainscore",
    csv_sha1="bc351933e1f21eee9704985c1b8231be6955d816",
    zip_sha1="6ad04c58de8cc8c28b309572cc41c86470f0c322",
    csv_version_id="jK6ddF6hF_oWmE5ccm4MyprQwjfeti3",
    zip_version_id="Eku5pHE1CNJBrcaRstb8PCYCjOliHQmY")

data_registry['Ferguson2024_color'] = lambda: load_assembly_from_s3(
    identifier='Ferguson2024_color',
    version_id="VQo0U9ag8r7r9DREexvSlAD_Z326Iumr",
    sha1="5b5d67fa3189db9984006910d1954586e6a5a9f3",
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Ferguson2024_color'),
)


# convergence:
stimulus_set_registry['Ferguson2024_convergence'] = lambda: load_stimulus_set_from_s3(
    identifier='Ferguson2024_convergence',
    bucket="brainio-brainscore",
    csv_sha1="bc351933e1f21eee9704985c1b8231be6955d816",
    zip_sha1="d65bdf6654e08c9107a20028281ab5e38a0be981",
    csv_version_id="PcRg7fdEJI.Ce3wkd0v6sTC3jSw6xiiq",
    zip_version_id="4EJaobPVM8STsvMKE.hEcePXcLAjB5VG")

data_registry['Ferguson2024_convergence'] = lambda: load_assembly_from_s3(
    identifier='Ferguson2024_convergence',
    version_id="l.nJf3IXYqi5euv5xsqS_ip7Bs0ZpZLX",
    sha1="5165c4b0da30826b89c2c242826bb79a4417b9a5",
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Ferguson2024_convergence'),
)


# eighth:
stimulus_set_registry['Ferguson2024_eighth'] = lambda: load_stimulus_set_from_s3(
    identifier='Ferguson2024_eighth',
    bucket="brainio-brainscore",
    csv_sha1="bc351933e1f21eee9704985c1b8231be6955d816",
    zip_sha1="6ee1931b316fd4ccf6eeb16788aa42bb7a36aa41",
    csv_version_id="fVbTijqnoE61rcXNCopHMhXdrTavCIjS",
    zip_version_id="ifwG3beZ0ePhQGqbo6S7D9Jj1LPCvwsJ")

data_registry['Ferguson2024_eighth'] = lambda: load_assembly_from_s3(
    identifier='Ferguson2024_eighth',
    version_id="VklOC2KrpgLJpD1.kGj6Y5D4kLYSwr3s",
    sha1="984f9498c42b14cfae6c7272a8707df96fea7ee2",
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Ferguson2024_eighth'),
)


# gray_easy:
stimulus_set_registry['Ferguson2024_gray_easy'] = lambda: load_stimulus_set_from_s3(
    identifier='Ferguson2024_gray_easy',
    bucket="brainio-brainscore",
    csv_sha1="bc351933e1f21eee9704985c1b8231be6955d816",
    zip_sha1="da76bdebf46fe0eb853ea1b877274b1f26f56dfc",
    csv_version_id="WCw44X7HWimdn3qLi2D9DSOm5i2bLyrd",
    zip_version_id="UJp9O0lHnMMPMFmwY29g5v1cHXvF1XpH")

data_registry['Ferguson2024_gray_easy'] = lambda: load_assembly_from_s3(
    identifier='Ferguson2024_gray_easy',
    version_id="gaK.0mU6IVHjkI6MG9eE5Hz5Jt7_gxc6",
    sha1="7b09c2f1e8199e680167cfeb124c28dc68c804ab",
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Ferguson2024_gray_easy'),
)


# gray_hard:
stimulus_set_registry['Ferguson2024_gray_hard'] = lambda: load_stimulus_set_from_s3(
    identifier='Ferguson2024_gray_hard',
    bucket="brainio-brainscore",
    csv_sha1="bc351933e1f21eee9704985c1b8231be6955d816",
    zip_sha1="93f02c051f4d89fe059345c0af7ba6fc83b65b35",
    csv_version_id="bxWCJhmQw9RYxQx8qzGSltZCnSY4UTRI",
    zip_version_id="WLtKonQVU9Og0ZbmVRJKx4Zzxb4INsT8")

data_registry['Ferguson2024_gray_hard'] = lambda: load_assembly_from_s3(
    identifier='Ferguson2024_gray_hard',
    version_id="KSpwyfIqK6uovFojNd2_w08lKUJvfOWl",
    sha1="2fa35d41e73053ece6d1f0120ca4dc9bc4a9d4ae",
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Ferguson2024_gray_hard'),
)


# half:
stimulus_set_registry['Ferguson2024_half'] = lambda: load_stimulus_set_from_s3(
    identifier='Ferguson2024_half',
    bucket="brainio-brainscore",
    csv_sha1="bc351933e1f21eee9704985c1b8231be6955d816",
    zip_sha1="6461a1d19e031943d42e97e8b557a97d14b18c55",
    csv_version_id="WGrCxoue4oPYUKz81t30jcScz1dWs5Dv",
    zip_version_id="9pvmNpTauZECPkemXEfLV_wYA9JZT0Iw")

data_registry['Ferguson2024_half'] = lambda: load_assembly_from_s3(
    identifier='Ferguson2024_half',
    version_id="Z2Mpv3qH9foT9qggDIxWVHoEuKb6mC.a",
    sha1="b65e14c5d62fee715438a613e55fffa5e6f76c40",
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Ferguson2024_half'),
)


# juncture:
stimulus_set_registry['Ferguson2024_juncture'] = lambda: load_stimulus_set_from_s3(
    identifier='Ferguson2024_juncture',
    bucket="brainio-brainscore",
    csv_sha1="bc351933e1f21eee9704985c1b8231be6955d816",
    zip_sha1="832102f1eaa713fdfd03c512df2b8feea422c61d",
    csv_version_id="J3wrdsSM9LMlGFoC3ks5ees_t1sKjvKc",
    zip_version_id="zNu6swQFgclcS8.miCuDBk4AQ4G54KT2")

data_registry['Ferguson2024_juncture'] = lambda: load_assembly_from_s3(
    identifier='Ferguson2024_juncture',
    version_id="RstO_IgzeE2UbmHbMw6RN7vV8doFZKBq",
    sha1="b18148383ef2158aa795b3cff8a8e237e08b5070",
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Ferguson2024_juncture'),
)


# lle:
stimulus_set_registry['Ferguson2024_lle'] = lambda: load_stimulus_set_from_s3(
    identifier='Ferguson2024_lle',
    bucket="brainio-brainscore",
    csv_sha1="bc351933e1f21eee9704985c1b8231be6955d816",
    zip_sha1="27817e955da9e4747d2aeb8757b7f6492bc7767e",
    csv_version_id="y3epQUp6h7zH5h8251G8DlYzwtk6VYxW",
    zip_version_id="RCPB0_kLL0GF3xrR0Nl.c11uAL8yYF8c")

data_registry['Ferguson2024_lle'] = lambda: load_assembly_from_s3(
    identifier='Ferguson2024_lle',
    version_id="nXWjKJJyGtX.67m.M03oRw7ysfP76e4e",
    sha1="08e98305657cd374d9ea103df0fe06783a70344a",
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Ferguson2024_lle'),
)


# llh (assuming 'llh' is correct and not a placeholder):
stimulus_set_registry['Ferguson2024_llh'] = lambda: load_stimulus_set_from_s3(
    identifier='Ferguson2024_llh',
    bucket="brainio-brainscore",
    csv_sha1="bc351933e1f21eee9704985c1b8231be6955d816",
    zip_sha1="56cdf86ecd0b52349b29b2ab0be89daeed9b0eb6",
    csv_version_id="n3gooGN6lqWT5c.Qa3.kpUGUwogDtQUT",
    zip_version_id="3A2EgFZ9Un_uFl43xqXIudDHUHdF7le1")

data_registry['Ferguson2024_llh'] = lambda: load_assembly_from_s3(
    identifier='Ferguson2024_llh',
    version_id="prACZ4sm395A5yfJEYDG77MfGMJhXaXv",
    sha1="864d49c00e777f3d464c6c0c59fee087c1de9037",
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Ferguson2024_llh'),
)


# quarter:
stimulus_set_registry['Ferguson2024_quarter'] = lambda: load_stimulus_set_from_s3(
    identifier='Ferguson2024_quarter',
    bucket="brainio-brainscore",
    csv_sha1="bc351933e1f21eee9704985c1b8231be6955d816",
    zip_sha1="c16c5ecf1f38af0d02255a78a6c438074ec8d446",
    csv_version_id="lnk7H5WiGe3oB0i5PMTrXA_q058kZSDz",
    zip_version_id="frHF3zSr4cCUEs7bVYjjaM3c0WQgwiA9")

data_registry['Ferguson2024_quarter'] = lambda: load_assembly_from_s3(
    identifier='Ferguson2024_quarter',
    version_id="_q5R_GoANyjQ8DWQsY.2HBtzW8DoSGpm",
    sha1="921b3b51208cdd5f163eca288ea83be47a2b482f",
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Ferguson2024_quarter'),
)


# round_f:
stimulus_set_registry['Ferguson2024_round_f'] = lambda: load_stimulus_set_from_s3(
    identifier='Ferguson2024_round_f',
    bucket="brainio-brainscore",
    csv_sha1="bc351933e1f21eee9704985c1b8231be6955d816",
    zip_sha1="7f456e390cb93805187266d05756beb9cf225e1d",
    csv_version_id="jheoU.xYIbSk2hFPhCue2MGmXckqMooe",
    zip_version_id="FtLIcpUQzHA_jPdRl_6iSJoqXZKDCeJn")

data_registry['Ferguson2024_round_f'] = lambda: load_assembly_from_s3(
    identifier='Ferguson2024_round_f',
    version_id="0E7lr44ha3rV7xpnWnE1MpDV79seDxCe",
    sha1="acb19ac865b45199a58609db31d3e885ff272fd4",
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Ferguson2024_round_f'),
)


# round_v:
stimulus_set_registry['Ferguson2024_round_v'] = lambda: load_stimulus_set_from_s3(
    identifier='Ferguson2024_round_v',
    bucket="brainio-brainscore",
    csv_sha1="bc351933e1f21eee9704985c1b8231be6955d816",
    zip_sha1="cebb84d2363c1539368e3e1b1bfd83305ad9ae13",
    csv_version_id="6_N_s3Cz_g32jncN0bWoDCh.1pWdKCv2",
    zip_version_id="r8e50KhAeIc0mKz1qE_xN2z4rMYaNsJ_")

data_registry['Ferguson2024_round_v'] = lambda: load_assembly_from_s3(
    identifier='Ferguson2024_round_v',
    version_id="VS.8.ocCdNugRJNU6ha2Wm3K1lK4vK5k",
    sha1="ce0361c4386dc7b8866d78023044b3009c84aa4b",
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Ferguson2024_round_v'),
)


# tilted_line:
stimulus_set_registry['Ferguson2024_tilted_line'] = lambda: load_stimulus_set_from_s3(
    identifier='Ferguson2024_tilted_line',
    bucket="brainio-brainscore",
    csv_sha1="bc351933e1f21eee9704985c1b8231be6955d816",
    zip_sha1="bb3d7bcb60ba586c8552266839187a59c2b3138f",
    csv_version_id="7mcYPI8IYpS9Rz7pLm6QOxBne29.WcWp",
    zip_version_id="5dvzTilCQkDUHG85qCCQZOhB6ZLpfU5_")

data_registry['Ferguson2024_tilted_line'] = lambda: load_assembly_from_s3(
    identifier='Ferguson2024_tilted_line',
    version_id="ae4Dbo9JU_PDwTqKGD1G4DQNrdh2cVE2",
    sha1="1806034da0c25e8625255eb94dc0a05c7e9cda1f",
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Ferguson2024_tilted_line'),
)