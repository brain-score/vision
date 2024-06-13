from brainio.assemblies import PropertyAssembly

from brainscore_vision import data_registry, stimulus_set_registry, load_stimulus_set
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3, load_stimulus_set_from_s3


BIBTEX = """@article{malania2007,
            author = {Malania, Maka and Herzog, Michael H. and Westheimer, Gerald},
            title = "{Grouping of contextual elements that affect vernier thresholds}",
            journal = {Journal of Vision},
            volume = {7},
            number = {2},
            pages = {1-1},
            year = {2007},
            issn = {1534-7362},
            doi = {10.1167/7.2.1},
            url = {https://doi.org/10.1167/7.2.1}
        }"""


data_registry['Malania2007_equal-2'] = lambda: load_assembly_from_s3(
    identifier='Malania2007_equal-2',
    version_id="yFXK8xjGjEmuYTSfS58rGS_ah3.NGg0X",
    sha1="277b2fbffed00e16b6a69b488f73eeda5abaaf10",
    bucket="brainio-brainscore",
    cls=PropertyAssembly,
    stimulus_set_loader=None,
)
data_registry['Malania2007_equal-16'] = lambda: load_assembly_from_s3(
    identifier='Malania2007_equal-16',
    version_id="SRZ7bs.Ek59GkeS084Pvdy38uTzFs4yw",
    sha1="ef49506238e8d2554918b113fbc60c133077186e",
    bucket="brainio-brainscore",
    cls=PropertyAssembly,
    stimulus_set_loader=None,
)
data_registry['Malania2007_long-2'] = lambda: load_assembly_from_s3(
    identifier='Malania2007_long-2',
    version_id="2c1lWuXthb3rymB3seTQX1jVqiKUTn1f",
    sha1="9076a5b693948c4992b6c8e753f04a7acd2014a1",
    bucket="brainio-brainscore",
    cls=PropertyAssembly,
    stimulus_set_loader=None,
)
data_registry['Malania2007_long-16'] = lambda: load_assembly_from_s3(
    identifier='Malania2007_long-16',
    version_id="qshNxhxjgusWyWiXnbfFN6gqjLgRh8fO",
    sha1="3106cf1f2fa9e66617ebf231df05d29077fc478f",
    bucket="brainio-brainscore",
    cls=PropertyAssembly,
    stimulus_set_loader=None,
)
data_registry['Malania2007_short-2'] = lambda: load_assembly_from_s3(
    identifier='Malania2007_short-2',
    version_id="8CQ9MupuljAgkkKUXs3hiOliHg8xoDxb",
    sha1="85fb65ad76de48033c704b9c5689771e1ea0457d",
    bucket="brainio-brainscore",
    cls=PropertyAssembly,
    stimulus_set_loader=None,
)
data_registry['Malania2007_short-4'] = lambda: load_assembly_from_s3(
    identifier='Malania2007_short-4',
    version_id=".ZUO0upSfQrWLPgd4oGwAaCbN4bz6S6H",
    sha1="75506be9a26ec38a223e41510f1a8cb32d5b0bc9",
    bucket="brainio-brainscore",
    cls=PropertyAssembly,
    stimulus_set_loader=None,
)
data_registry['Malania2007_short-6'] = lambda: load_assembly_from_s3(
    identifier='Malania2007_short-6',
    version_id="q4FugpNGkT_FQP..qIVzye83hAQR2xfS",
    sha1="2901be6b352e67550da040d79d744819365b8626",
    bucket="brainio-brainscore",
    cls=PropertyAssembly,
    stimulus_set_loader=None,
)
data_registry['Malania2007_short-8'] = lambda: load_assembly_from_s3(
    identifier='Malania2007_short-8',
    version_id="4_lcRl_I7Mp0RHxcfqZ9tkAZjVh.5oMU",
    sha1="6daf47b086cb969d75222e320f49453ed8437885",
    bucket="brainio-brainscore",
    cls=PropertyAssembly,
    stimulus_set_loader=None,
)
data_registry['Malania2007_short-16'] = lambda: load_assembly_from_s3(
    identifier='Malania2007_short-16',
    version_id="fFqEIyIC9CHzqTEmv0MitjCgpeMX5pxJ",
    sha1="8ae0898caad718b747f85fce5888416affc3a569",
    bucket="brainio-brainscore",
    cls=PropertyAssembly,
    stimulus_set_loader=None,
)
data_registry['Malania2007_vernier-only'] = lambda: load_assembly_from_s3(
    identifier='Malania2007_vernier-only',
    version_id="JLWf2pIR_UadQHqwtegJkC6XzWdbSNGi",
    sha1="1cf83e8b6141f8b0d67ea46994f342325f62001f",
    bucket="brainio-brainscore",
    cls=PropertyAssembly,
    stimulus_set_loader=None,
)


stimulus_set_registry['Malania2007_equal-2'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_equal-2',
    bucket="brainio-brainscore",
    csv_sha1="77e94b9b5122a83ebbaffb4a06fcab68ef652751",
    zip_sha1="99826d459f6920dafab72eed69eb2a90492ce796",
    csv_version_id="MlRpSz.4.jvVRFAZl8tGEum1P0Q0GtyS",
    zip_version_id="vHbAM_FjTbjp5U12BkAelJu4KW6PLYFn"
)
stimulus_set_registry['Malania2007_equal-2_fit'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_equal-2_fit',
    bucket="brainio-brainscore",
    csv_sha1="bafdfc855c164d3e5443d67dcf9eb7762443f964",
    zip_sha1="e52fec1a79ac8837e331b180c2a8a140840d6666",
    csv_version_id="PIXEW.2vHvjIBP0Q2KHIpnxns7t9o8Cf",
    zip_version_id="h7pp84CYFGLKlPhveD0L5ogePqisk_I7"
)
stimulus_set_registry['Malania2007_equal-16'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_equal-16',
    bucket="brainio-brainscore",
    csv_sha1="5fedcff56c302339c3451ae2edbcb846c39c3189",
    zip_sha1="b30dc2dc90e4f3d88775622e558db963765f38e0",
    csv_version_id="VmRGiQkhPALDwq74NpE2VpTiKTGn.30T",
    zip_version_id="c.DOlVULXZingRJ9gVY_NbZwRrj_xs_i"
)
stimulus_set_registry['Malania2007_equal-16_fit'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_equal-16_fit',
    bucket="brainio-brainscore",
    csv_sha1="3de3e5de19a638767a01ba68cb690dc746c29a77",
    zip_sha1="1728920c5ea4fb7b3a3cf3c076165aca65c8b751",
    csv_version_id="joAq8JBC_7axZDfLNFgoXFhTCLU_KKr_",
    zip_version_id="77JRwdldaHDr6TLW1NnB5HucIrkUCVg."
)
stimulus_set_registry['Malania2007_long-2'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_long-2',
    bucket="brainio-brainscore",
    csv_sha1="ba65316a63dc688d8dfb410219a28fd02850b991",
    zip_sha1="7fd431fbbd4a4dc0cd271624d3297c19a28a70b5",
    csv_version_id="_0fqObn6k5KvXurHMsuD4IqtrqbNskyo",
    zip_version_id="foL92ndVAAAETzMYHdmMtwIwKxXYhAB."
)
stimulus_set_registry['Malania2007_long-2_fit'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_long-2_fit',
    bucket="brainio-brainscore",
    csv_sha1="b91dd9261c1d47bdd37f9b60eb8066b7b719709f",
    zip_sha1="5be3e1cd57b59081103715b5d318505166e0045e",
    csv_version_id="mATh8lcVisdsDnPnU6ACE23iBPfpkLZA",
    zip_version_id="6nEviShTyCYQKrmxyjDyNov9Skc77eXT"
)
stimulus_set_registry['Malania2007_long-16'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_long-16',
    bucket="brainio-brainscore",
    csv_sha1="1f1b03319b81698ba5e7db389dcd4248f94e45ca",
    zip_sha1="97c70462a28905b58058c687880188d634d357f0",
    csv_version_id="4RtywQ40hfQA4N80g8lxEScAmMXFRg7E",
    zip_version_id="lJy2QosABzHtiA6BJaE4OqCn1w1Jhz2k"
)
stimulus_set_registry['Malania2007_long-16_fit'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_long-16_fit',
    bucket="brainio-brainscore",
    csv_sha1="d80a02c75b9908301c3c8dc9f7116fecf8e060ec",
    zip_sha1="d8819b94d3f502d7a382c8a0db0a34627132e5e2",
    csv_version_id="gOxY6tjnT7LO.FDeL1xkRmowl5wYeAia",
    zip_version_id="71UAPTnZscIuqdx2dhuW9V0O0DO_TgTM"
)
stimulus_set_registry['Malania2007_short-2'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_short-2',
    bucket="brainio-brainscore",
    csv_sha1="bf0252056d2084e855646f624700ab03c19cfc3d",
    zip_sha1="eee1270feb7443e7e315d8feb7fb0a6b6908f554",
    csv_version_id="zcJqM.ZPwJyiMRWa3RBdvv401yPnLQAp",
    zip_version_id="C8WZzAAQ0JGHAAKii4JpvlRhcUOhgSj."
)
stimulus_set_registry['Malania2007_short-2_fit'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_short-2_fit',
    bucket="brainio-brainscore",
    csv_sha1="73127d279a2cd254ae4f07b0053580851e84b00c",
    zip_sha1="918736349d714a4f784c29bf7e7d218b103e128d",
    csv_version_id="iwGRp3_ktAHfJ6r7ktSK9gsthDjKek70",
    zip_version_id="6RpplJ9UVXTlvhmFSXla0Qa20b44m8Ds"
)
stimulus_set_registry['Malania2007_short-4'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_short-4',
    bucket="brainio-brainscore",
    csv_sha1="816326d89d358f6592bd1f789e5c8d429fbca1cd",
    zip_sha1="ff57d976ef75ede9148a4097e90d6cf6c8054d34",
    csv_version_id="Waikk.bktXIncCUtCIAyB2EqynGk.H.F",
    zip_version_id="rl_muxI4UEpwXVaXuhsqroG..COGILvR"
)
stimulus_set_registry['Malania2007_short-4_fit'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_short-4_fit',
    bucket="brainio-brainscore",
    csv_sha1="3512cfd029f4e4299bc41ede519e691d80cfc3d5",
    zip_sha1="301386408dd1fb8556881f9a171be2d43dbfec6e",
    csv_version_id="UhisdJqiEmkQ_4zsUtAmaxtle2kMZdcD",
    zip_version_id="xt_v0xgCB8YUptyPB0yZFHIUcel5MF_x"
)
stimulus_set_registry['Malania2007_short-6'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_short-6',
    bucket="brainio-brainscore",
    csv_sha1="3d5dd9b48a56ba0c31de94b6221b97df962b6f8a",
    zip_sha1="120d90a143d1577d4745c3f69291d0db6c7e512e",
    csv_version_id="GwGHPJkMDdg8N_.boyj8qJ3ChsEx4w._",
    zip_version_id="gIN1O4yz.THvK0Ifm5M3AI58ZACE1QFh"
)
stimulus_set_registry['Malania2007_short-6_fit'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_short-6_fit',
    bucket="brainio-brainscore",
    csv_sha1="27a5be4fca190837fc5b75ed2cdbbffbf6b41338",
    zip_sha1="c88e05c6cadec88a2c9475b0735323a2b049bd75",
    csv_version_id="oMlj7wV85s00hJFE84ym0AJHLCfYHVA6",
    zip_version_id="oS.KrBTlcYAgr_lWyA_bIjVc2js_VeUe"
)
stimulus_set_registry['Malania2007_short-8'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_short-8',
    bucket="brainio-brainscore",
    csv_sha1="8fc35f607196b4c0cdcebd8102d17e3a637e5988",
    zip_sha1="a9215ed0cb0f0333582dda65f6afd7015c506ba5",
    csv_version_id="gzys8s7j7euMEl7JJpqBFLFHMpFjwbA7",
    zip_version_id="3fYb4Iruh3lRKUwC1APqFH4CNbE5DEuk"
)
stimulus_set_registry['Malania2007_short-8_fit'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_short-8_fit',
    bucket="brainio-brainscore",
    csv_sha1="aa4133a9fe19a3c9004a9cb5e6eb5a72564e4883",
    zip_sha1="beb9f068794708e41750202b78c438538a40a8fb",
    csv_version_id="7N1Z.uiagqBknJUSBQ4mVfHKWgocM5aA",
    zip_version_id="kcEOPOkvWymO0wX5j_QKxcNPl9sZsjFd"
)
stimulus_set_registry['Malania2007_short-16'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_short-16',
    bucket="brainio-brainscore",
    csv_sha1="addd260c9959f2f315db03c0a39c6c1b01fef685",
    zip_sha1="cba4c2866ec692fb808471df7c2fed446d9fb3fe",
    csv_version_id="Peu7WU5vanLoZNOFIAbuPzZNPDRgbCSX",
    zip_version_id="wFkJkZMC8Fs_HfPJy32CMKcHJWeQIUDB"
)
stimulus_set_registry['Malania2007_short-16_fit'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_short-16_fit',
    bucket="brainio-brainscore",
    csv_sha1="9b340fe242117482f6992f48a805297215ba9924",
    zip_sha1="4a90d511a3ceb3307a672177a3ad6b76521e65e5",
    csv_version_id="sYBPEmXDgbWipuepciLirlorQE3L8BLc",
    zip_version_id="pYvOkrLxadkQ67K3__wmciNwaCW.hyyN"
)
stimulus_set_registry['Malania2007_vernier-only'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_vernier-only',
    bucket="brainio-brainscore",
    csv_sha1="b2cb0f2ed32426b739f90187ae24ad4adf84110d",
    zip_sha1="0e177aea523adc320070196fbb777af4cdba2144",
    csv_version_id="c8wpZpqoMqdATlqdoq3srPUi_8fYg6a.",
    zip_version_id="28lHgxERhw32Ux6IBCxWWTtRwIaRrwo6"
)
