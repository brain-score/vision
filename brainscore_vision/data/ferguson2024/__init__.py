from brainio.assemblies import BehavioralAssembly
from brainscore_vision import data_registry, stimulus_set_registry, load_stimulus_set
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3, load_stimulus_set_from_s3

BIBTEX = """TBD"""

# circle_line:
stimulus_set_registry['Ferguson2024_circle_line'] = lambda: load_stimulus_set_from_s3(
    identifier='Ferguson2024_circle_line',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="fc59d23ccfb41b4f98cf02865fc335439d2ad222",
    zip_sha1="1f0065910b01a1a0e12611fe61252eafb9c534c3",
    csv_version_id="null",
    zip_version_id="null")

data_registry['Ferguson2024_circle_line'] = lambda: load_assembly_from_s3(
    identifier='Ferguson2024_circle_line',
    version_id="null",
    sha1="586da7b1c7cb5a60fe72bc148513e3159a27b134",
    bucket="brainscore-storage/brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Ferguson2024_circle_line'),
)


# color:
stimulus_set_registry['Ferguson2024_color'] = lambda: load_stimulus_set_from_s3(
    identifier='Ferguson2024_color',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="bc351933e1f21eee9704985c1b8231be6955d816",
    zip_sha1="6ad04c58de8cc8c28b309572cc41c86470f0c322",
    csv_version_id="null",
    zip_version_id="null")

data_registry['Ferguson2024_color'] = lambda: load_assembly_from_s3(
    identifier='Ferguson2024_color',
    version_id="null",
    sha1="5b5d67fa3189db9984006910d1954586e6a5a9f3",
    bucket="brainscore-storage/brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Ferguson2024_color'),
)


# convergence:
stimulus_set_registry['Ferguson2024_convergence'] = lambda: load_stimulus_set_from_s3(
    identifier='Ferguson2024_convergence',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="bc351933e1f21eee9704985c1b8231be6955d816",
    zip_sha1="d65bdf6654e08c9107a20028281ab5e38a0be981",
    csv_version_id="null",
    zip_version_id="null")

data_registry['Ferguson2024_convergence'] = lambda: load_assembly_from_s3(
    identifier='Ferguson2024_convergence',
    version_id="null",
    sha1="5165c4b0da30826b89c2c242826bb79a4417b9a5",
    bucket="brainscore-storage/brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Ferguson2024_convergence'),
)


# eighth:
stimulus_set_registry['Ferguson2024_eighth'] = lambda: load_stimulus_set_from_s3(
    identifier='Ferguson2024_eighth',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="bc351933e1f21eee9704985c1b8231be6955d816",
    zip_sha1="6ee1931b316fd4ccf6eeb16788aa42bb7a36aa41",
    csv_version_id="null",
    zip_version_id="null")

data_registry['Ferguson2024_eighth'] = lambda: load_assembly_from_s3(
    identifier='Ferguson2024_eighth',
    version_id="null",
    sha1="984f9498c42b14cfae6c7272a8707df96fea7ee2",
    bucket="brainscore-storage/brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Ferguson2024_eighth'),
)


# gray_easy:
stimulus_set_registry['Ferguson2024_gray_easy'] = lambda: load_stimulus_set_from_s3(
    identifier='Ferguson2024_gray_easy',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="bc351933e1f21eee9704985c1b8231be6955d816",
    zip_sha1="da76bdebf46fe0eb853ea1b877274b1f26f56dfc",
    csv_version_id="null",
    zip_version_id="null")

data_registry['Ferguson2024_gray_easy'] = lambda: load_assembly_from_s3(
    identifier='Ferguson2024_gray_easy',
    version_id="null",
    sha1="7b09c2f1e8199e680167cfeb124c28dc68c804ab",
    bucket="brainscore-storage/brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Ferguson2024_gray_easy'),
)


# gray_hard:
stimulus_set_registry['Ferguson2024_gray_hard'] = lambda: load_stimulus_set_from_s3(
    identifier='Ferguson2024_gray_hard',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="bc351933e1f21eee9704985c1b8231be6955d816",
    zip_sha1="93f02c051f4d89fe059345c0af7ba6fc83b65b35",
    csv_version_id="null",
    zip_version_id="null")

data_registry['Ferguson2024_gray_hard'] = lambda: load_assembly_from_s3(
    identifier='Ferguson2024_gray_hard',
    version_id="null",
    sha1="2fa35d41e73053ece6d1f0120ca4dc9bc4a9d4ae",
    bucket="brainscore-storage/brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Ferguson2024_gray_hard'),
)


# half:
stimulus_set_registry['Ferguson2024_half'] = lambda: load_stimulus_set_from_s3(
    identifier='Ferguson2024_half',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="bc351933e1f21eee9704985c1b8231be6955d816",
    zip_sha1="6461a1d19e031943d42e97e8b557a97d14b18c55",
    csv_version_id="null",
    zip_version_id="null")

data_registry['Ferguson2024_half'] = lambda: load_assembly_from_s3(
    identifier='Ferguson2024_half',
    version_id="null",
    sha1="b65e14c5d62fee715438a613e55fffa5e6f76c40",
    bucket="brainscore-storage/brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Ferguson2024_half'),
)


# juncture:
stimulus_set_registry['Ferguson2024_juncture'] = lambda: load_stimulus_set_from_s3(
    identifier='Ferguson2024_juncture',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="bc351933e1f21eee9704985c1b8231be6955d816",
    zip_sha1="832102f1eaa713fdfd03c512df2b8feea422c61d",
    csv_version_id="null",
    zip_version_id="null")

data_registry['Ferguson2024_juncture'] = lambda: load_assembly_from_s3(
    identifier='Ferguson2024_juncture',
    version_id="null",
    sha1="b18148383ef2158aa795b3cff8a8e237e08b5070",
    bucket="brainscore-storage/brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Ferguson2024_juncture'),
)


# lle:
stimulus_set_registry['Ferguson2024_lle'] = lambda: load_stimulus_set_from_s3(
    identifier='Ferguson2024_lle',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="bc351933e1f21eee9704985c1b8231be6955d816",
    zip_sha1="27817e955da9e4747d2aeb8757b7f6492bc7767e",
    csv_version_id="null",
    zip_version_id="null")

data_registry['Ferguson2024_lle'] = lambda: load_assembly_from_s3(
    identifier='Ferguson2024_lle',
    version_id="null",
    sha1="08e98305657cd374d9ea103df0fe06783a70344a",
    bucket="brainscore-storage/brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Ferguson2024_lle'),
)


# llh (assuming 'llh' is correct and not a placeholder):
stimulus_set_registry['Ferguson2024_llh'] = lambda: load_stimulus_set_from_s3(
    identifier='Ferguson2024_llh',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="bc351933e1f21eee9704985c1b8231be6955d816",
    zip_sha1="56cdf86ecd0b52349b29b2ab0be89daeed9b0eb6",
    csv_version_id="null",
    zip_version_id="null")

data_registry['Ferguson2024_llh'] = lambda: load_assembly_from_s3(
    identifier='Ferguson2024_llh',
    version_id="null",
    sha1="864d49c00e777f3d464c6c0c59fee087c1de9037",
    bucket="brainscore-storage/brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Ferguson2024_llh'),
)


# quarter:
stimulus_set_registry['Ferguson2024_quarter'] = lambda: load_stimulus_set_from_s3(
    identifier='Ferguson2024_quarter',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="bc351933e1f21eee9704985c1b8231be6955d816",
    zip_sha1="c16c5ecf1f38af0d02255a78a6c438074ec8d446",
    csv_version_id="null",
    zip_version_id="null")

data_registry['Ferguson2024_quarter'] = lambda: load_assembly_from_s3(
    identifier='Ferguson2024_quarter',
    version_id="null",
    sha1="921b3b51208cdd5f163eca288ea83be47a2b482f",
    bucket="brainscore-storage/brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Ferguson2024_quarter'),
)


# round_f:
stimulus_set_registry['Ferguson2024_round_f'] = lambda: load_stimulus_set_from_s3(
    identifier='Ferguson2024_round_f',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="bc351933e1f21eee9704985c1b8231be6955d816",
    zip_sha1="7f456e390cb93805187266d05756beb9cf225e1d",
    csv_version_id="null",
    zip_version_id="null")

data_registry['Ferguson2024_round_f'] = lambda: load_assembly_from_s3(
    identifier='Ferguson2024_round_f',
    version_id="null",
    sha1="acb19ac865b45199a58609db31d3e885ff272fd4",
    bucket="brainscore-storage/brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Ferguson2024_round_f'),
)


# round_v:
stimulus_set_registry['Ferguson2024_round_v'] = lambda: load_stimulus_set_from_s3(
    identifier='Ferguson2024_round_v',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="bc351933e1f21eee9704985c1b8231be6955d816",
    zip_sha1="cebb84d2363c1539368e3e1b1bfd83305ad9ae13",
    csv_version_id="null",
    zip_version_id="null")

data_registry['Ferguson2024_round_v'] = lambda: load_assembly_from_s3(
    identifier='Ferguson2024_round_v',
    version_id="null",
    sha1="ce0361c4386dc7b8866d78023044b3009c84aa4b",
    bucket="brainscore-storage/brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Ferguson2024_round_v'),
)


# tilted_line:
stimulus_set_registry['Ferguson2024_tilted_line'] = lambda: load_stimulus_set_from_s3(
    identifier='Ferguson2024_tilted_line',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="bc351933e1f21eee9704985c1b8231be6955d816",
    zip_sha1="bb3d7bcb60ba586c8552266839187a59c2b3138f",
    csv_version_id="null",
    zip_version_id="null")

data_registry['Ferguson2024_tilted_line'] = lambda: load_assembly_from_s3(
    identifier='Ferguson2024_tilted_line',
    version_id="null",
    sha1="1806034da0c25e8625255eb94dc0a05c7e9cda1f",
    bucket="brainscore-storage/brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Ferguson2024_tilted_line'),
)


'''
Pretraining Stimuli:
'''

# circle_line
stimulus_set_registry['Ferguson2024_circle_line_training_stimuli'] = lambda: load_stimulus_set_from_s3(
    identifier='Ferguson2024_circle_line_training_stimuli',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="098eb5999e9c4b723abc35ade862d2dc45899230",
    zip_sha1="ba9088601d8c79ea5ff3d513e1a76b1232491918",
    csv_version_id="null",
    zip_version_id="null")

# color
stimulus_set_registry['Ferguson2024_color_training_stimuli'] = lambda: load_stimulus_set_from_s3(
    identifier='Ferguson2024_color_training_stimuli',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="098eb5999e9c4b723abc35ade862d2dc45899230",
    zip_sha1="dcd5044c73e7523afc94f91543adb444a437f911",
    csv_version_id="null",
    zip_version_id="null")

# convergence
stimulus_set_registry['Ferguson2024_convergence_training_stimuli'] = lambda: load_stimulus_set_from_s3(
    identifier='Ferguson2024_convergence_training_stimuli',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="098eb5999e9c4b723abc35ade862d2dc45899230",
    zip_sha1="00eb401ddfc075a4bb448ec36b8a1c8f3ec1d6e4",
    csv_version_id="null",
    zip_version_id="null")

# eighth
stimulus_set_registry['Ferguson2024_eighth_training_stimuli'] = lambda: load_stimulus_set_from_s3(
    identifier='Ferguson2024_eighth_training_stimuli',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="098eb5999e9c4b723abc35ade862d2dc45899230",
    zip_sha1="26edaec1d0dd14343a117340283e091a2245f3aa",
    csv_version_id="null",
    zip_version_id="null")

# gray_easy
stimulus_set_registry['Ferguson2024_gray_easy_training_stimuli'] = lambda: load_stimulus_set_from_s3(
    identifier='Ferguson2024_gray_easy_training_stimuli',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="098eb5999e9c4b723abc35ade862d2dc45899230",
    zip_sha1="18211af83c680d5c916ec15b57b5b871494d6b28",
    csv_version_id="null",
    zip_version_id="null")

# gray hard
stimulus_set_registry['Ferguson2024_gray_hard_training_stimuli'] = lambda: load_stimulus_set_from_s3(
    identifier='Ferguson2024_gray_hard_training_stimuli',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="098eb5999e9c4b723abc35ade862d2dc45899230",
    zip_sha1="a54d84dbb548706bbfeb239113a1d92205dc3f67",
    csv_version_id="null",
    zip_version_id="null")

# half
stimulus_set_registry['Ferguson2024_half_training_stimuli'] = lambda: load_stimulus_set_from_s3(
    identifier='Ferguson2024_half_training_stimuli',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="098eb5999e9c4b723abc35ade862d2dc45899230",
    zip_sha1="0db665619307d5c532a3ccd4311611e5a3830a10",
    csv_version_id="null",
    zip_version_id="null")

# juncture
stimulus_set_registry['Ferguson2024_juncture_training_stimuli'] = lambda: load_stimulus_set_from_s3(
    identifier='Ferguson2024_juncture_training_stimuli',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="098eb5999e9c4b723abc35ade862d2dc45899230",
    zip_sha1="eb99fd862bec5e61900d037b6b38abf2a278c9f0",
    csv_version_id="null",
    zip_version_id="null")

# lle
stimulus_set_registry['Ferguson2024_lle_training_stimuli'] = lambda: load_stimulus_set_from_s3(
    identifier='Ferguson2024_lle_training_stimuli',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="098eb5999e9c4b723abc35ade862d2dc45899230",
    zip_sha1="a1d19d0b77f0eb17ae886a1b7ccc649c5e51d84e",
    csv_version_id="null",
    zip_version_id="null")

# llh
stimulus_set_registry['Ferguson2024_llh_training_stimuli'] = lambda: load_stimulus_set_from_s3(
    identifier='Ferguson2024_llh_training_stimuli',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="098eb5999e9c4b723abc35ade862d2dc45899230",
    zip_sha1="1550f9f71e6930caa15b96aaf811aa97d48d7267",
    csv_version_id="null",
    zip_version_id="null")

# quarter
stimulus_set_registry['Ferguson2024_quarter_training_stimuli'] = lambda: load_stimulus_set_from_s3(
    identifier='Ferguson2024_quarter_training_stimuli',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="098eb5999e9c4b723abc35ade862d2dc45899230",
    zip_sha1="22669e4a94718b3cbde3f5b2a493044bc091257e",
    csv_version_id="null",
    zip_version_id="null")

# round_f
stimulus_set_registry['Ferguson2024_round_f_training_stimuli'] = lambda: load_stimulus_set_from_s3(
    identifier='Ferguson2024_round_f_training_stimuli',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="098eb5999e9c4b723abc35ade862d2dc45899230",
    zip_sha1="e33855c899f78a115cf377a228e07d87baa554b7",
    csv_version_id="null",
    zip_version_id="null")

# round_v
stimulus_set_registry['Ferguson2024_round_v_training_stimuli'] = lambda: load_stimulus_set_from_s3(
    identifier='Ferguson2024_round_v_training_stimuli',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="098eb5999e9c4b723abc35ade862d2dc45899230",
    zip_sha1="b1555f8a140a12e01a87a2f4e452d5863be43a5b",
    csv_version_id="null",
    zip_version_id="null")

# tilted_line
stimulus_set_registry['Ferguson2024_tilted_line_training_stimuli'] = lambda: load_stimulus_set_from_s3(
    identifier='Ferguson2024_tilted_line_training_stimuli',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="098eb5999e9c4b723abc35ade862d2dc45899230",
    zip_sha1="e92533d8aded07ed90ef25650d0cf07c3a458be7",
    csv_version_id="null",
    zip_version_id="null")
