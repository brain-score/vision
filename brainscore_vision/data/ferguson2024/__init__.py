from brainio.assemblies import BehavioralAssembly
from brainscore_vision import data_registry, stimulus_set_registry, load_stimulus_set
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3, load_stimulus_set_from_s3

BIBTEX = """TBD"""

# circle_line:
stimulus_set_registry['Ferguson2024_circle_line'] = lambda: load_stimulus_set_from_s3(
    identifier='Ferguson2024_circle_line',
    bucket="brainio-brainscore",
    csv_sha1="9c97c155fd6039a95978be89eb604c6894c5fa16",
    zip_sha1="d166f1d3dc3d00c4f51a489e6fcf96dbbe778d2c",
    csv_version_id="1ZaFYwHPBkDOrgdrwGHYqMfJJBCWei21",
    zip_version_id="X62ivk_UuHgh7Sd7VwDxgnB8tWPK06gt")

data_registry['Ferguson2024_circle_line'] = lambda: load_assembly_from_s3(
    identifier='Ferguson2024_circle_line',
    version_id="RDjCFAFt_J5mMwFBN9Ifo0OyNPKlToqf",
    sha1="258862d82467614e45cc1e488a5ac909eb6e122d",
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Ferguson2024_circle_line'),
)


# color:
stimulus_set_registry['Ferguson2024_color'] = lambda: load_stimulus_set_from_s3(
    identifier='Ferguson2024_color',
    bucket="brainio-brainscore",
    csv_sha1="your_sha1_here",
    zip_sha1="your_zip_sha1_here",
    csv_version_id="your_csv_version_id_here",
    zip_version_id="your_zip_version_id_here")

data_registry['Ferguson2024_color'] = lambda: load_assembly_from_s3(
    identifier='Ferguson2024_color',
    version_id="your_version_id_here",
    sha1="your_sha1_here",
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Ferguson2024_color'),
)


# convergence:
stimulus_set_registry['Ferguson2024_convergence'] = lambda: load_stimulus_set_from_s3(
    identifier='Ferguson2024_convergence',
    bucket="brainio-brainscore",
    csv_sha1="your_sha1_here",
    zip_sha1="your_zip_sha1_here",
    csv_version_id="your_csv_version_id_here",
    zip_version_id="your_zip_version_id_here")

data_registry['Ferguson2024_convergence'] = lambda: load_assembly_from_s3(
    identifier='Ferguson2024_convergence',
    version_id="your_version_id_here",
    sha1="your_sha1_here",
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Ferguson2024_convergence'),
)


# eighth:
stimulus_set_registry['Ferguson2024_eighth'] = lambda: load_stimulus_set_from_s3(
    identifier='Ferguson2024_eighth',
    bucket="brainio-brainscore",
    csv_sha1="your_sha1_here",
    zip_sha1="your_zip_sha1_here",
    csv_version_id="your_csv_version_id_here",
    zip_version_id="your_zip_version_id_here")

data_registry['Ferguson2024_eighth'] = lambda: load_assembly_from_s3(
    identifier='Ferguson2024_eighth',
    version_id="your_version_id_here",
    sha1="your_sha1_here",
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Ferguson2024_eighth'),
)


# gray_easy:
stimulus_set_registry['Ferguson2024_gray_easy'] = lambda: load_stimulus_set_from_s3(
    identifier='Ferguson2024_gray_easy',
    bucket="brainio-brainscore",
    csv_sha1="your_sha1_here",
    zip_sha1="your_zip_sha1_here",
    csv_version_id="your_csv_version_id_here",
    zip_version_id="your_zip_version_id_here")

data_registry['Ferguson2024_gray_easy'] = lambda: load_assembly_from_s3(
    identifier='Ferguson2024_gray_easy',
    version_id="your_version_id_here",
    sha1="your_sha1_here",
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Ferguson2024_gray_easy'),
)


# gray_hard:
stimulus_set_registry['Ferguson2024_gray_hard'] = lambda: load_stimulus_set_from_s3(
    identifier='Ferguson2024_gray_hard',
    bucket="brainio-brainscore",
    csv_sha1="your_sha1_here",
    zip_sha1="your_zip_sha1_here",
    csv_version_id="your_csv_version_id_here",
    zip_version_id="your_zip_version_id_here")

data_registry['Ferguson2024_gray_hard'] = lambda: load_assembly_from_s3(
    identifier='Ferguson2024_gray_hard',
    version_id="your_version_id_here",
    sha1="your_sha1_here",
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Ferguson2024_gray_hard'),
)


# half:
stimulus_set_registry['Ferguson2024_half'] = lambda: load_stimulus_set_from_s3(
    identifier='Ferguson2024_half',
    bucket="brainio-brainscore",
    csv_sha1="your_sha1_here",
    zip_sha1="your_zip_sha1_here",
    csv_version_id="your_csv_version_id_here",
    zip_version_id="your_zip_version_id_here")

data_registry['Ferguson2024_half'] = lambda: load_assembly_from_s3(
    identifier='Ferguson2024_half',
    version_id="your_version_id_here",
    sha1="your_sha1_here",
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Ferguson2024_half'),
)


# juncture:
stimulus_set_registry['Ferguson2024_juncture'] = lambda: load_stimulus_set_from_s3(
    identifier='Ferguson2024_juncture',
    bucket="brainio-brainscore",
    csv_sha1="your_sha1_here",
    zip_sha1="your_zip_sha1_here",
    csv_version_id="your_csv_version_id_here",
    zip_version_id="your_zip_version_id_here")

data_registry['Ferguson2024_juncture'] = lambda: load_assembly_from_s3(
    identifier='Ferguson2024_juncture',
    version_id="your_version_id_here",
    sha1="your_sha1_here",
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Ferguson2024_juncture'),
)


# lle (assuming 'lle' is correct and not a placeholder):
stimulus_set_registry['Ferguson2024_lle'] = lambda: load_stimulus_set_from_s3(
    identifier='Ferguson2024_lle',
    bucket="brainio-brainscore",
    csv_sha1="your_sha1_here",
    zip_sha1="your_zip_sha1_here",
    csv_version_id="your_csv_version_id_here",
    zip_version_id="your_zip_version_id_here")

data_registry['Ferguson2024_lle'] = lambda: load_assembly_from_s3(
    identifier='Ferguson2024_lle',
    version_id="your_version_id_here",
    sha1="your_sha1_here",
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Ferguson2024_lle'),
)


# llh (assuming 'llh' is correct and not a placeholder):
stimulus_set_registry['Ferguson2024_llh'] = lambda: load_stimulus_set_from_s3(
    identifier='Ferguson2024_llh',
    bucket="brainio-brainscore",
    csv_sha1="your_sha1_here",
    zip_sha1="your_zip_sha1_here",
    csv_version_id="your_csv_version_id_here",
    zip_version_id="your_zip_version_id_here")

data_registry['Ferguson2024_llh'] = lambda: load_assembly_from_s3(
    identifier='Ferguson2024_llh',
    version_id="your_version_id_here",
    sha1="your_sha1_here",
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Ferguson2024_llh'),
)


# quarter:
stimulus_set_registry['Ferguson2024_quarter'] = lambda: load_stimulus_set_from_s3(
    identifier='Ferguson2024_quarter',
    bucket="brainio-brainscore",
    csv_sha1="your_sha1_here",
    zip_sha1="your_zip_sha1_here",
    csv_version_id="your_csv_version_id_here",
    zip_version_id="your_zip_version_id_here")

data_registry['Ferguson2024_quarter'] = lambda: load_assembly_from_s3(
    identifier='Ferguson2024_quarter',
    version_id="your_version_id_here",
    sha1="your_sha1_here",
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Ferguson2024_quarter'),
)


# round_f:
stimulus_set_registry['Ferguson2024_round_f'] = lambda: load_stimulus_set_from_s3(
    identifier='Ferguson2024_round_f',
    bucket="brainio-brainscore",
    csv_sha1="your_sha1_here",
    zip_sha1="your_zip_sha1_here",
    csv_version_id="your_csv_version_id_here",
    zip_version_id="your_zip_version_id_here")

data_registry['Ferguson2024_round_f'] = lambda: load_assembly_from_s3(
    identifier='Ferguson2024_round_f',
    version_id="your_version_id_here",
    sha1="your_sha1_here",
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Ferguson2024_round_f'),
)


# round_v:
stimulus_set_registry['Ferguson2024_round_v'] = lambda: load_stimulus_set_from_s3(
    identifier='Ferguson2024_round_v',
    bucket="brainio-brainscore",
    csv_sha1="your_sha1_here",
    zip_sha1="your_zip_sha1_here",
    csv_version_id="your_csv_version_id_here",
    zip_version_id="your_zip_version_id_here")

data_registry['Ferguson2024_round_v'] = lambda: load_assembly_from_s3(
    identifier='Ferguson2024_round_v',
    version_id="your_version_id_here",
    sha1="your_sha1_here",
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Ferguson2024_round_v'),
)


# tilted_line:
stimulus_set_registry['Ferguson2024_tilted_line'] = lambda: load_stimulus_set_from_s3(
    identifier='Ferguson2024_tilted_line',
    bucket="brainio-brainscore",
    csv_sha1="your_sha1_here",
    zip_sha1="your_zip_sha1_here",
    csv_version_id="your_csv_version_id_here",
    zip_version_id="your_zip_version_id_here")

data_registry['Ferguson2024_tilted_line'] = lambda: load_assembly_from_s3(
    identifier='Ferguson2024_tilted_line',
    version_id="your_version_id_here",
    sha1="your_sha1_here",
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Ferguson2024_tilted_line'),
)