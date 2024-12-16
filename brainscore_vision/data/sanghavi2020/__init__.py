from brainio.assemblies import NeuronRecordingAssembly
from brainscore_vision import data_registry, stimulus_set_registry, load_stimulus_set, load_ceiling
from brainscore_vision.data_helpers.s3 import load_stimulus_set_from_s3, load_assembly_from_s3
from brainscore_vision.metric_helpers.transformations import CrossValidation

BIBTEX_SANGHAVI = """  @misc{Sanghavi2020,
  title={SanghaviMurty2020},
  url={osf.io/CHWDK},
  DOI={10.17605/OSF.IO/CHWDK},
  publisher={OSF},
  author={Sanghavi, Sachi and DiCarlo, James J},
  year={2020},
  month={Nov}
}"""  # 'HvM' images from (Majaj et al., 2015)
BIBTEX_SANGHAVIMURTY = """  @misc{SanghaviMurty2020,
  title={SanghaviMurty2020},
  url={osf.io/fchme},
  DOI={10.17605/OSF.IO/FCHME},
  publisher={OSF},
  author={Sanghavi, Sachi and Murty, N A R and DiCarlo, James J},
  year={2020},
  month={Nov}
}"""  # BOLD5000 images
BIBTEX_SANGHAVIJOZWIK = """  @misc{SanghaviJozwik2020,
  title={SanghaviJozwik2020},
  url={osf.io/FHY36},
  DOI={10.17605/OSF.IO/FHY36},
  publisher={OSF},
  author={Sanghavi, Sachi and Jozwik, K M and DiCarlo, James J},
  year={2020},
  month={Nov}
}"""  # images from (Rust & DiCarlo, 2012)

# assemblies: dicarlo.Sanghavi2020 uses hvm
data_registry['Sanghavi2020'] = lambda: load_assembly_from_s3(
    identifier="dicarlo.Sanghavi2020",
    version_id="null",
    sha1="12e94e9dcda797c851021dfe818b64615c785866",
    bucket="brainscore-storage/brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('hvm'),
)

# assemblies: dicarlo.SanghaviJozwik2020 uses dicarlo.BOLD5000
data_registry['SanghaviJozwik2020'] = lambda: load_assembly_from_s3(
    identifier="dicarlo.SanghaviJozwik2020",
    version_id="null",
    sha1="c5841f1e7d2cf0544a6ee010e56e4e2eb0994ee0",
    bucket="brainscore-storage/brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('BOLD5000'),
)

# assemblies: dicarlo.SanghaviMurty2020 uses dicarlo.Rust2012
data_registry['SanghaviMurty2020'] = lambda: load_assembly_from_s3(
    identifier="dicarlo.SanghaviMurty2020",
    version_id="null",
    sha1="6cb8e054688066d1d86d4944e1385efc6a69ebd4",
    bucket="brainscore-storage/brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Rust2012'),
)

# assemblies: dicarlo.SanghaviMurty2020THINGS1 uses THINGS1
data_registry['SanghaviMurty2020THINGS1'] = lambda: load_assembly_from_s3(
    identifier="dicarlo.SanghaviMurty2020THINGS1",
    version_id="null",
    sha1="718def227d38c8425f449512e47a2df81c04de62",
    bucket="brainscore-storage/brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('THINGS1'),
)

# assemblies: dicarlo.SanghaviMurty2020THINGS2 uses THINGS2
data_registry['SanghaviMurty2020THINGS2'] = lambda: load_assembly_from_s3(
    identifier="dicarlo.SanghaviMurty2020THINGS2",
    version_id="null",
    sha1="80962139823cb145e2385c344e3945e99ed97fa2",
    bucket="brainscore-storage/brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('THINGS2'),
)

# stimulus set: BOLD5000 - can put in sanghavi
stimulus_set_registry['BOLD5000'] = lambda: load_stimulus_set_from_s3(
    identifier="BOLD5000",
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="51ac47ea03ed2e72864a95065494c2fabe779f6e",
    zip_sha1="1a2e6d782dcd21bbe60460d85d83b13fa76a9543",
    csv_version_id="null",
    zip_version_id="null")

# stimulus set: THINGS1
stimulus_set_registry['THINGS1'] = lambda: load_stimulus_set_from_s3(
    identifier="THINGS1",
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="e02469d805e1b9c1a18403e9b0b8a37ee6da5130",
    zip_sha1="fb716eb58c303157dd577a70caad5c8079e74e9b",
    csv_version_id="null",
    zip_version_id="null")

# stimulus set: THINGS2
stimulus_set_registry['THINGS2'] = lambda: load_stimulus_set_from_s3(
    identifier="THINGS2",
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="86c8beda8f495cd69ed047d3457902dd98e4904c",
    zip_sha1="e7918dd10102b67464bc652fdb3ced25ee1fbe7a",
    csv_version_id="null",
    zip_version_id="null")


def filter_neuroids(assembly, threshold):
    ceiler = load_ceiling('internal_consistency')
    ceiling = ceiler(assembly)
    ceiling = ceiling.raw
    ceiling = CrossValidation().aggregate(ceiling)
    pass_threshold = ceiling >= threshold
    assembly = assembly[{'neuroid': pass_threshold}]
    return assembly
