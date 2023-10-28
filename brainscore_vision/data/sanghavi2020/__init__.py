from brainio.assemblies import NeuronRecordingAssembly
from brainscore_vision import data_registry, stimulus_set_registry, load_stimulus_set, load_ceiling
from brainscore_vision.data_helpers.s3 import load_stimulus_set_from_s3, load_assembly_from_s3
from brainscore_vision.metric_helpers.transformations import CrossValidation

BIBTEX = """  @misc{Sanghavi_Murty_DiCarlo_2021,
  title={SanghaviMurty2020},
  url={osf.io/fchme},
  DOI={10.17605/OSF.IO/FCHME},
  publisher={OSF},
  author={Sanghavi, Sachi and Murty, N A R and DiCarlo, James J},
  year={2021},
  month={Nov}
}"""

# assemblies: dicarlo.Sanghavi2020 uses dicarlo.hvm
data_registry['dicarlo.Sanghavi2020'] = lambda: load_assembly_from_s3(
    identifier="dicarlo.Sanghavi2020",
    version_id="RZz2m5wUm.wgYyMEDY9UCMuyYjZKDuzw",
    sha1="12e94e9dcda797c851021dfe818b64615c785866",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('dicarlo.hvm'),
)

# assemblies: dicarlo.SanghaviJozwik2020 uses dicarlo.BOLD5000
data_registry['dicarlo.SanghaviJozwik2020'] = lambda: load_assembly_from_s3(
    identifier="dicarlo.SanghaviJozwik2020",
    version_id="j5AiLVh8xbchFP2CxVxFoifAeJy1vwHA",
    sha1="c5841f1e7d2cf0544a6ee010e56e4e2eb0994ee0",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('dicarlo.BOLD5000'),
)

# assemblies: dicarlo.SanghaviMurty2020 uses dicarlo.Rust2012
data_registry['dicarlo.SanghaviMurty2020'] = lambda: load_assembly_from_s3(
    identifier="dicarlo.SanghaviMurty2020",
    version_id="yvyTo2fM8kLsa7h7WMWC8jcqC2uAx.kp",
    sha1="6cb8e054688066d1d86d4944e1385efc6a69ebd4",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('dicarlo.Rust2012'),
)

# assemblies: dicarlo.SanghaviMurty2020THINGS1 uses dicarlo.THINGS1
data_registry['dicarlo.SanghaviMurty2020THINGS1'] = lambda: load_assembly_from_s3(
    identifier="dicarlo.SanghaviMurty2020THINGS1",
    version_id=".n3o2r4SKG4fO829jJPl9zz1UaSM7okH",
    sha1="718def227d38c8425f449512e47a2df81c04de62",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('dicarlo.THINGS1'),
)

# assemblies: dicarlo.SanghaviMurty2020THINGS2 uses dicarlo.THINGS2
data_registry['dicarlo.SanghaviMurty2020THINGS2'] = lambda: load_assembly_from_s3(
    identifier="dicarlo.SanghaviMurty2020THINGS2",
    version_id="z_MWZd12fk.AIcQzwRh6.vOIHhxInWRl",
    sha1="80962139823cb145e2385c344e3945e99ed97fa2",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('dicarlo.THINGS2'),
)

# stimulus set: dicarlo.BOLD5000 - can put in sanghavi
stimulus_set_registry['dicarlo.BOLD5000'] = lambda: load_stimulus_set_from_s3(
    identifier="dicarlo.BOLD5000",
    bucket="brainio-brainscore",
    csv_sha1="51ac47ea03ed2e72864a95065494c2fabe779f6e",
    zip_sha1="1a2e6d782dcd21bbe60460d85d83b13fa76a9543",
    csv_version_id="2gHs8J9iH7PzBOO24USSwOmmZZZez0K8",
    zip_version_id="r4gfAtAHaSj2WPc8kC5xkBpR2u4vkj_o")

# stimulus set: dicarlo.THINGS1
stimulus_set_registry['dicarlo.THINGS1'] = lambda: load_stimulus_set_from_s3(
    identifier="dicarlo.THINGS1",
    bucket="brainio-brainscore",
    csv_sha1="e02469d805e1b9c1a18403e9b0b8a37ee6da5130",
    zip_sha1="fb716eb58c303157dd577a70caad5c8079e74e9b",
    csv_version_id="PvdnGZPINuidsDNMZ3M2sZ4A3otjev1M",
    zip_version_id="c7UP4sqx2LsayqNxhScM.Omad5cUhw8g")

# stimulus set: dicarlo.THINGS2
stimulus_set_registry['dicarlo.THINGS2'] = lambda: load_stimulus_set_from_s3(
    identifier="dicarlo.THINGS2",
    bucket="brainio-brainscore",
    csv_sha1="86c8beda8f495cd69ed047d3457902dd98e4904c",
    zip_sha1="e7918dd10102b67464bc652fdb3ced25ee1fbe7a",
    csv_version_id="uZK7NNq7mfVFYTw2tdwrmJs7LivGnLRx",
    zip_version_id="xibNW6tiYOUZqhxO0XkeXammNehAjQY2")


def filter_neuroids(assembly, threshold):
    ceiler = load_ceiling('internal_consistency')
    ceiling = ceiler(assembly)
    ceiling = ceiling.raw
    ceiling = CrossValidation().aggregate(ceiling)
    ceiling = ceiling.sel(aggregation='center')
    pass_threshold = ceiling >= threshold
    assembly = assembly[{'neuroid': pass_threshold}]
    return assembly
