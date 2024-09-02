from brainscore_vision import stimulus_set_registry, data_registry
from brainscore_vision.data_helpers.s3 import load_stimulus_set_from_s3, load_assembly_from_s3

from brainio.assemblies import BehavioralAssembly

BIBTEX = """@article{DBLP:journals/corr/GoyalKMMWKHFYMH17,
author       = {Raghav Goyal and Samira Ebrahimi Kahou and Vincent Michalski and
                Joanna Materzynska and Susanne Westphal and Heuna Kim and
                Valentin Haenel and Ingo Fr{\"{u}}nd and Peter Yianilos and
                Moritz Mueller{-}Freitag and Florian Hoppe and Christian Thurau and
                Ingo Bax and Roland Memisevic},
title        = {The "something something" video database for learning and evaluating
              visual common sense},
journal      = {CoRR},
year         = {2017},
}"""


data_registry['SSV2ActivityRec2024'] = lambda: load_assembly_from_s3(
    identifier='SSV2ActivityRec2024',
    version_id="",
    sha1="",
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
)

