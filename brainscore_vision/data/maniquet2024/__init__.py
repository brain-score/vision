from brainio.assemblies import BehavioralAssembly
from brainscore_vision import stimulus_set_registry, data_registry
from brainscore_vision.data_helpers.s3 import (
    load_assembly_from_s3,
    load_stimulus_set_from_s3,
)

BIBTEX = """@article {Maniquet2024.04.02.587669,
        author = {Maniquet, Tim and de Beeck, Hans Op and Costantino, Andrea Ivan},
        title = {Recurrent issues with deep neural network models of visual recognition},
        elocation-id = {2024.04.02.587669},
        year = {2024},
        doi = {10.1101/2024.04.02.587669},
        publisher = {Cold Spring Harbor Laboratory},
        URL = {https://www.biorxiv.org/content/early/2024/04/10/2024.04.02.587669},
        eprint = {https://www.biorxiv.org/content/early/2024/04/10/2024.04.02.587669.full.pdf},
        journal = {bioRxiv}
}"""

# Human Stimulus Set
stimulus_set_registry["Maniquet2024"] = lambda: load_stimulus_set_from_s3(
    identifier="Maniquet2024",
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="ec61e1d7776a6c3b467fee862302edac8d4a156e",
    zip_sha1="bbdaf09528974c4ca3ee4cddbc91e0e03351291f",
    csv_version_id="null",
    zip_version_id="null",
)

# DNN test Stimulus Set
stimulus_set_registry["Maniquet2024-test"] = lambda: load_stimulus_set_from_s3(
    identifier="Maniquet2024-test",
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="993089ba4aaeffbc61303acb2a5171a5fa271ea5",
    zip_sha1="39f9aaf13fdd66d284bcea99f187bb0c065144e4",
    csv_version_id="null",
    zip_version_id="null",
)

#  DNN train Stimulus Set
stimulus_set_registry["Maniquet2024-train"] = lambda: load_stimulus_set_from_s3(
    identifier="Maniquet2024-train",
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="da965af3ae5ab6e49d46c28f682ef4b75d0a2045",
    zip_sha1="6685effb52f6870175988c47892b3f9a916a0375",
    csv_version_id="null",
    zip_version_id="null",
)

# Human Data Assembly (behavioural)
data_registry["Maniquet2024"] = lambda: load_assembly_from_s3(
    identifier="Maniquet2024",
    version_id="null",
    sha1="39b8b7b29fad080ebba6df8a46ac4426261342d5",
    bucket="brainscore-storage/brainio-brainscore",
    cls=BehavioralAssembly,
)
