from brainscore_vision import stimulus_set_registry
from brainscore_vision.data_helpers.s3 import load_stimulus_set_from_s3

BIBTEX = """"""


# stimulus set
stimulus_set_registry['dicarlo.ImageNetSlim15000'] = lambda: load_stimulus_set_from_s3(
    identifier="dicarlo.ImageNetSlim15000",
    bucket="brainio-brainscore",
    csv_sha1="56f79de36060147bc43ae89778e26d50f73c3a63",
    zip_sha1="633d08474e3067add834f4ddf552642c538db112",
    csv_version_id="GJ7u6Aq6nbjMAPL1YeHOxrGhhZVfJ4Uh",
    zip_version_id="oKEfsCxamcNmJpo7IvEsWok81ODaeIUW")
