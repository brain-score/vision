from brainscore_vision import stimulus_set_registry
from brainscore_vision.data_helpers.s3 import load_stimulus_set_from_s3

# stimulus set
stimulus_set_registry['ImageNetSlim15000'] = lambda: load_stimulus_set_from_s3(
    identifier="ImageNetSlim15000",
    bucket="brainio-brainscore",
    csv_sha1="56f79de36060147bc43ae89778e26d50f73c3a63",
    zip_sha1="633d08474e3067add834f4ddf552642c538db112",
    csv_version_id="iDLAbolZkyQ5c0Znj2mk0pirw.nhCW5j",
    zip_version_id="xqkshnRACHpOqp1sSRBmHI9PeCAcTCAg")
