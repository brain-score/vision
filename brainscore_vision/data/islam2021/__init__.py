from brainscore_vision import stimulus_set_registry
from brainscore_vision.data_helpers.s3 import load_stimulus_set_from_s3

BIBTEX = """@article{islam2021shape,
  title={Shape or texture: Understanding discriminative features in cnns},
  author={Islam, Md Amirul and Kowal, Matthew and Esser, Patrick and Jia, Sen and Ommer, Bjorn and Derpanis, Konstantinos G and Bruce, Neil},
  journal={arXiv preprint arXiv:2101.11604},
  year={2021}
}"""

stimulus_set_registry['Islam2021'] = lambda: load_stimulus_set_from_s3(
    identifier="neil.Islam2021",
    bucket="brainio-brainscore",
    filename_prefix='stimulus_',
    csv_sha1="93ab0f6386d8d5fb56640da45980a819c7dd6efc",
    zip_sha1="e55b673117a472f463e0705ac3e330ef8dfd938b",
    csv_version_id="rhCKHTDXKLBtvflwtQ1Ic4LVUk_nc59Q",
    zip_version_id="_mya.66OGJfbY0mwX5nRVPjaNjbt42bR")
