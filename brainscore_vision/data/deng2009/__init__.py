from brainscore_vision import data_registry
from brainscore_vision.utils.s3 import load_stimulus_set_from_s3


# TODO: add correct version id
# stimulus set
data_registry['fei-fei.Deng2009'] = lambda: load_stimulus_set_from_s3(
    identifier="fei-fei.Deng2009",
    bucket="brainio.contrib",
    csv_sha1="ff79dcf6b0d115e6e8aa8d0fbba3af11dc649e57",
    zip_sha1="78172d752d8216a00833cfa34be67c8532ad7330",
    csv_version_id="",
    zip_version_id="")
