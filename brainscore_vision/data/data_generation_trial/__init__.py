
from brainio.assemblies import NeuronRecordingAssembly
from brainscore_vision import load_stimulus_set
from brainscore_vision import stimulus_set_registry, data_registry
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3, load_stimulus_set_from_s3

stimulus_set_registry["Co3D"] = lambda: load_stimulus_set_from_s3(
    identifier="Co3D",
    bucket="brainio-brainscore",
    csv_version_id="pLwFhvszKFoHCLwWgNOb3dYw5WOApmet",
    csv_sha1="01d207b46e7f432f059ddbf59058108de43a1777",
    zip_version_id="MM2.ovwsfYZdpTgEYCqKzAhbcBzespal",
    zip_sha1="d15acd391ee026b95105efd787846a82cc4702f7",
    filename_prefix='stimulus_',
)

data_registry["Co3D"] = lambda: load_assembly_from_s3(
    identifier="Co3D",
    version_id="u8hQKHnjbxUk0Z3uPmLb7VXS3nYpr1RN",
    sha1="c364edc3b3dfe72efc615eb26c539438189bad57",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Co3D'),
)

    