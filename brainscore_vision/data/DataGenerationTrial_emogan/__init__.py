from brainio.assemblies import NeuronRecordingAssembly
from brainscore_vision import load_stimulus_set
from brainscore_vision import stimulus_set_registry, data_registry
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3, load_stimulus_set_from_s3

stimulus_set_registry["DataGenerationTrial_emogan"] = lambda: load_stimulus_set_from_s3(
    identifier="DataGenerationTrial_emogan",
    bucket="brainio-brainscore",
    csv_version_id="qX7K6GBJ8iWWsDYvXuyGrMsZ_7bz1893",
    csv_sha1="adc47833061332669bb5fe48709c050027f85fc7",
    zip_version_id="khCC5TK6hcoGnxYlnH1nik1_LcenYy7_",
    zip_sha1="1eed8230add1d098ff81a135c2b75eda19209047",
    filename_prefix='stimulus_',
)

data_registry["DataGenerationTrial_emogan"] = lambda: load_assembly_from_s3(
    identifier="DataGenerationTrial_emogan",
    version_id="6E9KZnONiZOz7_TPb38a40.XX819R2b4",
    sha1="07ae883c10d8aa886f92b547a7d580cda9c04c05",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('DataGenerationTrial_emogan'),
)

    