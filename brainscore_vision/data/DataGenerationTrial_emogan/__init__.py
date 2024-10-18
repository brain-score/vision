from brainio.assemblies import NeuronRecordingAssembly
from brainscore_vision import load_stimulus_set
from brainscore_vision import stimulus_set_registry, data_registry
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3, load_stimulus_set_from_s3

stimulus_set_registry["DataGenerationTrial_emogan"] = lambda: load_stimulus_set_from_s3(
    identifier="DataGenerationTrial_emogan",
    bucket="brainio-brainscore",
    csv_version_id="o_pbXUeXKWn0KldIlnuR.O_Sqxa0APiu",
    csv_sha1="d97769f7af6e7f185eaf33696ac6ecb406d0dd70",
    zip_version_id="i48c2HTwYpHE41nMHjNx.kvruDpa34wX",
    zip_sha1="4caa201128161a1736929d7d48f89b0074340daf",
    filename_prefix='stimulus_',
)

data_registry["DataGenerationTrial_emogan"] = lambda: load_assembly_from_s3(
    identifier="DataGenerationTrial_emogan",
    version_id="BEC8opBk8JVQ60usH9G11ez1V.6SFhDw",
    sha1="e2ecc0d271fedd2a7cfd94c9b60e321e64efef9a",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('DataGenerationTrial_emogan'),
)

    