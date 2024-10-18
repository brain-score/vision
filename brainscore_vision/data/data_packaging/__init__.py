from brainio.assemblies import NeuronRecordingAssembly
from brainscore_vision import load_stimulus_set
from brainscore_vision import stimulus_set_registry, data_registry
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3, load_stimulus_set_from_s3

stimulus_set_registry["DataGenerationTrial_emogan"] = lambda: load_stimulus_set_from_s3(
    identifier="DataGenerationTrial_emogan",
    bucket="brainio-brainscore",
    csv_version_id="rJjIUKeyAv7do1Omb47tY0Xh7kIWMzd8",
    csv_sha1="09667ef99b62b169eefbbe716781fc6d9cd2e07f",
    zip_version_id="w3SAy8dSwFUJxTKOmQg2OcVRDeI5o4ay",
    zip_sha1="8e2f46b148954ffdcebf349f90521b47282002f4",
    filename_prefix='stimulus_',
)

data_registry["DataGenerationTrial_emogan"] = lambda: load_assembly_from_s3(
    identifier="DataGenerationTrial_emogan",
    version_id="xWvkugISK7u_BeD8rx1JCyg_Sd.n5zGH",
    sha1="530c5e50e831c8d0fecbda243b0fcdf975114b7c",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('DataGenerationTrial_emogan'),
)

    