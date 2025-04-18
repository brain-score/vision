from brainio.assemblies import NeuronRecordingAssembly
from brainscore_vision import load_stimulus_set
from brainscore_vision import stimulus_set_registry, data_registry
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3, load_stimulus_set_from_s3

stimulus_set_registry["DataGenerationTrial_emogan"] = lambda: load_stimulus_set_from_s3(
    identifier="DataGenerationTrial_emogan",
    bucket="brainio-brainscore",
    csv_version_id="us10Kp6YL_nU38eW15OC9gSoCBlfqRbl",
    csv_sha1="76d58a0e2e1c2b615d388e8fe0eb4471d195173a",
    zip_version_id="_rn9rxzo2xIR0hMOP9vGP0W4uz8hPl9c",
    zip_sha1="75f50073573299dfe04b6eb9e187e5308a78f6e0",
    filename_prefix='stimulus_',
)

data_registry["DataGenerationTrial_emogan"] = lambda: load_assembly_from_s3(
    identifier="DataGenerationTrial_emogan",
    version_id="fyuEZyEDAZO3Ujpe60jql5LWdFlJ1EdP",
    sha1="530d5123d079da7b8d5e12c1c061329b6ae44ccb",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('DataGenerationTrial_emogan'),
)

    