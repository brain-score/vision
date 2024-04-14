
from brainio.assemblies import NeuronRecordingAssembly
from brainscore_vision import load_stimulus_set
from brainscore_vision import stimulus_set_registry, data_registry
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3, load_stimulus_set_from_s3

stimulus_set_registry["IAPS"] = lambda: load_stimulus_set_from_s3(
    identifier="IAPS",
    bucket="brainio-brainscore",
    csv_version_id="I90msPTpsEbRGMSnO.Ll76KOjN1waJMX",
    csv_sha1="2e6bcf4bffe5d710ce4c744c4dffdbdae256a333",
    zip_version_id="jFh.VNtU39NNFMoyss1fmoRFe9OyNv7x",
    zip_sha1="9450121d2ac447a3c370e0b2b2bc2a7d9157480b",
    filename_prefix='stimulus_',
)

data_registry["IAPS"] = lambda: load_assembly_from_s3(
    identifier="IAPS",
    version_id="J2gDD7VTuZ7qNT.JBfwFL9Z3emgNnRP3",
    sha1="9050b94675687b4267031bfa3ae09c70654203dd",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('IAPS'),
)

    