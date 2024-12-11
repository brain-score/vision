from brainio.assemblies import NeuronRecordingAssembly
from brainscore_vision import load_stimulus_set
from brainscore_vision import stimulus_set_registry, data_registry
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3, load_stimulus_set_from_s3

stimulus_set_registry["DataGenerationTrial_emogan"] = lambda: load_stimulus_set_from_s3(
    identifier="DataGenerationTrial_emogan",
    bucket="brainio-brainscore",
    csv_version_id=".m5NRndaBpbapEEAHeaXBdLefI_.cRis",
    csv_sha1="71c184010873c61bd95642e49d8710c79edb5af6",
    zip_version_id="IRE5PEG7ssU4NDClND8RGV_rK1kYVNju",
    zip_sha1="e1403c9b639eb80284bb8d2bda642d957b107832",
    filename_prefix='stimulus_',
)

data_registry["DataGenerationTrial_emogan"] = lambda: load_assembly_from_s3(
    identifier="DataGenerationTrial_emogan",
    version_id="AB_TIpyKlfpt.36E.Gsf5QY.RTCfMnZM",
    sha1="ef052488963396cf8994733672c377678ed4ea4e",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('DataGenerationTrial_emogan'),
)

    