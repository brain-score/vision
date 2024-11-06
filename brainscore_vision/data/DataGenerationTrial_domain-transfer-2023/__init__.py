from brainio.assemblies import NeuronRecordingAssembly
from brainscore_vision import load_stimulus_set
from brainscore_vision import stimulus_set_registry, data_registry
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3, load_stimulus_set_from_s3

stimulus_set_registry["DataGenerationTrial_domain-transfer-2023"] = lambda: load_stimulus_set_from_s3(
    identifier="DataGenerationTrial_domain-transfer-2023",
    bucket="brainio-brainscore",
    csv_version_id="agejo7lnPqWXTyP_9iCMThBEz2VCmRpw",
    csv_sha1="276755aca4d23478384970017bf3a06c2852b91c",
    zip_version_id="KdjwLh5UcjRwfAdf1bgVAmx0oFICvEg.",
    zip_sha1="266ce2d93fdeee3e1e6833d219d663eae2173e06",
    filename_prefix='stimulus_',
)

data_registry["DataGenerationTrial_domain-transfer-2023"] = lambda: load_assembly_from_s3(
    identifier="DataGenerationTrial_domain-transfer-2023",
    version_id="_bgL3mncuYGSNgfiSb4yqVMsVm5VKW3A",
    sha1="e93ce19f8b557441847e0e7905aa26c696e028f3",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('DataGenerationTrial_domain-transfer-2023'),
)

    