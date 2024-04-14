
from brainio.packaging import package_stimulus_set, package_data_assembly
from brainscore_vision import load_dataset, load_stimulus_set

def upload_stimulus_set_to_s3(stimuli):
    return package_stimulus_set(catalog_name=None, proto_stimulus_set=stimuli,
                                stimulus_set_identifier=stimuli.name, bucket_name="brainio-brainscore")
    
def upload_assembly_to_s3(assembly):
    return package_data_assembly(None, assembly, assembly_identifier=assembly.name,
                                 stimulus_set_identifier=assembly.name,
                                 assembly_class_name="NeuronRecordingAssembly",
                                 bucket_name="brainio-brainscore")


if __name__ == '__main__':
    ss = load_stimulus_set('IAPS')
    assembly = load_dataset('IAPS')

    