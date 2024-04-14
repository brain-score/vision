from pathlib import Path
from dandi_to_stimulus_set import convert_to_stimulus_set
from create_assembly import output_assembly


# currently hardcoded, change to take in as input
# NWB_METADATA = {
#     'assembly': {
#         'identifier': 'emogan',
#         'stimulus_set_identifier': 'emogan',
#         'region': {'IT'},
#         'presentation': 8550,
#         'neuroid': 27
#     },
#     'stimulus_set': {
#         'identifier': 'emogan',
#         'length': 171
#     }
# }
# NWB_METADATA = {
#     'assembly': {
#         'identifier': 'Co3D',
#         'stimulus_set_identifier': 'Co3D',
#         'region': {'IT'},
#         'presentation': 7588,
#         'neuroid': 36
#     },
#     'stimulus_set': {
#         'identifier': 'Co3D',
#         'length': 319
#     }
# }
NWB_METADATA = {
    'assembly': {
        'identifier': 'IAPS',
        'stimulus_set_identifier': 'IAPS',
        'region': {'IT'},
        'presentation': 30150,
        'neuroid': 36
    },
    'stimulus_set': {
        'identifier': 'IAPS',
        'length': 10
    }
}

class DataFactory:

    def __init__(self, directory: str, identifier: str):
        self.directory = directory
        self.identifier = identifier

    def __call__(self):

        # create and write data packaging code:
        data_packaging_code = self.generate_data_packaging_code()
        data_packaging_path = Path(f"{self.directory}/data_packaging.py")
        self.write_code_into_file(data_packaging_code, data_packaging_path)

        # create and write init code:
        init_code = self.generate_init_code()
        init_path = Path(f"{self.directory}/__init__.py")
        self.write_code_into_file(init_code, init_path)

        # create and write test code:
        test_code = self.generate_test_code()
        test_path = Path(f"{self.directory}/test.py")
        self.write_code_into_file(test_code, test_path)

    def generate_data_packaging_code(self) -> str:
        data_packaging_code = f"""
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
    ss = load_stimulus_set('{self.identifier}')
    assembly = load_dataset('{self.identifier}')

    """
        return data_packaging_code

    def generate_init_code(self) -> str:
        from data_packaging import upload_assembly_to_s3, upload_stimulus_set_to_s3
        stimuli = convert_to_stimulus_set()
        assembly = output_assembly()
        stimuli_info = upload_stimulus_set_to_s3(stimuli)
        assembly_info = upload_assembly_to_s3(assembly)

        init_code = f"""
from brainio.assemblies import NeuronRecordingAssembly
from brainscore_vision import load_stimulus_set
from brainscore_vision import stimulus_set_registry, data_registry
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3, load_stimulus_set_from_s3

stimulus_set_registry["{stimuli_info['identifier']}"] = lambda: load_stimulus_set_from_s3(
    identifier="{stimuli_info['identifier']}",
    bucket="{stimuli_info['bucket']}",
    csv_version_id="{stimuli_info['csv_version_id']}",
    csv_sha1="{stimuli_info['csv_sha1']}",
    zip_version_id="{stimuli_info['zip_version_id']}",
    zip_sha1="{stimuli_info['zip_sha1']}",
    filename_prefix='stimulus_',
)

data_registry["{assembly_info['identifier']}"] = lambda: load_assembly_from_s3(
    identifier="{assembly_info['identifier']}",
    version_id="{assembly_info['version_id']}",
    sha1="{assembly_info['sha1']}",
    bucket="{assembly_info['bucket']}",
    cls={assembly_info['cls']},
    stimulus_set_loader=lambda: load_stimulus_set('{assembly_info['identifier']}'),
)

    """
        return init_code

    def write_code_into_file(self, file_code: str, path: Path) -> None:
        with open(path, "w") as f:
            f.write(file_code)

    def generate_test_code(self) -> str:
        assembly_metadata = NWB_METADATA['assembly']
        ss_metadata = NWB_METADATA['stimulus_set']
        
        test_code = f"""
import pytest

from brainscore_vision import load_dataset, load_stimulus_set
from brainscore_vision.benchmark_helpers import check_standard_format


@pytest.mark.memory_intense
@pytest.mark.private_access
def test_assembly():
    assembly = load_dataset('{assembly_metadata['identifier']}')
    check_standard_format(assembly, nans_expected=True)
    assert assembly.attrs['stimulus_set_identifier'] == '{assembly_metadata['stimulus_set_identifier']}'
    assert set(assembly['region'].values) == {assembly_metadata['region']}
    assert len(assembly['presentation']) == {assembly_metadata['presentation']}
    assert len(assembly['neuroid']) == {assembly_metadata['neuroid']}


@pytest.mark.private_access
def test_stimulus_set():
    stimulus_set = load_stimulus_set('{ss_metadata['identifier']}')
    assert len(stimulus_set) == {ss_metadata['length']}

        """
        return test_code


if __name__ == '__main__':
    data_factory = DataFactory(directory='/Users/caroljiang/Downloads/vision/brainscore_vision/data/data_generation_trial', identifier=NWB_METADATA['stimulus_set']['identifier'])
    data_factory()