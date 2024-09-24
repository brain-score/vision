from pathlib import Path
from brainio.stimuli import StimulusSet
from dandi_to_stimulus_set import convert_to_stimulus_set, get_stimuli
from extract_nwb_data import generate_json_file, validate_nwb_file
from create_assembly import output_assembly, load_responses

import os, json
import zipfile

# currently hardcoded, change to take in as input
NWB_METADATA = {
    'dandiset_number': '000788', 
    'assembly': {
        'identifier': 'DataGenerationTrial_emogan',
        'stimulus_set_identifier': 'DataGenerationTrial_emogan',
        'region': {'IT'},
        'presentation': 8500,
        'neuroid': 18
    },
    'stimulus_set': {
        'identifier': 'DataGenerationTrial_emogan',
        'length': 170
    }
}
# NWB_METADATA = {
#     'dandiset_number': '000781', 
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
# NWB_METADATA = {
#     'assembly': {
#         'identifier': 'IAPS',
#         'stimulus_set_identifier': 'IAPS',
#         'region': {'IT'},
#         'presentation': 30150,
#         'neuroid': 36
#     },
#     'stimulus_set': {
#         'identifier': 'IAPS',
#         'length': 10
#     }
# }
# NWB_METADATA = {
#     'dandiset_number': '000786', 
#     'assembly': {
#         'identifier': 'domain-transfer-2023',
#         'stimulus_set_identifier': 'domain-transfer-2023',
#         'region': {'IT'},
#         'presentation': 7588,
#         'neuroid': 36
#     },
#     'stimulus_set': {
#         'identifier': 'domain-transfer-2023',
#         'length': 319
#     }
# }

class DataFactory:

    def __init__(self, directory: str, exp_path: str, identifier: str, dandiset_id: str, nwb_file_path: str):
        self.directory = directory
        self.exp_path = exp_path
        self.identifier = identifier

        self.dandiset_id = dandiset_id
        self.nwb_file_path = nwb_file_path

        # nwb_file_name   = os.listdir(os.path.join(self.exp_path, "sub-pico"))[0]
        # nwb_file_path   = os.path.join(os.path.join(self.exp_path, "sub-pico", nwb_file_name))
        self.nwb_file   = validate_nwb_file(self.dandiset_id, self.nwb_file_path)

        self.json_file_path = os.path.join(self.exp_path, "nwb_metadata.json")
        print('json path', self.json_file_path)
        generate_json_file(self.nwb_file, self.json_file_path)
        input("Validate the values in the generated JSON file are correct, or edit the file. Press Enter to continue.")
        with open(self.json_file_path, 'r') as f:
            self.params = json.load(f)


    def __call__(self):
        try: os.mkdir(os.path.join(self.directory, 'test_data_packaging'))
        except: pass 

        output_dir = os.path.join(self.directory, 'test_data_packaging')

        # create and write data packaging code:
        data_packaging_code = self.generate_data_packaging_code()
        data_packaging_path = Path(f"{output_dir}/data_packaging.py")
        self.write_code_into_file(data_packaging_code, data_packaging_path)

        # create and write init code:
        init_code = self.generate_init_code()
        init_path = Path(f"{output_dir}/__init__.py")
        self.write_code_into_file(init_code, init_path)

        # create and write test code:
        test_code = self.generate_test_code()
        test_path = Path(f"{output_dir}/test.py")
        self.write_code_into_file(test_code, test_path)

        # generate zip file
        path = Path(__file__).parent
        os.chdir(path)
        folder_to_zip = 'test_data_packaging'
        output_zip_file = 'test_data_packaging.zip'
        self.zip_files(folder_to_zip, output_zip_file)

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
        from test_data_packaging.data_packaging import upload_assembly_to_s3, upload_stimulus_set_to_s3
        stimuli = get_stimuli(self.dandiset_id, self.nwb_file, self.exp_path, self.params['exp_name'][4:])[0]
        assembly = load_responses(self.nwb_file, self.json_file_path, stimuli, use_QC_data = True, do_filter_neuroids = True, use_brainscore_filter_neuroids_method=True)
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

    def zip_files(self, folder_path, output_zip_path):
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, folder_path)
                    zipf.write(file_path, relative_path)


if __name__ == '__main__':
    directory = '/Users/caroljiang/Downloads/vision/brainscore_vision/data/data_generation_trial'
    exp_path = f"/Users/caroljiang/Downloads/vision/brainscore_vision/data/data_generation_trial/"
    dandiset_id = '000788'
    nwb_file_path = 'sub-pico/sub-pico_ecephys.nwb'
    # nwb_file_path = 'sub-pico/sub-pico_ecephys+image.nwb'
    data_factory = DataFactory(directory=directory, exp_path=exp_path, identifier=NWB_METADATA['stimulus_set']['identifier'], dandiset_id=dandiset_id, nwb_file_path=nwb_file_path)
    data_factory()