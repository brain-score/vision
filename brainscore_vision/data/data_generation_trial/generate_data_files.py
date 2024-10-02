from pathlib import Path
from dandi_to_stimulus_set import get_stimuli
from extract_nwb_data import generate_json_file, validate_nwb_file
from create_assembly import load_responses

import os, json
import zipfile


class DataFactory:
    def __init__(self, user_json: dict):
        self.user_json = user_json
        self.parse_json()

        self.nwb_file   = validate_nwb_file(self.nwb_file_path, dandiset_id=self.dandiset_id)
        self.json_file_path = os.path.join(self.exp_path, "nwb_metadata.json")
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

    def parse_json(self):
        self.dandiset_id = self.user_json['dandiset_id']
        self.directory = self.user_json['exp_path']
        self.exp_path = self.user_json['exp_path']
        self.nwb_file_path = self.user_json['nwb_file_path']
        self.identifier = self.user_json['identifier']

        self.assembly = self.user_json['assembly']
        self.stimulus_set = self.user_json['stimulus_set']

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
        test_code = f"""
import pytest

from brainscore_vision import load_dataset, load_stimulus_set
from brainscore_vision.benchmark_helpers import check_standard_format


@pytest.mark.memory_intense
@pytest.mark.private_access
def test_assembly():
    assembly = load_dataset('{self.identifier}')
    check_standard_format(assembly, nans_expected=True)
    assert assembly.attrs['stimulus_set_identifier'] == '{self.identifier}'
    assert set(assembly['region'].values) == {self.assembly['region']}
    assert len(assembly['presentation']) == {self.assembly['presentation']}
    assert len(assembly['neuroid']) == {self.assembly['neuroid']}


@pytest.mark.private_access
def test_stimulus_set():
    stimulus_set = load_stimulus_set('{self.identifier}')
    assert len(stimulus_set) == {self.stimulus_set['length']}

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
    # data_factory = DataFactory(directory=directory, exp_path=exp_path, identifier=NWB_METADATA['stimulus_set']['identifier'], dandiset_id=dandiset_id, nwb_file_path=nwb_file_path)

    with open('/Users/caroljiang/Downloads/vision/brainscore_vision/data/data_generation_trial/example_json.json', 'r') as file:
        user_json = json.load(file)
    data_factory = DataFactory(user_json=user_json)
    data_factory()