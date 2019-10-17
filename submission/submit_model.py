import json
import paramiko

remote_location = '/om/user/fgeiger/models'


# deliver dict, containg all the configuration parameter required:
# model_id, author,email(of the author), gpu_size(in MB), repo_name, zip_file_path
def push_configs(configs: dict):
    name = configs['name']
    model_file = f'{name}.json'
    file = open(model_file, 'w')
    # with as file:
    if configs['type'] is 'zip':
        zip_path = configs['zip_filepath']
        configs['zip_filepath'] = remote_location
        file.write(json.dumps(configs))
        file.close()
        load_to_remote_host(file.name, '%s/%s' % (remote_location, model_file),
                            '%s/%s' % (zip_path, configs['zip_filename']),
                            '%s/%s' % (remote_location, configs['zip_filename']))
    else:
        file.write(json.dumps(configs))
        file.close()
        load_to_remote_host(file.name, '%s/%s' % (remote_location, model_file))
    return model_file


def load_to_remote_host(source_config_file, target_config_file, source_zip_file=None, target_zip_file=None):
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(hostname='openmind7.mit.edu', username='fgeiger', password='Apfeldorf2696!')
    ftp_client = ssh_client.open_sftp()
    ftp_client.put(source_config_file, target_config_file)
    if source_zip_file is not None:
        ftp_client.put(source_zip_file, target_zip_file)
    ftp_client.close()


def trigger_jenkins_job(config_name, benchmarks: []):
    # TODO define the parameterized URL
    return


def on_submit():
    # TODO check some kind of user permission
    model_dict = {
        'name': 'candidate_models_zip',
        # Not beautiful but that's what it is
        'models': ['alexnet', 'squeezenet1_0', 'squeezenet1_1', 'resnet-18', 'resnet-34', 'vgg-16', 'vgg-19', 'vggface',
                   'xception', 'densenet-121', 'densenet-169', 'densenet-201', 'inception_v1', 'inception_v2',
                   'inception_v3', 'inception_v4', 'inception_resnet_v2', 'resnet-50_v1', 'resnet-101_v1',
                   'resnet-152_v1', 'resnet-50_v2', 'resnet-101_v2', 'resnet-152_v2', 'nasnet_mobile', 'nasnet_large',
                   'pnasnet_large', 'bagnet9', 'bagnet17', 'bagnet33', 'CORnet-Z', 'CORnet-R', 'CORnet-S',
                   'resnet50-SIN', 'resnet50-SIN_IN', 'resnet50-SIN_IN_IN', 'resnext101_32x8d_wsl',
                   'resnext101_32x16d_wsl', 'resnext101_32x32d_wsl', 'resnext101_32x48d_wsl',
                   'fixres_resnext101_32x48d_wsl', 'mobilenet_v1_1.0_224', 'mobilenet_v1_1.0_192',
                   'mobilenet_v1_1.0_160', 'mobilenet_v1_1.0_128', 'mobilenet_v1_0.75_224', 'mobilenet_v1_0.75_192',
                   'mobilenet_v1_0.75_160', 'mobilenet_v1_0.75_128', 'mobilenet_v1_0.5_224', 'mobilenet_v1_0.5_192',
                   'mobilenet_v1_0.5_160', 'mobilenet_v1_0.5_128', 'mobilenet_v1_0.25_224', 'mobilenet_v1_0.25_192',
                   'mobilenet_v1_0.25_160', 'mobilenet_v1_0.25_128', 'mobilenet_v2_1.4_224', 'mobilenet_v2_1.3_224',
                   'mobilenet_v2_1.0_224', 'mobilenet_v2_1.0_192', 'mobilenet_v2_1.0_160', 'mobilenet_v2_1.0_128',
                   'mobilenet_v2_1.0_96', 'mobilenet_v2_0.75_224', 'mobilenet_v2_0.75_192', 'mobilenet_v2_0.75_160',
                   'mobilenet_v2_0.75_128', 'mobilenet_v2_0.75_96', 'mobilenet_v2_0.5_224', 'mobilenet_v2_0.5_192',
                   'mobilenet_v2_0.5_160', 'mobilenet_v2_0.5_128', 'mobilenet_v2_0.5_96', 'mobilenet_v2_0.35_224',
                   'mobilenet_v2_0.35_192', 'mobilenet_v2_0.35_160', 'mobilenet_v2_0.35_128', 'mobilenet_v2_0.35_96'],
        'model_type': 'BaseModel',  # | 'BrainModel'
        'author': 'DiCarloLab',
        'email': 'fgeiger@mit.edu',
        'gpu_size': '8000',
        'repo_name': 'candidate_models',
        'type': 'git',  # | 'zip'
        'git_url': 'https://github.com/brain-score/candidate_models.git',
        # 'zip_filepath': '/home/fgeiger/repos',
        # 'zip_filename': 'alexnet.zip',
        'publish_results': 'True'  # | False

    }
    push_configs(model_dict)


if __name__ == '__main__':
    on_submit()
