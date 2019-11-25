import json
import paramiko

remote_location = '/om2/group/dicarlo/jenkins/configs'


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
    ssh_client.connect(hostname='braintree-cpu-1.mit.edu', username='fgeiger', password='??')
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
        'model_type': 'BaseModel',  # | 'BrainModel'
        'author': 'DiCarloLab',
        'email': 'fgeiger@mit.edu',
        'gpu_size': '8000',
        'type': 'git',  # | 'zip'
        'git_url': 'https://github.com/brain-score/candidate_models.git',
        # or
        # 'zip_filepath': '/home/fgeiger/repos', this can be a static path
        # 'zip_filename': 'alexnet.zip',
    }
    push_configs(model_dict)


if __name__ == '__main__':
    on_submit()
