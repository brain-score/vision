import logging
import os
import subprocess
import sys
import zipfile
from importlib import import_module
from pathlib import Path

from brainscore_vision.submission.configuration import BaseConfig
from brainscore_vision.submission.models import Submission

logger = logging.getLogger(__name__)


def prepare_module(submission: Submission, config: BaseConfig):
    config_path = config.config_path
    logger.info('Start executing models in repo submission_%s' % submission.id)
    repo = extract_zip_file(submission.id, config_path, config.work_dir)
    package = 'models.brain_models' if submission.model_type == 'BrainModel' else 'models.base_models'
    logger.info(f'We work with {submission.model_type} and access {package} in the submission folder')
    return install_project(repo, package), os.path.basename(repo)


def extract_zip_file(id, config_path, work_dir):
    logger.info(f'Unpack zip file')
    zip_file = Path(f'{config_path}/submission_{id}.zip')
    with zipfile.ZipFile(zip_file, 'r') as model_repo:
        model_repo.extractall(path=str(work_dir))
    # Use the single directory in the zip file
    full_path = Path(work_dir).absolute()
    submission_directory = find_submission_directory(work_dir)
    return full_path / submission_directory


def find_submission_directory(work_dir):
    """
    Find the single directory inside a directory that corresponds to the submission file.
    Ignores hidden directories, e.g. those prefixed with `.` and `_`
    """
    list = os.listdir(work_dir)
    candidates = []
    for item in list:
        if not item.startswith('.') and not item.startswith('_'):
            candidates.append(item)
    if len(candidates) is 1:
        return candidates[0]
    logger.error('The zip file structure is not correct, we try to detect the correct directory')
    if 'sample-model-submission' in candidates:
        return 'sample-model-submission'
    logger.error('The submission file contains too many entries and can therefore not be installed')
    raise Exception('The submission file contains too many entries and can therefore not be installed')


def install_project(repo, package):
    install_repo_command = [sys.executable, "-m",
                            "pip", "install",
                            "-v", "--default-timeout=3600",
                            str(repo),
                            "--user"]
    logger.info(f"Install submitted repository: {install_repo_command}")
    try:
        subprocess.check_output(install_repo_command,
                                env=os.environ,
                                stderr=subprocess.STDOUT)
        sys.path.insert(0, str(repo))
        logger.info(f'System paths {sys.path}')
        return import_module(package)
    except ImportError:
        return __import__(package)
    except subprocess.CalledProcessError as e:
        logger.error('Installation of submitted models failed!')
        logger.info(e.output)
        raise e


def deinstall_project(module):
    if 'models.brain_models' in sys.modules:
        del sys.modules['models.brain_models']
    if 'models.base_models' in sys.modules:
        del sys.modules['models.base_models']
    subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "-v", module],
                    env=os.environ)
