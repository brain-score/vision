import logging
import os
import subprocess
import sys
import zipfile
from importlib import import_module
from pathlib import Path

from brainscore.submission.configuration import BaseConfig
from brainscore.submission.models import Submission

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
    zip_file = Path('%s/submission_%s.zip' % (config_path, id))
    with zipfile.ZipFile(zip_file, 'r') as model_repo:
        model_repo.extractall(path=str(work_dir))
    #     Use the single directory in the zip file
    full_path = Path(work_dir).absolute()
    return Path('%s/%s' % (str(full_path), find_correct_dir(work_dir)))


def find_correct_dir(work_dir):
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
    logger.info('Start installing submitted the repository')
    try:
        assert 0 == subprocess.call([sys.executable, "-m", "pip", "install", "-v", repo], env=os.environ)
        sys.path.insert(0, str(repo))
        logger.info(f'System paths {sys.path}')
        return import_module(package)
    except ImportError:
        return __import__(package)
    except AssertionError as e:
        logger.error('Installation of submitted models failed!')
        raise e


def deinstall_project(module):
    if 'models.brain_models' in sys.modules:
        del sys.modules['models.brain_models']
    if 'models.base_models' in sys.modules:
        del sys.modules['models.base_models']
    subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "-v", module], env=os.environ)
