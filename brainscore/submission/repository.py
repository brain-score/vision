import logging
import os
import subprocess
import sys
import zipfile
from importlib import import_module

import git

from brainscore.submission.configuration import SubmissionConfig, BaseConfig
from pathlib import Path

from brainscore.submission.models import Submission

logger = logging.getLogger(__name__)


def prepare_module(submission:Submission, config:BaseConfig):
    config_path = config.config_path
    logger.info('Start executing models in repo submission_%s' % submission.id)
    repo = extract_zip_file(submission.id, config_path, config.work_dir)
    package = 'models.brain_models' if submission.model_type == 'BrainModel' else 'models.base_models'
    return install_project(repo, package)


def extract_zip_file(id, config_path, work_dir):
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
    return candidates[0]


def clone_repo(config, work_dir):
    git.Git(work_dir).clone(config['git_url'])
    return Path('%s/%s' % (work_dir, os.listdir(work_dir)[0]))


def install_project(repo, package):
    try:
        assert 0 == subprocess.call([sys.executable, "-m", "pip", "install", "-v", repo], env=os.environ)
        sys.path.insert(1, str(repo))
        logger.info(f'System paths {sys.path}')
        return import_module(package)
    except ImportError:
        return __import__(package)
