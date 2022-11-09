import argparse
import logging
import sys
from pathlib import Path

import fire

from brainscore_vision.submission.evaluation import run_evaluation

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--log_level', type=str, default='INFO')
parser.add_argument('config_dir', type=str, help='The configuration directory for the repo containing the models')
parser.add_argument('work_dir', type=str, help='A working directory to unpack/clone the model repo')
parser.add_argument('jenkins_id', type=int,
                    help='The id of the current jenkins run')
parser.add_argument('--db_secret', type=str,
                    help='The name of the database credential secret loaded from AWS', default='brainscore-1-ohio-cred')
parser.add_argument('--models', type=str, nargs='*', default=None,
                    help='An optional list of the models to benchmark, if it does not exist all models are scored')
parser.add_argument('--benchmarks', type=str, nargs='*', default=None,
                    help='An optional list of the benchmarks to run, if it does not exist all benchmarks are run')
args, remaining_args = parser.parse_known_args()
logging.basicConfig(stream=sys.stdout, level=logging.getLevelName(args.log_level),
                    format='%(asctime)-15s %(levelname)s:%(name)s:%(message)s')
for disable_logger in ['s3transfer', 'botocore', 'boto3', 'urllib3', 'peewee', 'PIL']:
    logging.getLogger(disable_logger).setLevel(logging.WARNING)


def score_model_console():
    logger.info('Start scoring model process..')
    assert Path(args.config_dir).is_dir(), f'Configuration directory {args.config_dir} does not exist'
    assert Path(args.work_dir).is_dir(), f'Work directory {args.work_dir} is not a valid directory'
    assert args.db_secret is not None, 'The db connection file does not exist'
    assert isinstance(args.jenkins_id, int)
    logger.info(f'Benchmarks configured: {args.benchmarks}')
    logger.info(f'Models configured: {args.models}')
    run_evaluation(args.config_dir, args.work_dir, args.jenkins_id, db_secret=args.db_secret,
                   models=args.models, benchmarks=args.benchmarks)


logger.info(f"Running {' '.join(sys.argv)}")
fire.Fire(command='score_model_console')
