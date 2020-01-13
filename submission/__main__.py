import argparse
import logging
import os
import sys
from pathlib import Path

import fire

from submission.evaluation import run_evaluation

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--log_level', type=str, default='INFO')
parser.add_argument('config_file', type=str, help='The configuration file for the repo containing the models')
parser.add_argument('work_dir', type=str, help='A working directory to unpack/clone the model repo')
parser.add_argument('db_config', type=str,
                    help='A configuration file containing database details to write the output to')
parser.add_argument('jenkins_id', type=str,
                    help='The id of the current jenkins run')
parser.add_argument('--models', type=str, nargs='*', default=None,
                    help='An optional list of the models to benchmark, if it doesn\'t exist all models are socred')
parser.add_argument('--benchmarks', type=str, nargs='*', default=None,
                    help='An optional list of the benchmarks to run, if it doesn\'t exist all benchmarks are run')
args, remaining_args = parser.parse_known_args()
logging.basicConfig(stream=sys.stdout, level=logging.getLevelName(args.log_level),
                    format='%(asctime)-15s %(levelname)s:%(name)s:%(message)s')
for disable_logger in ['s3transfer', 'botocore', 'boto3', 'urllib3', 'peewee', 'PIL']:
    logging.getLogger(disable_logger).setLevel(logging.WARNING)


def score_model_console():
    logger.info('Start scoring model process..')
    assert Path(args.config_file).is_file(), 'Configuration file doesn\'t exist'
    assert Path(args.work_dir).is_dir(), 'Work directory is not a valid directory'
    assert Path(args.db_config).is_file(), 'The db connection file doesn\'t exist'
    logger.info(f'Benchmarks configured: {args.benchmarks}')
    logger.info(f'Models configured: {args.models}')
    run_evaluation(args.config_file, args.work_dir, args.db_config, args.jenkins_id,
                 models=args.models, benchmarks=args.benchmarks)


logger.info(f"Running {' '.join(sys.argv)}")
fire.Fire(command='score_model_console')