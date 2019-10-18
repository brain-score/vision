import argparse
import logging
import os
import sys

import fire

from submission.score_model import score_models

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--log_level', type=str, default='INFO')
parser.add_argument('config_file', type=str, help='The configuration file for the repo containing the models')
parser.add_argument('work_dir', type=str, help='A working directory to unpack/clone the model repo')
parser.add_argument('db_config', type=str,
                    help='A configuration file containing database details to write the output to')
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
    print('Start scoring model proces..')
    assert os.path.exists(args.config_file) or os.path.exists(os.path.realpath(args.config_file)), 'Configuration file doesn\'t exist'
    assert os.path.exists(args.work_dir) or os.path.exists(os.path.realpath(args.work_dir)), 'Work directory is not a valid directory'
    assert os.path.exists(args.db_config) or os.path.exists(os.path.realpath(args.db_config)), 'The db connection file doesn\'t exist'
    # assert args.models is list
    # assert args.benchmarks is list
    score_models(args.config_file, args.work_dir, args.db_config, all_models=(args.models is None),
                 all_benchmarks=(args.benchmarks is None),
                 models=args.models, benchmarks=args.benchmarks)


logger.info(f"Running {' '.join(sys.argv)}")
fire.Fire(command='score_model_console')