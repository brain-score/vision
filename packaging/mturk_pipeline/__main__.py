import logging
import sys
import os
import argparse
import fire
from mturk_custom_logger import setup_logger
from results_to_assembly import csv_to_assembly
from images_to_stimulus_set import path_to_stimulus_set
from tests.test_mturk_pipeline.test_benchmark_creation import TestStimulusSet, TestAssembly
import json

"""
This file will create a brain-score benchmark. It automatically packages a set of stimuli into a BrainIO stimulus set, 
converts a raw MTurk CSV (or any other CSV with specific columns) into a BrainIO assembly, and with an input metric
creates a benchmark.py file. Any three of these functions, images_to_stimulus_set, results_to_assembly, and 
assembly_to_benchmark can be called separately from the CLI or from a run configuration, such as Fire (default). 
A sample run (via PyCharm's run configurator) to create a stimulus set might look like this:

images_to_stimulus_set --stimulus_set_name test_set-1 --images_path images/test_set_1 --stimuli_type image --num_images 8


"""
logger = setup_logger()
parser = argparse.ArgumentParser()
parser.add_argument('--log_level', type=str, default='INFO')
args, remaining_args = parser.parse_known_args()
for disable_logger in ['s3transfer', 'botocore', 'boto3', 'urllib3', 'peewee', 'PIL', 'brainio', 'brainscore']:
    logging.getLogger(disable_logger).setLevel(logging.WARNING)
sys.stdout = open(os.devnull, 'w')

"""
Function to convert images, located at a certain path, package them into a BrainIO StimulusSet, test them, and 
upload them for storage into S3. Tests are automatically run after packaging and upload.

"""


def images_to_stimulus_set(parameter_config: str) -> None:
    """

     :param parameter_config: a JSON file of parameters to use in this function call. It will automatically parse
                             the stimulus_set's arguments and ignore others.
     :return: static method (None return type)
     """

    # Load parameters from JSON file and set them
    with open(parameter_config, 'r') as f:
        params = json.load(f)
    stimulus_set_params = params['stimulus_set']
    stimulus_set_name = stimulus_set_params['stimulus_set_name']
    stimuli_type = stimulus_set_params['stimuli_type']
    stimuli_path = stimulus_set_params['stimuli_path']
    num_stimuli = stimulus_set_params['num_stimuli']

    logger.debug(f"Converting MTurk raw images into stimulus_set...")
    stimulus_set = path_to_stimulus_set(stimulus_set_name=stimulus_set_name, stimuli_path=stimuli_path)
    tester = TestStimulusSet(name=stimulus_set_name, stimuli_type=stimuli_type, num_stimuli=num_stimuli)
    logger.info(f"Stimulus Set successfully created.")
    logger.debug(f"Testing newly created stimulus_set...")
    try:
        tester(stimulus_set)
    except AssertionError as e:
        logger.error(f"Error- stimulus set tests failed: \n\t {e.args[0]}")
        return
    logger.info(f"Stimulus Set tests passed.")
    logger.info(f"Done. Stimuli at {stimuli_path} converted to stimulus set, uploaded to BrainIO.")


"""
Function to automatically create an assembly from a .csv file of results. The column values are fixed, but can 
be changed via source code editing. Tests are run automatically after packaging. 

"""


def results_to_assembly(parameter_config: str) -> None:
    """

    :param parameter_config: a JSON file of parameters to use in this function call. It will automatically parse
                            the assembly's arguments and ignore others.
    :return: static method (None return type)
    """

    # Load parameters from JSON file and set them
    with open(parameter_config, 'r') as f:
        params = json.load(f)
    assembly_params = params['assembly']
    stimulus_set_name = assembly_params['stimulus_set_name']
    assembly_name = assembly_params['assembly_name']
    csv_file_path = assembly_params['csv_file_path']
    num_subjects = assembly_params['num_subjects']
    num_reps = assembly_params['num_reps']
    num_stimuli = assembly_params['num_stimuli']

    # creation and testing
    logger.debug(f"Converting MTurk Results at {csv_file_path} to BrainIO assembly...")
    assembly = csv_to_assembly(stimulus_set_name, assembly_name, csv_file_path)
    tester = TestAssembly(assembly_name=assembly_name, num_subjects=num_subjects, num_reps=num_reps,
                          num_stimuli=num_stimuli)
    logger.info(f"Assembly successfully created.")
    logger.debug(f"Testing newly created assembly...")
    try:
        tester(assembly)
    except AssertionError as e:
        logger.error(f"Assembly tests failed: \n\t {e.args[0]}")
        return
    logger.info(f"Assembly tests passed.")
    logger.info(f"Assembly creation finished. New assembly named {assembly_name} now on BrainIO.")

# needs to be done still
# def assembly_to_benchmark(model, *benchmarks):
#     scores = OrderedDict()
#     for benchmark in benchmarks:
#         _model = brain_translated_pool[model]
#         score = score_model_function(model_identifier=model, benchmark_identifier=benchmark, model=_model)
#         scores[benchmark] = score
#     print(scores)


logger.debug(f"Running {' '.join(sys.argv)}")
fire.Fire(command=remaining_args)
