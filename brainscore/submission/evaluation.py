import zipfile
from importlib import import_module

import datetime
import git
import json
import logging
import os
import pandas as pd
import subprocess
import sys
from pathlib import Path

from brainscore import score_model
from brainscore.benchmarks import evaluation_benchmark_pool
from brainscore.submission.ml_pool import MLBrainPool, ModelLayers
from brainscore.utils import LazyLoad

logger = logging.getLogger(__name__)

all_benchmarks_list = [benchmark for benchmark in evaluation_benchmark_pool.keys()
                       if benchmark != 'dicarlo.Kar2019-ost']


def run_evaluation(config_file, work_dir, db_connection_config, jenkins_id, models=None,
                   benchmarks=None):
    config_file = Path(config_file).resolve()
    work_dir = Path(work_dir).resolve()
    db_conn = connect_db(db_connection_config)
    with open(config_file) as file:
        configs = json.load(file)
    logger.info(f'Run with following configurations: {str(configs)}')
    if configs['type'] == 'zip':
        config_path = config_file.parent
        logger.info('Start executing models in repo %s' % (configs['zip_filename']))
        repo = extract_zip_file(configs, config_path, work_dir)
    else:
        logger.info(f'Start executing models in repo {configs["git_url"]}')
        repo = clone_repo(configs, work_dir)
    package = 'models.brain_models' if configs['model_type'] == 'BrainModel' else 'models.base_models'
    module = install_project(repo, package)
    test_benchmarks = all_benchmarks_list if benchmarks is None or len(benchmarks) == 0 else benchmarks
    ml_brain_pool = {}
    test_models = module.get_model_list() if models is None or len(models) == 0 else models
    if configs['model_type'] == 'BaseModel':
        logger.info(f"Start working with base models")
        layers = {}
        base_model_pool = {}
        for model in test_models:
            function = lambda: module.get_model(model)
            base_model_pool[model] = LazyLoad(function)
            try:
                layers[model] = module.get_layers(model)
            except Exception:
                logging.warning(f'Could not retrieve layer for model {model} -- skipping model')
        model_layers = ModelLayers(layers)
        ml_brain_pool = MLBrainPool(base_model_pool, model_layers)
    else:
        logger.info(f"Start working with brain models")
        for model in test_models:
            ml_brain_pool[model] = module.get_model(model)
    data = []
    try:
        for model in test_models:
            for benchmark in test_benchmarks:
                try:
                    logger.info(f"Scoring {model} on benchmark {benchmark}")
                    score = score_model(model, benchmark, ml_brain_pool[model])
                    logger.info(f'Running benchmark {benchmark} on model {model} produced this score: {score}')
                    if not hasattr(score, 'ceiling'):
                        raw = score.sel(aggregation='center').item(0)
                        ceiled = None
                        error = None
                    else:
                        assert score.raw.sel(aggregation='center') is not None
                        raw = score.raw.sel(aggregation='center').item(0)
                        ceiled = score.sel(aggregation='center').item(0)
                        error = score.sel(aggregation='error').item(0)
                    finished = datetime.datetime.now()
                    result = {
                        'Model': model,
                        'Benchmark': benchmark,
                        'raw_result': raw,
                        'ceiled_result': ceiled,
                        'error': error,
                        'finished_time': finished
                    }
                    data.append(result)
                    store_score(db_conn, {**result, **{'jenkins_id': jenkins_id,
                                                       'email': configs['email'],
                                                       'name': configs['name']}})

                except Exception as e:
                    logging.error(f'Could not run model {model} because of following error')
                    logging.error(e, exc_info=True)
                    data.append({
                        'Model': model, 'Benchmark': benchmark,
                        'raw_result': 0, 'ceiled_result': 0,
                        'error': 0, 'finished_time': datetime.datetime.now()
                    })
    finally:
        df = pd.DataFrame(data)
        # This is the result file we send to the user after the scoring process is done
        df.to_csv(f'result_{jenkins_id}.csv', index=None, header=True)
        db_conn.close()


def connect_db(db):
    with open(db) as file:
        db_configs = json.load(file)
    import psycopg2
    return psycopg2.connect(host=db_configs['hostname'], user=db_configs['user_name'], password=db_configs['password'],
                            dbname=db_configs['database'])


def store_score(dbConnection, score):
    insert = '''insert into benchmarks_score
            (model, benchmark, score_raw, score_ceiled, error, timestamp, jenkins_job_id, user_id, name)   
            VALUES(%s,%s,%s,%s,%s,%s, %s, %s, %s)'''
    logging.info(f'Run results: {score}')
    cur = dbConnection.cursor()
    args = [score['Model'], score['Benchmark'],
            score['raw_result'], score['ceiled_result'],
            score['error'], score['finished_time'],
            score['jenkins_id'], score['email'],
            score['name']]
    cur.execute(insert, args)
    dbConnection.commit()
    return


def extract_zip_file(config, config_path, work_dir):
    zip_file = Path('%s/%s' % (config_path, config['zip_filename']))
    with zipfile.ZipFile(zip_file, 'r') as model_repo:
        model_repo.extractall(path=work_dir)
    #     Use the single directory in the zip file
    return Path('%s/%s' % (work_dir, os.listdir(work_dir)[0]))


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
