import datetime
import json
import logging
import os
import subprocess
import sys
import zipfile
import git

from importlib import import_module, reload

from brainscore.utils import LazyLoad

from submission import score_model
from submission.ml_pool import MLBrainPool, ModelLayers

logger = logging.getLogger(__name__)

all_benchmarks_list = [
    'movshon.FreemanZiemba2013.V1-pls', 'movshon.FreemanZiemba2013.V2-pls',
    'movshon.FreemanZiemba2013.V1-rdm', 'movshon.FreemanZiemba2013.V2-rdm',
    'dicarlo.Majaj2015.V4-pls', 'dicarlo.Majaj2015.IT-pls',
    'dicarlo.Majaj2015.V4-rdm', 'dicarlo.Majaj2015.IT-rdm',
    'dicarlo.Rajalingham2018-i2n',
    'dicarlo.Kar2019-ost',
    'fei-fei.Deng2009-top1'
]


def score_models(config_file, work_dir, db_connection_config, jenkins_id, models=None,
                 benchmarks=None):
    config_file = config_file if os.path.isfile(config_file) else os.path.realpath(config_file)
    db_conn = connect_db(db_connection_config)
    with open(config_file) as file:
        configs = json.load(file)
    print(configs)
    if configs['type'] == 'zip':
        logger.info('Start executing models in repo %s %s' % (configs['zip_filepath'], configs['zip_filename']))
        repo = extract_zip_file(configs, work_dir)
    else:
        logger.info('Start executing models in repo %s' % (configs['git_url']))
        repo = clone_repo(configs, work_dir)
    package = 'models.brain_models' if configs['model_type'] is 'BrainModel' else 'models.base_models'
    module = install_project(repo, package)
    test_benchmarks = all_benchmarks_list if benchmarks is None or len(benchmarks) == 0 else benchmarks
    ml_brain_pool = {}
    if configs['model_type'] == 'BaseModel':
        test_models = module.base_models.get_model_list() if models is None or len(models) == 0 else models
        logger.info(f"Start working with base models")
        layers = {}
        base_model_pool = {}
        for model in test_models:
            function = lambda: module.base_models.get_model(model)
            base_model_pool[model] = LazyLoad(function)
            if module.base_models.get_layers != None:
                layers[model] = module.base_models.get_layers(model)
        model_layers = ModelLayers(layers)
        ml_brain_pool = MLBrainPool(base_model_pool, model_layers)
    else:
        logger.info(f"Start working with brain models")
        test_models = module.brain_models.get_model_list() if models is None or len(benchmarks) == 0 else models
        for model in test_models:
            ml_brain_pool[model] = module.brain_models.get_model(model)
    file = open('result.txt', 'w')

    file.write(f'Executed benchmarks in this order: {test_benchmarks}')
    try:
        for model in test_models:
            scores = []
            for benchmark in test_benchmarks:
                try:
                    logger.info(f"Scoring {model} on benchmark {benchmark}")
                    score = score_model(model, benchmark, ml_brain_pool[model])
                    scores.append(score.sel(aggregation='center').values)
                    logger.info(f'Running benchmark {benchmark} on model {model} produced this score: {score}')
                    store_score(db_conn, (model, benchmark, score.raw.sel(aggregation='center').item(0), score.sel(aggregation='center').item(0), score.sel(aggregation='error').item(0), datetime.datetime.now(), jenkins_id))
                except Exception as e:
                    logging.error(f'Could not run model {model} because of following error')
                    logging.error(e, exc_info=True)
            file.write(f'Results for model{model}: {str(scores)}')
    finally:
        file.close()
        db_conn.close()


def connect_db(db):
    with open(db) as file:
        db_configs = json.load(file)
    print(f'somethign1!!!{str(db_configs)}')
    import psycopg2
    return psycopg2.connect(host=db_configs['hostname'], user=db_configs['user_name'], password=db_configs['password'], dbname=db_configs['database'])


def store_score(dbConnection, score):
    insert = '''insert into benchmarks_score(model, benchmark, score_raw, score_ceiled, error, timestamp, jenkins_job_id)   
            VALUES(%s,%s,%s,%s,%s,%s, %s)'''
    print(score)
    cur = dbConnection.cursor()
    cur.execute(insert, score)
    dbConnection.commit()
    return


def extract_zip_file(config, work_dir):
    zip_file = '%s/%s' % (config['zip_filepath'], config['zip_filename'])
    zip_file = zip_file if os.path.isfile(zip_file) else os.path.realpath(zip_file)
    with zipfile.ZipFile(zip_file, 'r') as model_repo:
        model_repo.extractall(path=work_dir)
    #     Use the single directory in the zip file
    path = '%s/%s' % (work_dir, os.listdir(work_dir)[0])
    path = path if os.path.isfile(path) else os.path.realpath(path)
    return path


def clone_repo(config, work_dir):
    git.Git(work_dir).clone(config['git_url'])
    return '%s/%s' % (work_dir, os.listdir(work_dir)[0])


def install_project(repo,package):
    try:
        repo_name = repo.split('/')[-1]
        # subprocess.call([sys.executable, f"{repo}/setup.py", "install", f'--install-dir={git_install_dir}'])
        print(os.environ["PYTHONPATH"])
        subprocess.call([sys.executable, "-m", "pip", "install", repo], env=os.environ)
        sys.path.insert(1, repo)
        print(sys.path)
        return import_module(f'{repo_name}.{package}')
    except ImportError:
        return __import__(package)
