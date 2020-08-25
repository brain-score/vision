import datetime
import json
import logging
from pathlib import Path

import bibtexparser as bibtexparser
import pandas as pd
from peewee import DoesNotExist

from brainscore import score_model
from brainscore.benchmarks import evaluation_benchmark_pool, benchmark_pool
from brainscore.submission.configuration import object_decoder, MultiConfig
from brainscore.submission.database import connect_db
from brainscore.submission.ml_pool import MLBrainPool, ModelLayers
from brainscore.submission.models import Model, Score, BenchmarkInstance, BenchmarkType, Reference
from brainscore.submission.repository import prepare_module, deinstall_project
from brainscore.utils import LazyLoad

logger = logging.getLogger(__name__)

all_benchmarks_list = [benchmark for benchmark in evaluation_benchmark_pool.keys()
                       if benchmark not in ['dicarlo.Kar2019-ost', 'fei-fei.Deng2009-top1']]


def run_evaluation(config_dir, work_dir, jenkins_id, db_secret, models=None,
                   benchmarks=None):
    connect_db(db_secret)
    config_file = Path(f'{config_dir}/submission_{jenkins_id}.json').resolve()
    with open(config_file) as file:
        configs = json.load(file)
    configs['config_file'] = str(config_file)
    submission_config = object_decoder(configs, work_dir, config_file.parent, db_secret, jenkins_id)

    logger.info(f'Run with following configurations: {str(configs)}')
    test_benchmarks = all_benchmarks_list if benchmarks is None or len(benchmarks) == 0 else benchmarks
    data = []
    if isinstance(submission_config, MultiConfig):
        # We rerun existing models, which potentially are defined in different submissions
        for submission_entry in submission_config.submission_entries.values():
            repo = None
            try:
                module, repo = prepare_module(submission_entry, submission_config)
                logger.info('Successfully installed repository')
                models = []
                for model_entry in submission_config.models:
                    if model_entry.submission.id == submission_entry.id:
                        models.append(model_entry)
                assert len(models) > 0
                sub_data = run_submission(module, models, test_benchmarks, submission_entry)
                data = data + sub_data
                deinstall_project(repo)
            except Exception as e:
                if repo is not None:
                    deinstall_project(repo)
                logging.error(f'Could not install submission because of following error')
                logging.error(e, exc_info=True)
                raise e
    else:
        submission_entry = submission_config.submission
        repo = None
        try:
            module, repo = prepare_module(submission_entry, submission_config)
            logger.info('Successfully installed repository')
            test_models = module.get_model_list() if models is None or len(models) == 0 else models
            assert len(test_models) > 0
            model_entries = []
            logger.info(f'Create model instances')
            for model_name in test_models:
                reference = None
                if hasattr(module, 'get_bibtex'):
                    bibtex_string = module.get_bibtex(model_name)
                    reference = get_reference(bibtex_string)
                model_entries.append(
                    Model.create(name=model_name, owner=submission_entry.submitter, public=submission_config.public,
                                 reference=reference, submission=submission_entry))
            data = run_submission(module, model_entries, test_benchmarks, submission_entry)
            deinstall_project(repo)
        except Exception as e:
            if repo is not None:
                deinstall_project(repo)
            submission_entry.status = 'failure'
            submission_entry.save()
            logging.error(f'Could not install submission because of following error')
            logging.error(e, exc_info=True)
            raise e
    df = pd.DataFrame(data)
    # This is the result file we send to the user after the scoring process is done
    df.to_csv(f'result_{jenkins_id}.csv', index=None, header=True)


def run_submission(module, test_models, test_benchmarks, submission_entry):
    ml_brain_pool = get_ml_pool(test_models, module, submission_entry)
    data = []
    success = True
    try:
        for model_entry in test_models:
            model_id = model_entry.name
            for benchmark_name in test_benchmarks:
                score_entry = None
                try:
                    start = datetime.datetime.now()
                    benchmark_entry = get_benchmark_instance(benchmark_name)
                    assert Score.get_or_none(benchmark=benchmark_entry, model=model_entry) is None
                    score_entry = Score.create(benchmark=benchmark_entry, start_timestamp=start, model=model_entry)
                    logger.info(f"Scoring {model_id} on benchmark {benchmark_name}")
                    model = ml_brain_pool[model_id]
                    score = score_model(model_id, benchmark_name, model)
                    logger.info(f'Running benchmark {benchmark_name} on model {model_id} produced this score: {score}')
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
                    layer_commitment = str(
                        model.layer_model.region_layer_map) if submission_entry.model_type == 'BaseModel' else ''
                    result = {
                        'Model': model_id,
                        'Benchmark': benchmark_name,
                        'raw_result': raw,
                        'ceiled_result': ceiled,
                        'error': error,
                        'finished_time': finished,
                        'comment': f"layers: {layer_commitment}"
                    }
                    data.append(result)
                    score_entry.end_timestamp = finished
                    score_entry.error = error
                    score_entry.score_ceiled = ceiled
                    score_entry.score_raw = raw
                    score_entry.save()
                except Exception as e:
                    success = False
                    error = f'Benchmark {benchmark_name} failed for model {model_id} because of this error: {e}'
                    logging.error(f'Could not run model {model_id} because of following error')
                    logging.error(e, exc_info=True)
                    data.append({
                        'Model': model_id, 'Benchmark': benchmark_name,
                        'raw_result': 0, 'ceiled_result': 0,
                        'error': error, 'finished_time': datetime.datetime.now()
                    })
                    if score_entry:
                        score_entry.comment = error
                        score_entry.save()
    finally:
        if success:
            submission_entry.status = 'success'
            logger.info(f'Submission is stored as successful')
        else:
            submission_entry.status = 'failure'
            logger.info(f'Submission was not entirely successful (some benchmarks could not be executed)')
        submission_entry.save()
        return data


def get_ml_pool(test_models, module, submission):
    ml_brain_pool = {}
    if submission.model_type == 'BaseModel':
        logger.info(f"Start working with base models")
        layers = {}
        base_model_pool = {}
        for model in test_models:
            function = lambda model_inst=model.name: module.get_model(model_inst)
            base_model_pool[model.name] = LazyLoad(function)
            try:
                layers[model.name] = module.get_layers(model.name)
            except Exception:
                logging.warning(f'Could not retrieve layer for model {model} -- skipping model')
        model_layers = ModelLayers(layers)
        ml_brain_pool = MLBrainPool(base_model_pool, model_layers)
    else:
        logger.info(f"Start working with brain models")
        for model in test_models:
            ml_brain_pool[model.name] = module.get_model(model.name)
    return ml_brain_pool


def get_benchmark_instance(benchmark_name):
    benchmark = benchmark_pool[benchmark_name]
    benchmark_type, created = BenchmarkType.get_or_create(identifier=benchmark_name, order=999)
    if created:
        try:
            parent = BenchmarkType.get(identifier=benchmark.parent)
            benchmark_type.parent = parent
            benchmark_type.save()
        except DoesNotExist:
            logger.error(
                f'Couldn\'t connect benchmark {benchmark_name} to parent {benchmark.parent} since parent doesn\'t exist')
        if hasattr(benchmark, 'bibtex') and benchmark.bibtex is not None:
            bibtex_string = benchmark.bibtex
            benchmark_type.reference = get_reference(bibtex_string)
            benchmark_type.save()
    bench_inst, created = BenchmarkInstance.get_or_create(benchmark=benchmark_type, version=benchmark.version)
    if created:
        # the version has changed and the benchmark instance was not yet in the database
        ceiling = benchmark.ceiling
        bench_inst.ceiling = ceiling.sel(aggregation='center')
        bench_inst.ceiling_error = ceiling.sel(aggregation='error')
        bench_inst.save()
    return bench_inst


def get_reference(bibtex_string):
    parsed = bibtexparser.loads(bibtex_string)
    entry = list(parsed.entries)[0]
    ref, create = Reference.get_or_create(bibtex=bibtex_string, author=entry.get('author', ''), url=entry.get('url' ,''),
                                          year=entry.get('year', ""))
    return ref
