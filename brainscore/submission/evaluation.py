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
from brainscore.submission.repository import prepare_module
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
    if isinstance(submission_config, MultiConfig):
        # We rerun existing models, which potentially are defined in different submissions
        for submission in submission_config.submissions.values():
            module = prepare_module(submission, submission_config)
            models = []
            for model in submission_config.models:
                if model.submission.id == submission.id:
                    models.append(model)
            assert len(models) > 0
            run_submission(module, models, test_benchmarks, submission, submission_config.jenkins_id)
    else:
        submission = submission_config.submission
        try:
            module = prepare_module(submission, submission_config)
            test_models = module.get_model_list() if models is None or len(models) == 0 else models
            assert len(test_models) > 0
            model_instances = []
            for model in test_models:
                reference = None
                if hasattr(module, 'get_bibtex'):
                    bibtex_string = module.get_bibtex(model)
                    reference = get_reference(bibtex_string)
                model_instances.append(
                    Model.create(name=model, owner=submission.submitter, public=submission_config.public,
                                 reference=reference, submission=submission))
            run_submission(module, model_instances, test_benchmarks, submission, submission.id)
        except Exception as e:
            submission.status = 'failure'
            submission.save()
            logging.error(f'Could not install submission because of following error')
            logging.error(e, exc_info=True)
            raise e


def run_submission(module, test_models, test_benchmarks, submission, jenkins_id):
    ml_brain_pool = get_ml_pool(test_models, module, submission)
    data = []
    success = True
    try:
        for modelInst in test_models:
            model_id = modelInst.name
            for benchmark in test_benchmarks:
                scoreInst = None
                try:
                    start = datetime.datetime.now()
                    bench_inst = get_benchmark_instance(benchmark)
                    scoreInst = Score.create(benchmark=bench_inst, start_timestamp=start, model=modelInst)
                    logger.info(f"Scoring {model_id} on benchmark {benchmark}")
                    model = ml_brain_pool[model_id]
                    score = score_model(model_id, benchmark, model)
                    logger.info(f'Running benchmark {benchmark} on model {model_id} produced this score: {score}')
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
                        model.layer_model.region_layer_map) if submission.model_type == 'BaseModel' else ''
                    result = {
                        'Model': model_id,
                        'Benchmark': benchmark,
                        'raw_result': raw,
                        'ceiled_result': ceiled,
                        'error': error,
                        'finished_time': finished,
                        'layer': layer_commitment
                    }
                    data.append(result)
                    scoreInst.end_timestamp = finished
                    scoreInst.error = error
                    scoreInst.score_ceiled = ceiled
                    scoreInst.score_raw = raw
                    scoreInst.save()
                except Exception as e:
                    success = False
                    error = f'Benchmark {benchmark} failed for model {model_id} because of this error: {e}'
                    logging.error(f'Could not run model {model_id} because of following error')
                    logging.error(e, exc_info=True)
                    data.append({
                        'Model': model_id, 'Benchmark': benchmark,
                        'raw_result': 0, 'ceiled_result': 0,
                        'error': error, 'finished_time': datetime.datetime.now()
                    })
                    if scoreInst:
                        scoreInst.comment = error
                        scoreInst.save()
    finally:
        df = pd.DataFrame(data)
        if success:
            submission.status = 'success'
        else:
            submission.status = 'failure'
        submission.save()
        # This is the result file we send to the user after the scoring process is done
        df.to_csv(f'result_{jenkins_id}.csv', index=None, header=True)


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


def get_benchmark_instance(benchmark):
    bench = benchmark_pool[benchmark]
    benchmark_type, created = BenchmarkType.get_or_create(identifier=benchmark, order=999)
    if created:
        try:
            parent = BenchmarkType.get(identifier=bench.parent)
            benchmark_type.parent = parent
            benchmark_type.save()
        except DoesNotExist:
            logger.error(
                f'Couldn\'t connect benchmark {benchmark} to parent {bench.parent} since parent doesn\'t exist')
        if hasattr(bench, 'bibtex') and bench.bibtex is not None:
            bibtex_string = bench.bibtex
            benchmark_type.reference = get_reference(bibtex_string)
            benchmark_type.save()
    bench_inst, created = BenchmarkInstance.get_or_create(benchmark=benchmark_type, version=bench.version)
    if created:
        # the version has changed and the benchmark instance was not yet in the database
        ceiling = bench.ceiling
        bench_inst.ceiling = ceiling.sel(aggregation='center')
        bench_inst.ceiling_error = ceiling.sel(aggregation='error')
        bench_inst.save()
    return bench_inst


def get_reference(bibtex_string):
    parsed = bibtexparser.loads(bibtex_string)
    entry = list(parsed.entries)[0]
    ref, create= Reference.get_or_create(bibtex=bibtex_string, author=entry['author'], url=entry['url'], year=entry['year'])
    return ref
