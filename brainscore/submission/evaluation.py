import datetime
import json
import logging
from pathlib import Path

from pybtex.database.input import bibtex
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

all_benchmarks_list = [benchmark for benchmark in evaluation_benchmark_pool.keys()]

SCORE_COMMENT_MAX_LENGTH = 1000


def run_evaluation(config_dir, work_dir, jenkins_id, db_secret, models=None,
                   benchmarks=None):
    data = []
    try:
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
                    model_entry, created = Model.get_or_create(name=model_name, owner=submission_entry.submitter,
                                                               defaults={'public': submission_config.public,
                                                                         'submission': submission_entry})
                    if hasattr(module, 'get_bibtex') and created:
                        bibtex_string = module.get_bibtex(model_name)
                        reference = get_reference(bibtex_string)
                        model_entry.reference = reference
                        model_entry.save()
                    model_entries.append(model_entry)
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
    finally:
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
                    # Check if the model is already scored on the benchmark
                    score_entry, created = Score.get_or_create(benchmark=benchmark_entry, model=model_entry,
                                                               defaults={'start_timestamp': start, })
                    if not created and score_entry.score_raw is not None:
                        logger.warning(f'A score for model {model_id} and benchmark {benchmark_name} already exists')
                        raw = score_entry.score_raw
                        ceiled = score_entry.score_ceiled
                        error = score_entry.error
                        finished = score_entry.end_timestamp
                        comment = score_entry.comment
                    else:
                        if not created:
                            score_entry.start_timestamp = datetime.datetime.now()
                            score_entry.comment = None
                            logger.warning('An entry already exists but was not evaluated successful, we rerun!')
                        logger.info(f"Scoring {model_id}, id {model_entry.id} on benchmark {benchmark_name}")
                        model = ml_brain_pool[model_id]
                        score = score_model(model_id, benchmark_name, model)
                        logger.info(f'Running benchmark {benchmark_name} on model {model_id} (id {model_entry.id}) '
                                    f'produced this score: {score}')
                        if not hasattr(score, 'ceiling'):  # many engineering benchmarks do not have a primate ceiling
                            raw = score.sel(aggregation='center').item(0)
                            ceiled = None
                            error = None
                        else:  # score has a ceiling. Store ceiled as well as raw value
                            assert score.raw.sel(aggregation='center') is not None
                            raw = score.raw.sel(aggregation='center').item(0)
                            ceiled = score.sel(aggregation='center').item(0)
                            error = score.sel(aggregation='error').item(0)
                        finished = datetime.datetime.now()
                        comment = f"layers: {model.layer_model.region_layer_map}" \
                            if submission_entry.model_type == 'BaseModel' else ''
                        score_entry.end_timestamp = finished
                        score_entry.error = error
                        score_entry.score_ceiled = ceiled
                        score_entry.score_raw = raw
                        score_entry.comment = comment
                        score_entry.save()
                    result = {
                        'Model': model_id,
                        'Benchmark': benchmark_name,
                        'raw_result': raw,
                        'ceiled_result': ceiled,
                        'error': error,
                        'finished_time': finished,
                        'comment': comment,
                    }
                    data.append(result)
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
                        score_entry.comment = error if len(error) <= SCORE_COMMENT_MAX_LENGTH else \
                            error[:int(SCORE_COMMENT_MAX_LENGTH / 2) - 5] + ' [...] ' + \
                            error[-int(SCORE_COMMENT_MAX_LENGTH / 2) + 5:]
                        score_entry.save()
    finally:
        if success:
            submission_entry.status = 'successful'
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
            logger.exception(f'Could not connect benchmark {benchmark_name} to parent {benchmark.parent} '
                             f'since parent does not exist')
        if hasattr(benchmark, 'bibtex') and benchmark.bibtex is not None:
            bibtex_string = benchmark.bibtex
            ref = get_reference(bibtex_string)
            if ref:
                benchmark_type.reference = ref
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
    def parse_bib(bibtex_str):
        bib_parser = bibtex.Parser()
        entry = bib_parser.parse_string(bibtex_str)
        entry = entry.entries
        assert len(entry) == 1
        entry = list(entry.values())[0]
        return entry

    try:
        entry = parse_bib(bibtex_string)
        ref, create = Reference.get_or_create(url=entry.fields['url'],
                                              defaults={'bibtex': bibtex_string,
                                                        'author': entry.persons["author"][0].last()[0],
                                                        'year': entry.fields['year']})
        return ref
    except Exception:
        logger.exception('Could not load reference from bibtex string')
        return None
