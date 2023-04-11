import logging
import os
import tempfile
from datetime import datetime

import pytest
from _pytest import tmpdir

from brainscore.submission.configuration import object_decoder, BaseConfig
from brainscore.submission.database import connect_db
from brainscore.submission.evaluation import get_reference, get_benchmark_instance, get_ml_pool, \
    run_submission
from brainscore.submission.models import Reference, BenchmarkType, Submission, Model, BenchmarkInstance, Score
from brainscore.submission.repository import prepare_module, extract_zip_file, find_submission_directory
from model_tools.brain_transformation import ModelCommitment
from tests.test_submission import base_model
from tests.test_submission.test_db import clear_schema, init_user

logger = logging.getLogger(__name__)
database = 'brainscore-ohio-test'  # test database


@pytest.mark.memory_intense
@pytest.mark.private_access
class TestSubmission:

    @classmethod
    def setup_class(cls):
        logger.info('Connect to database')
        connect_db(database)
        clear_schema()

    def setup_method(self):
        logger.info('Initialize database')
        init_user()

    def teardown_method(self):
        logger.info('Clean database')
        clear_schema()

    def test_get_reference(self):
        bibtex = """@Article{Freeman2013,
                                author={Freeman, Jeremy
                                and Ziemba, Corey M.
                                and Heeger, David J.
                                and Simoncelli, Eero P.
                                and Movshon, J. Anthony},
                                title={A functional and perceptual signature of the second visual area in primates},
                                journal={Nature Neuroscience},
                                year={2013},
                                month={Jul},
                                day={01},
                                volume={16},
                                number={7},
                                pages={974-981},
                                abstract={The authors examined neuronal responses in V1 and V2 to synthetic texture stimuli that replicate higher-order statistical dependencies found in natural images. V2, but not V1, responded differentially to these textures, in both macaque (single neurons) and human (fMRI). Human detection of naturalistic structure in the same images was predicted by V2 responses, suggesting a role for V2 in representing natural image structure.},
                                issn={1546-1726},
                                doi={10.1038/nn.3402},
                                url={https://doi.org/10.1038/nn.3402}
                                }
                            """
        ref = get_reference(bibtex)
        assert isinstance(ref, Reference)
        assert ref.url == 'https://doi.org/10.1038/nn.3402'
        assert ref.year == '2013'
        assert ref.author is not None
        ref2 = get_reference(bibtex)
        assert ref2.id == ref.id

    def test_get_benchmark_instance(self):
        instance = get_benchmark_instance('dicarlo.MajajHong2015.V4-pls')
        type = BenchmarkType.get(identifier=instance.benchmark)
        assert instance.ceiling is not None
        assert instance.ceiling_error is not None
        assert not type.parent
        BenchmarkType.create(identifier='IT', order=3)
        instance2 = get_benchmark_instance('dicarlo.MajajHong2015.IT-pls')
        assert instance2.benchmark.parent.identifier == 'IT'

    def get_test_models(self):
        submission = Submission.create(id=33, jenkins_id=33, submitter=1, timestamp=datetime.now(),
                                       model_type='BaseModel', status='running')
        model_instances = []
        model_instances.append(
            Model.create(name='alexnet', owner=submission.submitter, public=False,
                         submission=submission))
        return model_instances, submission

    def test_get_ml_pool(self):
        model_instances, submission = self.get_test_models()
        ml_pool = get_ml_pool(model_instances, base_model, submission)
        assert len(ml_pool) == 1
        assert isinstance(ml_pool['alexnet'], ModelCommitment)

    def test_run_submission(self):
        model_instances, submission = self.get_test_models()
        run_submission(base_model, model_instances, test_benchmarks=['dicarlo.MajajHong2015.IT-pls'],
                       submission_entry=submission)
        bench_inst = BenchmarkInstance.get(benchmark_type_id='dicarlo.MajajHong2015.IT-pls')
        assert not isinstance(bench_inst, list)
        assert Score.get(benchmark=bench_inst)


@pytest.mark.memory_intense
@pytest.mark.private_access
class TestConfig:
    @classmethod
    def setup_class(cls):
        logger.info('Connect to database')
        connect_db(database)
        clear_schema()
        init_user()

    @classmethod
    def teardown_class(cls):
        logger.info('Connect to database')
        clear_schema()

    def test_base_config(self):
        config = {"model_type": "BaseModel",
                  "user_id": 1,
                  "public": "False",
                  "competition": "cosyne2022"}
        submission_config = object_decoder(config, 'work_dir', 'config_path', 'db_secret', 33)
        assert submission_config.db_secret == 'db_secret'
        assert submission_config.work_dir == 'work_dir'
        assert submission_config.jenkins_id == 33
        assert submission_config.submission is not None
        assert not submission_config.public

    def test_resubmit_config(self):
        model = Model.create(id=19, name='alexnet', public=True, submission=33, owner=1)
        config = {
            "model_ids": [model.id],
            "user_id": 1,
            "competition": "cosyne2022"
        }
        submission_config = object_decoder(config, 'work_dir', 'config_path', 'db_secret', 33)
        assert len(submission_config.submission_entries) == 1
        assert len(submission_config.models) == 1
        assert submission_config.models[0].name == 'alexnet'


@pytest.mark.memory_intense
@pytest.mark.private_access
class TestRepository:
    working_dir = None
    config_dir = str(os.path.join(os.path.dirname(__file__), 'configs/'))

    @classmethod
    def setup_class(cls):
        connect_db(database)
        clear_schema()
        init_user()

    @classmethod
    def tear_down_class(cls):
        clear_schema()

    def setup_method(self):
        tmpdir = tempfile.mkdtemp()
        TestRepository.working_dir = tmpdir

    def tear_down_method(self):
        os.rmdir(TestRepository.working_dir)

    def test_prepare_module(self):
        config = BaseConfig(TestRepository.working_dir, 33, '', TestRepository.config_dir)
        submission = Submission.create(id=33, submitter=1, timestamp=datetime.now(),
                                       model_type='BaseModel', status='running')
        module, repo = prepare_module(submission, config)
        assert hasattr(module, 'get_model_list')
        assert hasattr(module, 'get_model')
        assert hasattr(module, 'get_bibtex')
        assert repo == 'candidate_models'
        assert module.get_model('alexnet') is not None

    def test_extract_zip_file(self):
        path = extract_zip_file(33, TestRepository.config_dir, TestRepository.working_dir)
        assert str(path) == f'{TestRepository.working_dir}/candidate_models'

    def test_find_correct_dir(self):
        f1 = open(f'{TestRepository.working_dir}/.temp', "w+")
        f2 = open(f'{TestRepository.working_dir}/_MACOS', "w+")
        f3 = open(f'{TestRepository.working_dir}/candidate_models', "w+")
        dir = find_submission_directory(TestRepository.working_dir)
        assert dir == 'candidate_models'
        exception = False
        try:
            f4 = open(f'{TestRepository.working_dir}/candidate_models2', "w+")
            dir = find_submission_directory(TestRepository.working_dir)
        except:
            exception = True
        assert exception
