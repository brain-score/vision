import logging

import numpy as np
import pytest
from pytest import approx

""" the mock import has to be before importing endpoints so that the database is properly mocked """
from .mock_config import test_database

from brainscore_core.submission import database_models
from brainscore_core.submission.database import connect_db
from brainscore_core.submission.database_models import clear_schema
from brainscore_vision.submission.endpoints import run_scoring

logger = logging.getLogger(__name__)


@pytest.mark.private_access
class TestRunScoring:
    @classmethod
    def setup_class(cls):
        logger.info('Connect to database')
        connect_db(test_database)
        clear_schema()

    def setup_method(self):
        logger.info('Initialize database entries')
        database_models.User.create(id=1, email='test@brainscore.com', password='abcde',
                                    is_active=True, is_staff=False, is_superuser=False, last_login='2022-10-14 9:25:00')

    def teardown_method(self):
        logger.info('Clean database')
        clear_schema()

    def test_successful_run(self):
        args_dict = {'jenkins_id': 62, 'user_id': 1, 'model_type': 'brainmodel',
                     'public': True, 'competition': None, 'specified_only': True,
                     'new_models': ['alexnet'], 'new_benchmarks': ['MajajHong2015public.IT-pls']}
        run_scoring(args_dict)
        score_entries = database_models.Score.select()
        score_entries = list(score_entries)
        assert len(score_entries) == 1
        score_entry = score_entries[0]
        assert score_entry.score_ceiled == approx(.5079817, abs=0.005)
        assert score_entry.comment.startswith('layers:')

    @pytest.mark.travis_slow
    def test_two_models_one_benchmark(self):
        args_dict = {'jenkins_id': 62, 'user_id': 1, 'model_type': 'brainmodel',
                     'public': True, 'competition': None, 'specified_only': True,
                     'new_models': ['pixels', 'alexnet'],
                     'new_benchmarks': ['MajajHong2015public.IT-pls']}
        run_scoring(args_dict)
        score_entries = database_models.Score.select()
        assert len(score_entries) == 2
        score_values = [entry.score_ceiled for entry in score_entries]
        assert all(np.array(score_values) > 0)

    @pytest.mark.travis_slow
    def test_one_model_two_benchmarks(self):
        args_dict = {'jenkins_id': 62, 'user_id': 1, 'model_type': 'brainmodel',
                     'public': True, 'competition': None, 'specified_only': True,
                     'new_models': ['alexnet'],
                     'new_benchmarks': ['MajajHong2015public.IT-pls', 'Rajalingham2018-i2n']}
        run_scoring(args_dict)
        score_entries = database_models.Score.select()
        assert len(score_entries) == 2
        score_values = [entry.score_ceiled for entry in score_entries]
        assert all(np.array(score_values) > 0)
        score_MajajHong = database_models.Score.get(benchmark__benchmark_type_id='MajajHong2015.IT.public-pls')
        assert score_MajajHong.score_ceiled == approx(.5079817, abs=0.005)
        score_Rajalingham = database_models.Score.get(benchmark__benchmark_type_id='Rajalingham2018-i2n')
        assert score_Rajalingham.score_ceiled == approx(.3701702, abs=0.005)

    @pytest.mark.travis_slow
    def test_two_models_two_benchmarks(self):
        args_dict = {'jenkins_id': 62, 'user_id': 1, 'model_type': 'brainmodel',
                     'public': True, 'competition': None, 'specified_only': True,
                     'new_models': ['pixels', 'alexnet'],
                     'new_benchmarks': ['MajajHong2015public.IT-pls', 'Rajalingham2018-i2n']}
        run_scoring(args_dict)
        score_entries = database_models.Score.select()
        assert len(score_entries) == 4
        score_values = [entry.score_ceiled for entry in score_entries]
        assert all(np.array(score_values) > 0)

    def test_benchmark_does_not_exist(self):
        args_dict = {'jenkins_id': 62, 'user_id': 1, 'model_type': 'brainmodel',
                     'public': True, 'competition': None, 'new_models': ['alexnet'],
                     'new_benchmarks': ['idonotexist'], 'specified_only': True}
        run_scoring(args_dict)
        score_entries = database_models.Score.select()
        assert len(score_entries) == 0

    def test_competition_field_set(self):
        args_dict = {'jenkins_id': 62, 'user_id': 1, 'model_type': 'brainmodel',
                     'public': True, 'competition': 'cosyne2022', 'new_models': ['alexnet'],
                     'new_benchmarks': ['MajajHong2015public.IT-pls'], 'specified_only': True}
        run_scoring(args_dict)
        model_entries = database_models.Model.select()
        assert len(model_entries) == 1
        assert model_entries[0].competition == "cosyne2022"

    def test_competition_field_not_set(self):
        args_dict = {'jenkins_id': 62, 'user_id': 1, 'model_type': 'brainmodel',
                     'public': True, 'competition': None, 'new_models': ['alexnet'],
                     'new_benchmarks': ['MajajHong2015public.IT-pls'], 'specified_only': True}
        run_scoring(args_dict)
        model_entries = database_models.Model.select()
        assert len(model_entries) == 1
        assert model_entries[0].competition is None
