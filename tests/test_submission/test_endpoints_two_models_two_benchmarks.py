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
@pytest.mark.travis_slow
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
