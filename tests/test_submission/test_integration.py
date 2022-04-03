import csv
import logging
import os
from datetime import datetime

import pytest

from brainscore.submission.database import connect_db
from brainscore.submission.evaluation import run_evaluation
from brainscore.submission.models import Score, Model, Submission
from tests.test_submission.test_db import clear_schema, init_user, init_benchmark_parents

logger = logging.getLogger(__name__)


#
#     Integration tests for the submission systems, executing 4 submissions:
#     1: ID:33 Working submission, executing one benchmark on Alexnet (zip + json)
#     2: ID:34 Rerunning Alexnet on another benchmark (only json)
#     3: ID:35 Failing installation submission (zip + json)
#     4: ID:36 Submission is installable, but model (Alexnet) is not scoreable (zip + json)
#

@pytest.mark.memory_intense
@pytest.mark.private_access
class TestIntegration:
    database = 'brainscore-ohio-test'  # test database

    @classmethod
    def setup_class(cls):
        logger.info('Connect to database')
        connect_db(TestIntegration.database)
        clear_schema()

    def setup_method(self):
        logger.info('Initialize database')
        init_user()
        init_benchmark_parents()

    def teardown_method(self):
        logger.info('Clean database')
        clear_schema()

    def compare(self, a, b):
        return abs(a - b) <= 0.0001

    def test_competition_field(self, tmpdir):
        working_dir = str(tmpdir.mkdir('sub'))
        config_dir = str(os.path.join(os.path.dirname(__file__), 'configs/'))
        run_evaluation(config_dir, working_dir, 33, TestIntegration.database, models=['alexnet'],
                       benchmarks=['dicarlo.MajajHong2015.IT-pls'])
        model = Model.get()
        assert model.competition == "cosyne2022"

    def test_competition_field_none(self, tmpdir):
        working_dir = str(tmpdir.mkdir('sub'))
        config_dir = str(os.path.join(os.path.dirname(__file__), 'configs/'))
        submission = Submission.create(id=33, submitter=1, timestamp=datetime.now(),
                                       model_type='BaseModel', status='running')
        model = Model.create(name='alexnet', owner=submission.submitter, public=False,
                             submission=submission)
        with open(f'{config_dir}submission_40.json', 'w') as rerun:
            rerun.write(f"""{{
                "model_ids": [{model.id}], "user_id": 1, "competition": null}}""")
        run_evaluation(config_dir, working_dir, 40, TestIntegration.database,
                       benchmarks=['dicarlo.Rajalingham2018-i2n'])
        model = Model.get()
        assert model.competition is None

    def test_evaluation(self, tmpdir):
        working_dir = str(tmpdir.mkdir('sub'))
        config_dir = str(os.path.join(os.path.dirname(__file__), 'configs/'))
        run_evaluation(config_dir, working_dir, 33, TestIntegration.database, models=['alexnet'],
                       benchmarks=['dicarlo.MajajHong2015.IT-pls'])
        with open('result_33.csv') as results:
            csv_reader = csv.reader(results, delimiter=',')
            next(csv_reader)  # header row
            result_row = next(csv_reader)
            assert result_row[0] == 'alexnet'
            assert result_row[1] == 'dicarlo.MajajHong2015.IT-pls'
            assert self.compare(float(result_row[2]), 0.5857491098187586)
            assert self.compare(float(result_row[3]), 0.5079816726934638)
            assert self.compare(float(result_row[4]), 0.003155449372125895)
        scores = Score.select()
        assert len(scores) == 1
        # successful score comment should inform about which layers were used for which regions
        assert scores[0].comment.startswith("layers:")

    def test_rerun_evaluation(self, tmpdir):
        working_dir = str(tmpdir.mkdir('sub'))
        config_dir = str(os.path.join(os.path.dirname(__file__), 'configs/'))
        submission = Submission.create(id=33, submitter=1, timestamp=datetime.now(),
                                       model_type='BaseModel', status='running')
        model = Model.create(name='alexnet', owner=submission.submitter, public=False,
                             submission=submission)
        with open(f'{config_dir}submission_34.json', 'w') as rerun:
            rerun.write(f"""{{
            "model_ids": [{model.id}], "user_id": 1}}""")
        run_evaluation(config_dir, working_dir, 34, TestIntegration.database,
                       benchmarks=['dicarlo.Rajalingham2018-i2n'])
        with open('result_34.csv') as results:
            csv_reader = csv.reader(results, delimiter=',')
            next(csv_reader)  # header row
            result_row = next(csv_reader)
            assert result_row[0] == 'alexnet'
            assert result_row[1] == 'dicarlo.Rajalingham2018-i2n'
            assert self.compare(float(result_row[2]), 0.25771746331458695)
            assert self.compare(float(result_row[3]), 0.3701702418190641)
            assert self.compare(float(result_row[4]), 0.011129032024657565)

    def test_failure_evaluation(self, tmpdir):
        working_dir = str(tmpdir.mkdir('sub'))
        config_dir = str(os.path.join(os.path.dirname(__file__), 'configs/'))
        with pytest.raises(Exception):
            run_evaluation(config_dir, working_dir, 35, TestIntegration.database, models=['alexnet'],
                           benchmarks=['dicarlo.Rajalingham2018-i2n'])

    def test_model_failure_evaluation(self, tmpdir):
        # os.environ['RESULTCACHING_DISABLE'] = 'brainscore.score_model,model_tools'
        working_dir = str(tmpdir.mkdir('sub'))
        config_dir = str(os.path.join(os.path.dirname(__file__), 'configs/'))
        run_evaluation(config_dir, working_dir, 36, TestIntegration.database, models=['alexnet'],
                       benchmarks=['movshon.FreemanZiemba2013.V1-pls'])
        with open('result_36.csv') as results:
            csv_reader = csv.reader(results, delimiter=',')
            next(csv_reader)  # header row
            result_row = next(csv_reader)
            assert result_row[0] == 'alexnet'
            assert result_row[1] == 'movshon.FreemanZiemba2013.V1-pls'
            assert result_row[2] == '0'
            assert result_row[3] == '0'
        model = Model.get()
        score = Score.get(model=model)
        assert score.comment is not None  # When there's a problem, the comment field contains an error message
        # os.environ['RESULTCACHING_DISABLE'] = '0'
