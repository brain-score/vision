import os
from datetime import datetime

import pytest
from _pytest import tmpdir

from brainscore.submission.database import connect_db
from brainscore.submission.evaluation import run_evaluation
from brainscore.submission.models import User, Score, Model, BenchmarkInstance, BenchmarkType, Submission, Reference


def init_user():
    User.create(id=1, email='test@brainscore.com', is_active=True, is_staff=False, is_superuser=False,
                last_login=datetime.now(), password='abcde')
    User.create(id=2, email='admin@brainscore.com', is_active=True, is_staff=True, is_superuser=True,
                last_login=datetime.now(), password='abcdef')


def init_benchmark():
    BenchmarkType.create(identifier='neural', order=0, )
    BenchmarkType.create(identifier='behavior', order=1, )


def clear_schema():
    Score.delete().execute()
    Model.delete().execute()
    Submission.delete().execute()
    BenchmarkInstance.delete().execute()
    BenchmarkType.delete().execute()
    Reference.delete().execute()
    User.delete().execute()


@pytest.mark.parametrize('database', ['brainscore-ohio-test', 'configs/django_generated.sqlit3'])
def test_evaluation( database):
    connect_db(database)
    clear_schema()
    init_user()
    working_dir = str(tmpdir.tmpdir('sub'))
    config_dir = str(os.path.join(os.path.dirname(__file__), 'configs/'))
    run_evaluation(config_dir, working_dir, 33, 'brainscore-ohio-test', models=['alexnet'],
                   benchmarks=['dicarlo.Majaj2015.IT-pls'])
    scores = Score.get().all()
    assert len(scores) == 1
    assert scores[0].comment is None  # If comment is none the score was successfully stored, otherwise there would be an error message there
