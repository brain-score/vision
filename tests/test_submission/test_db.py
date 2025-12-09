from datetime import datetime

import pytest
from brainscore_core.submission.database import connect_db
from brainscore_core.submission.database_models import (
    BenchmarkType,
    Model,
    User,
    clear_schema,
    Submission
)


def init_user():
    User.create(id=1, email='test@brainscore.com', is_active=True, is_staff=False, is_superuser=False,
                last_login=datetime.now(), password='abcde')
    User.create(id=2, email='admin@brainscore.com', is_active=True, is_staff=True, is_superuser=True,
                last_login=datetime.now(), password='abcdef')


def init_benchmark_parents():
    BenchmarkType.create(identifier='average_vision', order=0, owner_id=2, domain ="vision")
    BenchmarkType.create(identifier='behavior', parent='average_vision', order=1, owner_id=2, domain ="vision")
    BenchmarkType.create(identifier='neural_vision', parent='average_vision', order=0, owner_id=2, domain ="vision")
    BenchmarkType.create(identifier='V1', parent='neural_vision', order=0, owner_id=2, domain ="vision")
    BenchmarkType.create(identifier='V2', parent='neural_vision', order=1, owner_id=2, domain ="vision")
    BenchmarkType.create(identifier='V4', parent='neural_vision', order=2, owner_id=2, domain ="vision")
    BenchmarkType.create(identifier='IT', parent='neural_vision', order=3, owner_id=2, domain ="vision")


def init_models():
    
    # Create a test submission
    submission = Submission.create(
        jenkins_id=1, 
        submitter_id=1, 
        model_type='artificialsubject',
        status='successful'
    )
    
    Model.create(name='dummy_model_1', owner=1, public=True, domain='vision', submission=submission)
    Model.create(name='dummy_model_2', owner=2, public=True, domain='vision', submission=submission)
    Model.create(name='dummy_model_3', owner=1, public=True, domain='vision', submission=submission)


@pytest.mark.memory_intense
@pytest.mark.private_access
@pytest.mark.parametrize('database', ['brainscore-ohio-test'])  # test database
def test_db(database):
    connect_db(database)
    clear_schema()
    init_user()
    init_benchmark_parents()
    init_models()

    user_entries = list(User.select())
    model_entries = list(Model.select())
    benchmark_type_entries = list(BenchmarkType.select())

    assert len(user_entries) == 2
    assert len(model_entries) == 3
    assert len(benchmark_type_entries) == 7
