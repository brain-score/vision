import brainio
import pytest
import tracemalloc
from brainscore_vision.benchmark_helpers import screen


@pytest.fixture
def brainio_home(tmp_path, monkeypatch):
    monkeypatch.setattr(brainio.fetch, "_local_data_path", str(tmp_path))
    yield tmp_path


@pytest.fixture
def resultcaching_home(tmp_path, monkeypatch):
    monkeypatch.setenv('RESULTCACHING_HOME', str(tmp_path))
    yield tmp_path


@pytest.fixture
def brainscore_home(tmp_path, monkeypatch):
    monkeypatch.setattr(screen, "root_path", tmp_path)
    yield tmp_path

@pytest.fixture(autouse=True)
def trace_memory():
    tracemalloc.start()
    yield # Run the test
    _, peak = tracemalloc.get_traced_memory()
    print(f"Max memory used: {peak / 1024 / 1024:.2f} MB")
    tracemalloc.stop()




