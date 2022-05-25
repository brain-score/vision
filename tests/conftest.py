import brainio
import pytest

from brainscore.benchmarks import screen


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




