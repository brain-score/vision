"""Cache-key revisions for stored activations.

The property under test throughout: this module must never be able to fail a
scoring run. Every resolution path that cannot produce a revision degrades to
the bare identifier, which reproduces the behaviour that shipped before it.
"""
import subprocess
import unittest.mock as mock
from pathlib import Path

import pytest

from brainscore_vision.model_helpers.activations import cache_key
from brainscore_vision.model_helpers.activations.cache_key import (
    _content_revision, _git_revision, model_cache_identifier,
    stimulus_set_cache_identifier)


@pytest.fixture(autouse=True)
def _clear_revision_cache():
    """_plugin_revision is lru_cached per process; isolate the tests."""
    cache_key._plugin_revision.cache_clear()
    yield
    cache_key._plugin_revision.cache_clear()


@pytest.fixture
def revisioning_on(monkeypatch):
    """Revisioning is opt-in; most tests here are about what it does when on."""
    monkeypatch.setenv('BRAINSCORE_CACHE_PLUGIN_REVISION', '1')


class TestDisabledByDefault:
    """result_caching is enabled by default (RESULTCACHING_DISABLE defaults to
    '0'), so developers have warm local caches keyed on bare identifiers.
    Revisioning must not silently invalidate them."""

    def test_off_by_default(self, monkeypatch):
        monkeypatch.delenv('BRAINSCORE_CACHE_PLUGIN_REVISION', raising=False)
        monkeypatch.setenv('BRAINSCORE_MODEL_PLUGIN_SHA', 'a' * 40)
        assert model_cache_identifier('alexnet') == 'alexnet', \
            "keys must be byte-for-byte unchanged unless revisioning is opted into"
        assert stimulus_set_cache_identifier('Papale2025') == 'Papale2025'

    def test_off_does_not_even_look_up_the_plugin(self, monkeypatch):
        monkeypatch.delenv('BRAINSCORE_CACHE_PLUGIN_REVISION', raising=False)
        with mock.patch.object(cache_key, '_locate_plugin_dir') as locate:
            model_cache_identifier('alexnet')
            locate.assert_not_called()

    @pytest.mark.parametrize('value,expected_on', [
        ('1', True), ('true', True), ('TRUE', True), ('yes', True),
        ('0', False), ('false', False), ('', False), ('no', False),
    ])
    def test_flag_parsing(self, monkeypatch, value, expected_on):
        monkeypatch.setenv('BRAINSCORE_CACHE_PLUGIN_REVISION', value)
        assert cache_key.revision_enabled() is expected_on


class TestEnvironmentOverride:
    def test_env_var_wins_without_touching_git(self, revisioning_on, monkeypatch):
        monkeypatch.setenv('BRAINSCORE_MODEL_PLUGIN_SHA', 'a' * 40)
        with mock.patch.object(cache_key, '_locate_plugin_dir') as locate:
            assert model_cache_identifier('alexnet') == 'alexnet@' + 'a' * 12
            locate.assert_not_called(), "env var should short-circuit plugin lookup"

    def test_data_plugin_uses_its_own_env_var(self, revisioning_on, monkeypatch):
        monkeypatch.setenv('BRAINSCORE_DATA_PLUGIN_SHA', 'b' * 40)
        assert stimulus_set_cache_identifier('Papale2025') == 'Papale2025@' + 'b' * 12

    def test_model_env_var_does_not_leak_into_stimuli(self, revisioning_on, monkeypatch):
        """A shared env var would make every stimulus set key on the model."""
        monkeypatch.setenv('BRAINSCORE_MODEL_PLUGIN_SHA', 'a' * 40)
        monkeypatch.delenv('BRAINSCORE_DATA_PLUGIN_SHA', raising=False)
        with mock.patch.object(cache_key, '_locate_plugin_dir', return_value=None):
            assert stimulus_set_cache_identifier('Papale2025') == 'Papale2025'


class TestDegradesInsteadOfFailing:
    """None of these may raise, and none may invent a revision."""

    def test_unresolvable_plugin_returns_bare_identifier(self, revisioning_on, monkeypatch):
        monkeypatch.delenv('BRAINSCORE_MODEL_PLUGIN_SHA', raising=False)
        with mock.patch.object(cache_key, '_locate_plugin_dir', return_value=None):
            assert model_cache_identifier('nonexistent_model') == 'nonexistent_model'

    def test_locate_plugin_raising_is_swallowed(self, revisioning_on, monkeypatch):
        monkeypatch.delenv('BRAINSCORE_MODEL_PLUGIN_SHA', raising=False)
        from brainscore_core.plugin_management import import_plugin
        with mock.patch.object(import_plugin.ImportPlugin, 'locate_plugin',
                               side_effect=AssertionError("No registrations found")):
            assert model_cache_identifier('alexnet') == 'alexnet'

    def test_git_failure_falls_through_to_content_hash(self, revisioning_on, tmp_path, monkeypatch):
        monkeypatch.delenv('BRAINSCORE_MODEL_PLUGIN_SHA', raising=False)
        (tmp_path / 'model.py').write_text('weights = "v1"')
        with mock.patch.object(cache_key, '_locate_plugin_dir', return_value=tmp_path), \
             mock.patch.object(cache_key, '_git_revision', return_value=None):
            result = model_cache_identifier('alexnet')
        assert result.startswith('alexnet@') and len(result) == len('alexnet@') + 12

    def test_everything_failing_still_returns_identifier(self, revisioning_on, tmp_path, monkeypatch):
        monkeypatch.delenv('BRAINSCORE_MODEL_PLUGIN_SHA', raising=False)
        with mock.patch.object(cache_key, '_locate_plugin_dir', return_value=tmp_path), \
             mock.patch.object(cache_key, '_git_revision', return_value=None), \
             mock.patch.object(cache_key, '_content_revision', return_value=None):
            assert model_cache_identifier('alexnet') == 'alexnet'

    def test_falsy_identifier_passes_through(self, revisioning_on):
        """`stimuli_identifier=False` is how callers disable storing."""
        assert stimulus_set_cache_identifier(False) is False
        assert stimulus_set_cache_identifier(None) is None
        assert stimulus_set_cache_identifier('') == ''


class TestGitRevisionIsScopedToThePlugin:
    def test_uses_a_path_scoped_git_log(self, tmp_path):
        """Repo HEAD would invalidate every model on every unrelated commit."""
        captured = {}

        def fake_run(cmd, **kwargs):
            captured['cmd'] = cmd
            return subprocess.CompletedProcess(cmd, 0, stdout='c' * 40 + '\n', stderr='')

        with mock.patch('subprocess.run', side_effect=fake_run):
            assert _git_revision(tmp_path) == 'c' * 12
        assert captured['cmd'][:4] == ['git', 'log', '-1', '--format=%H']
        assert '--' in captured['cmd'] and str(tmp_path) in captured['cmd'], \
            "must be scoped to the plugin path, not the whole repository"

    def test_untracked_path_returns_none(self, tmp_path):
        """Empty git output means untracked; must not be read as a revision."""
        completed = subprocess.CompletedProcess([], 0, stdout='\n', stderr='')
        with mock.patch('subprocess.run', return_value=completed):
            assert _git_revision(tmp_path) is None

    def test_git_absent_returns_none(self, tmp_path):
        with mock.patch('subprocess.run', side_effect=FileNotFoundError('git')):
            assert _git_revision(tmp_path) is None


class TestContentRevisionTracksContent:
    def test_changing_a_source_file_changes_the_revision(self, tmp_path):
        f = tmp_path / 'model.py'
        f.write_text('weights = "v1"')
        first = _content_revision(tmp_path)
        f.write_text('weights = "v2"')
        assert _content_revision(tmp_path) != first, \
            "a revised plugin must not reuse the previous cache key"

    def test_identical_contents_give_identical_revisions(self, tmp_path):
        a, b = tmp_path / 'a', tmp_path / 'b'
        for d in (a, b):
            d.mkdir()
            (d / 'model.py').write_text('weights = "v1"')
        assert _content_revision(a) == _content_revision(b)

    def test_pycache_is_ignored(self, tmp_path):
        (tmp_path / 'model.py').write_text('weights = "v1"')
        before = _content_revision(tmp_path)
        cachedir = tmp_path / '__pycache__'
        cachedir.mkdir()
        (cachedir / 'model.cpython-311.pyc').write_bytes(b'\x00\x01compiled')
        assert _content_revision(tmp_path) == before, \
            "recompiling must not invalidate the cache"

    def test_filename_is_part_of_the_hash(self, tmp_path):
        (tmp_path / 'a.py').write_text('x = 1')
        first = _content_revision(tmp_path)
        (tmp_path / 'a.py').unlink()
        (tmp_path / 'b.py').write_text('x = 1')
        assert _content_revision(tmp_path) != first


class TestKeyActuallyChangesTheStoredIdentifier:
    """The point of the change: two revisions must not share a cache key."""

    def test_two_revisions_produce_different_keys(self, revisioning_on, tmp_path, monkeypatch):
        monkeypatch.delenv('BRAINSCORE_MODEL_PLUGIN_SHA', raising=False)
        (tmp_path / 'model.py').write_text('weights = "v1"')
        with mock.patch.object(cache_key, '_locate_plugin_dir', return_value=tmp_path), \
             mock.patch.object(cache_key, '_git_revision', return_value=None):
            key_v1 = model_cache_identifier('alexnet')
            cache_key._plugin_revision.cache_clear()
            (tmp_path / 'model.py').write_text('weights = "v2"')
            key_v2 = model_cache_identifier('alexnet')
        assert key_v1 != key_v2, "revising a plugin must land under a new cache key"
        assert key_v1.startswith('alexnet@') and key_v2.startswith('alexnet@')


class TestLogNoise:
    """Layer commitment calls from_paths repeatedly; an unresolved revision
    must not emit a line per call, and the routine stimulus-set case must not
    emit a warning at all."""

    def test_unresolved_model_warns_once_not_per_call(self, revisioning_on, monkeypatch, caplog):
        monkeypatch.delenv('BRAINSCORE_MODEL_PLUGIN_SHA', raising=False)
        with mock.patch.object(cache_key, '_locate_plugin_dir', return_value=None):
            with caplog.at_level('WARNING'):
                for _ in range(25):
                    model_cache_identifier('unresolvable_model')
        warnings = [r for r in caplog.records if r.levelname == 'WARNING']
        assert len(warnings) == 1, f"expected one warning, got {len(warnings)}"

    def test_unresolved_stimulus_set_does_not_warn(self, revisioning_on, monkeypatch, caplog):
        monkeypatch.delenv('BRAINSCORE_DATA_PLUGIN_SHA', raising=False)
        with mock.patch.object(cache_key, '_locate_plugin_dir', return_value=None):
            with caplog.at_level('WARNING'):
                stimulus_set_cache_identifier('SomeBenchmarkLocalStimuli')
        assert [r for r in caplog.records if r.levelname == 'WARNING'] == [], \
            "unresolvable stimulus sets are routine; warning on them is log noise"


class TestExtractorIntegration:
    """The enriched identifiers must reach the cache key and nothing else.

    `_from_paths_stored` uses `identifier` / `stimuli_identifier` solely to
    build the storage key -- it forwards neither into `_from_paths` -- so
    enriching them must not perturb the returned assembly. These use a stub
    extractor rather than a real model: the point is the call path, and
    loading models here would make the suite unrunnable locally.
    """

    def _stub_extractor(self):
        from brainscore_vision.model_helpers.activations.core import ActivationsExtractorHelper
        extractor = ActivationsExtractorHelper.__new__(ActivationsExtractorHelper)
        extractor.identifier = 'alexnet'
        extractor._logger = mock.Mock()
        extractor._from_paths_stored = mock.Mock(return_value='ASSEMBLY')
        extractor._from_paths = mock.Mock(return_value='ASSEMBLY')
        extractor._reduce_paths = lambda paths: paths
        extractor._expand_paths = lambda a, original_paths: a
        return extractor

    def test_enriched_identifiers_are_passed_to_the_stored_call(self, revisioning_on, monkeypatch):
        monkeypatch.setenv('BRAINSCORE_MODEL_PLUGIN_SHA', 'a' * 40)
        monkeypatch.setenv('BRAINSCORE_DATA_PLUGIN_SHA', 'b' * 40)
        extractor = self._stub_extractor()
        result = extractor.from_paths(stimuli_paths=['x.png'], layers=['fc'],
                                      stimuli_identifier='Papale2025')
        assert result == 'ASSEMBLY', "the returned assembly must be untouched"
        kwargs = extractor._from_paths_stored.call_args[1]
        assert kwargs['identifier'] == 'alexnet@' + 'a' * 12
        assert kwargs['stimuli_identifier'] == 'Papale2025@' + 'b' * 12

    def test_model_identifier_attribute_is_not_mutated(self, revisioning_on, monkeypatch):
        """Only the cache key carries the revision; self.identifier is used
        elsewhere (assembly metadata, logging) and must stay clean."""
        monkeypatch.setenv('BRAINSCORE_MODEL_PLUGIN_SHA', 'a' * 40)
        extractor = self._stub_extractor()
        extractor.from_paths(stimuli_paths=['x.png'], layers=['fc'],
                             stimuli_identifier='Papale2025')
        assert extractor.identifier == 'alexnet'

    def test_unstored_path_is_untouched(self, revisioning_on):
        """No stimuli_identifier -> no storing -> cache_key never involved."""
        extractor = self._stub_extractor()
        result = extractor.from_paths(stimuli_paths=['x.png'], layers=['fc'],
                                      stimuli_identifier=None)
        assert result == 'ASSEMBLY'
        extractor._from_paths_stored.assert_not_called()
        extractor._from_paths.assert_called_once()
