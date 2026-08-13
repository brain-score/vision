"""Tests for the activation-cache stimulus-identifier contract.

Regression guard for the Zerbe rdm/cka family, which lost caching entirely
because the benchmark synthesised an identifier that no registered plugin
matched. See :mod:`brainscore_vision.benchmark_helpers.cache_contract`.
"""
from __future__ import annotations

import os

import pytest

from brainscore_vision.benchmark_helpers.cache_contract import (
    assert_stimulus_identifier_is_cacheable, stimulus_identifier_is_cacheable)


class TestCacheableCheck:
    def test_registered_set_is_cacheable(self):
        assert stimulus_identifier_is_cacheable('Hebart2023') is True

    def test_unregistered_name_is_not(self):
        assert stimulus_identifier_is_cacheable('NotARegisteredStimulusSet') is False

    def test_derivative_of_a_registered_set_is_cacheable(self):
        """The `--<marker>` convention: marker stays in the key, base resolves."""
        assert stimulus_identifier_is_cacheable('Zerbe2026_fmri_stim_full--rdm-sub-01') is True

    def test_screen_suffix_composes_with_a_marker(self):
        """place_on_screen appends to whatever it is given, so both stack."""
        assert stimulus_identifier_is_cacheable(
            'Zerbe2026_fmri_stim_full--rdm-sub-01--target8.00--source9.20') is True

    def test_derivative_of_an_unregistered_base_is_not(self):
        """The convention must not manufacture revisions for unknown sets."""
        assert stimulus_identifier_is_cacheable('NotRegistered--marker') is False

    def test_the_exact_pre_fix_name_is_rejected(self):
        """`Zerbe2026_fmri_rdm_full_sub-01` is what silently disabled caching."""
        assert stimulus_identifier_is_cacheable('Zerbe2026_fmri_rdm_full_sub-01') is False


class TestAssertionMessage:
    def test_raises_with_actionable_guidance(self):
        with pytest.raises(AssertionError) as exc:
            assert_stimulus_identifier_is_cacheable('Zerbe2026_fmri_rdm_full_sub-01')
        message = str(exc.value)
        assert 'recomputed at full cost' in message
        assert '<registered identifier>--<marker>' in message

    def test_passes_silently_when_cacheable(self):
        assert_stimulus_identifier_is_cacheable('Hebart2023')  # must not raise


class TestCheckIsNotVacuous:
    """The check forces revisioning on. Without that it would pass for every
    identifier, since revisioning is opt-in and off by default -- a green test
    that guarantees nothing."""

    def test_still_detects_with_revisioning_unset(self, monkeypatch):
        monkeypatch.delenv('BRAINSCORE_CACHE_PLUGIN_REVISION', raising=False)
        assert stimulus_identifier_is_cacheable('NotARegisteredStimulusSet') is False

    def test_restores_the_env_var_it_borrowed(self, monkeypatch):
        monkeypatch.setenv('BRAINSCORE_CACHE_PLUGIN_REVISION', '0')
        stimulus_identifier_is_cacheable('Hebart2023')
        assert os.environ['BRAINSCORE_CACHE_PLUGIN_REVISION'] == '0'

    def test_leaves_no_var_behind_when_there_was_none(self, monkeypatch):
        monkeypatch.delenv('BRAINSCORE_CACHE_PLUGIN_REVISION', raising=False)
        stimulus_identifier_is_cacheable('Hebart2023')
        assert 'BRAINSCORE_CACHE_PLUGIN_REVISION' not in os.environ


class TestLaionFmriIdentifiers:
    """Every stimulus identifier the laion_fmri benchmarks synthesise.

    Kept in sync with the three `stim.identifier = ...` assignments in
    benchmarks/laion_fmri/benchmark.py.
    """

    PREFIX = 'Zerbe2026_fmri'

    @pytest.mark.parametrize('identifier', [
        f'{PREFIX}_stim_full--rdm-sub-01',          # merged tau pool, per subject
        f'{PREFIX}_stim_full--tau-train-sub-01',    # ridge split, persubject pool
        f'{PREFIX}_stim_full--tau-test',            # ridge split, shared pool
        f'{PREFIX}_stim_full--ood-test',
    ])
    def test_synthesised_identifiers_are_cacheable(self, identifier):
        assert_stimulus_identifier_is_cacheable(identifier)
