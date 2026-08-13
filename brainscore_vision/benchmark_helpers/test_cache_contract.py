"""Tests for the activation-cache stimulus-identifier contract.

Regression guard for the Zerbe rdm/cka family, which lost caching entirely
because the benchmark synthesised an identifier that no registered plugin
matched. See :mod:`brainscore_vision.benchmark_helpers.cache_contract`.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path

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


class TestNoBenchmarkSynthesisesAnUncacheableIdentifier:
    """Repo-wide sweep: every synthesised stimulus identifier must derive from a
    registered set.

    A per-plugin check would only cover the plugin someone thought to write it
    for. This sweep is what surfaced imagenet_c -- the slowest jobs in the suite
    -- silently recomputing activations on every run.

    Uses ``ast`` rather than a regex: Python merges adjacent string literals into
    a single node, so an identifier split across lines for width is read whole. A
    regex sees only the first fragment and reports a false offender.
    """

    def _assignments(self):
        root = Path(__file__).resolve().parent.parent / 'benchmarks'
        found = []
        for path in root.rglob('*.py'):
            if 'data_packaging' in str(path):
                continue
            try:
                tree = ast.parse(path.read_text(errors='replace'))
            except SyntaxError:  # pragma: no cover - not our contract to enforce
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Assign):
                    continue
                for target in node.targets:
                    if not (isinstance(target, ast.Attribute) and target.attr == 'identifier'):
                        continue
                    owner = getattr(target.value, 'id', '') or getattr(target.value, 'attr', '')
                    if 'stim' not in owner.lower():
                        continue
                    if isinstance(node.value, ast.JoinedStr):
                        # keep placeholder POSITIONS: joining only the constant
                        # fragments lets separators either side of a placeholder
                        # merge, so 'a-{x}-{y}' reads as containing '--'
                        literal = ''.join(
                            v.value if isinstance(v, ast.Constant) else '{}'
                            for v in node.value.values)
                        synthesised = any(isinstance(v, ast.FormattedValue)
                                          for v in node.value.values)
                        found.append((path.relative_to(root), owner, literal, synthesised))
        assert found, "no stimulus identifier assignments found -- has the layout changed?"
        return found

    def test_every_synthesised_identifier_carries_a_marker(self):
        offenders = [(str(path), owner, literal)
                     for path, owner, literal, synthesised in self._assignments()
                     # a placeholder-free name is a registered set used directly
                     if synthesised and '--' not in literal]
        assert not offenders, (
            "these benchmarks synthesise a stimulus identifier with no "
            "'--<marker>', so no data-plugin revision resolves, the shared "
            "activation cache refuses them, and every extraction recomputes at "
            "full cost:\n"
            + "\n".join(f"    {p}: {o}.identifier -> {lit!r}" for p, o, lit in offenders)
            + "\n  Name them '<registered identifier>--<marker>'.")

    def test_the_sweep_actually_finds_the_known_assignments(self):
        """A sweep that silently matches nothing would pass forever."""
        paths = {str(p) for p, _o, _l, _s in self._assignments()}
        for expected in ('laion_fmri/benchmark.py', 'allen2022_fmri/benchmark.py',
                         'imagenet_c/benchmark.py'):
            assert expected in paths, f"sweep no longer sees {expected}"
