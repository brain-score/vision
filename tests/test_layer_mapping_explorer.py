"""Tests for the Layer Mapping Explorer tool."""

import csv
import json
import os
import tempfile

import pytest

from brainscore_vision.tools.layer_mapping_explorer import (
    suggest_mapping,
    save_outputs,
    explore_layer_mapping,
    batch_generate_mappings,
)


# ---------- Unit tests (no model loading) ----------

class TestImports:
    """Verify that internal imports resolve correctly (catches module vs function errors)."""

    def test_import_plugin_is_callable(self):
        from brainscore_core.plugin_management.import_plugin import import_plugin
        assert callable(import_plugin)


class TestSuggestMapping:

    def test_picks_highest_scoring_layer_per_region(self):
        scores = {
            'V1': {'layer.1': 0.3, 'layer.2': 0.8, 'layer.3': 0.5},
            'IT': {'layer.1': 0.9, 'layer.2': 0.4, 'layer.3': 0.6},
        }
        mapping = suggest_mapping(scores)
        assert mapping == {'V1': 'layer.2', 'IT': 'layer.1'}

    def test_single_layer(self):
        scores = {'V4': {'only_layer': 0.42}}
        mapping = suggest_mapping(scores)
        assert mapping == {'V4': 'only_layer'}

    def test_negative_scores(self):
        scores = {'IT': {'a': -0.1, 'b': -0.5}}
        mapping = suggest_mapping(scores)
        assert mapping == {'IT': 'a'}

    def test_all_regions(self):
        scores = {
            'V1': {'l1': 0.9, 'l2': 0.1},
            'V2': {'l1': 0.7, 'l2': 0.3},
            'V4': {'l1': 0.2, 'l2': 0.8},
            'IT': {'l1': 0.1, 'l2': 0.9},
        }
        mapping = suggest_mapping(scores)
        assert mapping == {'V1': 'l1', 'V2': 'l1', 'V4': 'l2', 'IT': 'l2'}


class TestSaveOutputs:

    @pytest.fixture
    def sample_data(self):
        scores = {
            'V1': {'features.2': 0.7, 'features.5': 0.3},
            'V4': {'features.2': 0.4, 'features.5': 0.8},
        }
        mapping = {'V1': 'features.2', 'V4': 'features.5'}
        return scores, mapping

    def test_creates_json_file(self, sample_data, tmp_path):
        scores, mapping = sample_data
        save_outputs(scores, mapping, str(tmp_path), 'test_model')
        json_path = tmp_path / 'region_layer_map.json'
        assert json_path.exists()

    def test_json_is_valid_and_matches_mapping(self, sample_data, tmp_path):
        scores, mapping = sample_data
        save_outputs(scores, mapping, str(tmp_path), 'test_model')
        with open(tmp_path / 'region_layer_map.json') as f:
            loaded = json.load(f)
        assert loaded == mapping

    def test_json_loadable_as_region_layer_map(self, sample_data, tmp_path):
        """The JSON format must be a plain {region: layer} dict, which is
        what ModelCommitment.load_region_layer_map_json returns."""
        scores, mapping = sample_data
        save_outputs(scores, mapping, str(tmp_path), 'test_model')
        with open(tmp_path / 'region_layer_map.json') as f:
            loaded = json.load(f)
        # Must be a flat dict of string -> string
        assert isinstance(loaded, dict)
        for region, layer in loaded.items():
            assert isinstance(region, str)
            assert isinstance(layer, str)
        # Must be indexable like region_layer_map[region]
        assert loaded['V1'] == 'features.2'
        assert loaded['V4'] == 'features.5'

    def test_creates_csv_file(self, sample_data, tmp_path):
        scores, mapping = sample_data
        save_outputs(scores, mapping, str(tmp_path), 'test_model')
        csv_path = tmp_path / 'test_model_layer_scores.csv'
        assert csv_path.exists()

    def test_csv_has_correct_columns(self, sample_data, tmp_path):
        scores, mapping = sample_data
        save_outputs(scores, mapping, str(tmp_path), 'test_model')
        with open(tmp_path / 'test_model_layer_scores.csv') as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header == ['layer', 'V1', 'V4']

    def test_csv_has_one_row_per_layer(self, sample_data, tmp_path):
        scores, mapping = sample_data
        save_outputs(scores, mapping, str(tmp_path), 'test_model')
        with open(tmp_path / 'test_model_layer_scores.csv') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            rows = list(reader)
        assert len(rows) == 2  # two layers
        layer_names = [row[0] for row in rows]
        assert 'features.2' in layer_names
        assert 'features.5' in layer_names

    def test_csv_score_values_are_correct(self, sample_data, tmp_path):
        scores, mapping = sample_data
        save_outputs(scores, mapping, str(tmp_path), 'test_model')
        with open(tmp_path / 'test_model_layer_scores.csv') as f:
            reader = csv.DictReader(f)
            rows = {row['layer']: row for row in reader}
        assert float(rows['features.2']['V1']) == 0.7
        assert float(rows['features.5']['V4']) == 0.8

    def test_creates_output_dir_if_missing(self, sample_data, tmp_path):
        nested = tmp_path / 'a' / 'b' / 'c'
        scores, mapping = sample_data
        save_outputs(scores, mapping, str(nested), 'test_model')
        assert (nested / 'region_layer_map.json').exists()

    def test_four_region_csv(self, tmp_path):
        scores = {
            'V1': {'l1': 0.1, 'l2': 0.2, 'l3': 0.3},
            'V2': {'l1': 0.4, 'l2': 0.5, 'l3': 0.6},
            'V4': {'l1': 0.7, 'l2': 0.8, 'l3': 0.9},
            'IT': {'l1': 0.15, 'l2': 0.25, 'l3': 0.35},
        }
        mapping = suggest_mapping(scores)
        save_outputs(scores, mapping, str(tmp_path), 'mymodel')
        with open(tmp_path / 'mymodel_layer_scores.csv') as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)
        assert header == ['layer', 'V1', 'V2', 'V4', 'IT']
        assert len(rows) == 3


# ---------- Integration test (needs model + benchmark data) ----------

@pytest.mark.slow
class TestExploreLayerMappingIntegration:
    """Integration tests using alexnet with public benchmarks.
    These tests are slow — they load a model and download benchmark data.
    Run with: pytest -m slow
    """

    def test_returns_correct_structure(self):
        scores = explore_layer_mapping(
            model_identifier='alexnet',
            layers=['features.2', 'features.5', 'features.12'],
            regions=['V4', 'IT'],
        )
        assert isinstance(scores, dict)
        assert set(scores.keys()) == {'V4', 'IT'}
        for region, layer_scores in scores.items():
            assert isinstance(layer_scores, dict)
            assert set(layer_scores.keys()) == {'features.2', 'features.5', 'features.12'}
            for layer, score in layer_scores.items():
                assert isinstance(score, float)

    def test_scores_are_plausible(self):
        scores = explore_layer_mapping(
            model_identifier='alexnet',
            layers=['features.2', 'features.12'],
            regions=['IT'],
        )
        for layer, score in scores['IT'].items():
            assert -1.0 <= score <= 1.0, f"Score {score} for {layer} out of range"

    def test_suggest_mapping_from_real_scores(self):
        scores = explore_layer_mapping(
            model_identifier='alexnet',
            layers=['features.2', 'features.12'],
            regions=['IT'],
        )
        mapping = suggest_mapping(scores)
        assert 'IT' in mapping
        assert mapping['IT'] in ('features.2', 'features.12')

    def test_full_pipeline_saves_files(self, tmp_path):
        scores = explore_layer_mapping(
            model_identifier='alexnet',
            layers=['features.2', 'features.12'],
            regions=['IT'],
        )
        mapping = suggest_mapping(scores)
        save_outputs(scores, mapping, str(tmp_path), 'alexnet')
        assert (tmp_path / 'region_layer_map.json').exists()
        assert (tmp_path / 'alexnet_layer_scores.csv').exists()
