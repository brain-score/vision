import os

import numpy as np
import pytest
from pytest import approx

from mkgu.assemblies import NeuroidAssembly
from mkgu.metrics.rdm import RSA, RDMSimilarity, RDMMetric
from tests.test_metrics import load_hvm


class TestRSA(object):
    def test_hvm(self):
        hvm_it_v6_obj = self._load_neural_data()
        assert hvm_it_v6_obj.shape == (64, 168)
        self._test_hvm(hvm_it_v6_obj)

    def test_hvm_T(self):
        hvm_it_v6_obj = self._load_neural_data().T
        assert hvm_it_v6_obj.shape == (168, 64)
        self._test_hvm(hvm_it_v6_obj)

    def _test_hvm(self, hvm_it_v6_obj):
        loaded = np.load(os.path.join(os.path.dirname(__file__), "it_rdm.p"), encoding="latin1")
        rsa_characterization = RSA()
        rsa = rsa_characterization(hvm_it_v6_obj)
        np.testing.assert_array_equal(rsa.shape, [64, 64])
        np.testing.assert_array_equal(rsa.dims, ['presentation', 'presentation'])
        assert rsa.values == approx(loaded, abs=1e-6)

    def _load_neural_data(self):
        data = load_hvm(group=lambda hvm: hvm.multi_groupby(["category_name", "object_name"]))
        data = data.sel(region='IT')
        return data


class TestRDMSimilarity(object):
    def test_2d_equal20(self):
        rdm = np.random.rand(20, 20)  # not mirrored across diagonal, but fine for unit test
        np.fill_diagonal(rdm, 0)
        rdm = NeuroidAssembly(
            rdm, coords={'presentation': list(range(20)), 'object_name': ('presentation', ['A', 'B'] * 10)},
            dims=['presentation', 'presentation'])
        similarity = RDMSimilarity()
        score = similarity(rdm, rdm)
        assert score == approx(1.)

    def test_2d_equal100(self):
        rdm = np.random.rand(100, 100)  # not mirrored across diagonal, but fine for unit test
        np.fill_diagonal(rdm, 0)
        rdm = NeuroidAssembly(
            rdm, coords={'presentation': list(range(100)), 'object_name': ('presentation', ['A', 'B'] * 50)},
            dims=['presentation', 'presentation'])
        similarity = RDMSimilarity()
        score = similarity(rdm, rdm)
        assert score == approx(1.)


class TestRDMMetric(object):
    @pytest.yield_fixture(autouse=True)
    def run_around_tests(self):
        self.metric = RDMMetric()
        yield  # run test function

    def test_equal_hvm(self):
        hvm = load_hvm().sel(region='IT')
        score = self.metric(hvm, hvm)
        assert score.center == approx(1.)

    def test_3d_equal(self):
        # presentation x neuroid
        values = np.broadcast_to(np.random.rand(20, 30, 1), [20, 30, 3]).copy()
        # can't stack because xarray gets confused with repeated dimensions
        assembly1 = NeuroidAssembly(values, coords={
            'image_id': ('presentation', list(range(20))), 'object_name': ('presentation', ['A', 'B'] * 10),
            'neuroid_id': ('neuroid', list(range(30))),
            'dim1': list(range(3))},
                                    dims=['presentation', 'neuroid', 'dim1'])
        assembly2 = NeuroidAssembly(values, coords={
            'image_id': ('presentation', list(range(20))), 'object_name': ('presentation', ['A', 'B'] * 10),
            'neuroid_id': ('neuroid', list(range(30))),
            'dim2': list(range(3))},
                                    dims=['presentation', 'neuroid', 'dim2'])
        score = self.metric(assembly1, assembly2)
        np.testing.assert_array_equal(score.center.shape, [3, 3])
        np.testing.assert_array_almost_equal(score.center, np.broadcast_to(1, [3, 3]))

    def test_3d_diag(self):
        # presentation x neuroid
        values = np.random.rand(20, 30, 3)
        assembly1 = NeuroidAssembly(values, coords={
            'image_id': ('presentation', list(range(20))), 'object_name': ('presentation', ['A', 'B'] * 10),
            'neuroid_id': ('neuroid', list(range(30))),
            'dim1': list(range(3))},
                                    dims=['presentation', 'neuroid', 'dim1'])
        assembly2 = NeuroidAssembly(values, coords={
            'image_id': ('presentation', list(range(20))), 'object_name': ('presentation', ['A', 'B'] * 10),
            'neuroid_id': ('neuroid', list(range(30))),
            'dim2': list(range(3))},
                                    dims=['presentation', 'neuroid', 'dim2'])
        score = self.metric(assembly1, assembly2)
        np.testing.assert_array_equal(score.center.shape, [3, 3])
        assert len(score.values['split']) * 3 == (score.values == approx(1)).sum()
        diags_indexer = score.center['dim1'] == score.center['dim2']
        diag_values = score.center.values[diags_indexer.values]
        np.testing.assert_array_almost_equal(diag_values, 1)

    def test_3d_equal_presentation_last(self):
        values = np.broadcast_to(np.random.rand(20, 30, 1), [20, 30, 3]).copy()
        assembly1 = NeuroidAssembly(values.T, coords={
            'image_id': ('presentation', list(range(20))), 'object_name': ('presentation', ['A', 'B'] * 10),
            'neuroid_id': ('neuroid', list(range(30))),
            'dim1': list(range(3))},
                                    dims=['dim1', 'neuroid', 'presentation'])
        assembly2 = NeuroidAssembly(values.T, coords={
            'image_id': ('presentation', list(range(20))), 'object_name': ('presentation', ['A', 'B'] * 10),
            'neuroid_id': ('neuroid', list(range(30))),
            'dim2': list(range(3))},
                                    dims=['dim2', 'neuroid', 'presentation'])
        scores = self.metric(assembly1, assembly2)
        np.testing.assert_array_equal(scores.center.shape, [3, 3])
        np.testing.assert_array_almost_equal(scores.center, 1)


def test_np_load():
    print(os.getcwd())
    p_path = os.path.join(os.path.dirname(__file__), "it_rdm.p")
    it_rdm = np.load(p_path, encoding="latin1")
    print(it_rdm)
    assert it_rdm.shape == (64, 64)
