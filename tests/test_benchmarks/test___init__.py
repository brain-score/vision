import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest
from PIL import Image
from pytest import approx

from brainio.assemblies import BehavioralAssembly, NeuroidAssembly, PropertyAssembly
from brainscore_vision.benchmarks import benchmark_pool, public_benchmark_pool, evaluation_benchmark_pool, \
    engineering_benchmark_pool
from brainscore_vision.model_interface import BrainModel
from tests.test_benchmarks import PrecomputedFeatures


class TestPoolList:
    """ ensures that the right benchmarks are in the right benchmark pool """

    @pytest.mark.parametrize('benchmark', [
        'movshon.FreemanZiemba2013.V1-pls',
        'movshon.FreemanZiemba2013public.V1-pls',
        'dicarlo.MajajHong2015.IT-pls',
        'dicarlo.MajajHong2015public.IT-pls',
        'dicarlo.Rajalingham2018-i2n',
        'dicarlo.Rajalingham2018public-i2n',
        'fei-fei.Deng2009-top1',
    ])
    def test_contained_global(self, benchmark):
        assert benchmark in benchmark_pool

    @pytest.mark.parametrize('benchmark', [
        'movshon.FreemanZiemba2013public.V1-pls',
        'dicarlo.MajajHong2015public.IT-pls',
        'dicarlo.Rajalingham2018public-i2n',
        'fei-fei.Deng2009-top1',
    ])
    def test_contained_public(self, benchmark):
        assert benchmark in public_benchmark_pool

    # TODO: is this ever used? should something similar be made in test_helper
    def test_exact_evaluation_pool(self):
        assert set(evaluation_benchmark_pool.keys()) == {
            # V1
            'movshon.FreemanZiemba2013.V1-pls',
            # V2
            'movshon.FreemanZiemba2013.V2-pls',
            # V4
            'dicarlo.MajajHong2015.V4-pls',
            # IT
            'dicarlo.MajajHong2015.IT-pls',
            'dicarlo.Kar2019-ost',
            # behavior
            'dicarlo.Rajalingham2018-i2n',
            'brendel.Geirhos2021colour-error_consistency',
            'brendel.Geirhos2021contrast-error_consistency',
            'brendel.Geirhos2021cueconflict-error_consistency',
            'brendel.Geirhos2021edge-error_consistency',
            'brendel.Geirhos2021eidolonI-error_consistency',
            'brendel.Geirhos2021eidolonII-error_consistency',
            'brendel.Geirhos2021eidolonIII-error_consistency',
            'brendel.Geirhos2021falsecolour-error_consistency',
            'brendel.Geirhos2021highpass-error_consistency',
            'brendel.Geirhos2021lowpass-error_consistency',
            'brendel.Geirhos2021phasescrambling-error_consistency',
            'brendel.Geirhos2021powerequalisation-error_consistency',
            'brendel.Geirhos2021rotation-error_consistency',
            'brendel.Geirhos2021silhouette-error_consistency',
            'brendel.Geirhos2021stylized-error_consistency',
            'brendel.Geirhos2021sketch-error_consistency',
            'brendel.Geirhos2021uniformnoise-error_consistency',
        }

    def test_engineering_pool(self):
        assert set(engineering_benchmark_pool.keys()) == {
            'fei-fei.Deng2009-top1',
            'katz.BarbuMayo2019-top1',
            'dietterich.Hendrycks2019-noise-top1', 'dietterich.Hendrycks2019-blur-top1',
            'dietterich.Hendrycks2019-weather-top1', 'dietterich.Hendrycks2019-digital-top1',
            'brendel.Geirhos2021colour-top1',
            'brendel.Geirhos2021contrast-top1',
            'brendel.Geirhos2021cueconflict-top1',
            'brendel.Geirhos2021edge-top1',
            'brendel.Geirhos2021eidolonI-top1',
            'brendel.Geirhos2021eidolonII-top1',
            'brendel.Geirhos2021eidolonIII-top1',
            'brendel.Geirhos2021falsecolour-top1',
            'brendel.Geirhos2021highpass-top1',
            'brendel.Geirhos2021lowpass-top1',
            'brendel.Geirhos2021phasescrambling-top1',
            'brendel.Geirhos2021powerequalisation-top1',
            'brendel.Geirhos2021rotation-top1',
            'brendel.Geirhos2021silhouette-top1',
            'brendel.Geirhos2021stylized-top1',
            'brendel.Geirhos2021sketch-top1',
            'brendel.Geirhos2021uniformnoise-top1',
            'kornblith.Hermann2020cueconflict-shape_bias',
            'kornblith.Hermann2020cueconflict-shape_match',
        }


@pytest.mark.private_access
class TestStandardized:
    @pytest.mark.parametrize('benchmark, expected', [
        pytest.param('movshon.FreemanZiemba2013.V1-pls', approx(.873345, abs=.001),
                     marks=[pytest.mark.memory_intense]),
        pytest.param('movshon.FreemanZiemba2013.V2-pls', approx(.824836, abs=.001),
                     marks=[pytest.mark.memory_intense]),
        pytest.param('movshon.FreemanZiemba2013.V1-rdm', approx(.918672, abs=.001),
                     marks=[pytest.mark.memory_intense]),
        pytest.param('movshon.FreemanZiemba2013.V2-rdm', approx(.856968, abs=.001),
                     marks=[pytest.mark.memory_intense]),
        pytest.param('dicarlo.MajajHong2015.V4-pls', approx(.89503, abs=.001),
                     marks=pytest.mark.memory_intense),
        pytest.param('dicarlo.MajajHong2015.IT-pls', approx(.821841, abs=.001),
                     marks=pytest.mark.memory_intense),
        pytest.param('dicarlo.MajajHong2015.V4-rdm', approx(.936473, abs=.001),
                     marks=pytest.mark.memory_intense),
        pytest.param('dicarlo.MajajHong2015.IT-rdm', approx(.887618, abs=.001),
                     marks=pytest.mark.memory_intense),
        pytest.param('dicarlo.Rajalingham2020.IT-pls', approx(.561013, abs=.001),
                     marks=[pytest.mark.memory_intense, pytest.mark.slow]),
    ])
    def test_ceilings(self, benchmark, expected):
        benchmark = benchmark_pool[benchmark]
        ceiling = benchmark.ceiling
        assert ceiling.sel(aggregation='center') == expected

    @pytest.mark.parametrize('benchmark, visual_degrees, expected', [
        pytest.param('movshon.FreemanZiemba2013.V1-pls', 4, approx(.668491, abs=.001),
                     marks=[pytest.mark.memory_intense]),
        pytest.param('movshon.FreemanZiemba2013.V2-pls', 4, approx(.553155, abs=.001),
                     marks=[pytest.mark.memory_intense]),
        pytest.param('tolias.Cadena2017-pls', 2, approx(.577474, abs=.005),
                     marks=pytest.mark.private_access),
        pytest.param('dicarlo.MajajHong2015.V4-pls', 8, approx(.923713, abs=.001),
                     marks=pytest.mark.memory_intense),
        pytest.param('dicarlo.MajajHong2015.IT-pls', 8, approx(.823433, abs=.001),
                     marks=pytest.mark.memory_intense),
        pytest.param('dicarlo.Rajalingham2020.IT-pls', 8, approx(.693463, abs=.005),
                     marks=[pytest.mark.memory_intense, pytest.mark.slow]),
    ])
    def test_self_regression(self, benchmark, visual_degrees, expected):
        benchmark = benchmark_pool[benchmark]
        score = benchmark(PrecomputedFeatures(benchmark._assembly, visual_degrees=visual_degrees)).raw
        assert score.sel(aggregation='center') == expected
        raw_values = score.attrs['raw']
        assert hasattr(raw_values, 'neuroid')
        assert hasattr(raw_values, 'split')
        assert len(raw_values['split']) == 10

    @pytest.mark.parametrize('benchmark, visual_degrees, expected', [
        pytest.param('movshon.FreemanZiemba2013.V1-rdm', 4, approx(1, abs=.001),
                     marks=[pytest.mark.memory_intense]),
        pytest.param('movshon.FreemanZiemba2013.V2-rdm', 4, approx(1, abs=.001),
                     marks=[pytest.mark.memory_intense]),
        pytest.param('dicarlo.MajajHong2015.V4-rdm', 8, approx(1, abs=.001),
                     marks=pytest.mark.memory_intense),
        pytest.param('dicarlo.MajajHong2015.IT-rdm', 8, approx(1, abs=.001),
                     marks=pytest.mark.memory_intense),
    ])
    def test_self_rdm(self, benchmark, visual_degrees, expected):
        benchmark = benchmark_pool[benchmark]
        score = benchmark(PrecomputedFeatures(benchmark._assembly, visual_degrees=visual_degrees)).raw
        assert score.sel(aggregation='center') == expected
        raw_values = score.attrs['raw']
        assert hasattr(raw_values, 'split')
        assert len(raw_values['split']) == 10


@pytest.mark.private_access
class TestPrecomputed:
    @pytest.mark.memory_intense
    @pytest.mark.parametrize('benchmark, expected', [
        ('movshon.FreemanZiemba2013.V1-pls', approx(.466222, abs=.005)),
        ('movshon.FreemanZiemba2013.V2-pls', approx(.459283, abs=.005)),
    ])
    def test_FreemanZiemba2013(self, benchmark, expected):
        self.run_test(benchmark=benchmark, file='alexnet-freemanziemba2013.aperture-private.nc', expected=expected)

    @pytest.mark.memory_intense
    @pytest.mark.parametrize('benchmark, expected', [
        ('dicarlo.MajajHong2015.V4-pls', approx(.490236, abs=.005)),
        ('dicarlo.MajajHong2015.IT-pls', approx(.584053, abs=.005)),
    ])
    def test_MajajHong2015(self, benchmark, expected):
        self.run_test(benchmark=benchmark, file='alexnet-majaj2015.private-features.12.nc', expected=expected)

    @pytest.mark.memory_intense
    @pytest.mark.slow
    @pytest.mark.parametrize('benchmark, expected', [
        ('dicarlo.Rajalingham2020.IT-pls', approx(.147549, abs=.01)),
    ])
    def test_Rajalingham2020(self, benchmark, expected):
        self.run_test(benchmark=benchmark, file='alexnet-rajalingham2020-features.12.nc', expected=expected)

    def run_test(self, benchmark, file, expected):
        benchmark = benchmark_pool[benchmark]
        precomputed_features = Path(__file__).parent / file
        precomputed_features = NeuroidAssembly.from_files(
            precomputed_features,
            stimulus_set_identifier=benchmark._assembly.stimulus_set.identifier,
            stimulus_set=None)
        precomputed_features = precomputed_features.stack(presentation=['stimulus_path'])
        precomputed_paths = list(map(lambda f: Path(f).name, precomputed_features['stimulus_path'].values))
        # attach stimulus set meta
        stimulus_set = benchmark._assembly.stimulus_set
        expected_stimulus_paths = [stimulus_set.get_stimulus(image_id) for image_id in stimulus_set['stimulus_id']]
        expected_stimulus_paths = list(map(lambda f: Path(f).name, expected_stimulus_paths))
        assert set(precomputed_paths) == set(expected_stimulus_paths)
        for column in stimulus_set.columns:
            precomputed_features[column] = 'presentation', stimulus_set[column].values
        precomputed_features = PrecomputedFeatures(precomputed_features,
                                                   visual_degrees=10,  # doesn't matter, features are already computed
                                                   )
        # score
        score = benchmark(precomputed_features).raw
        assert score.sel(aggregation='center') == expected

    @pytest.mark.memory_intense
    @pytest.mark.private_access
    @pytest.mark.slow
    def test_Kar2019ost_cornet_s(self):
        benchmark = benchmark_pool['dicarlo.Kar2019-ost']
        precomputed_features = Path(__file__).parent / 'cornet_s-kar2019.nc'
        precomputed_features = NeuroidAssembly.from_files(
            precomputed_features,
            stimulus_set_identifier=benchmark._assembly.stimulus_set.identifier,
            stimulus_set=benchmark._assembly.stimulus_set)
        precomputed_features = PrecomputedFeatures(precomputed_features, visual_degrees=8)
        # score
        score = benchmark(precomputed_features).raw
        assert score.sel(aggregation='center') == approx(.316, abs=.005)

    def test_Rajalingham2018public(self):
        benchmark = benchmark_pool['dicarlo.Rajalingham2018public-i2n']
        # load features
        precomputed_features = Path(__file__).parent / 'CORnetZ-rajalingham2018public.nc'
        precomputed_features = BehavioralAssembly.from_files(
            precomputed_features,
            stimulus_set_identifier=benchmark._assembly.stimulus_set.identifier,
            stimulus_set=benchmark._assembly.stimulus_set)
        precomputed_features = PrecomputedFeatures(precomputed_features,
                                                   visual_degrees=8,  # doesn't matter, features are already computed
                                                   )
        # score
        score = benchmark(precomputed_features).raw
        assert score.sel(aggregation='center') == approx(.136923, abs=.005)

    @pytest.mark.memory_intense
    @pytest.mark.slow
    @pytest.mark.parametrize('benchmark, expected', [

    ])
    def test_Marques2020(self, benchmark, expected):
        self.run_test_properties(
            benchmark=benchmark,
            files={'dicarlo.Marques2020_blank': 'alexnet-dicarlo.Marques2020_blank.nc',
                   'dicarlo.Marques2020_receptive_field': 'alexnet-dicarlo.Marques2020_receptive_field.nc',
                   'dicarlo.Marques2020_orientation': 'alexnet-dicarlo.Marques2020_orientation.nc',
                   'dicarlo.Marques2020_spatial_frequency': 'alexnet-dicarlo.Marques2020_spatial_frequency.nc',
                   'dicarlo.Marques2020_size': 'alexnet-dicarlo.Marques2020_size.nc',
                   'movshon.FreemanZiemba2013_properties': 'alexnet-movshon.FreemanZiemba2013_properties.nc',
                   },
            expected=expected)

    def run_test_properties(self, benchmark, files, expected):
        benchmark = benchmark_pool[benchmark]
        from brainscore_vision import get_stimulus_set

        stimulus_identifiers = np.unique(np.array(['dicarlo.Marques2020_blank', 'dicarlo.Marques2020_receptive_field',
                                                   'dicarlo.Marques2020_orientation',
                                                   benchmark._assembly.stimulus_set.identifier]))
        precomputed_features = {}
        for current_stimulus in stimulus_identifiers:
            stimulus_set = get_stimulus_set(current_stimulus)
            path = Path(__file__).parent / files[current_stimulus]
            features = PropertyAssembly.from_files(path,
                                                   stimulus_set_identifier=stimulus_set.identifier,
                                                   stimulus_set=stimulus_set)
            features = features.stack(presentation=['stimulus_path'])
            precomputed_features[current_stimulus] = features
            precomputed_paths = [Path(f).name for f in precomputed_features[current_stimulus]['stimulus_path'].values]
            # attach stimulus set meta
            expected_stimulus_paths = [stimulus_set.get_stimulus(stimulus_id)
                                       for stimulus_id in stimulus_set['stimulus_id']]
            expected_stimulus_paths = list(map(lambda f: Path(f).name, expected_stimulus_paths))
            assert set(precomputed_paths) == set(expected_stimulus_paths)
            for column in stimulus_set.columns:
                precomputed_features[current_stimulus][column] = 'presentation', stimulus_set[column].values

        precomputed_features = PrecomputedFeatures(precomputed_features, visual_degrees=8)
        # score
        score = benchmark(precomputed_features).raw
        assert score.sel(aggregation='center') == expected


class TestVisualDegrees:
    @pytest.mark.parametrize('benchmark, candidate_degrees, image_id, expected', [
        pytest.param('movshon.FreemanZiemba2013.V1-pls', 14, 'c3a633a13e736394f213ddf44bf124fe80cabe07',
                     approx(.31429, abs=.0001), marks=[pytest.mark.private_access]),
        pytest.param('movshon.FreemanZiemba2013.V1-pls', 6, 'c3a633a13e736394f213ddf44bf124fe80cabe07',
                     approx(.22966, abs=.0001), marks=[pytest.mark.private_access]),
        pytest.param('movshon.FreemanZiemba2013public.V1-pls', 14, '21041db1f26c142812a66277c2957fb3e2070916',
                     approx(.314561, abs=.0001), marks=[]),
        pytest.param('movshon.FreemanZiemba2013public.V1-pls', 6, '21041db1f26c142812a66277c2957fb3e2070916',
                     approx(.23113, abs=.0001), marks=[]),
        pytest.param('movshon.FreemanZiemba2013.V2-pls', 14, 'c3a633a13e736394f213ddf44bf124fe80cabe07',
                     approx(.31429, abs=.0001), marks=[pytest.mark.private_access]),
        pytest.param('movshon.FreemanZiemba2013.V2-pls', 6, 'c3a633a13e736394f213ddf44bf124fe80cabe07',
                     approx(.22966, abs=.0001), marks=[pytest.mark.private_access]),
        pytest.param('movshon.FreemanZiemba2013public.V2-pls', 14, '21041db1f26c142812a66277c2957fb3e2070916',
                     approx(.314561, abs=.0001), marks=[]),
        pytest.param('movshon.FreemanZiemba2013public.V2-pls', 6, '21041db1f26c142812a66277c2957fb3e2070916',
                     approx(.23113, abs=.0001), marks=[]),
        pytest.param('dicarlo.MajajHong2015.V4-pls', 14, '40a786ed8e13db10185ddfdbe07759d83a589e1c',
                     approx(.251345, abs=.0001), marks=[pytest.mark.private_access]),
        pytest.param('dicarlo.MajajHong2015.V4-pls', 6, '40a786ed8e13db10185ddfdbe07759d83a589e1c',
                     approx(.0054886, abs=.0001), marks=[pytest.mark.private_access]),
        pytest.param('dicarlo.MajajHong2015public.V4-pls', 14, '8a72e2bfdb8c267b57232bf96f069374d5b21832',
                     approx(.25071, abs=.0001), marks=[]),
        pytest.param('dicarlo.MajajHong2015public.V4-pls', 6, '8a72e2bfdb8c267b57232bf96f069374d5b21832',
                     approx(.00460, abs=.0001), marks=[]),
        pytest.param('dicarlo.MajajHong2015.IT-pls', 14, '40a786ed8e13db10185ddfdbe07759d83a589e1c',
                     approx(.251345, abs=.0001), marks=[pytest.mark.private_access]),
        pytest.param('dicarlo.MajajHong2015.IT-pls', 6, '40a786ed8e13db10185ddfdbe07759d83a589e1c',
                     approx(.0054886, abs=.0001), marks=[pytest.mark.private_access]),
        pytest.param('dicarlo.MajajHong2015public.IT-pls', 14, '8a72e2bfdb8c267b57232bf96f069374d5b21832',
                     approx(.25071, abs=.0001), marks=[]),
        pytest.param('dicarlo.MajajHong2015public.IT-pls', 6, '8a72e2bfdb8c267b57232bf96f069374d5b21832',
                     approx(.00460, abs=.0001), marks=[]),
        pytest.param('dicarlo.Kar2019-ost', 14, '6d19b24c29832dfb28360e7731e3261c13a4287f',
                     approx(.225021, abs=.0001), marks=[pytest.mark.private_access]),
        pytest.param('dicarlo.Kar2019-ost', 6, '6d19b24c29832dfb28360e7731e3261c13a4287f',
                     approx(.001248, abs=.0001), marks=[pytest.mark.private_access]),
        pytest.param('dicarlo.Rajalingham2018-i2n', 14, '0223bf9e5db0edad21976b16494fe9396a5ef145',
                     approx(.225023, abs=.0001), marks=[pytest.mark.private_access]),
        pytest.param('dicarlo.Rajalingham2018-i2n', 6, '0223bf9e5db0edad21976b16494fe9396a5ef145',
                     approx(.002244, abs=.0001), marks=[pytest.mark.private_access]),
        pytest.param('dicarlo.Rajalingham2018public-i2n', 14, '0020cef91bd626e9fbbabd853494ee444e5c9ecb',
                     approx(.22486, abs=.0001), marks=[]),
        pytest.param('dicarlo.Rajalingham2018public-i2n', 6, '0020cef91bd626e9fbbabd853494ee444e5c9ecb',
                     approx(.00097, abs=.0001), marks=[]),
        pytest.param('tolias.Cadena2017-pls', 14, '0fe27ddd5b9ea701e380063dc09b91234eba3551', approx(.32655, abs=.0001),
                     marks=[pytest.mark.private_access]),
        pytest.param('tolias.Cadena2017-pls', 6, '0fe27ddd5b9ea701e380063dc09b91234eba3551', approx(.29641, abs=.0001),
                     marks=[pytest.mark.private_access]),
    ])
    def test_amount_gray(self, benchmark, candidate_degrees, image_id, expected, brainio_home, resultcaching_home,
                         brainscore_home):
        benchmark = benchmark_pool[benchmark]

        class DummyCandidate(BrainModel):
            class StopException(Exception):
                pass

            def visual_degrees(self):
                return candidate_degrees

            def look_at(self, stimuli, number_of_trials=1):
                image = stimuli.get_stimulus(image_id)
                image = Image.open(image)
                image = np.array(image)
                amount_gray = 0
                for index in np.ndindex(image.shape[:2]):
                    color = image[index]
                    gray = [128, 128, 128]
                    if (color == gray).all():
                        amount_gray += 1
                assert amount_gray / image.size == expected
                raise self.StopException()

            def start_task(self, task: BrainModel.Task, fitting_stimuli):
                pass

            def start_recording(self, recording_target: BrainModel.RecordingTarget, time_bins=List[Tuple[int]]):
                pass

        candidate = DummyCandidate()
        try:
            benchmark(candidate)  # just call to get the stimuli
        except DummyCandidate.StopException:  # but stop early
            pass


class TestNumberOfTrials:
    @pytest.mark.private_access
    @pytest.mark.parametrize('benchmark_identifier', [
        # V1
        'movshon.FreemanZiemba2013.V1-pls',
        # V2
        'movshon.FreemanZiemba2013.V2-pls',
        # V4
        'dicarlo.MajajHong2015.V4-pls',
        # IT
        'dicarlo.MajajHong2015.IT-pls',
        'dicarlo.Kar2019-ost',
        # behavior
        'dicarlo.Rajalingham2018-i2n',  # Geirhos2021 are single-trial, i.e. not included here
    ])
    def test_repetitions(self, benchmark_identifier):
        """ Tests that benchmarks have repetitions in the stimulus_set """
        benchmark = benchmark_pool[benchmark_identifier]

        class AssertRepeatCandidate(BrainModel):
            class StopException(Exception):
                pass

            def identifier(self) -> str:
                return 'assert-repeat-candidate'

            def visual_degrees(self):
                return 8

            def look_at(self, stimuli, number_of_trials=1):
                assert number_of_trials > 1
                raise self.StopException()

            def start_task(self, task: BrainModel.Task, fitting_stimuli):
                pass

            def start_recording(self, recording_target: BrainModel.RecordingTarget, time_bins=List[Tuple[int]]):
                pass

        candidate = AssertRepeatCandidate()
        try:
            benchmark(candidate)  # just call to get the stimuli
        except AssertRepeatCandidate.StopException:  # but stop early
            pass


def lstrip_local(path):
    parts = path.split(os.sep)
    brainio_index = parts.index('.brainio')
    path = os.sep.join(parts[brainio_index:])
    return path
