# from pathlib import Path
#
# import pytest
# from pytest import approx
#
# from brainio.assemblies import BehavioralAssembly
# from brainscore import benchmark_pool
# from tests.test_benchmarks import PrecomputedFeatures
#
#  Jerry: Anything marked with FILL needs to be filled in!
# class TestJacob2020OcclusionDepthOrdering:
#
#     @pytest.mark.parametrize('benchmark', [
#         'FILL',
#         'FILL',
#     ])
#     def test_in_pool(self, benchmark):
#         FILL
#
#     @pytest.mark.parametrize('benchmark, expected_ceiling', [
#         ('FILL', approx(FILL, abs=0.0001)),
#         ('FILL', approx(FILL, abs=0.001)),
#     ])
#     def test_benchmark_ceiling(self, benchmark, expected_ceiling):
#         benchmark = benchmark_pool[benchmark]
#         ceiling = benchmark.ceiling
#         assert ceiling.sel(aggregation='center') == expected_ceiling
#
#     @pytest.mark.parametrize('benchmark, model, expected_raw_score', [
#         ('FILL', 'alexnet', approx(FILL, abs=0.0001)),
#         ('FILL', 'alexnet', approx(FILL, abs=0.0001)),
#         ('FILL', 'resnet-18', approx(FILL, abs=0.0001)),
#         ('FILL', 'resnet-18', approx(FILL, abs=0.0001)),
#     ])
#     def test_model_raw_score(self, benchmark, model, expected_raw_score):
#         extra = benchmark.split("_")[-1]
#         precomputed_features = Path(__file__).parent / \
#                                 f"model_identifier={model},benchmark_identifier={extra}-IT,visual_degrees=8.nc"
#         benchmark = benchmark_pool[benchmark]
#         precomputed_features = BehavioralAssembly.from_files(file_path=precomputed_features)
#         precomputed_features = PrecomputedFeatures(precomputed_features,
#                                                    visual_degrees=8.0,  # doesn't matter, features are already computed
#                                                    )
#         score = benchmark(precomputed_features)
#         raw_score = score.raw
#
#         # division by ceiling <= 1 should result in higher score
#         assert score.sel(aggregation='center') >= raw_score.sel(aggregation='center')
#         assert raw_score.sel(aggregation='center') == expected_raw_score
#
#     @pytest.mark.parametrize('benchmark, model, expected_ceiled_score', [
#         ('FILL', 'alexnet', approx(0.FILL, abs=0.0001)),
#         ('FILL', 'alexnet', approx(0.FILL, abs=0.0001)),
#         ('FILL', 'resnet-18', approx(FILL, abs=0.0001)),
#         ('FILL', 'resnet-18', approx(FILL, abs=0.0001)),
#     ])
#     def test_model_ceiled_score(self, benchmark, model, expected_ceiled_score):
#         # load features
#          extra = benchmark.split("_")[-1]
# #        precomputed_features = Path(__file__).parent / \
# #                                 f"model_identifier={model},benchmark_identifier={extra}-IT,visual_degrees=8.nc"
#         benchmark = benchmark_pool[benchmark]
#         precomputed_features = BehavioralAssembly.from_files(file_path=precomputed_features)
#         precomputed_features = PrecomputedFeatures(precomputed_features,
#                                                    visual_degrees=8.0,
#                                                    # doesn't matter, features are already computed
#                                                    )
#         score = benchmark(precomputed_features)
#         assert score[0] == expected_ceiled_score