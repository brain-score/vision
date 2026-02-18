from brainscore_core import Metric
from brainscore_vision import load_metric, load_dataset
from brainscore_vision.benchmark_helpers.neural_common import NeuralBenchmark, average_repetition

import brainscore_vision.metrics.predictor_consistency
from brainscore_vision.metrics.predictor_consistency.ceiling import SplitHalfPredictorConsistency

VISUAL_DEGREES = 8
NUMBER_OF_TRIALS = 50
BIBTEX = """@article{muzellec2025reverse,
  title={Reverse Predictivity: Going Beyond One-Way Mapping to Compare Artificial Neural Network Models and Brains},
  author={Muzellec, Sabine and Kar, Kohitij},
  journal={bioRxiv},
  pages={2025--08},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}"""

crossvalidation_kwargs = dict(stratification_coord="object_name")

MY_METRIC_ID = "reverse_pls"


def _MajajHong2015PublicRegion(
    region: str,
    identifier_metric_suffix: str,
    similarity_metric: Metric,
):
    assembly_repetition = load_assembly(average_repetitions=False, region=region, access="public")
    assembly = load_assembly(average_repetitions=True, region=region, access="public")

    benchmark_identifier = f"MajajHong2015public.{region}"

    predictor_ceiler = SplitHalfPredictorConsistency()

    return NeuralBenchmark(
        identifier=f"{benchmark_identifier}-{identifier_metric_suffix}",
        version=4,
        assembly=assembly,
        similarity_metric=similarity_metric,
        visual_degrees=VISUAL_DEGREES,
        number_of_trials=NUMBER_OF_TRIALS,
        ceiling_func=lambda: predictor_ceiler(assembly_repetition),
        parent=region,
        bibtex=BIBTEX,
    )


def MajajHongV4PublicBenchmark():
    similarity_metric = load_metric(MY_METRIC_ID, crossvalidation_kwargs=crossvalidation_kwargs)
    return _MajajHong2015PublicRegion(
        region="V4",
        identifier_metric_suffix=MY_METRIC_ID,
        similarity_metric=similarity_metric,
    )


def MajajHongITPublicBenchmark():
    similarity_metric = load_metric(MY_METRIC_ID, crossvalidation_kwargs=crossvalidation_kwargs)
    return _MajajHong2015PublicRegion(
        region="IT",
        identifier_metric_suffix=MY_METRIC_ID,
        similarity_metric=similarity_metric,
    )


def load_assembly(average_repetitions: bool, region: str, access: str = "public"):
    assembly = load_dataset(f"MajajHong2015.{access}")

    if "time_bin" in assembly.dims:
        assembly = assembly.squeeze("time_bin")

    assembly = assembly.sel(region=region)
    assembly["region"] = ("neuroid", [region] * len(assembly["neuroid"]))

    assembly.load()
    assembly = assembly.transpose("presentation", "neuroid", ...)

    if average_repetitions:
        assembly = average_repetition(assembly)

    return assembly
