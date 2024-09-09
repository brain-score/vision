from brainscore_vision import score
import os

os.environ["RESULTCACHING_DISABLE"] = "1"


def test_score(model_identifier, benchmark_identifier, expected_score):
    actual_score = score(model_identifier=model_identifier, benchmark_identifier=benchmark_identifier,
                         conda_active=True)
    assert actual_score == expected_score


actual_score = score(model_identifier="resnet18_imagenet21kP", benchmark_identifier="NSD.V1.pls",
                         conda_active=True)
print(actual_score)