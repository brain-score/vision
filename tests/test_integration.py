import subprocess
import sys
from pathlib import Path

import pytest
from pytest import approx


def test_can_import():
    # noinspection PyUnresolvedReferences
    import brainscore_vision


@pytest.mark.parametrize("model_identifier, benchmark_identifier, expected_score", [
    ("pixels", "dicarlo.MajajHong2015public.IT-pls", approx(0.01538053, abs=0.0005)),
])
def test_model_benchmark_score(model_identifier, benchmark_identifier, expected_score):
    from brainscore_vision import score
    actual_score = score(model_identifier=model_identifier, benchmark_identifier=benchmark_identifier)
    assert actual_score == expected_score


def test_model_benchmark_commandline_score():
    process = subprocess.run(
        [
            sys.executable,
            "brainscore_vision",
            "score",
            "--model_identifier=pixels",
            "--benchmark_identifier=dicarlo.MajajHong2015public.IT-pls",
        ],
        cwd=Path(__file__).parent.parent,
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, "Process failed"
    assert "error" not in process.stderr.lower()
    output = process.stdout
    assert "Score" in output
    assert "0.10192326" in output
    assert "<xarray.Score ()>\narray(0.01538053)" in output
    assert "model_identifier:      pixels" in output
    assert "benchmark_identifier:  dicarlo.MajajHong2015public.IT-pls" in output
