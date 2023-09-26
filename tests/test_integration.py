import pytest
import subprocess
import sys
from pathlib import Path
from pytest import approx

from brainscore_vision import score


@pytest.mark.parametrize("model_identifier, benchmark_identifier, expected_score", [
    ("pixels", "dicarlo.MajajHong2015.IT-pls", approx(0.36144805, abs=0.0005)),
])
def test_score(model_identifier, benchmark_identifier, expected_score):
    actual_score = score(model_identifier=model_identifier, benchmark_identifier=benchmark_identifier)
    assert actual_score == expected_score


def test_commandline_score():
    process = subprocess.run(
        [
            sys.executable,
            "brainscore_vision",
            "score",
            "--model_identifier=pixels",
            "--benchmark_identifier=dicarlo.MajajHong2015.IT-pls",
        ],
        cwd=Path(__file__).parent.parent,
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, "Process failed"
    assert "error" not in process.stderr.lower()
    output = process.stdout
    assert "Score" in output
    assert "0.0285" in output
    assert "<xarray.Score ()>\narray(0.0285022)" in output
    assert "model_identifier:      pixels" in output
    assert "benchmark_identifier:  dicarlo.MajajHong2015.IT-pls" in output
