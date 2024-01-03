# These tests ensure that the example notebooks execute without errors

import logging
from pathlib import Path

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor

_logger = logging.getLogger(__name__)


def run_notebook(notebook_path):
    _logger.info(f"Running {notebook_path}")
    with open(notebook_path, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    proc = ExecutePreprocessor(timeout=30 * 60, kernel_name='python3')
    proc.allow_errors = True
    proc.preprocess(nb, {'metadata': {'path': '/'}})

    errors = []
    for cell in nb.cells:
        if 'outputs' in cell:
            for output in cell['outputs']:
                if output.output_type == 'error':
                    errors.append(output)

    return nb, errors


@pytest.mark.memory_intense
@pytest.mark.parametrize('filename', [
    'score.ipynb',
    'data_metrics_benchmarks.ipynb',
    'models.ipynb',
])
def test_notebook(filename):
    filepath = Path(__file__).parent / '..' / 'examples' / filename
    nb, errors = run_notebook(filepath)
    assert not errors, f"Encountered errors: {errors}"
