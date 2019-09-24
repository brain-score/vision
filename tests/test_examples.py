import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path


def run_notebook(notebook_path):
    print(f"Running {notebook_path}")
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    proc = ExecutePreprocessor(timeout=600, kernel_name='python3')
    proc.allow_errors = True
    proc.preprocess(nb, {'metadata': {'path': '/'}})

    errors = []
    for cell in nb.cells:
        if 'outputs' in cell:
            for output in cell['outputs']:
                if output.output_type == 'error':
                    errors.append(output)

    return nb, errors


@pytest.mark.parametrize('filename', [
    pytest.param('data.ipynb', marks=pytest.mark.memory_intense),
    pytest.param('metrics.ipynb', marks=pytest.mark.memory_intense),
    pytest.param('benchmarks.ipynb', marks=[]),
])
def test_notebook(filename):
    filepath = Path(__file__).parent / '..' / 'examples' / filename
    nb, errors = run_notebook(filepath)
    assert not errors, f"Encountered errors: {errors}"
