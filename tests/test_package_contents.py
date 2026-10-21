"""Keep dataset builders independent of their authors' directory layouts."""
import ast
from pathlib import Path
import re

import brainscore_vision


def test_dataset_builders_have_no_personal_path_literals():
    root = Path(brainscore_vision.__file__).parent / 'data'
    personal = re.compile(r'/(?:Users|home|Volumes|braintree)/')
    failures = []
    for path in root.rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if personal.search(node.value):
                    failures.append(f'{path.relative_to(root)}:{node.lineno}')
    assert not failures, failures
