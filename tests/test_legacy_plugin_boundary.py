"""Enforcement check E2a: no unified plugin inside the legacy plugin folders.

Contract rule 11.2; Architecture and Ownership Map, section 6, E2a.

Signal: every ``.py`` file under the legacy plugin folders (``models``,
``benchmarks``, ``data``, ``metrics``) is parsed with ``ast`` (never
imported). A file fails when it

- imports the unified package ``brainscore`` (``import brainscore``,
  ``from brainscore[.x] import``; not ``brainscore_core`` or a domain package);
- names ``BrainScoreModel``;
- assigns a registry key (``*_registry['...']``) that contains ``unified``;
- defines a class whose body declares ``in_channels``, ``out_channels`` and
  ``participate`` (a subject declared outside the legacy adapter; torch
  layers also declare the two channel names, so ``participate`` is required).

Cannot detect: a registration built through a helper in another package that
returns a unified subject; a computed registry key; a subject declared with
``in_channels=``/``out_channels=`` keyword arguments (these cannot be told
apart from ``torch.nn.Conv2d`` arguments); imports through ``importlib``.
The registries are not imported: importing every plugin needs model
libraries and weights.

The allow-list holds today's violations. It may only shrink: an entry that no
longer violates fails the test, so it must be removed.
"""
import ast
from pathlib import Path

import pytest

import brainscore_vision as legacy_package

PLUGIN_FOLDERS = ('models', 'benchmarks', 'data', 'metrics')

# (path inside the package, signal): reason
ALLOWED = {
    ('data/igustibagus2024/merged_assembly/create_merged_assembly_and_stim.py', 'import brainscore'):
        'Brain-Score 1.x packaging script (the old package was also named brainscore); same on master',
    ('data/igustibagus2024/merged_assembly/create_stimulus_set_with_background-id.py', 'import brainscore'):
        'Brain-Score 1.x packaging script (the old package was also named brainscore); same on master',
    ('data/igustibagus2024/merged_assembly/helpers_background_id.py', 'import brainscore'):
        'Brain-Score 1.x packaging script (the old package was also named brainscore); same on master',
}


def unified_signals(source):
    """Set of E2a signals found in one source file."""
    found = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            if any(alias.name.split('.')[0] == 'brainscore' for alias in node.names):
                found.add('import brainscore')
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module.split('.')[0] == 'brainscore':
                found.add('import brainscore')
        if (isinstance(node, ast.Name) and node.id == 'BrainScoreModel') or \
                (isinstance(node, ast.Attribute) and node.attr == 'BrainScoreModel') or \
                (isinstance(node, ast.alias) and node.name == 'BrainScoreModel'):
            found.add('BrainScoreModel')
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) \
                and isinstance(node.slice.value, str) and 'unified' in node.slice.value:
            target = node.value
            name = target.id if isinstance(target, ast.Name) else getattr(target, 'attr', '')
            if name.endswith('_registry'):
                found.add('unified registry key')
        if isinstance(node, ast.ClassDef):
            declared = set()
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    declared.add(item.name)
                elif isinstance(item, ast.Assign):
                    declared.update(t.id for t in item.targets if isinstance(t, ast.Name))
                elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    declared.add(item.target.id)
            if {'in_channels', 'out_channels', 'participate'} <= declared:
                found.add('subject declaration')
    return found


def violations(package_root):
    root = Path(package_root)
    return {(path.relative_to(root).as_posix(), signal)
            for folder in PLUGIN_FOLDERS if (root / folder).is_dir()
            for path in sorted((root / folder).rglob('*.py'))
            for signal in unified_signals(path.read_text(encoding='utf-8', errors='replace'))}


def check(package_root, allowed):
    found = violations(package_root)
    new = sorted(found - set(allowed))
    stale = sorted(set(allowed) - found)
    assert not new, f'unified plugin code in the legacy plugin folders: {new}'
    assert not stale, f'allow-list entries that no longer violate; remove them: {stale}'


def test_no_unified_plugin_in_legacy_folders():
    check(Path(legacy_package.__file__).parent, ALLOWED)


def test_check_fails_on_a_new_violation(tmp_path):
    for path, _ in ALLOWED:
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text('import brainscore\n')
    (tmp_path / 'models' / 'new').mkdir(parents=True)
    (tmp_path / 'models' / 'new' / '__init__.py').write_text(
        "from brainscore import BrainScoreModel\nmodel_registry['x-unified'] = lambda: BrainScoreModel()\n")
    with pytest.raises(AssertionError, match='unified plugin code'):
        check(tmp_path, ALLOWED)


def test_check_fails_on_a_stale_entry(tmp_path):
    (tmp_path / 'models').mkdir()
    (tmp_path / 'models' / 'clean.py').write_text('import brainscore_core\n')
    with pytest.raises(AssertionError, match='no longer violate'):
        check(tmp_path, ALLOWED)


@pytest.mark.parametrize('source, signal', [
    ('import brainscore', 'import brainscore'),
    ('from brainscore.benchmarks import x', 'import brainscore'),
    ('from brainscore import BrainScoreModel', 'BrainScoreModel'),
    ("benchmark_registry['MajajHong2015.IT-unified'] = f", 'unified registry key'),
    ('class S:\n    in_channels = ()\n    out_channels = ()\n    def participate(self, s): pass', 'subject declaration'),
])
def test_signal_catches(source, signal):
    assert signal in unified_signals(source)


@pytest.mark.parametrize('source', [
    'import brainscore_core',
    'from brainscore_vision import model_registry',
    "model_registry['alexnet'] = f",
    'nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3)',
    'class Conv:\n    in_channels = 3\n    out_channels = 8',
])
def test_signal_allows(source):
    assert not unified_signals(source)
