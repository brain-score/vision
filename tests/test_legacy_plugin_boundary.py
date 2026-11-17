"""Enforcement check E2a: no unified plugin inside the legacy plugin folders, and no
test of this repository that imports the unified package.

Contract rule 11.2; Architecture and Ownership Map, section 6, E2a.

Signal: every ``.py`` file under the legacy plugin folders (``models``,
``benchmarks``, ``data``, ``metrics``) is parsed with ``ast`` (never
imported). A file fails when it

- imports the unified package ``brainscore`` (``import brainscore``,
  ``from brainscore[.x] import``; not ``brainscore_core`` or a domain package).
  The Brain-Score 1.x package has the same import name. A file whose every
  reference into ``brainscore`` is a 1.x name (``BRAINSCORE_1X``: names the
  unified package does not define) imports the 1.x package, not the unified one;
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

The allow-list holds today's violations; its target is empty. It may only
shrink: an entry that no longer violates fails the test, so it must be removed.
"""
import ast
from pathlib import Path

import pytest

import brainscore_vision as legacy_package

PLUGIN_FOLDERS = ('models', 'benchmarks', 'data', 'metrics')

# Names of the Brain-Score 1.x package, which had the import name of the unified
# package. The unified package defines none of them (checked below when it is installed).
BRAINSCORE_1X = ('brainscore.get_assembly', 'brainscore.get_stimulus_set',
                 'brainscore.benchmarks.screen', 'brainscore.benchmarks._neural_common')

# (path inside the package, signal): reason. The target is empty.
ALLOWED = {}


def _is_1x(reference):
    return any(reference == name or reference.startswith(name + '.') for name in BRAINSCORE_1X)


def _brainscore_references(tree):
    """Dotted names reached in ``brainscore``, or None when the package is not imported.

    ``from brainscore.a import b`` gives ``brainscore.a.b``; an explicit
    ``import brainscore.a`` gives ``brainscore.a`` (the imported module is
    evidence of its own); after an import each longest attribute chain on the
    bound name gives one name. A use of the bound name that is not an attribute
    chain (``getattr(brainscore, ...)``, passing the module) and a bare
    ``import brainscore`` that names nothing give ``brainscore``.
    """
    references, bound = [], {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split('.')[0] == 'brainscore':
                    bound[alias.asname or 'brainscore'] = alias.asname and alias.name or 'brainscore'
                    if alias.name != 'brainscore':
                        references.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module \
                and node.module.split('.')[0] == 'brainscore':
            references += [f'{node.module}.{alias.name}' for alias in node.names]
    if not bound and not references:
        return None
    chains, roots = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            base = node
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name) and base.id in bound:
                roots.add(id(base))
                text = ast.unparse(node)
                chains.add(bound[base.id] + text[len(base.id):])
    other_use = any(isinstance(node, ast.Name) and node.id in bound and id(node) not in roots
                    for node in ast.walk(tree))
    longest = {c for c in chains if not any(o.startswith(c + '.') for o in chains)}
    references += sorted(longest)
    if bound and (other_use or not longest):
        references.append('brainscore')
    return references


def unified_signals(source):
    """Set of E2a signals found in one source file."""
    tree = ast.parse(source)
    found = set()
    references = _brainscore_references(tree)
    if references is not None and not all(_is_1x(reference) for reference in references):
        found.add('import brainscore')
    for node in ast.walk(tree):
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


# Test files of this repository that import the unified package ``brainscore``: (path in tests/,
# reason). The target is empty: the dependency goes from unified to this package, never back,
# and the old Jenkins job runs these tests where the unified package is not installed.
ALLOWED_TESTS = {}


def unified_test_imports(tests_root):
    """Test files (paths relative to ``tests_root``) that import the unified package,
    at module level or inside a function (the same signal as for the plugin folders)."""
    root = Path(tests_root)
    return {path.relative_to(root).as_posix() for path in sorted(root.rglob('*.py'))
            if 'import brainscore' in unified_signals(path.read_text(encoding='utf-8', errors='replace'))}


def test_no_test_imports_the_unified_package():
    found = unified_test_imports(Path(__file__).parent)
    assert not sorted(found - set(ALLOWED_TESTS)), \
        f'tests that import the unified package (move them to unified): {sorted(found - set(ALLOWED_TESTS))}'
    assert not sorted(set(ALLOWED_TESTS) - found), \
        f'allow-list entries that no longer import it; remove them: {sorted(set(ALLOWED_TESTS) - found)}'


def test_the_test_check_finds_an_import_inside_a_test(tmp_path):
    (tmp_path / 'test_x.py').write_text('def test_x():\n    import brainscore\n    brainscore.score(1, 2)\n')
    (tmp_path / 'test_y.py').write_text('import brainscore_core\n')
    assert unified_test_imports(tmp_path) == {'test_x.py'}


def test_no_unified_plugin_in_legacy_folders():
    check(Path(legacy_package.__file__).parent, ALLOWED)


def test_check_fails_on_a_new_violation(tmp_path):
    (tmp_path / 'models' / 'new').mkdir(parents=True)
    (tmp_path / 'models' / 'new' / '__init__.py').write_text(
        "from brainscore import BrainScoreModel\nmodel_registry['x-unified'] = lambda: BrainScoreModel()\n")
    with pytest.raises(AssertionError, match='unified plugin code'):
        check(tmp_path, ALLOWED)


def test_check_fails_on_a_stale_entry(tmp_path):
    (tmp_path / 'models').mkdir()
    (tmp_path / 'models' / 'clean.py').write_text('import brainscore_core\n')
    with pytest.raises(AssertionError, match='no longer violate'):
        check(tmp_path, {('models/gone.py', 'import brainscore'): 'stale entry'})


@pytest.mark.parametrize('source, signal', [
    ('import brainscore', 'import brainscore'),
    ('from brainscore.benchmarks import x', 'import brainscore'),
    ('import brainscore\nbrainscore.load_model("x")', 'import brainscore'),
    ('import brainscore.tools\nbrainscore.get_assembly("x")', 'import brainscore'),
    ('import brainscore.tools as t\nimport brainscore\nbrainscore.get_assembly("x")', 'import brainscore'),
    ('import brainscore\nbrainscore.get_assembly("x")\ngetattr(brainscore, "tools")', 'import brainscore'),
    ('import brainscore as bs\nbs.get_stimulus_set("x")\nbs.load_benchmark("y")', 'import brainscore'),
    ('from brainscore import *', 'import brainscore'),
    ('import brainscore\nbrainscore.get_assembly("a")\nbrainscore.load_model("x")', 'import brainscore'),
    ('from brainscore.benchmarks.screen import place_on_screen\nfrom brainscore import load_model',
     'import brainscore'),
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
    'import brainscore\nassembly = brainscore.get_assembly("dicarlo.Sanghavi2021.domain_transfer")',
    'import brainscore\nfrom brainscore.benchmarks.screen import place_on_screen\n'
    'stimuli = brainscore.get_stimulus_set("dicarlo.domain_transfer")',
    'from brainscore.benchmarks._neural_common import average_repetition',
    'import brainscore.benchmarks.screen\nbrainscore.benchmarks.screen.place_on_screen(s)',
    'import brainscore as bs\nbs.get_assembly("x")',
])
def test_signal_allows(source):
    assert not unified_signals(source)


def test_the_unified_package_defines_no_brainscore_1x_name():
    """The 1.x names stay distinct while both packages are called ``brainscore``."""
    from importlib.util import find_spec
    spec = find_spec('brainscore')
    root = Path(list(spec.submodule_search_locations)[0]) if spec and spec.submodule_search_locations else None
    if root is None or not (root / 'brainscore_model.py').exists():
        pytest.skip('the unified package is not installed')
    top = {node.name for node in ast.parse((root / '__init__.py').read_text()).body
           if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    top |= {target.id for node in ast.parse((root / '__init__.py').read_text()).body
            if isinstance(node, ast.Assign) for target in node.targets if isinstance(target, ast.Name)}
    for name in BRAINSCORE_1X:
        parts = name.split('.')[1:]
        assert not (len(parts) == 1 and parts[0] in top), name
        assert not (root.joinpath(*parts).with_suffix('.py').exists()
                    or root.joinpath(*parts).is_dir()), name

