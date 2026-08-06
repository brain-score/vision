"""Content revisions for the stored-activation cache key.

``ActivationsExtractorHelper._from_paths_stored`` is decorated with
``@store_xarray``, which builds its cache key from the call arguments —
principally ``identifier`` (the model) and ``stimuli_identifier``. Neither of
those changes when the *contents* behind them change: a model plugin can be
revised to load different weights or apply different preprocessing while still
registering as ``alexnet``, and a stimulus set can be re-uploaded under the
same name.

A cache keyed on the bare identifiers would then hand back activations
produced by code that no longer exists, and the resulting score would look
entirely normal. Appending a short revision string means a plugin revision
lands under a *different* key rather than silently overwriting the meaning of
the old one.

Off by default — see :func:`revision_enabled`
-----------------------------------------------
``result_caching`` is enabled by default (``RESULTCACHING_DISABLE`` defaults to
``'0'``), so developers already have warm local caches keyed on the bare
identifiers. Revisioning unconditionally would turn every one of those cold,
and would keep re-invalidating them on each commit that touches the plugin.

Worse for local use, ``git log -1 -- <dir>`` reports the last *commit* touching
the directory, so uncommitted working-tree edits resolve to the same revision:
locally the revision is simultaneously too eager (churns on commit) and not
eager enough (blind to the edit you are actually testing).

So this is opt-in via ``BRAINSCORE_CACHE_PLUGIN_REVISION=1``. Production enables
it together with the shared S3 backend, where the checkout is clean and the
cache is long-lived and shared — the setting in which a stale hit actually
matters. With it unset, keys are byte-for-byte what they were before this
module existed.

Resolution order, per plugin:

1. ``BRAINSCORE_<TYPE>_PLUGIN_SHA`` environment variable. The scoring
   orchestrator knows the revision already and can inject it, which avoids a
   ``git`` call inside every container.
2. The git commit that last touched the plugin directory. Only that plugin's
   history counts — using the repository HEAD would invalidate every model's
   cache on every unrelated commit.
3. A content hash of the plugin directory, for installs that are not a git
   checkout.
4. Nothing. The identifier is returned unchanged, which reproduces the
   pre-existing (revision-blind) key, and a warning is logged.

Step 4 is deliberately not an error: this module must never be able to fail a
scoring run. It degrades to exactly the behaviour that shipped before it.
"""
import hashlib
import logging
import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Optional

_logger = logging.getLogger(__name__)

# Length of the revision appended to an identifier. 12 hex chars is the git
# short-sha convention and is far beyond collision risk for this population.
_REVISION_CHARS = 12

# Files whose contents cannot change what a model computes.
_IGNORED_SUFFIXES = ('.pyc', '.pyo', '.md')
_IGNORED_DIRS = ('__pycache__', '.git', '.pytest_cache')

# Opt-in switch. Unset => keys are exactly what they were before this module.
_ENABLE_VAR = 'BRAINSCORE_CACHE_PLUGIN_REVISION'


def revision_enabled() -> bool:
    """True if cache keys should carry plugin revisions.

    Deliberately opt-in; see the module docstring. Any consumer that shares a
    cache across machines or across plugin revisions (i.e. the production S3
    backend) must enable this, and should refuse to start without it.
    """
    return os.environ.get(_ENABLE_VAR, '0').strip().lower() in ('1', 'true', 'yes')


def model_cache_identifier(identifier: str) -> str:
    """Model identifier with its plugin revision appended, for cache keying."""
    return _with_revision(identifier, plugin_type='models',
                          env_var='BRAINSCORE_MODEL_PLUGIN_SHA', unresolved_is_notable=True)


def stimulus_set_cache_identifier(stimuli_identifier: str) -> str:
    """Stimulus-set identifier with its data-plugin revision appended.

    The identifier arriving here is usually a screen-converted derivative:
    ``place_on_screen`` rewrites it to
    ``<base>--target<degrees>--source<degrees>``. Those visual-degree
    parameters genuinely change the activations, so they stay in the key; only
    the *revision* is resolved from the base stimulus set.

    Stimulus sets register under ``stimulus_set_registry``, not the
    ``data_registry`` that ``ImportPlugin`` would infer from the ``data``
    plugin directory, hence the explicit prefix.
    """
    return _with_revision(stimuli_identifier, plugin_type='data',
                          env_var='BRAINSCORE_DATA_PLUGIN_SHA', unresolved_is_notable=False,
                          registry_prefix='stimulus_set', base_of=_strip_screen_suffix)


# place_on_screen builds `f"{identifier}--target{...:.2f}--source{...}"`.
_SCREEN_SUFFIX = '--target'


def _strip_screen_suffix(stimuli_identifier: str) -> str:
    """Base stimulus-set identifier, with any screen conversion suffix removed."""
    return stimuli_identifier.split(_SCREEN_SUFFIX, 1)[0]


def _with_revision(identifier, plugin_type: str, env_var: str, unresolved_is_notable: bool,
                   registry_prefix: Optional[str] = None, base_of=None):
    if not revision_enabled():
        return identifier
    if not identifier or not isinstance(identifier, str):
        # `stimuli_identifier` is False when the caller disables storing.
        return identifier
    # The revision is resolved from the base plugin, but the *full* identifier
    # stays in the key: derived parameters (e.g. visual degrees) change the
    # activations and must not collapse onto the same entry.
    lookup_identifier = base_of(identifier) if base_of else identifier
    revision = _plugin_revision(plugin_type=plugin_type, identifier=lookup_identifier,
                               env_var=env_var, unresolved_is_notable=unresolved_is_notable,
                               registry_prefix=registry_prefix)
    return f"{identifier}@{revision}" if revision else identifier


@lru_cache(maxsize=None)
def _plugin_revision(plugin_type: str, identifier: str, env_var: str,
                     unresolved_is_notable: bool = True,
                     registry_prefix: Optional[str] = None) -> Optional[str]:
    """Short revision string for a plugin, or None if it cannot be determined.

    Cached per process, which also means the "unresolved" log line is emitted
    once per plugin rather than on every ``from_paths`` call — layer
    commitment calls into here repeatedly and a per-call warning would be
    hundreds of lines in the container log.
    """
    from_env = os.environ.get(env_var)
    if from_env:
        return from_env.strip()[:_REVISION_CHARS]

    plugin_dir = _locate_plugin_dir(plugin_type=plugin_type, identifier=identifier,
                                    registry_prefix=registry_prefix)
    revision = None
    if plugin_dir is not None:
        revision = _git_revision(plugin_dir) or _content_revision(plugin_dir)
    if not revision:
        # An unresolved *model* revision means the cache cannot tell two
        # revisions of the same plugin apart, which is the failure this module
        # exists to prevent -- worth surfacing. An unresolved stimulus set is
        # routine.
        log = _logger.warning if unresolved_is_notable else _logger.debug
        log(f"Could not resolve a {plugin_type} revision for '{identifier}'; the activation "
            f"cache key cannot distinguish revisions of this plugin. "
            f"Set {env_var} to make it explicit.")
    return revision


def _locate_plugin_dir(plugin_type: str, identifier: str,
                       registry_prefix: Optional[str] = None) -> Optional[Path]:
    try:
        from brainscore_core.plugin_management import import_plugin as _import_plugin
        from brainscore_core.plugin_management.import_plugin import ImportPlugin
        importer = ImportPlugin(library_root='brainscore_vision', plugin_type=plugin_type,
                                identifier=identifier, registry_prefix=registry_prefix)
        # locate_plugin scans every plugin directory and warns about each one
        # missing an __init__.py. Those are pre-existing repo issues, not
        # anything this lookup can act on, and they would appear in every
        # container log. Silence them for the duration of the scan only.
        resolver_logger = logging.getLogger(_import_plugin.__name__)
        previous_level = resolver_logger.level
        resolver_logger.setLevel(logging.ERROR)
        try:
            dirname = importer.locate_plugin()
        finally:
            resolver_logger.setLevel(previous_level)
        plugin_dir = Path(importer.plugins_dir) / dirname
        return plugin_dir if plugin_dir.is_dir() else None
    except Exception:
        # Unregistered identifier, ambiguous registration, or a layout this
        # resolver does not understand. Not fatal — the caller degrades.
        _logger.debug(f"Could not locate {plugin_type} plugin dir for '{identifier}'", exc_info=True)
        return None


def _git_revision(plugin_dir: Path) -> Optional[str]:
    """Commit that last touched ``plugin_dir``, or None outside a git checkout.

    Scoped to the directory on purpose. The repository HEAD would change on
    every unrelated commit and invalidate the whole cache each time.
    """
    try:
        result = subprocess.run(
            ['git', 'log', '-1', '--format=%H', '--', str(plugin_dir)],
            cwd=str(plugin_dir), capture_output=True, text=True, timeout=30, check=True,
        )
    except Exception:
        _logger.debug(f"git revision unavailable for {plugin_dir}", exc_info=True)
        return None
    sha = result.stdout.strip()
    # An empty result means the path is untracked — fall through to the
    # content hash rather than returning a revision that means "unknown".
    return sha[:_REVISION_CHARS] if len(sha) == 40 else None


def _content_revision(plugin_dir: Path) -> Optional[str]:
    """sha256 over the plugin's source files, for non-git installs."""
    digest = hashlib.sha256()
    try:
        for path in sorted(p for p in plugin_dir.rglob('*') if p.is_file()):
            if any(part in _IGNORED_DIRS for part in path.parts):
                continue
            if path.suffix in _IGNORED_SUFFIXES:
                continue
            digest.update(str(path.relative_to(plugin_dir)).encode())
            digest.update(path.read_bytes())
    except Exception:
        _logger.debug(f"content revision failed for {plugin_dir}", exc_info=True)
        return None
    return digest.hexdigest()[:_REVISION_CHARS]
