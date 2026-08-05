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
_IGNORED_SUFFIXES = ('.pyc', '.pyo', '.md', '.txt.orig')
_IGNORED_DIRS = ('__pycache__', '.git', '.pytest_cache')


def model_cache_identifier(identifier: str) -> str:
    """Model identifier with its plugin revision appended, for cache keying."""
    return _with_revision(identifier, plugin_type='models',
                          env_var='BRAINSCORE_MODEL_PLUGIN_SHA', unresolved_is_notable=True)


def stimulus_set_cache_identifier(stimuli_identifier: str) -> str:
    """Stimulus-set identifier with its data-plugin revision appended.

    Stimulus sets reach the extractor through several routes (a registered
    data plugin, a benchmark-local assembly, a screen-converted derivative),
    so a revision is often unavailable. That is expected rather than notable —
    an unresolved stimulus set simply keys as it does today.
    """
    return _with_revision(stimuli_identifier, plugin_type='data',
                          env_var='BRAINSCORE_DATA_PLUGIN_SHA', unresolved_is_notable=False)


def _with_revision(identifier, plugin_type: str, env_var: str, unresolved_is_notable: bool):
    if not identifier or not isinstance(identifier, str):
        # `stimuli_identifier` is False when the caller disables storing.
        return identifier
    revision = _plugin_revision(plugin_type=plugin_type, identifier=identifier, env_var=env_var,
                               unresolved_is_notable=unresolved_is_notable)
    return f"{identifier}@{revision}" if revision else identifier


@lru_cache(maxsize=None)
def _plugin_revision(plugin_type: str, identifier: str, env_var: str,
                     unresolved_is_notable: bool = True) -> Optional[str]:
    """Short revision string for a plugin, or None if it cannot be determined.

    Cached per process, which also means the "unresolved" log line is emitted
    once per plugin rather than on every ``from_paths`` call — layer
    commitment calls into here repeatedly and a per-call warning would be
    hundreds of lines in the container log.
    """
    from_env = os.environ.get(env_var)
    if from_env:
        return from_env.strip()[:_REVISION_CHARS]

    plugin_dir = _locate_plugin_dir(plugin_type=plugin_type, identifier=identifier)
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


def _locate_plugin_dir(plugin_type: str, identifier: str) -> Optional[Path]:
    try:
        from brainscore_core.plugin_management.import_plugin import ImportPlugin
        importer = ImportPlugin(library_root='brainscore_vision', plugin_type=plugin_type,
                                identifier=identifier)
        dirname = importer.locate_plugin()
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
