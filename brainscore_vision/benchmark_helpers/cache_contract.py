"""Check that a benchmark's stimulus identifiers can carry a plugin revision.

The shared activation cache keys on ``stimuli_identifier``. An identifier that
cannot be traced back to a registered plugin cannot carry a revision, and since
brain-score refuses to write an unrevisioned key to a shared cache, every
extraction for that stimulus set is recomputed at full cost -- silently, apart
from one warning line in a scoring log.

That is exactly what happened to the Zerbe rdm and cka variants: the benchmark
synthesised ``Zerbe2026_fmri_rdm_full_sub-01`` for a merged train+test pool,
which is not a registered plugin, so caching was disabled for the entire family
the cache existed to serve.

The rule for a synthesised stimulus set is to name it after the registered set
it derives from::

    stim.identifier = f"{registered_identifier}--{marker}"

Everything from the first ``--`` on stays in the cache key, so derivatives stay
distinct from each other; only the part before it is used to look up the
revision. ``place_on_screen`` already follows this convention, and its suffix
composes with a benchmark's own marker.

Call :func:`assert_stimulus_identifier_is_cacheable` from a plugin's tests to
catch this at PR time instead of in a production scoring log.
"""
from __future__ import annotations

import os
from contextlib import contextmanager

__all__ = ['stimulus_identifier_is_cacheable', 'assert_stimulus_identifier_is_cacheable']


@contextmanager
def _revisioning_enabled():
    """Force revisioning on for the duration of the check.

    Revisioning is opt-in and off by default, and with it off every identifier
    resolves to itself -- so a check that did not force it on would pass
    unconditionally, which is worse than no check at all.
    """
    from brainscore_vision.model_helpers.activations import cache_key

    var = 'BRAINSCORE_CACHE_PLUGIN_REVISION'
    previous = os.environ.get(var)
    os.environ[var] = '1'
    # the resolver memoises per (plugin_type, identifier); clear so a check
    # cannot be answered from a lookup made while revisioning was off
    cache_key._plugin_revision.cache_clear()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = previous
        cache_key._plugin_revision.cache_clear()


def stimulus_identifier_is_cacheable(stimuli_identifier: str) -> bool:
    """True if ``stimuli_identifier`` can carry a data-plugin revision."""
    from brainscore_vision.model_helpers.activations.cache_key import (
        stimulus_set_cache_identifier)

    with _revisioning_enabled():
        return stimulus_set_cache_identifier(stimuli_identifier) is not None


def assert_stimulus_identifier_is_cacheable(stimuli_identifier: str) -> None:
    """Raise ``AssertionError`` if this identifier would disable caching.

    :param stimuli_identifier: the value a benchmark assigns to
        ``stimulus_set.identifier``, exactly as the extractor will see it.
    """
    from brainscore_vision.model_helpers.activations.cache_key import (
        base_stimulus_identifier)

    if stimulus_identifier_is_cacheable(stimuli_identifier):
        return
    base = base_stimulus_identifier(stimuli_identifier)
    raise AssertionError(
        f"stimulus identifier {stimuli_identifier!r} cannot carry a data-plugin "
        f"revision, so the shared activation cache will refuse it and every "
        f"extraction will be recomputed at full cost.\n"
        f"  looked up: {base!r} -- not a registered data or stimulus_set plugin.\n"
        f"  If this set is synthesised from a registered one, name it "
        f"'<registered identifier>--<marker>'; the marker stays in the cache key, "
        f"the part before it resolves the revision."
    )
