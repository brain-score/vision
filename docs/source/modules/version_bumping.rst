Version Bumping
===============

Brain-Score Vision releases automatically, following
`Semantic Versioning <https://semver.org/>`_ (MAJOR.MINOR.PATCH). Every squash merge to ``master``
is judged on its own by the
`release workflow <https://github.com/brain-score/vision/blob/master/.github/workflows/release.yml>`_,
which calls the shared release workflow in `brain-score/core <https://github.com/brain-score/core>`_.

How It Works
------------

1. **Path gate:**
   A merge can release only if it changes a file under ``brainscore_vision/`` outside the plugin
   directories (``benchmarks``, ``data``, ``metrics`` and ``models``), or changes the
   ``[project] dependencies`` in ``pyproject.toml``. Plugin-only, test, docs and workflow changes
   never release.

2. **Release level from the PR title:**
   PR titles must follow ``type(scope): subject``; a check on every PR enforces this and comments
   with the allowed types.

   - **MINOR:** ``feat``.
   - **PATCH:** ``fix`` or ``perf``.
   - **MAJOR:** ``feat!:`` or a ``BREAKING CHANGE:`` footer, plus the ``major update`` label.
     Without the label it releases as minor.
   - Any other type (``refactor``, ``docs``, ``test``, ``ci``, ``chore``, ``build``, ``plugin``,
     ``model``, ``benchmark``, ``data``, ``metric``) does not release.

3. **Version, tag and release:**
   `python-semantic-release <https://python-semantic-release.readthedocs.io/>`_ updates the version
   in ``pyproject.toml``, commits ``chore(release): X.Y.Z`` to ``master``, tags ``vX.Y.Z`` and
   creates a GitHub release with notes. There is no separate version-bump PR.

PyPI Publishing
---------------

The same workflow builds the package and publishes it to PyPI through trusted publishing. You can
always find the latest package on the `PyPI project page <https://pypi.org/project/brainscore-vision/>`_.
