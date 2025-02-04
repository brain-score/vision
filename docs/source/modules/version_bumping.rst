Version Bumping
===============

Brain-Score Vision now features an automated version bumping process that follows
`Semantic Versioning <https://semver.org/>`_ (MAJOR.MINOR.PATCH) and is triggered via a GitHub Action.
This ensures that any significant changes to the codebase lead to an appropriate version update.

How It Works
------------

1. **Pull Request Analysis:**
   When a pull request (PR) is submitted, a GitHub Action (see the
   `workflow file <https://github.com/brain-score/vision/blob/master/.github/workflows/bump_version.yml>`_)
   analyzes the changes. If any files outside the plugin directories—namely, ``benchmarks``,
   ``data``, ``metrics``, and ``models``—are modified, the version bump process is initiated. Changes confined to
   plugin directories are considered non-breaking and therefore do not necessitate a version increment.

2. **Determining the Bump Type:**
   The type of version bump is determined by labels applied to the PR:

   - **MAJOR:** Apply the ``major update`` label to trigger an increase in the major version.
   - **MINOR:** Apply the ``minor update`` label to trigger an increase in the minor version.
   - **PATCH:** If neither label is applied, the version will automatically increment the patch number.

3. **Version Increment and PR Creation:**
   The tool `bump-my-version <https://github.com/callowayproject/bump-my-version>`_ uses the latest
   version tag to calculate the new version number. After bumping the version, a new PR is automatically
   created containing the updated version. This PR is auto-approved, undergoes status checks, and is then
   merged automatically.

Version Releases
----------------

After the version is bumped, release notes are automatically generated. These notes include all commit
details since the previous version bump and can be viewed in the
`Releases section <https://github.com/brain-score/vision/releases>`_ of the repository.

PyPI Publishing
---------------

A version bump also triggers a publishing job that builds the new package and uploads it to PyPI.
You can always find the latest package available on PyPI at the
`PyPI project page <https://pypi.org/project/brainscore-vision/>`_.