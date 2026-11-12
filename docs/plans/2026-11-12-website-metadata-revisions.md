# Website metadata revisions

A website metadata revision must pass schema and source-policy validation and
reach the existing merged-PR publisher. It must not trigger legacy enrichment,
load a model, or create a second metadata PR.

The merge workflow identifies schema v2 from either metadata.yml or
metadata.yaml in the merged repository. It excludes those directories from the
legacy handler. For a mixed revision, only legacy directories reach that handler.
Missing metadata retains the existing generation path. Duplicate suffixes fail
before generation. Code changes retain their existing scoring path.

The companion core change returns after validation for existing schema-v2
metadata. It also extends the existing same-repository website title exception
to web_metadata_ branches. Other PR titles retain validation.

Source kind, contributor identity, and assertion confidence remain separate.
Website revisions keep probable unless a later evidence review changes the
assertion. This fix does not change model facts or assertion statuses.

Regression tests execute the workflow step with temporary metadata files.
They cover both suffixes, v2 and legacy schemas, mixed revisions, absent
metadata, code changes, and duplicate files. No test loads a model.
