# Model Metadata v2 Design

## Decision

Use a versioned `metadata.yml` document as the canonical source of truth and
project each validated document into typed PostgreSQL tables. Retain the exact
document in `jsonb` for lossless round-tripping, but do not use `jsonb` as the
primary analytics interface. Stable, frequently filtered facts such as
architecture family, parameter count, recurrence, input size, supervision, and
visual degrees have typed columns. Repeated concepts such as datasets,
training stages, artifacts, lineage, contributors, references, intended-use
items, and provenance have focused child tables.

This hybrid is preferable to the two obvious alternatives. A JSONB-only table
would mirror YAML easily, but model comparison queries would depend on brittle
JSON paths, inconsistent types, and expensive expression indexes. A completely
normalized representation of every possible metadata leaf would provide strong
relational constraints, but make normal schema evolution costly and make it
difficult to preserve new or domain-specific fields. Model Metadata v2 uses
normalization where it improves queryability and integrity, and the validated
document for long-tail content and forward compatibility.

The database is a derived query projection. Curators edit repository files,
not database rows. An importer validates a document, computes its content hash,
stores an immutable copy, replaces the current relational projection in one
transaction, and only then marks the new document current. This avoids two
independent sources of truth and makes every production value traceable to a
repository path and revision.

The contract is defined by
[`model-metadata-v2.schema.json`](../model_metadata/model-metadata-v2.schema.json).
The proposed, unapplied relational projection is in
[`postgres-schema-v2.sql`](../model_metadata/postgres-schema-v2.sql).

## YAML contract

Each document describes exactly one Brain-Score registry identifier. The top
level is organized by concepts needed for model cards and comparison: identity,
architecture, lineage, interface, preprocessing, training, evaluation,
artifacts, intended use, authorship, licenses, references, and provenance. It
does not contain Brain-Score results, ranks, composite scores, or benchmark
scores; those remain result data and are joined at read time.

All values use their natural type. Counts are integers, booleans are booleans,
and lists are YAML lists. Approximate counts use `{value, exact}` rather than a
string such as `"~304M"`. Unknown facts are omitted. They are never represented
by strings such as `"N/A"`, `"unknown"`, or `"not documented"`. When a curated
field is unavailable, its provenance assertion has status `undocumented`. This
keeps database columns type-safe while preserving completeness information for
the UI.

Example:

```yaml
schema_version: "2.0.0"
schema_url: "https://raw.githubusercontent.com/brain-score/vision/master/docs/model_metadata/model-metadata-v2.schema.json"
model:
  identifier: "convnext_xxlarge:clip_laion2b_soup_ft_in1k"
  display_name: "convnext_xxlarge.clip_laion2b_soup_ft_in1k"
  domain: "vision"
architecture:
  family: "convolutional_neural_network"
  description: "ConvNeXt (pure CNN)"
  parameter_count:
    value: 845500000
    exact: true
  recurrent: false
interface:
  inputs:
    - modality: "image"
      shape:
        channels: 3
        height: 256
        width: 256
training:
  datasets:
    - identifier: "laion-2b"
      name: "LAION-2B"
      role: "pretraining"
    - identifier: "imagenet-1k"
      name: "ImageNet-1k"
      role: "fine_tuning"
artifacts:
  - role: "weights"
    provider: "timm / OpenCLIP / LAION"
provenance:
  sources:
    curation_workbook:
      type: "curation"
      title: "Brainscore Model Metadata.xlsx"
  assertions:
    - path: "/architecture/parameter_count"
      status: "verified"
      source: "curation_workbook"
```

## Provenance and completeness

Provenance is separate from values so that ordinary metadata stays readable.
Each assertion identifies a JSON Pointer path, a status, and a source key.
Statuses are deliberately limited to `verified`, `inferred`, and
`undocumented`, matching the planned model-card summary. Sources may point to
code, checkpoints, papers, datasets, model cards, or a curation artifact. The
database stores assertions separately so the UI can aggregate counts without
walking JSON at request time.

The workbook uses green, yellow, and red cell fills as field-level confidence.
The generator maps these to verified, inferred, and undocumented. Text that
explicitly says assumed, presumed, inferred, or unconfirmed is treated as
inferred even when its fill is ambiguous. Text that says a fact is not
documented, and values such as `N/A`, are undocumented. This conversion makes a
formerly visual-only convention machine-readable.

Every workbook column that supplies a model name or model ID is processed. The
generator matches the column to the model registry, which supplies the
canonical identifier when the workbook ID is blank. All curated fields,
including architecture, can be empty. Empty or unknown values are omitted from
typed YAML and receive an undocumented provenance assertion. This avoids
fabricated placeholder values while allowing every current workbook model to
use the metadata system. When multiple workbook columns match one registry
model, the later column is primary and blank or unknown values are filled from
earlier columns. Conflicting documented values remain resolved in favor of the
later curation.

Completeness percentages should be computed from the schema and assertions,
not authored into YAML. Otherwise they become stale when the schema changes.

## Repository layout and shared plugins

For a plugin directory that registers one model, the path is the expected:

```text
brainscore_vision/models/<plugin>/metadata.yml
```

Some plugin directories, notably `timm_models` and `scaling_models`, register
many identifiers. A single directory-level document would recreate the legacy
`models:` mapping and make unrelated model edits conflict. Moving implementation
code into 20 new plugins solely for metadata would also be disruptive. Shared
plugins therefore use one logical model directory per registry identifier:

```text
brainscore_vision/models/<plugin>/metadata/<url-encoded-identifier>/metadata.yml
```

The identifier itself remains unmodified inside the document. URL encoding is
used only for the portable filesystem path, so characters such as `:` do not
break Windows checkouts. Import discovery is recursive and selects v2
`metadata.yml` files. Existing legacy `metadata.yaml` files can remain during
rollout for compatibility, but a v2 importer must not merge their values into a
v2 document. The formats have different semantics and should have an explicit
cutover date.

## PostgreSQL projection

The proposed schema creates a new `model_metadata_v2` namespace and does not
alter the existing tables. `model_entity` gives each `(domain, identifier)` one
canonical identity. `model_binding` associates that identity with one or more
rows in the existing `brainscore_model` table, avoiding duplicated metadata for
historical or owner-specific leaderboard records. `metadata_document` stores
immutable validated documents; a partial unique index permits only one current
document per entity.

`model_profile` contains card-header and comparison fields. The remaining
tables model repeated data without an unbounded entity-attribute-value design.
Only provenance uses paths because provenance is inherently about arbitrary
document assertions. A GIN index on the exact payload supports occasional
long-tail searches, while ordinary page and analytics queries use B-tree
indexes over typed columns.

| Model-card element | YAML source | PostgreSQL source |
|---|---|---|
| At-a-glance specifications | `architecture`, `interface`, `training` | `model_profile`, `preprocessing` |
| Training pipeline | `training.stages`, `training.datasets` | `training_stage`, `model_dataset_usage` |
| Provenance summary | `provenance.assertions` | `verification_summary` view |
| Lineage and siblings | `lineage` | `model_relationship` |
| Creators and organizations | `authorship` | `party`, `contribution` |
| Software and data licenses | `licenses`, `artifacts` | `artifact` and document payload |
| References and citations | `references` | `reference`, `model_reference` |
| Composite and benchmark scores | Not metadata | Existing score tables |

## Import and failure behavior

Ingestion must be deterministic and atomic. Discovery first rejects duplicate
`(domain, identifier)` documents. Each file is parsed with YAML safe mode,
validated against the schema version named in the file, and checked for
cross-field rules that JSON Schema cannot express: assertion source keys must
exist, registry identifiers must match the plugin registry, training-stage
orders must be unique, and a source-code artifact path must resolve inside the
repository.

The importer then computes SHA-256 over a canonical representation. If that
hash is already current, ingestion is a no-op. Otherwise it locks the entity,
inserts a new immutable document, replaces the typed projection and child rows,
marks the old document non-current, marks the new document current, and commits.
Any parse, validation, mapping, or database error rolls back the whole model.
One invalid model must fail the batch in CI; a production backfill may support a
separate explicitly requested quarantine mode, but must never silently import a
partial document.

The importer should emit file, identifier, JSON Pointer, invalid value, and
expected constraint in each error. It should also report duplicate documents,
orphan database bindings, unresolved lineage targets, and assertions whose
source keys do not exist. No importer or DDL has been run as part of this design.

## Evolution, testing, and rollout

Schema versions use semantic versioning. A patch clarifies validation without
changing accepted structure. A minor version adds optional fields or controlled
values. A major version removes, renames, or changes the meaning of fields.
Importers must select the schema by the exact version and persist both version
and schema URL with every document. Database migrations follow the projection,
not every YAML addition: a new long-tail optional field can remain in the
document until it becomes important enough to query frequently.

CI should perform four checks: parse every `metadata.yml` in safe mode; validate
it against the declared JSON Schema; enforce the cross-file and registry rules;
and generate the relational projection into a temporary PostgreSQL database to
exercise constraints and representative card queries. Generator tests should
cover number coercion, `N/A` omission, confidence-color mapping, shared-plugin
paths, and exact identifier matching. A fixture should verify that regenerated
files are byte-for-byte stable.

Rollout should be staged. First merge the schema, generator, and completed v2
files. Next implement read-only ingestion in CI and compare its projection with
the legacy metadata. Then deploy the new tables and importer behind a feature
flag, backfill current documents, and render model-card sections from v2 only
when a current validated document exists. Finally, after parity and monitoring,
stop reading legacy `metadata.yaml` and remove it in a separate logical change.
This branch deliberately contains no database migration and performs no
database writes.
