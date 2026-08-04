# Bundled-weight → S3 migration (2026-06-08)

This branch (`kp/migrate-bundled-weights-to-s3`) moves 6 model checkpoints out of
the source tree and into the canonical `brainscore-storage` S3 bucket so they
load via `brainscore_core.supported_data_standards.brainio.s3.load_weight_file`
at runtime instead of being packaged into the published wheel.

Until the files are uploaded to S3, **the affected models will fail to load at
score time** with a 404 from `load_weight_file`. Upload first, then merge.

## Why

These 6 `.pth` files added ~277 MB to the wheel and pushed it past PyPI's 100 MB
per-file limit, silently blocking every release since 2.3.19. See
`feedback_no_direct_pushes_vision_master` memory for chronology.

## What needs to land on S3

All paths are relative to the `brainscore-storage` bucket, under
`brainscore-vision/models/<model_id>/`. The `version_id` field in
`load_weight_file` is `"null"` (string) because we are not using S3 versioning
for these uploads.

| Model | S3 key | SHA1 | Size |
|---|---|---|---|
| `pr1_stl10_disentangled_alpha04` | `brainscore-vision/models/pr1_stl10_disentangled_alpha04/weights.pth` | `5c8d66c99421bce81edee06dd00e3c93f4fb5b8d` | 46,748,875 |
| `20260408_2033_stl10_disentangled_alpha04` | `brainscore-vision/models/20260408_2033_stl10_disentangled_alpha04/weights.pth` | `5c8d66c99421bce81edee06dd00e3c93f4fb5b8d` | 46,748,875 |
| `20260408_2033_stl10_disentangled_alpha00` | `brainscore-vision/models/20260408_2033_stl10_disentangled_alpha00/weights.pth` | `ab26a6087022fae981ed46168d2035dfead81962` | 46,748,875 |
| `20260408_2033_stl10_combined_alpha04` | `brainscore-vision/models/20260408_2033_stl10_combined_alpha04/weights.pth` | `4d3c9af0d1a1b179edd2518fe8f7e5f56e5d1b1f` | 46,082,419 |
| `resnet18_fair` | `brainscore-vision/models/resnet18_fair/model_weights.pth` | `04b939f367b2044a3e1dff55e861330c8ff7ef52` | 45,196,043 |
| `aughost_fair_resnet18` | `brainscore-vision/models/aughost_fair_resnet18/model_weights.pth` | `a185f4ba89ed8dda847588c5ddab0e909bebe2e6` | 45,194,290 |

Note: the first two rows are byte-identical (same SHA1). The branch deliberately
keeps two separate S3 keys so each model directory remains self-describing.
Storage cost of the duplicate is ~$0.001/month and saves a cross-model symlink
that would obscure provenance.

## Upload checklist

Run before merging this PR. Requires the same AWS profile that owns
`brainscore-storage` (mit-brainscore, account `616043263657`, region
`us-east-2`).

Restore the deleted files locally first — they're still in the prior commit on
this branch:

```bash
cd "/Users/kartik/Brain-Score 2026/vision"
git checkout HEAD~1 -- \
  brainscore_vision/models/pr1_stl10_disentangled_alpha04/weights.pth \
  brainscore_vision/models/20260408_2033_stl10_disentangled_alpha04/weights.pth \
  brainscore_vision/models/20260408_2033_stl10_disentangled_alpha00/weights.pth \
  brainscore_vision/models/20260408_2033_stl10_combined_alpha04/weights.pth \
  brainscore_vision/models/resnet18_fair/model_weights.pth \
  brainscore_vision/models/aughost_fair_resnet18/model_weights.pth
```

Then upload each one (the trailing key in `s3://...` matches the
`folder_name/relative_path` from `model.py`):

```bash
aws s3 cp brainscore_vision/models/pr1_stl10_disentangled_alpha04/weights.pth \
  s3://brainscore-storage/brainscore-vision/models/pr1_stl10_disentangled_alpha04/weights.pth

aws s3 cp brainscore_vision/models/20260408_2033_stl10_disentangled_alpha04/weights.pth \
  s3://brainscore-storage/brainscore-vision/models/20260408_2033_stl10_disentangled_alpha04/weights.pth

aws s3 cp brainscore_vision/models/20260408_2033_stl10_disentangled_alpha00/weights.pth \
  s3://brainscore-storage/brainscore-vision/models/20260408_2033_stl10_disentangled_alpha00/weights.pth

aws s3 cp brainscore_vision/models/20260408_2033_stl10_combined_alpha04/weights.pth \
  s3://brainscore-storage/brainscore-vision/models/20260408_2033_stl10_combined_alpha04/weights.pth

aws s3 cp brainscore_vision/models/resnet18_fair/model_weights.pth \
  s3://brainscore-storage/brainscore-vision/models/resnet18_fair/model_weights.pth

aws s3 cp brainscore_vision/models/aughost_fair_resnet18/model_weights.pth \
  s3://brainscore-storage/brainscore-vision/models/aughost_fair_resnet18/model_weights.pth
```

Then revert the locally restored files so the working tree matches the branch
tip:

```bash
git checkout -- brainscore_vision/models/*/weights.pth brainscore_vision/models/*/model_weights.pth 2>/dev/null
git clean -f brainscore_vision/models/*/weights.pth brainscore_vision/models/*/model_weights.pth
```

## Verification

Quick spot-check that the S3 objects are reachable and match the expected
SHA1s before relying on `load_weight_file`:

```bash
for path in \
  pr1_stl10_disentangled_alpha04/weights.pth \
  20260408_2033_stl10_disentangled_alpha04/weights.pth \
  20260408_2033_stl10_disentangled_alpha00/weights.pth \
  20260408_2033_stl10_combined_alpha04/weights.pth \
  resnet18_fair/model_weights.pth \
  aughost_fair_resnet18/model_weights.pth \
; do
  aws s3api head-object --bucket brainscore-storage \
    --key "brainscore-vision/models/$path" \
    --query 'ContentLength' --output text
done
```

Then run each model's test once to confirm the load path works end-to-end:

```bash
DOMAIN=vision pytest brainscore_vision/models/pr1_stl10_disentangled_alpha04/test.py -v
# ... repeat for each
```

## After merging

- Future model submissions should use `load_weight_file` from day one; do not
  commit `.pth` (or any large binary) into a model directory. A pre-commit
  hook blocking files >5 MB under `brainscore_vision/models/` is in the
  follow-up backlog.
- The wheel size will drop from 282 MB to ~5 MB once this branch lands. PyPI
  publication should resume normally, no size-limit increase needed.
