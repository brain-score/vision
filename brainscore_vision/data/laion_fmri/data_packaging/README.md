# LAION-fMRI data packaging pipeline

End-to-end pipeline to rebuild every Brain-Score artifact (neural assemblies + stimuli) from the raw LAION-fMRI release. Used to produce what's currently on S3, and to verify reproducibility for audits or future re-uploads.

99% of users **do not need to run any of this** — `_run_score(...)` downloads the pre-built assemblies from S3 automatically. This pipeline is for:

- Producing the canonical S3 artifacts from raw GLMsingle output
- Verifying that the S3 artifacts can be regenerated bit-for-bit-of-data from the upstream release
- Re-uploading after upstream changes (new subjects, schema migrations, etc.)

## One-command full rebuild

```bash
# Prereqs:
#   - `pip install "laion-fmri @ git+https://github.com/ViCCo-Group/LAION-fMRI.git@main"`
#   - `laion-fmri config --data-dir ~/laion-fmri` already run
#   - Signed DUA at https://laion-fmri.hebartlab.com/request -> received a request_id
#   - AWS credentials configured (for the semantic-verification download)

python -m brainscore_vision.data.laion_fmri.data_packaging.rebuild_assemblies \
    --request-id <YOUR_LAION_FMRI_REQUEST_ID>
```

Runs all 7 steps in order, idempotent, ~2-3 hours total (dominated by the upstream GLMsingle download). See `rebuild_assemblies.py --help` for flags.

The `--request-id` flag sets `LAION_FMRI_REQUEST_ID` for the stimulus-download step (step 5). You can alternatively `export LAION_FMRI_REQUEST_ID=<id>` and omit the flag, or use the cached token from `laion-fmri request-access`. Pass `--skip-stimuli` to skip the DUA-gated portion entirely if you only need the (CC0) neural assemblies.

Memory note: the semantic-verification step (step 7) loads up to two ~950 MB DataArrays at a time and aggressively GCs between files; peak RSS measured at ~4 GB for the full 5-subject sweep (~3 GB single-subject). If you're memory-constrained, run with `--skip-semantic-check` and verify subject-by-subject afterward via `--subjects sub-01`.

## Scripts (in execution order)

| Script | Input | Output | Purpose |
|---|---|---|---|
| `build_assembly.py` | `~/laion-fmri/derivatives/glmsingle-tedana/<sub>/...` | `<out>/sub-XX.nc` | Per-subject GLMsingle betas → masked, z-scored NeuroidAssembly across V1-V4 + IT |
| `prepare_shared.py` | `<out>/sub-XX.nc` | `<out>/shared_sub-XX.nc` | Filter to shared 1,492-stim pool; strip trial bookkeeping; add `subject_id_pres` |
| `prepare_persubject.py` | `<out>/sub-XX.nc` | `<out>/persubject_sub-XX.nc` | Filter to per-subject 6,204-stim pool (1,121 shared non-OOD + 4,712 unique + 371 OOD); no schema cleanup (load-time surgery handles it) |
| `repackage_for_s3.py` | `<out>/{shared,persubject}_sub-XX.nc` | `<out>/{shared,persubject}_sub-XX_brainscore.nc` | Collapse to single `data` data_var (required by `load_assembly_from_s3`) |
| `get_local_stimuli.py` | `~/laion-fmri/stimuli/task-images_stimuli.h5` | `~/laion-fmri/stimuli/images_extracted/...` | Extract 25,052 DUA-gated JPEGs + manifest |
| `upload_to_s3.py` | `<out>/{shared,persubject}_sub-XX_brainscore.nc` | S3 bucket + printed registry snippets | Push assemblies; print `_S3_ASSEMBLIES_SHARED` / `_S3_ASSEMBLIES_PERSUBJECT` dicts to paste into `_helpers/assemblies.py` |
| `rebuild_assemblies.py` | all of the above | orchestrates + semantic-verifies | One-stop pipeline runner with downstream verification |

## Semantic verification (step 7 of `rebuild_assemblies.py`)

Bit-exact sha1 reproducibility across xarray/HDF5 versions is **not** feasible (serialization byte-order varies). Instead, the rebuild downloads the pinned S3 artifact for each (family, subject) and compares data + every coord element-wise.

**Known cosmetic divergence:** published `shared_sub-XX_v3_brainscore.nc` files carry `subject_id_pres = 'sub-XX_v3'` (the original packaging derived sub_id from a `sub-XX_v3.nc` filename stem). Fresh rebuilds produce clean `'sub-XX'`. The value is only used as a per-subject namespace identifier; runtime scoring is identical either way. Expect 5 WARN lines for this on every rebuild — one per shared subject.

If anything else differs, that's a real divergence worth investigating before considering the rebuild canonical.

## Re-uploading after an intentional change

1. Modify the relevant `prepare_*.py` / `repackage_for_s3.py` and rebuild.
2. Run `upload_to_s3.py` — it prints registry snippets with new `version_id` + `sha1`.
3. Paste those into `_helpers/assemblies.py` (replacing `_S3_ASSEMBLIES_SHARED` / `_S3_ASSEMBLIES_PERSUBJECT`).
4. Re-run the test suite to confirm runtime loaders pick up the new versions.
