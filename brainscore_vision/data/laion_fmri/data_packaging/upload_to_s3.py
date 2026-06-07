"""Upload per-subject LAION-fMRI assemblies to Brain-Score S3.

Mirrors the ``assy_<identifier>.nc`` naming convention used by Allen2022 and
others, so :func:`brainscore_core.brainio.s3.load_assembly_from_s3` finds them
out of the box.

Walks both pool families produced by the rebuild pipeline:

  - ``shared_sub-XX_brainscore.nc``      ->  ``Zerbe2026_fmri_full_sub-XX_Assembly``
  - ``persubject_sub-XX_brainscore.nc``  ->  ``Zerbe2026_fmri_persubject_sub-XX_Assembly``

Does NOT upload stimuli (gated by the LAION-fMRI Data Use Agreement -- stays
local via the manifest in ``stimuli/images_extracted/``).

Output: prints the data-registry snippets to paste into
``vision/brainscore_vision/data/laion_fmri/_helpers/assemblies.py``
(``_S3_ASSEMBLIES_SHARED`` and ``_S3_ASSEMBLIES_PERSUBJECT`` dicts).
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("upload_to_s3")

DEFAULT_BUCKET = "brainscore-storage"
DEFAULT_PREFIX = "brainscore-vision/benchmarks/Zerbe2026_fmri"
SUBJECTS = ("sub-01", "sub-03", "sub-05", "sub-06", "sub-07")


@dataclass(frozen=True)
class Family:
    """One pool family: input-file pattern + identifier template + registry dict name."""
    name: str           # "shared" / "persubject"
    file_prefix: str    # "shared_" / "persubject_"
    identifier_tmpl: str  # f-string with {sub}
    registry_var: str   # "_S3_ASSEMBLIES_SHARED" / "_S3_ASSEMBLIES_PERSUBJECT"


FAMILIES = (
    Family(name="shared", file_prefix="shared_",
           identifier_tmpl="Zerbe2026_fmri_full_{sub}_Assembly",
           registry_var="_S3_ASSEMBLIES_SHARED"),
    Family(name="persubject", file_prefix="persubject_",
           identifier_tmpl="Zerbe2026_fmri_persubject_{sub}_Assembly",
           registry_var="_S3_ASSEMBLIES_PERSUBJECT"),
)


def sha1_of(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def head_version(bucket: str, key: str) -> str:
    out = subprocess.check_output([
        "aws", "s3api", "head-object",
        "--bucket", bucket, "--key", key,
        "--query", "VersionId", "--output", "text",
    ]).decode().strip()
    return out


def upload(local: Path, bucket: str, key: str) -> None:
    log.info("  uploading -> s3://%s/%s", bucket, key)
    subprocess.check_call(["aws", "s3", "cp", str(local), f"s3://{bucket}/{key}"])


def _resolve_local(in_dir: Path, family: Family, sub: str) -> Path:
    """Prefer the post-repackage file (``*_brainscore.nc``); fall back to legacy."""
    repackaged = in_dir / f"{family.file_prefix}{sub}_brainscore.nc"
    legacy = in_dir / f"{family.file_prefix}{sub}.nc"
    if repackaged.exists():
        return repackaged
    if legacy.exists():
        log.warning("[%s/%s] using legacy (pre-repackage) file %s -- "
                    "expected %s. Run `repackage_for_s3.py` first to match "
                    "the published schema.", family.name, sub, legacy.name,
                    repackaged.name)
        return legacy
    raise FileNotFoundError(
        f"Neither {repackaged} nor {legacy} exists. "
        f"Run prepare_{family.name}.py and repackage_for_s3.py first."
    )


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path,
                   default=Path.home() / "laion-fmri/assemblies",
                   help="Directory holding S3-ready .nc files. "
                        "Default: ~/laion-fmri/assemblies/")
    p.add_argument("--bucket", default=DEFAULT_BUCKET)
    p.add_argument("--prefix", default=DEFAULT_PREFIX)
    p.add_argument("--subjects", nargs="*", default=list(SUBJECTS))
    p.add_argument("--families", nargs="*", default=[f.name for f in FAMILIES],
                   choices=[f.name for f in FAMILIES],
                   help="Which pool families to upload. Default: both.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    log.info("Target: s3://%s/%s/", args.bucket, args.prefix)

    selected = [f for f in FAMILIES if f.name in args.families]
    manifests: dict[str, list[dict]] = {f.name: [] for f in selected}

    for family in selected:
        log.info("== Family: %s ==", family.name)
        for sub in args.subjects:
            local_path = _resolve_local(args.in_dir, family, sub)
            sha1 = sha1_of(local_path)
            identifier = family.identifier_tmpl.format(sub=sub)
            s3_key = f"{args.prefix}/assy_{identifier}.nc"

            log.info("[%s/%s] sha1=%s  size=%.1f MB", family.name, sub,
                     sha1, local_path.stat().st_size / 1e6)
            if args.dry_run:
                log.info("  DRY RUN: aws s3 cp %s s3://%s/%s",
                         local_path, args.bucket, s3_key)
                version_id = "<populate-after-upload>"
            else:
                upload(local_path, args.bucket, s3_key)
                version_id = head_version(args.bucket, s3_key)
                log.info("  version_id=%s", version_id)

            manifests[family.name].append({
                "subject": sub, "identifier": identifier,
                "version_id": version_id, "sha1": sha1,
                "size_mb": local_path.stat().st_size / 1e6,
            })

    print("\n# Registry snippets -- paste into "
          "vision/brainscore_vision/data/laion_fmri/_helpers/assemblies.py:\n")
    for family in selected:
        print(f"{family.registry_var} = {{")
        for m in manifests[family.name]:
            print(f'    "{m["subject"]}": dict(')
            print(f'        identifier="{m["identifier"]}",')
            print(f'        version_id="{m["version_id"]}",')
            print(f'        sha1="{m["sha1"]}",')
            print(f'    ),')
        print("}\n")


if __name__ == "__main__":
    main()
