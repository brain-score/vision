"""Exercise metadata routing from the actual merge workflow without model loads."""

import os
from pathlib import Path
import subprocess

import pytest
import yaml

WORKFLOW = yaml.safe_load(
    (
        Path(__file__).resolve().parents[2] / ".github/workflows/score_new_plugins.yml"
    ).read_text()
)
STEPS = WORKFLOW["jobs"]["process_submission"]["steps"]


def run_metadata_status(
    tmp_path: Path, documents: dict[str, dict[str, str]], changed_files: list[str]
) -> subprocess.CompletedProcess:
    """Execute the workflow step against temporary plugin directories."""
    for directory, files in documents.items():
        plugin = tmp_path / directory
        plugin.mkdir(parents=True)
        for filename, content in files.items():
            (plugin / filename).write_text(content)
    output = tmp_path / "outputs"
    step = next(step for step in STEPS if step.get("id") == "check_metadata")
    return subprocess.run(
        ["bash", "-e", "-c", step["run"]],
        cwd=tmp_path,
        env={
            **os.environ,
            "CHANGED_FILES": " ".join(changed_files),
            "PLUGIN_DIRS": ",".join(documents),
            "GITHUB_OUTPUT": str(output),
        },
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("suffix", ["yml", "yaml"])
@pytest.mark.parametrize("v2", [True, False])
def test_metadata_revision_routes_by_schema(
    tmp_path: Path, suffix: str, v2: bool
) -> None:
    directory = "brainscore_vision/models/example"
    filename = f"metadata.{suffix}"
    content = "schema_version: '2.0'\nmodels: {}\n" if v2 else "models: {}\n"
    result = run_metadata_status(
        tmp_path, {directory: {filename: content}}, [f"{directory}/{filename}"]
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "outputs").read_text().splitlines() == [
        f"RUN_METADATA={str(not v2).lower()}",
        f"LEGACY_PLUGIN_DIRS={'' if v2 else directory}",
    ]


def test_mixed_revision_passes_only_legacy_directories(tmp_path: Path) -> None:
    v2 = "brainscore_vision/models/revised"
    legacy = "brainscore_vision/models/legacy"
    result = run_metadata_status(
        tmp_path,
        {
            v2: {"metadata.yaml": "schema_version: '2.0'\nmodels: {}\n"},
            legacy: {"metadata.yml": "models: {}\n"},
        },
        [f"{v2}/metadata.yaml", f"{legacy}/metadata.yml"],
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "outputs").read_text().splitlines() == [
        "RUN_METADATA=true",
        f"LEGACY_PLUGIN_DIRS={legacy}",
    ]


def test_new_plugin_without_metadata_keeps_legacy_generation(tmp_path: Path) -> None:
    directory = "brainscore_vision/models/new"
    result = run_metadata_status(tmp_path, {directory: {}}, [f"{directory}/model.py"])
    assert result.returncode == 0, result.stderr
    assert "RUN_METADATA=true" in (tmp_path / "outputs").read_text()


@pytest.mark.parametrize("suffix", ["yml", "yaml"])
def test_code_change_with_existing_metadata_needs_no_generation(
    tmp_path: Path, suffix: str
) -> None:
    directory = "brainscore_vision/models/example"
    result = run_metadata_status(
        tmp_path,
        {directory: {f"metadata.{suffix}": "models: {}\n"}},
        [f"{directory}/model.py"],
    )
    assert result.returncode == 0, result.stderr
    assert "RUN_METADATA=false" in (tmp_path / "outputs").read_text()


def test_duplicate_metadata_files_fail_before_generation(tmp_path: Path) -> None:
    directory = "brainscore_vision/models/example"
    result = run_metadata_status(
        tmp_path,
        {directory: {"metadata.yml": "models: {}", "metadata.yaml": "models: {}"}},
        [f"{directory}/metadata.yaml"],
    )
    assert result.returncode != 0
    assert "Both metadata suffixes exist" in result.stderr


def test_legacy_job_requires_legacy_work_and_uses_filtered_directories() -> None:
    job = WORKFLOW["jobs"]["process_metadata"]
    assert "METADATA_ONLY == 'true'" in job["if"]
    assert "RUN_METADATA == 'true'" in job["if"]
    assert job["with"]["plugin-dir"] == (
        "${{ needs.process_submission.outputs.LEGACY_PLUGIN_DIRS }}"
    )


@pytest.mark.parametrize(
    ("filename", "metadata_only"),
    [("metadata.yml", True), ("metadata.yaml", True), ("metadata.yaml.bak", False)],
)
def test_metadata_only_filter_matches_complete_filenames(
    tmp_path: Path, filename: str, metadata_only: bool
) -> None:
    step = next(step for step in STEPS if step.get("id") == "check_metadata_only")
    script = step["run"].replace(
        "${{ steps.getpluginfo.outputs.PLUGIN_INFO }}",
        "'{\"modifies_plugins\": true}'",
    )
    script = script.replace(
        "${{ env.CHANGED_FILES }}", f"brainscore_vision/models/example/{filename}"
    )
    output = tmp_path / "outputs"
    result = subprocess.run(
        ["bash", "-e", "-c", script],
        env={**os.environ, "GITHUB_OUTPUT": str(output)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert f"only_metadata={str(metadata_only).lower()}" in output.read_text()
