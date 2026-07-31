#!/usr/bin/env python3
"""Generate v2 model metadata from the curation workbook.

The workbook is read without third-party dependencies. Every workbook column
that names a registered model is eligible. Empty fields are omitted from typed
metadata and represented as undocumented provenance.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import urllib.parse
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree as ET


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS = {"x": MAIN_NS}
CURATED_FIELD_ROWS = range(1, 34)
REGISTRY_PATTERN = re.compile(r"model_registry\[\s*(['\"])(.*?)\1\s*\]")
UNKNOWN_VALUES = {
    "",
    "n/a",
    "na",
    "not applicable",
    "not documented",
    "not publicly documented",
    "unknown",
}

FIELD_PATHS = {
    "model_name": "/model/display_name",
    "base model": "/lineage/base_models",
    "model_version": "/model/version",
    "model_ID": "/model/identifier",
    "architecture_family": "/architecture/family",
    "model_recipe/model_process (steps that model takes)": "/training/process",
    "parameter_count": "/architecture/parameter_count",
    "trainable_layer_count": "/architecture/trainable_layers",
    "input_resolution": "/interface/inputs/0/shape",
    "recurrent": "/architecture/recurrent",
    "Dataset_source (training_data)": "/training/datasets",
    "dataset_size": "/training/dataset_summary",
    "supervision_type": "/training/supervision",
    "training_objective": "/training/objectives",
    "weights_provider": "/artifacts/0/provider",
    "checkpoint_identifier": "/artifacts/0/identifier",
    "preprocessing_recipe (Maybe reconsider how we store this)": "/preprocessing",
    "learning rate": "/training/hyperparameters/learning_rate",
    "Batch size": "/training/hyperparameters/batch_size",
    "Test dataset": "/evaluation/test_datasets",
    "Validation dataset": "/evaluation/validation_datasets",
    "Loss function": "/training/loss",
    "Recommended applications": "/intended_use/applications",
    "Target users": "/intended_use/users",
    "Known weaknesses": "/intended_use/limitations",
    "Biases": "/intended_use/biases",
    "Expected input and output format": "/interface/description",
    "Tokenizer?": "/interface/tokenizer",
    "Creator": "/authorship/creators",
    "Organization": "/authorship/organizations",
    "License": "/licenses",
    "Confidence": "/provenance/curation_confidence",
    "Visual Degrees": "/interface/visual_degrees",
}

DATASETS = OrderedDict(
    [
        ("laion-aesthetic", ("LAION-Aesthetic", ("laion-aesthetic", "laion aesthetic"))),
        ("laion-2b", ("LAION-2B", ("laion-2b", "laion2b", "laion-2b-en"))),
        ("wit-400m", ("WIT-400M", ("wit-400m", "webimagetext"))),
        ("imagenet-22k", ("ImageNet-22k", ("imagenet-22k", "imagenet-21,841"))),
        ("imagenet-21k", ("ImageNet-21k", ("imagenet-21k", "in21k"))),
        ("imagenet-12k", ("ImageNet-12k", ("imagenet-12k", "in12k"))),
        (
            "imagenet-1k",
            (
                "ImageNet-1k",
                (
                    "imagenet-1k",
                    "imagenet 1k",
                    "imagenet (ilsvrc 2012)",
                    "imagenet ilsvrc 2012",
                    "ilsvrc2012",
                    "ilsvrc-2012",
                ),
            ),
        ),
    ]
)

BASE_MODEL_MAPPINGS = {
    "AdvProp EfficientNet": (
        "advprop-efficientnet",
        "AdvProp EfficientNet",
        "variant_of",
    ),
    "AlexNet": ("alexnet", "AlexNet", "variant_of"),
    "AlexNet, AlexNet-SIN": ("AlexNet_SIN", "AlexNet SIN", "variant_of"),
    "Standard torchvision ResNet-50": (
        "torchvision-resnet50",
        "Torchvision ResNet-50",
        "variant_of",
    ),
    "CLIP-ConvNeXt-Base image tower": (
        "clip-convnext-base",
        "CLIP ConvNeXt-Base",
        "fine_tuned_from",
    ),
    "ConvNeXt-Base (trained from scratch, not fine-tuned from another checkpoint)": (
        "convnext-base",
        "ConvNeXt-Base",
        "variant_of",
    ),
    "ConvNeXt-Tiny (trained from scratch, not fine-tuned)": (
        "convnext-tiny",
        "ConvNeXt-Tiny",
        "variant_of",
    ),
    "ConvNeXt-Tiny pretrained on ImageNet-12k, then fine-tuned": (
        "convnext-tiny",
        "ConvNeXt-Tiny",
        "fine_tuned_from",
    ),
    "ConvNeXt-Tiny (standard supervised baseline)": (
        "convnext-tiny",
        "ConvNeXt-Tiny",
        "variant_of",
    ),
    "ConvNeXt-XLarge pretrained on ImageNet-22k by the original ConvNeXt paper authors, fine-tuned on ImageNet-1k": (
        "convnext-xlarge",
        "ConvNeXt-XLarge",
        "fine_tuned_from",
    ),
    "CLIP-ConvNeXt-XXLarge image tower from OpenCLIP, pretrained on LAION-2B, model-souped, fine-tuned on ImageNet-1k": (
        "clip-convnext-xxlarge",
        "CLIP ConvNeXt-XXLarge",
        "fine_tuned_from",
    ),
    "ResNet-50": ("resnet-50", "ResNet-50", "derived_from"),
    "CORnet-S": ("CORnet-S", "CORnet-S", "derived_from"),
    "Hybrid ResNet-stem + ViT-Tiny": (
        "hybrid-resnet-vit-tiny",
        "Hybrid ResNet-stem + ViT-Tiny",
        "derived_from",
    ),
    "Vision Transformer, relative-position variant": (
        "relative-position-vit",
        "Relative-position Vision Transformer",
        "variant_of",
    ),
    "CLIP ViT-L/14 (OpenAI, WIT-400M pretraining)": (
        "openai-clip-vit-l14",
        "OpenAI CLIP ViT-L/14",
        "fine_tuned_from",
    ),
    "CLIP ViT-L/14 (OpenAI, WIT-400M)": (
        "openai-clip-vit-l14",
        "OpenAI CLIP ViT-L/14",
        "fine_tuned_from",
    ),
    "CLIP ViT-L/14 via OpenCLIP, pretrained on LAION-2B": (
        "openclip-vit-l14-laion2b",
        "OpenCLIP ViT-L/14 (LAION-2B)",
        "fine_tuned_from",
    ),
    "Vision Transformer architecture (Dosovitskiy et al. 2020); trained from scratch, no separate base checkpoint": (
        "vit-l32",
        "Vision Transformer L/32",
        "variant_of",
    ),
    "OpenCLIP ViT-H/14 image encoder (laion/CLIP-ViT-H-14-laion2B-s32B-b79K)": (
        "openclip-vit-h14-laion2b",
        "OpenCLIP ViT-H/14 (LAION-2B)",
        "fine_tuned_from",
    ),
    "OpenAI CLIP ViT-B/16 image encoder": (
        "openai-clip-vit-b16",
        "OpenAI CLIP ViT-B/16",
        "fine_tuned_from",
    ),
}


def column_index(cell_ref: str) -> int:
    match = re.match(r"([A-Z]+)", cell_ref)
    if not match:
        raise ValueError(f"Invalid cell reference: {cell_ref}")
    index = 0
    for character in match.group(1):
        index = index * 26 + ord(character) - ord("A") + 1
    return index - 1


def text_content(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return "".join(node.text or "" for node in element.iter(f"{{{MAIN_NS}}}t"))


def read_workbook(path: Path) -> tuple[list[list[Any]], list[list[int]], dict[int, str | None]]:
    with zipfile.ZipFile(path) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared_strings = [text_content(item) for item in shared_root.findall("x:si", NS)]

        styles_root = ET.fromstring(archive.read("xl/styles.xml"))
        fills_element = styles_root.find("x:fills", NS)
        fills: list[str | None] = []
        for fill in fills_element if fills_element is not None else []:
            color = fill.find("x:patternFill/x:fgColor", NS)
            fills.append(color.attrib.get("rgb") if color is not None else None)
        cell_styles_element = styles_root.find("x:cellXfs", NS)
        style_fill_colors: dict[int, str | None] = {}
        for index, style in enumerate(
            cell_styles_element if cell_styles_element is not None else []
        ):
            fill_index = int(style.attrib.get("fillId", 0))
            style_fill_colors[index] = fills[fill_index] if fill_index < len(fills) else None

        sheet_root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        rows: list[list[Any]] = []
        styles: list[list[int]] = []
        for row in sheet_root.findall(".//x:sheetData/x:row", NS):
            values_by_column: dict[int, Any] = {}
            styles_by_column: dict[int, int] = {}
            for cell in row.findall("x:c", NS):
                index = column_index(cell.attrib["r"])
                styles_by_column[index] = int(cell.attrib.get("s", 0))
                cell_type = cell.attrib.get("t")
                value_node = cell.find("x:v", NS)
                if cell_type == "s" and value_node is not None:
                    value: Any = shared_strings[int(value_node.text or 0)]
                elif cell_type == "inlineStr":
                    value = text_content(cell.find("x:is", NS))
                elif cell_type == "b" and value_node is not None:
                    value = value_node.text == "1"
                elif value_node is None or value_node.text is None:
                    value = None
                else:
                    raw = value_node.text
                    try:
                        number = float(raw)
                        value = int(number) if number.is_integer() else number
                    except ValueError:
                        value = raw
                values_by_column[index] = value
            width = max(values_by_column, default=-1) + 1
            rows.append([values_by_column.get(index) for index in range(width)])
            styles.append([styles_by_column.get(index, 0) for index in range(width)])
    return rows, styles, style_fill_colors


def cell(rows: list[list[Any]], row: int, column: int) -> Any:
    return rows[row][column] if column < len(rows[row]) else None


def is_unknown(value: Any) -> bool:
    if value is None:
        return True
    if not isinstance(value, str):
        return False
    normalized = value.strip().lower().rstrip(".")
    return normalized in UNKNOWN_VALUES


def string_value(value: Any) -> str | None:
    if is_unknown(value):
        return None
    return str(value).strip()


def parse_parameter_count(value: Any) -> dict[str, Any] | None:
    if is_unknown(value):
        return None
    if isinstance(value, int):
        return {"value": value, "exact": True}
    if isinstance(value, float):
        return {"value": round(value), "exact": value.is_integer()}
    text = str(value).strip()
    exact_integer = re.match(r"^([\d,]+)(?:\s*\(|$)", text)
    if exact_integer:
        integer = int(exact_integer.group(1).replace(",", ""))
        exact = "~" not in text[: exact_integer.end()]
        return {"value": integer, "exact": exact}
    scaled = re.search(r"(~?)(\d+(?:\.\d+)?)\s*([MB])\b", text, re.IGNORECASE)
    if scaled:
        scale = 1_000_000 if scaled.group(3).upper() == "M" else 1_000_000_000
        return {"value": round(float(scaled.group(2)) * scale), "exact": False}
    return {"description": text}


def parse_trainable_layers(value: Any) -> dict[str, Any] | None:
    if is_unknown(value):
        return None
    if isinstance(value, int):
        return {"count": value}
    text = str(value).strip()
    if text.isdigit():
        return {"count": int(text)}
    return {"description": text}


def parse_resolution(value: Any) -> dict[str, int] | None:
    if is_unknown(value):
        return None
    matches = re.findall(r"(\d+)\s*[x×]\s*(\d+)", str(value), re.IGNORECASE)
    if not matches:
        return None
    width, height = matches[-1]
    return {"channels": 3, "height": int(height), "width": int(width)}


def parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if is_unknown(value):
        return None
    normalized = str(value).strip().lower()
    if normalized in {"yes", "true"}:
        return True
    if normalized in {"no", "false"}:
        return False
    return None


def parse_visual_degrees(value: Any) -> float | None:
    if is_unknown(value):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    return float(match.group()) if match else None


def parse_curation_confidence(value: Any) -> str | None:
    text = string_value(value)
    if not text:
        return None
    normalized = re.sub(r"[^a-z]+", "_", text.lower()).strip("_")
    if normalized not in {"low", "medium", "medium_high", "high"}:
        raise ValueError(f"Unsupported curation confidence: {text}")
    return normalized


def parse_base_model(value: Any) -> dict[str, str] | None:
    description = string_value(value)
    if not description or description.lower().startswith("none ("):
        return None
    mapping = BASE_MODEL_MAPPINGS.get(description)
    if not mapping:
        return {"name": description}
    identifier, name, relationship = mapping
    return {
        "identifier": identifier,
        "name": name,
        "relationship": relationship,
    }


def architecture_family(value: str) -> str:
    lowered = value.lower()
    if "v1 front-end" in lowered or "biologically" in lowered or "gabor" in lowered:
        return "hybrid_biological_convolutional"
    if "resnet-vit" in lowered or "hybrid" in lowered and "vit" in lowered:
        return "hybrid_convolutional_transformer"
    if "recurrent" in lowered and ("cnn" in lowered or "conv" in lowered):
        return "recurrent_convolutional_neural_network"
    if "transformer" in lowered or "vit" in lowered:
        return "vision_transformer"
    if "pixel" in lowered:
        return "raw_pixels"
    if "cnn" in lowered or "conv" in lowered:
        return "convolutional_neural_network"
    return "other"


def dataset_role(source: str, match_start: int, dataset_index: int, dataset_count: int) -> str:
    prefix = source[max(0, match_start - 40) : match_start].lower()
    cues = [
        (prefix.rfind(token), role)
        for role, tokens in (
            ("fine_tuning", ("fine-tune", "finetune", "stage2", "stage 2")),
            ("pretraining", ("pretrain", "stage1", "stage 1")),
        )
        for token in tokens
    ]
    cue_position, cue_role = max(cues)
    if cue_position >= 0:
        return cue_role
    if dataset_count > 1:
        return "pretraining" if dataset_index == 0 else "fine_tuning"
    return "training"


def parse_datasets(source_value: Any) -> list[dict[str, Any]]:
    source = string_value(source_value)
    if not source:
        return []
    lowered = source.lower()
    found: list[tuple[int, str, str]] = []
    for identifier, (name, aliases) in DATASETS.items():
        positions = [lowered.find(alias) for alias in aliases if lowered.find(alias) >= 0]
        if positions:
            found.append((min(positions), identifier, name))
    found.sort()
    datasets: list[dict[str, Any]] = []
    for index, (position, identifier, name) in enumerate(found):
        datasets.append(
            {
                "identifier": identifier,
                "name": name,
                "role": dataset_role(source, position, index, len(found)),
            }
        )
    if not datasets:
        datasets.append({"name": source, "role": "training"})
    return datasets


def supervision(value: Any) -> dict[str, str] | None:
    description = string_value(value)
    if not description:
        return None
    lowered = description.lower()
    if "contrastive" in lowered and "supervised" in lowered:
        kind = "mixed"
    elif "weakly" in lowered or "natural-language" in lowered:
        kind = "weakly_supervised"
    elif "self-supervised" in lowered or "self supervised" in lowered:
        kind = "self_supervised"
    elif "supervised" in lowered:
        kind = "supervised"
    elif "unsupervised" in lowered:
        kind = "unsupervised"
    else:
        kind = "other"
    return {"type": kind, "description": description}


def objectives(value: Any) -> list[dict[str, str]]:
    description = string_value(value)
    if not description:
        return []
    lowered = description.lower()
    kinds: list[str] = []
    if any(token in lowered for token in ("contrastive", "infonce", "image-text")):
        kinds.append("contrastive")
    if any(token in lowered for token in ("classification", "class", "cross-entropy")):
        kinds.append("classification")
    if not kinds:
        kinds.append("other")
    return [{"type": kind, "description": description} for kind in kinds]


def list_value(value: Any) -> list[str]:
    text = string_value(value)
    if not text:
        return []
    return [part.strip() for part in re.split(r"\s*;\s*", text) if part.strip()]


def license_value(value: Any) -> list[dict[str, str]]:
    text = string_value(value)
    if not text:
        return []
    spdx = None
    lowered = text.lower()
    if "apache-2.0" in lowered or "apache 2.0" in lowered:
        spdx = "Apache-2.0"
    elif "bsd-3-clause" in lowered or "bsd 3-clause" in lowered:
        spdx = "BSD-3-Clause"
    elif "cc by-nc-sa 4.0" in lowered:
        spdx = "CC-BY-NC-SA-4.0"
    license_record = {"scope": "unspecified", "name": text}
    if spdx:
        license_record["spdx"] = spdx
    return [license_record]


def provenance_status(value: Any, fill_color: str | None) -> str:
    if is_unknown(value):
        return "undocumented"
    text = str(value).lower()
    if any(token in text for token in ("not publicly documented", "not published")):
        return "undocumented"
    if fill_color == "FFFF0000":
        return "undocumented"
    if fill_color == "FFFFE599":
        return "inferred"
    if any(token in text for token in ("assumed", "presumed", "inferred", "unconfirmed")):
        return "inferred"
    if fill_color == "FF93C47D":
        return "verified"
    return "undocumented"


def scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            raise ValueError("YAML output does not support non-finite numbers")
        return str(value)
    if value is None:
        return "null"
    return json.dumps(str(value), ensure_ascii=False)


def yaml_lines(value: Any, indent: int = 0) -> list[str]:
    prefix = " " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, child in value.items():
            rendered_key = key if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", key) else scalar(key)
            if isinstance(child, (dict, list)) and child:
                lines.append(f"{prefix}{rendered_key}:")
                lines.extend(yaml_lines(child, indent + 2))
            elif isinstance(child, (dict, list)):
                lines.append(f"{prefix}{rendered_key}: {'{}' if isinstance(child, dict) else '[]'}")
            else:
                lines.append(f"{prefix}{rendered_key}: {scalar(child)}")
        return lines
    if isinstance(value, list):
        lines = []
        for child in value:
            if isinstance(child, (dict, list)):
                lines.append(f"{prefix}-")
                lines.extend(yaml_lines(child, indent + 2))
            else:
                lines.append(f"{prefix}- {scalar(child)}")
        return lines
    return [f"{prefix}{scalar(value)}"]


def registry_entries(models_root: Path) -> list[tuple[str, Path]]:
    entries: list[tuple[str, Path]] = []
    for init_path in models_root.glob("*/__init__.py"):
        source = init_path.read_text(encoding="utf-8", errors="replace")
        entries.extend((match.group(2), init_path.parent) for match in REGISTRY_PATTERN.finditer(source))
    return entries


def match_model(rows: list[list[Any]], column: int, entries: list[tuple[str, Path]]) -> tuple[str, Path]:
    candidates = {
        str(cell(rows, row, column))
        for row in (0, 1, 4)
        if cell(rows, row, column) not in (None, "")
    }
    matches = [(identifier, directory) for identifier, directory in entries if identifier in candidates]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one registry match for workbook column {cell(rows, 0, column)!r}; "
            f"found {matches!r}"
        )
    return matches[0]


def output_path(identifier: str, directory: Path, entries: list[tuple[str, Path]]) -> Path:
    directory_entries = [registered for registered, path in entries if path == directory]
    if len(directory_entries) == 1:
        return directory / "metadata.yml"
    safe_identifier = urllib.parse.quote(identifier, safe="-_.")
    return directory / "metadata" / safe_identifier / "metadata.yml"


def build_metadata(
    rows: list[list[Any]],
    styles: list[list[int]],
    style_fill_colors: dict[int, str | None],
    column: int,
    identifier: str,
    directory: Path,
) -> dict[str, Any]:
    fields = {str(cell(rows, row, 0)).strip(): cell(rows, row, column) for row in CURATED_FIELD_ROWS}
    repository_path = str(directory.relative_to(directory.parents[2]))
    repository_url = f"https://github.com/brain-score/vision/tree/master/{repository_path}"

    model: dict[str, Any] = {
        "identifier": identifier,
        "display_name": string_value(fields["model_name"]) or identifier,
        "domain": "vision",
    }
    version = string_value(fields["model_version"])
    if version:
        model["version"] = version
    aliases = []
    for candidate in (cell(rows, 0, column), fields["model_name"]):
        if candidate and str(candidate) != identifier and "/" not in str(candidate) and " " not in str(candidate):
            aliases.append(str(candidate))
    if aliases:
        model["aliases"] = sorted(set(aliases))

    architecture_description = string_value(fields["architecture_family"])
    architecture: dict[str, Any] = {}
    if architecture_description:
        architecture.update(
            {
                "family": architecture_family(architecture_description),
                "description": architecture_description,
            }
        )
    parameter_count = parse_parameter_count(fields["parameter_count"])
    if parameter_count:
        architecture["parameter_count"] = parameter_count
    trainable_layers = parse_trainable_layers(fields["trainable_layer_count"])
    if trainable_layers:
        architecture["trainable_layers"] = trainable_layers
    recurrent = parse_bool(fields["recurrent"])
    if recurrent is not None:
        architecture["recurrent"] = recurrent

    metadata: dict[str, Any] = {
        "schema_version": "2.0.0",
        "schema_url": "https://raw.githubusercontent.com/brain-score/vision/master/docs/model_metadata/model-metadata-v2.schema.json",
        "model": model,
    }
    if architecture:
        metadata["architecture"] = architecture

    base_model = parse_base_model(fields["base model"])
    if base_model:
        metadata["lineage"] = {"base_models": [base_model]}

    input_record: dict[str, Any] = {"modality": "image"}
    resolution = parse_resolution(fields["input_resolution"])
    if resolution:
        input_record["shape"] = resolution
    interface: dict[str, Any] = {"inputs": [input_record]}
    interface_description = string_value(fields["Expected input and output format"])
    if interface_description:
        interface["description"] = interface_description
    tokenizer = parse_bool(fields["Tokenizer?"])
    if tokenizer is not None:
        interface["tokenizer"] = tokenizer
    visual_degrees = parse_visual_degrees(fields["Visual Degrees"])
    if visual_degrees is not None:
        interface["visual_degrees"] = visual_degrees
        interface["visual_degrees_description"] = str(fields["Visual Degrees"]).strip()
    metadata["interface"] = interface

    preprocessing_description = string_value(
        fields["preprocessing_recipe (Maybe reconsider how we store this)"]
    )
    if preprocessing_description:
        metadata["preprocessing"] = {"description": preprocessing_description}

    training: dict[str, Any] = {}
    process = string_value(fields["model_recipe/model_process (steps that model takes)"])
    if process:
        training["process"] = process
    datasets = parse_datasets(fields["Dataset_source (training_data)"])
    dataset_size = fields["dataset_size"]
    if datasets and isinstance(dataset_size, int):
        datasets[-1]["sample_count"] = dataset_size
        datasets[-1]["sample_unit"] = "images"
    if datasets:
        training["datasets"] = datasets
    dataset_summary = None if isinstance(dataset_size, int) else string_value(dataset_size)
    if dataset_summary:
        training["dataset_summary"] = dataset_summary
    supervision_record = supervision(fields["supervision_type"])
    if supervision_record:
        training["supervision"] = supervision_record
    objective_records = objectives(fields["training_objective"])
    if objective_records:
        training["objectives"] = objective_records
    hyperparameters: dict[str, Any] = {}
    learning_rate = string_value(fields["learning rate"])
    if learning_rate:
        hyperparameters["learning_rate"] = {"description": learning_rate}
    batch_size = fields["Batch size"]
    if isinstance(batch_size, int):
        hyperparameters["batch_size"] = {"value": batch_size}
    else:
        batch_description = string_value(batch_size)
        if batch_description:
            hyperparameters["batch_size"] = {"description": batch_description}
    if hyperparameters:
        training["hyperparameters"] = hyperparameters
    loss = string_value(fields["Loss function"])
    if loss:
        training["loss"] = {"description": loss}
    metadata["training"] = training

    evaluation: dict[str, Any] = {}
    test_dataset = string_value(fields["Test dataset"])
    if test_dataset:
        evaluation["test_datasets"] = [{"description": test_dataset}]
    validation_dataset = string_value(fields["Validation dataset"])
    if validation_dataset:
        evaluation["validation_datasets"] = [{"description": validation_dataset}]
    if evaluation:
        metadata["evaluation"] = evaluation

    artifacts: list[dict[str, Any]] = []
    weights: dict[str, Any] = {"role": "weights"}
    provider = string_value(fields["weights_provider"])
    checkpoint = string_value(fields["checkpoint_identifier"])
    if provider:
        weights["provider"] = provider
    if checkpoint:
        weights["identifier"] = checkpoint
    if len(weights) > 1:
        artifacts.append(weights)
    artifacts.append(
        {
            "role": "source_code",
            "provider": "Brain-Score",
            "url": repository_url,
            "path": repository_path,
        }
    )
    metadata["artifacts"] = artifacts

    intended_use = {
        "applications": list_value(fields["Recommended applications"]),
        "users": list_value(fields["Target users"]),
        "limitations": list_value(fields["Known weaknesses"]),
        "biases": list_value(fields["Biases"]),
    }
    intended_use = {key: value for key, value in intended_use.items() if value}
    if intended_use:
        metadata["intended_use"] = intended_use

    authorship = {
        "creators": list_value(fields["Creator"]),
        "organizations": list_value(fields["Organization"]),
    }
    authorship = {key: value for key, value in authorship.items() if value}
    if authorship:
        metadata["authorship"] = authorship

    licenses = license_value(fields["License"])
    if licenses:
        metadata["licenses"] = licenses

    source_url = cell(rows, 34, column) if len(rows) > 34 else None
    if source_url:
        metadata["references"] = [{"role": "primary", "url": str(source_url)}]

    assertions = []
    for row in CURATED_FIELD_ROWS:
        field_name = str(cell(rows, row, 0)).strip()
        value = cell(rows, row, column)
        style_id = styles[row][column] if column < len(styles[row]) else 0
        assertions.append(
            {
                "path": FIELD_PATHS[field_name],
                "status": provenance_status(value, style_fill_colors.get(style_id)),
                "source": "curation_workbook",
            }
        )
    provenance = {
        "sources": {
            "curation_workbook": {
                "type": "curation",
                "title": "Brainscore Model Metadata.xlsx",
            },
            "implementation": {
                "type": "source_code",
                "url": repository_url,
                "path": repository_path,
            },
        },
        "assertions": assertions,
    }
    curation_confidence = parse_curation_confidence(fields["Confidence"])
    if curation_confidence:
        provenance["curation_confidence"] = curation_confidence
    metadata["provenance"] = provenance
    return metadata


def validate_generated(metadata: dict[str, Any]) -> None:
    if metadata.get("schema_version") != "2.0.0":
        raise ValueError("schema_version must be 2.0.0")
    model = metadata.get("model", {})
    if not model.get("identifier") or model.get("domain") != "vision":
        raise ValueError("model.identifier and vision domain are required")
    architecture_family_value = metadata.get("architecture", {}).get("family")
    if architecture_family_value is not None and architecture_family_value not in {
        "convolutional_neural_network",
        "vision_transformer",
        "recurrent_convolutional_neural_network",
        "hybrid_convolutional_transformer",
        "hybrid_biological_convolutional",
        "raw_pixels",
        "other",
    }:
        raise ValueError("architecture.family is invalid")
    assertions = metadata.get("provenance", {}).get("assertions", [])
    if len(assertions) != len(CURATED_FIELD_ROWS):
        raise ValueError("every curated workbook field must have a provenance assertion")


def resolve_schema_reference(root_schema: dict[str, Any], reference: str) -> dict[str, Any]:
    if not reference.startswith("#/"):
        raise ValueError(f"Only local schema references are supported: {reference}")
    resolved: Any = root_schema
    for component in reference[2:].split("/"):
        key = component.replace("~1", "/").replace("~0", "~")
        resolved = resolved[key]
    return resolved


def validate_json_schema(
    value: Any,
    schema: dict[str, Any],
    root_schema: dict[str, Any],
    path: str = "$",
) -> None:
    if "$ref" in schema:
        validate_json_schema(
            value,
            resolve_schema_reference(root_schema, schema["$ref"]),
            root_schema,
            path,
        )
        return

    if "const" in schema and value != schema["const"]:
        raise ValueError(f"{path}: expected constant {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path}: {value!r} is not in {schema['enum']!r}")

    schema_type = schema.get("type")
    type_matches = {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }
    if schema_type and not type_matches[schema_type]:
        raise ValueError(f"{path}: expected {schema_type}, got {type(value).__name__}")

    if "anyOf" in schema:
        branch_errors = []
        for branch in schema["anyOf"]:
            try:
                validate_json_schema(value, branch, root_schema, path)
                break
            except ValueError as error:
                branch_errors.append(str(error))
        else:
            raise ValueError(f"{path}: no anyOf branch matched: {branch_errors}")

    if isinstance(value, dict):
        for required_key in schema.get("required", []):
            if required_key not in value:
                raise ValueError(f"{path}: missing required property {required_key!r}")
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        for key, child in value.items():
            child_path = f"{path}/{key}"
            if key in properties:
                validate_json_schema(child, properties[key], root_schema, child_path)
            elif additional is False:
                raise ValueError(f"{child_path}: additional property is not allowed")
            elif isinstance(additional, dict):
                validate_json_schema(child, additional, root_schema, child_path)
        if len(value) < schema.get("minProperties", 0):
            raise ValueError(f"{path}: too few properties")

    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            raise ValueError(f"{path}: too few items")
        if "items" in schema:
            for index, child in enumerate(value):
                validate_json_schema(child, schema["items"], root_schema, f"{path}/{index}")

    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            raise ValueError(f"{path}: string is too short")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            raise ValueError(f"{path}: string does not match {schema['pattern']!r}")
        if schema.get("format") == "uri":
            parsed = urllib.parse.urlparse(value)
            if not parsed.scheme:
                raise ValueError(f"{path}: expected an absolute URI")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise ValueError(f"{path}: value is below minimum {schema['minimum']}")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            raise ValueError(
                f"{path}: value must be greater than {schema['exclusiveMinimum']}"
            )


def model_columns(rows: list[list[Any]]) -> Iterable[int]:
    width = len(rows[0])
    for column in range(4, width):
        fields = {
            str(cell(rows, row, 0)).strip(): cell(rows, row, column)
            for row in CURATED_FIELD_ROWS
        }
        if string_value(fields.get("model_name")) or string_value(fields.get("model_ID")):
            yield column


def merge_model_columns(
    rows: list[list[Any]], styles: list[list[int]], columns: list[int]
) -> tuple[list[list[Any]], list[list[int]]]:
    merged_rows: list[list[Any]] = []
    merged_styles: list[list[int]] = []
    for row_index in range(len(rows)):
        value = cell(rows, row_index, columns[-1])
        style = (
            styles[row_index][columns[-1]]
            if columns[-1] < len(styles[row_index])
            else 0
        )
        if is_unknown(value):
            for column in reversed(columns[:-1]):
                candidate = cell(rows, row_index, column)
                if not is_unknown(candidate):
                    value = candidate
                    style = styles[row_index][column] if column < len(styles[row_index]) else 0
                    break
        merged_rows.append([cell(rows, row_index, 0), value])
        merged_styles.append([0, style])
    return merged_rows, merged_styles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workbook", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    models_root = repo_root / "brainscore_vision" / "models"
    rows, styles, style_fill_colors = read_workbook(args.workbook)
    entries = registry_entries(models_root)
    schema = json.loads(
        (repo_root / "docs" / "model_metadata" / "model-metadata-v2.schema.json").read_text(
            encoding="utf-8"
        )
    )

    grouped_columns: OrderedDict[tuple[str, Path], list[int]] = OrderedDict()
    for column in model_columns(rows):
        identifier, directory = match_model(rows, column, entries)
        grouped_columns.setdefault((identifier, directory), []).append(column)

    written: list[Path] = []
    for (identifier, directory), columns in grouped_columns.items():
        model_rows, model_styles = merge_model_columns(rows, styles, columns)
        metadata = build_metadata(
            model_rows,
            model_styles,
            style_fill_colors,
            1,
            identifier,
            directory,
        )
        validate_generated(metadata)
        validate_json_schema(metadata, schema, schema)
        target = output_path(identifier, directory, entries)
        if not args.dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("\n".join(yaml_lines(metadata)) + "\n", encoding="utf-8")
        written.append(target.relative_to(repo_root))

    print(f"{'Would generate' if args.dry_run else 'Generated'} {len(written)} metadata files:")
    for path in written:
        print(path)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise
