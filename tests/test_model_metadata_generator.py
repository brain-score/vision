import runpy
from pathlib import Path
from unittest import TestCase


GENERATOR = runpy.run_path(
    Path(__file__).parents[1] / "scripts" / "generate_model_metadata.py"
)
model_columns = GENERATOR["model_columns"]
merge_model_columns = GENERATOR["merge_model_columns"]
parse_datasets = GENERATOR["parse_datasets"]
parse_curation_confidence = GENERATOR["parse_curation_confidence"]
parse_visual_degrees = GENERATOR["parse_visual_degrees"]


def workbook_rows():
    rows = [[None] * 6 for _ in range(34)]
    rows[1][0] = "model_name"
    rows[2][0] = "base model"
    rows[4][0] = "model_ID"
    rows[5][0] = "architecture_family"
    return rows


class TestModelColumns(TestCase):
    def test_optional_blank_does_not_exclude_model(self):
        rows = workbook_rows()
        rows[1][4] = "AlexNet"

        self.assertEqual(list(model_columns(rows)), [4])

    def test_empty_column_is_ignored(self):
        rows = workbook_rows()

        self.assertEqual(list(model_columns(rows)), [])

    def test_duplicate_model_uses_latest_with_fallback(self):
        rows = workbook_rows()
        styles = [[0] * 6 for _ in range(34)]
        rows[1][4] = "Earlier"
        rows[2][4] = "Base model"
        rows[1][5] = "Latest"

        merged_rows, _ = merge_model_columns(rows, styles, [4, 5])

        self.assertEqual(merged_rows[1][1], "Latest")
        self.assertEqual(merged_rows[2][1], "Base model")


class TestDatasetRoles(TestCase):
    def test_uses_nearest_training_stage(self):
        datasets = parse_datasets("Stage1: WIT-400M; Stage2: ImageNet-1k")

        self.assertEqual(
            [dataset["role"] for dataset in datasets],
            ["pretraining", "fine_tuning"],
        )


class TestAdditionalWorkbookFields(TestCase):
    def test_parses_visual_degrees_with_description(self):
        self.assertEqual(
            parse_visual_degrees("8 degrees (VOneNet family default convention)"),
            8.0,
        )

    def test_normalizes_curation_confidence(self):
        self.assertEqual(parse_curation_confidence("Medium/High"), "medium_high")
