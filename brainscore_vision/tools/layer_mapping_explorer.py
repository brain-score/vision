"""
Layer Mapping Explorer — generate region_layer_map.json for Brain-Score models.

Wraps the existing LayerScores infrastructure to score all (layer, region) pairs
and produce a recommended mapping. The submitter reviews the CSV evidence and
commits the JSON decision.

Usage:
    python -m brainscore_vision.tools.layer_mapping_explorer \
        --model alexnet \
        --layers features.2,features.5,features.8,features.11 \
        --output-dir ./mappings/
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

from brainscore_vision.model_helpers.brain_transformation import STANDARD_REGION_BENCHMARKS
from brainscore_vision.model_helpers.brain_transformation.neural import LayerScores
from brainscore_vision.model_helpers.activations.pca import LayerPCA

logger = logging.getLogger(__name__)

DEFAULT_REGIONS = ['V1', 'V2', 'V4', 'IT']


def explore_layer_mapping(
    model_identifier: str,
    layers: List[str],
    regions: List[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Run LayerScores for each region and return scores.

    Returns {region: {layer: score, ...}, ...} sorted by score descending.
    """
    if regions is None:
        regions = list(DEFAULT_REGIONS)

    # Load model to get activations_model and visual_degrees
    from brainscore_vision import model_registry
    from brainscore_core.plugin_management import import_plugin
    import_plugin('brainscore_vision', 'models', model_identifier)
    model = model_registry[model_identifier]()

    activations_model = model.activations_model
    visual_degrees = model.visual_degrees()

    # Hook PCA to match LayerSelection behavior (scores are comparable to
    # what ModelCommitment would produce during auto layer search)
    pca_hooked = LayerPCA.is_hooked(activations_model)
    pca_handle = None
    original_id = None
    if not pca_hooked:
        pca_handle = LayerPCA.hook(activations_model, n_components=1000)
        original_id = activations_model.identifier
        activations_model.identifier = original_id + "-pca_1000"

    scoring_id = model_identifier + ("-pca_1000" if not pca_hooked else "")

    layer_scoring = LayerScores(
        model_identifier=scoring_id,
        activations_model=activations_model,
        visual_degrees=visual_degrees,
    )

    try:
        results = {}
        for region in regions:
            benchmark = STANDARD_REGION_BENCHMARKS[region]
            scores = layer_scoring(
                benchmark=benchmark,
                benchmark_identifier=region,
                layers=layers,
                prerun=True,
            )
            region_scores = {}
            for layer in layers:
                score_val = float(scores.sel(layer=layer).item())
                region_scores[layer] = round(score_val, 6)
            results[region] = dict(
                sorted(region_scores.items(), key=lambda x: x[1], reverse=True)
            )
    finally:
        if pca_handle is not None:
            pca_handle.remove()
            activations_model.identifier = original_id

    return results


def suggest_mapping(scores: Dict[str, Dict[str, float]]) -> Dict[str, str]:
    """Pick best layer per region from explore results."""
    mapping = {}
    for region, layer_scores in scores.items():
        best_layer = max(layer_scores, key=layer_scores.get)
        mapping[region] = best_layer
    return mapping


def save_outputs(
    scores: Dict[str, Dict[str, float]],
    mapping: Dict[str, str],
    output_dir: str,
    model_identifier: str,
) -> None:
    """
    Save two files:
    1. region_layer_map.json — the recommended mapping (best layer per region).
       This is what BrainScoreModel loads.
    2. {model_identifier}_layer_scores.csv — full score matrix with columns:
       layer, V1, V2, V4, IT (one row per layer). This is for the submitter
       to review all options before committing.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # region_layer_map.json
    json_path = output_path / 'region_layer_map.json'
    with json_path.open('w') as f:
        json.dump(mapping, f, indent=2)
        f.write('\n')

    # layer_scores.csv — rows in original layer order (from first region)
    csv_path = output_path / f'{model_identifier}_layer_scores.csv'
    regions = list(scores.keys())
    all_layers = list(next(iter(scores.values())).keys())

    with csv_path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['layer'] + regions)
        for layer in all_layers:
            row = [layer] + [scores[region].get(layer, '') for region in regions]
            writer.writerow(row)

    logger.info(f"Saved mapping to {json_path}")
    logger.info(f"Saved scores to {csv_path}")


def batch_generate_mappings(
    model_identifiers: List[str],
    layers_per_model: Optional[Dict[str, List[str]]] = None,
    regions: List[str] = None,
    output_dir: str = './mappings',
) -> None:
    """
    Generate region_layer_map.json and layer_scores.csv for multiple models.

    Output structure:
        output_dir/
        ├── alexnet/
        │   ├── region_layer_map.json
        │   └── alexnet_layer_scores.csv
        └── ...

    Args:
        model_identifiers: list of model identifiers to process.
        layers_per_model: optional {model_id: [layers]} override. If a model
            is not in this dict (or the dict is None), uses the model's own
            layer list from ModelCommitment.
        regions: brain regions to score against.
        output_dir: root output directory.
    """
    from brainscore_vision import model_registry
    from brainscore_core.plugin_management import import_plugin

    for model_id in model_identifiers:
        print(f"\n{'=' * 60}")
        print(f"Processing: {model_id}")
        print(f"{'=' * 60}")

        # Resolve layers
        if layers_per_model and model_id in layers_per_model:
            layers = layers_per_model[model_id]
        else:
            import_plugin('brainscore_vision', 'models', model_id)
            model = model_registry[model_id]()
            layers = model.layers

        try:
            scores = explore_layer_mapping(model_id, layers, regions)
            mapping = suggest_mapping(scores)
            model_output_dir = str(Path(output_dir) / model_id)
            save_outputs(scores, mapping, model_output_dir, model_id)
            _print_results(model_id, scores, mapping)
        except Exception as e:
            print(f"ERROR: {model_id} failed: {e}")
            continue


def _print_results(
    model_identifier: str,
    scores: Dict[str, Dict[str, float]],
    mapping: Dict[str, str],
) -> None:
    """Print formatted results table to stdout."""
    regions = list(scores.keys())
    all_layers = set()
    for region_scores in scores.values():
        all_layers.update(region_scores.keys())
    all_layers = sorted(all_layers)

    col_width = max(len(layer) for layer in all_layers) + 2
    region_width = 10
    header = f"{'layer':<{col_width}}" + "".join(
        f"{r:>{region_width}}" for r in regions
    )

    print(f"\nScores for {model_identifier}:")
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for layer in all_layers:
        row = f"{layer:<{col_width}}"
        for region in regions:
            score = scores[region].get(layer, float('nan'))
            marker = " *" if mapping.get(region) == layer else "  "
            row += f"{score:>{region_width - 2}.4f}{marker}"
        print(row)

    print("-" * len(header))
    print("* = suggested best layer for region")
    print(f"\nSuggested mapping:")
    for region, layer in mapping.items():
        print(f"  {region}: {layer}")


def main():
    parser = argparse.ArgumentParser(
        description='Explore layer-to-region mappings for Brain-Score models',
        prog='python -m brainscore_vision.tools.layer_mapping_explorer',
    )
    parser.add_argument(
        '--model', help='Model identifier (e.g. alexnet)'
    )
    parser.add_argument(
        '--layers',
        help='Comma-separated layer names (e.g. features.2,features.5)',
    )
    parser.add_argument(
        '--regions',
        default=','.join(DEFAULT_REGIONS),
        help=f'Comma-separated region names (default: {",".join(DEFAULT_REGIONS)})',
    )
    parser.add_argument(
        '--output-dir',
        default='./mappings',
        help='Directory to save outputs (default: ./mappings)',
    )
    parser.add_argument(
        '--batch',
        nargs='*',
        help='Batch mode: space-separated model identifiers. '
             'Uses each model\'s own layer list.',
    )

    args = parser.parse_args()
    regions = [r.strip() for r in args.regions.split(',')]

    if args.batch is not None:
        model_ids = args.batch if args.batch else []
        if not model_ids and args.model:
            model_ids = [args.model]
        if not model_ids:
            parser.error('--batch requires model identifiers')
        batch_generate_mappings(model_ids, regions=regions, output_dir=args.output_dir)
    else:
        if not args.model:
            parser.error('--model is required in single-model mode')
        if not args.layers:
            parser.error('--layers is required in single-model mode')
        layers = [l.strip() for l in args.layers.split(',')]

        print(f"Exploring layer mappings for {args.model}...")
        print(f"Layers: {layers}")
        print(f"Regions: {regions}")

        scores = explore_layer_mapping(args.model, layers, regions)
        mapping = suggest_mapping(scores)
        save_outputs(scores, mapping, args.output_dir, args.model)
        _print_results(args.model, scores, mapping)

        print(f"\nOutputs saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
