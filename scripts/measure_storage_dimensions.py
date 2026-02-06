#!/usr/bin/env python
"""
Empirically measure dimension sizes for Brain-Score storage estimation.

This script loads real neural data and model activations to provide
accurate dimension sizes for storage cost calculations.

Usage:
    conda activate vision-2026
    python scripts/measure_storage_dimensions.py

Output:
    - Neural benchmark dimension sizes (presentations, neuroids, time_bins)
    - Model activation dimension sizes (neuroids per layer)
    - Data types and memory footprints
"""

import json
from pathlib import Path
from typing import Any

import numpy as np


def get_assembly_info(assembly: Any) -> dict:
    """Extract dimension and dtype information from a NeuroidAssembly."""
    info = {
        "dims": list(assembly.dims),
        "shape": list(assembly.shape),
        "sizes": dict(assembly.sizes),
        "dtype": str(assembly.values.dtype),
        "bytes_per_element": assembly.values.dtype.itemsize,
        "total_elements": int(np.prod(assembly.shape)),
        "total_bytes": int(assembly.values.nbytes),
        "total_mb": round(assembly.values.nbytes / (1024 ** 2), 3),
    }

    # Extract coordinate metadata if available
    if "neuroid" in assembly.dims:
        neuroid_coords = {}
        for coord_name, coord_data in assembly.coords.items():
            if hasattr(coord_data, "dims") and "neuroid" in coord_data.dims:
                unique_vals = np.unique(coord_data.values)
                neuroid_coords[coord_name] = {
                    "n_unique": len(unique_vals),
                    "examples": list(unique_vals[:5]) if len(unique_vals) <= 10 else list(unique_vals[:3]) + ["..."],
                }
        info["neuroid_coordinates"] = neuroid_coords

    return info


def measure_neural_benchmarks() -> dict:
    """Measure dimensions from various neural benchmark datasets."""
    from brainscore_vision import load_dataset

    results = {}

    # List of datasets to measure
    benchmarks_to_check = [
        # fMRI benchmarks (human, voxel-based)
        ("Coggan2024_fMRI", "fMRI"),
        # Electrophysiology benchmarks (monkey, neuron-based)
        ("MajajHong2015.public", "electrophysiology"),
        ("MajajHong2015.private", "electrophysiology"),
        # TVSD (THINGS Vision Stream Datasets)
        ("Papale2025_train", "electrophysiology_TVSD"),
        ("Papale2025_test", "electrophysiology_TVSD"),
        ("Gifford2022_train", "EEG_TVSD"),
        ("Gifford2022_test", "EEG_TVSD"),
        ("Hebart2023_fmri_train", "fMRI_TVSD"),
        ("Hebart2023_fmri_test", "fMRI_TVSD"),
    ]

    for benchmark_id, data_type in benchmarks_to_check:
        try:
            print(f"\nLoading {benchmark_id}...")
            assembly = load_dataset(benchmark_id)
            info = get_assembly_info(assembly)
            info["data_type"] = data_type
            info["source"] = f"load_dataset('{benchmark_id}')"
            results[benchmark_id] = info
            print(f"  Shape: {info['shape']}")
            print(f"  Dims: {info['dims']}")
            print(f"  Neuroids: {info['sizes'].get('neuroid', 'N/A')}")
            print(f"  Dtype: {info['dtype']}")
        except Exception as e:
            print(f"  Error loading {benchmark_id}: {e}")
            results[benchmark_id] = {"error": str(e)}

    return results


def measure_model_activations() -> dict:
    """Measure activation dimensions from models at different layers."""
    from brainscore_vision import load_model
    from PIL import Image

    results = {}

    # Models and their IT-equivalent layers
    models_to_check = [
        ("alexnet", ["features.2", "features.5", "features.12", "classifier.2"]),
    ]

    # Create dummy images for activation extraction
    print("\nCreating test stimuli...")
    test_images = []

    # Create simple synthetic images (224x224 RGB)
    for i in range(10):  # 10 test images
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        test_images.append(img)

    for model_id, layers in models_to_check:
        try:
            print(f"\nLoading model {model_id}...")
            model = load_model(model_id)

            model_results = {
                "identifier": model_id,
                "layers": {},
            }

            # Get activations for each layer
            print(f"  Extracting activations for {len(test_images)} images...")

            # Access the underlying activations model
            if hasattr(model, "activations_model"):
                activations_model = model.activations_model
            elif hasattr(model, "_activations_model"):
                activations_model = model._activations_model
            else:
                print(f"  Warning: Cannot find activations_model for {model_id}")
                print(f"  Available attributes: {[a for a in dir(model) if not a.startswith('_')]}")
                results[model_id] = {"error": "Cannot find activations_model"}
                continue

            for layer in layers:
                try:
                    # Get activations using the activations model
                    activations = activations_model(test_images, layers=[layer])

                    layer_info = {
                        "shape": list(activations.shape),
                        "dims": list(activations.dims) if hasattr(activations, "dims") else ["presentation", "neuroid"],
                        "dtype": str(activations.values.dtype) if hasattr(activations, "values") else str(type(activations)),
                        "n_stimuli": activations.shape[0],
                        "n_neuroids": int(np.prod(activations.shape[1:])),  # Flatten spatial dims
                        "bytes_per_element": 4,  # Typically float32
                    }
                    layer_info["bytes_per_stimulus"] = layer_info["n_neuroids"] * layer_info["bytes_per_element"]
                    layer_info["mb_per_stimulus"] = round(layer_info["bytes_per_stimulus"] / (1024 ** 2), 6)

                    model_results["layers"][layer] = layer_info
                    print(f"    {layer}: {layer_info['n_neuroids']} neuroids, {layer_info['shape']}")

                except Exception as e:
                    print(f"    {layer}: Error - {e}")
                    model_results["layers"][layer] = {"error": str(e)}

            results[model_id] = model_results

        except Exception as e:
            print(f"  Error loading {model_id}: {e}")
            results[model_id] = {"error": str(e)}

    return results


def measure_benchmark_metadata() -> dict:
    """
    Query benchmark metadata from the database or registry to get
    expected dimension sizes without loading full data.
    """
    # This would query brainscore_benchmark_data_meta and brainscore_benchmark_stimuli_meta
    # For now, we'll document known values from the codebase exploration

    known_benchmarks = {
        "Coggan2024_fMRI": {
            "source": "brainscore_vision/data/coggan2024_fMRI/data_packaging.py",
            "n_stimuli": 24,  # 8 objects × 3 occlusion levels
            "regions": ["V1", "V2", "V4", "IT"],
            "n_subjects": "variable (loaded from dataset)",
            "data_type": "fMRI (voxels)",
        },
        "MajajHong2015": {
            "source": "brainscore_vision/benchmarks/majajhong2015/benchmark.py",
            "n_stimuli_public": 3200,
            "n_stimuli_private": 2560,
            "regions": ["V4", "IT"],
            "species": "Macaque",
            "data_type": "electrophysiology (neurons)",
        },
        "Papale2025": {
            "source": "brainscore_vision/benchmarks/papale2025/benchmark.py",
            "description": "TVSD monkey electrophysiology (spiking activity)",
            "n_stimuli_train": 22248,
            "n_stimuli_test": 3000,
            "regions": ["V1", "V4", "IT"],
            "species": "Macaque",
            "n_subjects": 2,
            "data_type": "electrophysiology (spiking)",
        },
        "Gifford2022": {
            "source": "brainscore_vision/benchmarks/gifford2022/benchmark.py",
            "description": "TVSD human EEG recordings",
            "regions": ["IT"],
            "species": "Human",
            "n_subjects": 10,
            "n_electrodes": 17,
            "n_time_points": 100,
            "data_type": "EEG",
        },
        "Hebart2023_fmri": {
            "source": "brainscore_vision/benchmarks/hebart2023_fmri/benchmark.py",
            "description": "TVSD human fMRI recordings",
            "regions": ["V1", "V2", "V4", "IT"],
            "species": "Human",
            "data_type": "fMRI (voxels)",
        },
    }

    return known_benchmarks


def summarize_for_storage_estimation(
    neural_results: dict,
    model_results: dict,
) -> dict:
    """
    Create a summary suitable for storage cost estimation.
    """
    summary = {
        "neural_data": {
            "typical_neuroid_counts": {},
            "typical_stimulus_counts": {},
            "typical_dtypes": {},
        },
        "model_activations": {
            "typical_neuroid_counts_by_layer": {},
            "it_layer_neuroids": {},
        },
        "recommendations": {},
    }

    # Summarize neural data
    for benchmark_id, info in neural_results.items():
        if "error" not in info:
            n_neuroids = info["sizes"].get("neuroid", 0)
            n_presentations = info["sizes"].get("presentation", 0)
            summary["neural_data"]["typical_neuroid_counts"][benchmark_id] = n_neuroids
            summary["neural_data"]["typical_stimulus_counts"][benchmark_id] = n_presentations
            summary["neural_data"]["typical_dtypes"][benchmark_id] = info["dtype"]

    # Summarize model activations
    for model_id, info in model_results.items():
        if "error" not in info and "layers" in info:
            for layer, layer_info in info["layers"].items():
                if "error" not in layer_info:
                    key = f"{model_id}.{layer}"
                    summary["model_activations"]["typical_neuroid_counts_by_layer"][key] = layer_info["n_neuroids"]

                    # Identify IT-equivalent layers (features.12 for alexnet)
                    if "12" in layer or "IT" in layer.upper():
                        summary["model_activations"]["it_layer_neuroids"][model_id] = layer_info["n_neuroids"]

    # Generate recommendations
    neural_neuroids = list(summary["neural_data"]["typical_neuroid_counts"].values())
    model_neuroids = list(summary["model_activations"]["it_layer_neuroids"].values())

    if neural_neuroids:
        summary["recommendations"]["neural_neuroid_estimate"] = {
            "min": min(neural_neuroids),
            "max": max(neural_neuroids),
            "mean": int(np.mean(neural_neuroids)),
            "note": "Use for per-neuroid storage estimates (neural benchmarks)",
        }

    if model_neuroids:
        summary["recommendations"]["model_neuroid_estimate"] = {
            "min": min(model_neuroids),
            "max": max(model_neuroids),
            "mean": int(np.mean(model_neuroids)),
            "note": "Use for per-neuroid storage estimates (model predictions)",
        }

    return summary


def main():
    print("=" * 60)
    print("Brain-Score Storage Dimension Measurement")
    print("=" * 60)

    results = {
        "neural_benchmarks": {},
        "model_activations": {},
        "known_metadata": {},
        "summary": {},
    }

    # Measure neural benchmarks
    print("\n" + "=" * 60)
    print("SECTION 1: Neural Benchmark Dimensions")
    print("=" * 60)
    try:
        results["neural_benchmarks"] = measure_neural_benchmarks()
    except Exception as e:
        print(f"Error measuring neural benchmarks: {e}")
        results["neural_benchmarks"] = {"error": str(e)}

    # Measure model activations
    print("\n" + "=" * 60)
    print("SECTION 2: Model Activation Dimensions")
    print("=" * 60)
    try:
        results["model_activations"] = measure_model_activations()
    except Exception as e:
        print(f"Error measuring model activations: {e}")
        results["model_activations"] = {"error": str(e)}

    # Get known metadata
    print("\n" + "=" * 60)
    print("SECTION 3: Known Benchmark Metadata")
    print("=" * 60)
    results["known_metadata"] = measure_benchmark_metadata()
    for benchmark, meta in results["known_metadata"].items():
        print(f"\n{benchmark}:")
        for key, val in meta.items():
            print(f"  {key}: {val}")

    # Generate summary
    print("\n" + "=" * 60)
    print("SECTION 4: Summary for Storage Estimation")
    print("=" * 60)
    results["summary"] = summarize_for_storage_estimation(
        results["neural_benchmarks"],
        results["model_activations"],
    )

    # Print summary
    print("\nNeural Data Neuroid Counts:")
    for k, v in results["summary"]["neural_data"]["typical_neuroid_counts"].items():
        print(f"  {k}: {v}")

    print("\nModel IT-Layer Neuroid Counts:")
    for k, v in results["summary"]["model_activations"]["it_layer_neuroids"].items():
        print(f"  {k}: {v}")

    print("\nRecommendations:")
    for k, v in results["summary"]["recommendations"].items():
        print(f"  {k}: {v}")

    # Save results to JSON
    output_path = Path(__file__).parent / "storage_dimension_measurements.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
