import os
import re
import sys
import logging
import requests
import contextlib
import torch.nn as nn
from typing import Optional, Dict, Any, List
from brainscore_core.plugin_management.import_plugin import import_plugin
from brainscore_core.plugin_management.domain_plugin_interface import DomainPluginInterface
from brainscore_vision import load_benchmark

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# output suppression to ensure github action logs don't have random print statements
@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


class VisionDomainPlugin(DomainPluginInterface):
    """
    Vision-specific implementation of the domain plugin interface.
    
    This class provides vision-specific implementations for loading benchmarks
    and creating metadata using brainscore_vision framework.
    """

    def __init__(self, benchmark_type: str = "neural"):
        """
        Initialize the vision domain plugin.
        
        :param benchmark_type: str, type of benchmarks to handle ("neural", "behavioral", "engineering").
        """
        self.benchmark_type = benchmark_type

    def load_benchmark(self, identifier: str) -> Optional[object]:
        """
        Load a benchmark using brainscore_vision and return the benchmark instance.

        :param identifier: str, the unique name of the benchmark to load.
        :return: Optional[object], the benchmark instance if successfully loaded, otherwise None.

        Notes:
        - Uses import_plugin to dynamically load the benchmark from brainscore_vision.benchmark.
        - Retrieves the benchmark instance from benchmark_registry using the given identifier.
        - Returns None if an error occurs during benchmark loading.
        - Prints an error message if the benchmark fails to load.
        """
        try:
            return load_benchmark(identifier)
        except Exception as e:
            error_message = f"ERROR: Failed to load benchmark '{identifier}': {e}"
            print(error_message, file=sys.stderr)
            return None

    def create_stimuli_metadata(self, plugin: Any, plugin_dir_name: str) -> Dict[str, Any]:
        """
        Create stimuli metadata for vision benchmarks.
        
        :param plugin: The benchmark plugin instance.
        :param plugin_dir_name: str, name of the plugin directory.
        :return: Dict[str, Any], stimuli metadata dictionary.
        """
        def get_num_stimuli(stimulus_set):
            try:
                num_stimuli = len(stimulus_set)
                return num_stimuli
            except TypeError:
                return None

        def total_size_mb(stimulus_set):
            try:
                size = round(float(stimulus_set.memory_usage(deep=True).sum() / (1024 ** 2)), 4)
                return size
            except AttributeError:
                return None

        try:
            stimulus_set = plugin._assembly.stimulus_set
        except AttributeError:
            try:
                stimulus_set = plugin.stimulus_set
            except AttributeError:
                stimulus_set = None

        new_metadata = {
            "num_stimuli": get_num_stimuli(stimulus_set),
            "datatype": "image",
            "stimuli_subtype": None,
            "total_size_mb": total_size_mb(stimulus_set),
            "brainscore_link": f"https://github.com/brain-score/vision/tree/master/brainscore_vision/data/{plugin_dir_name}",
            "extra_notes": None
        }
        return new_metadata

    def create_data_metadata(self, benchmark: Any, benchmark_dir_name: str) -> Dict[str, Any]:
        """
        Create data metadata for vision benchmarks.
        
        :param benchmark: The benchmark instance.
        :param benchmark_dir_name: str, name of the benchmark directory.
        :return: Dict[str, Any], data metadata dictionary.
        """
        try:
            assembly = benchmark._assembly
        except AttributeError:
            try:
                assembly = benchmark.assembly
            except AttributeError:
                assembly = None

        def get_hemisphere(assembly):
            try:
                hemisphere_values = set(assembly.hemisphere.values)
                # If multiple hemispheres, prefer "L" for consistency
                # Appears to be the case for Rajalingham2020 where there is both 'L' and 'R' but yaml contains 'L' but test produce 'R'
                if "L" in hemisphere_values:
                    return "L"
                elif hemisphere_values:
                    return sorted(list(hemisphere_values))[0]  # Return first alphabetically for consistency
                else:
                    return None
            except AttributeError:
                return None

        def get_num_subjects(assembly):
            try:
                num_subjects = len(set(assembly.subject.values))
                return num_subjects
            except AttributeError:
                return None

        def get_region(assembly):
            try:
                region = list(set(assembly.region.values))[0]
                return region
            except AttributeError:
                return None

        def get_datatype():
            if self.benchmark_type == "engineering":
                return "engineering"
            elif self.benchmark_type == "behavioral":
                return "behavioral"
            else:  # either neural or unspecified will return None
                return None

        new_metadata = {
            "benchmark_type": self.benchmark_type,
            "task": None,
            "region": get_region(assembly),
            "hemisphere": get_hemisphere(assembly),
            "num_recording_sites": None,
            "duration_ms": None,
            "species": None,
            "datatype": get_datatype(),
            "num_subjects": get_num_subjects(assembly),
            "pre_processing": None,
            "brainscore_link": f"https://github.com/brain-score/vision/tree/master/brainscore_vision/data/{benchmark_dir_name}",
            "extra_notes": None,
            "data_publicly_available": True
        }

        return new_metadata

    def create_metric_metadata(self, plugin: Any, plugin_dir_name: str) -> Dict[str, Any]:
        """
        Create metric metadata for vision benchmarks.
        
        :param plugin: The benchmark plugin instance.
        :param plugin_dir_name: str, name of the plugin directory.
        :return: Dict[str, Any], metric metadata dictionary.
        """
        new_metadata = {
            "type": None,
            "reference": None,
            "public": None,
            "brainscore_link": f"https://github.com/brain-score/vision/tree/master/brainscore_vision/benchmarks/{plugin_dir_name}",
            "extra_notes": None
        }

        return new_metadata

    def find_registered_benchmarks(self, root_folder: str) -> List[str]:
        """
        Find all registered benchmarks inside __init__.py files within a given root directory.

        :param root_folder: str, the root directory to search for benchmark registrations.
        :return: List[str], a list of benchmark names found in benchmark_registry assignments.

        Notes:
        - Recursively searches for __init__.py files in the specified directory.
        - Extracts benchmark names assigned to benchmark_registry[...] using regex.
        - Logs an error message if any __init__.py file cannot be read.
        """
        registered_benchmarks = []
        init_file_path = os.path.join(root_folder, "__init__.py")

        # Ensure that root_folder is a benchmark directory (must contain __init__.py)
        if not os.path.isfile(init_file_path):
            print(f"ERROR: {root_folder} does not contain an `__init__.py` file.", file=sys.stderr)
            return []
        try:
            with open(init_file_path, "r", encoding="utf-8") as file:
                content = file.read()
            # Match both lambda and class assignments to benchmark_registry
            matches = re.findall(r'benchmark_registry\[\s*["\'](.*?)["\']\s*\]\s*=\s*([^\s\n]+)', content, re.DOTALL)
            if matches:
                # Extract just the benchmark names (first group in each match)
                registered_benchmarks.extend([match[0] for match in matches])
        except Exception as e:
            print(f"ERROR: Could not read {init_file_path}: {e}", file=sys.stderr)
        return registered_benchmarks

    # ============================================================================
    # MODEL-RELATED METHODS
    # ============================================================================

    def load_model(self, identifier: str) -> Optional[object]:
        """Load a vision model using brainscore_vision."""
        try:
            import_plugin('brainscore_vision', 'models', identifier)
            from brainscore_vision import model_registry
            with suppress_output():
                model_instance = model_registry[identifier]()
            return model_instance
        except Exception as e:
            print(f"ERROR: Failed to load model '{identifier}': {e}")
            return None

    def find_registered_models(self, root_folder: str) -> List[str]:
        """Find registered models in vision __init__.py files."""
        registered_models = []
        init_file_path = os.path.join(root_folder, "__init__.py")

        if not os.path.isfile(init_file_path):
            print(f"ERROR: {root_folder} does not contain an `__init__.py` file.")
            return []

        try:
            with open(init_file_path, "r", encoding="utf-8") as file:
                content = file.read()
            matches = re.findall(r'model_registry\[\s*["\'](.*?)["\']\s*\]\s*=\s*\\?\s*lambda\s*:', content, re.DOTALL)
            if matches:
                registered_models.extend(matches)
        except Exception as e:
            print(f"ERROR: Could not read {init_file_path}: {e}")
        
        return registered_models

    def extract_model_for_analysis(self, model: Any) -> Any:
        """Extract the underlying PyTorch model from vision model wrapper."""
        try:
            # Vision models have this structure: model.activations_model._model
            return model.activations_model._model
        except AttributeError:
            # Fallback: return the model as-is if extraction fails
            return model

    def detect_model_architecture(self, model: Any, model_name: str) -> str:
        """Detect vision model architecture types."""
        tags = {"DCNN"}  # Default for vision models
        
        # Vision-specific hardcoded patterns
        if re.search(r'cor[_-]*net', model_name, re.IGNORECASE):
            tags.add("RNN")
            tags.add("SKIP_CONNECTIONS")  # hardcode cornets
        
        # Detect transformer components
        if any(
                isinstance(layer, (nn.MultiheadAttention, nn.LayerNorm)) or
                'transformer' in layer.__class__.__name__.lower() or
                'attention' in layer.__class__.__name__.lower()
                for layer in model.modules()
        ):
            tags.add("Transformer")
        
        # Detect RNN components
        if any(isinstance(layer, (nn.RNN, nn.LSTM, nn.GRU)) for layer in model.modules()):
            tags.add("RNN")
        
        return ", ".join(sorted(tags))

    def get_model_family(self, model_name: str) -> Optional[str]:
        """Extract vision model family using vision-specific patterns."""
        families = []

        # Vision-specific model families
        known_families = {
            "resnet": r'resnet',
            "resnext": r'resnext',
            "alexnet": r'alexnet',
            "efficientnet": r'efficientnet|effnet',
            "convnext": r'convnext',
            "vit": r'vit|visiontransformer',
            "densenet": r'densenet',
            "nasnet": r'nasnet',
            "pnasnet": r'pnasnet',
            "inception": r'inception',
            "swin": r'swin',
            "mobilenet": r'mobilenet|mobilevit',
            "mvit": r'mvit',
            "slowfast": r'slowfast',
            "i3d": r'i3d',
            "x3d": r'x3d',
            "timesformer": r'timesformer',
            "s3d": r's3d',
            "r3d": r'r3d',
            "r2plus1d": r'r2plus1d',
            "deit": r'deit',
            "cornet": r'cornet',  # Vision-specific
            "vgg": r'vgg',
            "clip": r'clip',
            "cvt": r'cvt',
            "vone": r'vone'  # Vision-specific
        }
        
        for family, pattern in known_families.items():
            if re.search(pattern, model_name, re.IGNORECASE):
                families.append(family)
        
        return ", ".join(sorted(families)) if families else None

    def get_huggingface_link(self, model_name: str) -> Optional[str]:
        """Check if a Hugging Face model repository exists."""
        sanitized_model_name = model_name.replace(":", "-").replace("/", "-")
        hf_url = f"https://huggingface.co/{sanitized_model_name}"
        try:
            response = requests.head(hf_url, timeout=1)
            if response.status_code == 200:
                return hf_url
            logger.info(f"HuggingFace link for {model_name} found.")
        except requests.RequestException as e:
            logger.info(f"WARNING: checking HuggingFace link for '{model_name}': {e} failed.")
        return None

    def create_model_metadata(self, model: Any, model_name: str, model_dir_name: str) -> Dict[str, Any]:
        """Create comprehensive model metadata for vision models."""
        architecture_type = self.detect_model_architecture(model, model_name)
        
        return {
            "architecture": architecture_type,
            "model_family": self.get_model_family(model_name),
            "total_parameter_count": sum(p.numel() for p in model.parameters()),
            "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total_layers": sum(1 for _ in model.modules()),
            "trainable_layers": sum(1 for p in model.parameters() if p.requires_grad and p.dim() > 1),
            "model_size_mb": round(sum(p.element_size() * p.numel() for p in model.parameters()) / 1e6, 2),
            "training_dataset": None,
            "task_specialization": None,
            "brainscore_link": f"https://github.com/brain-score/vision/tree/master/brainscore_vision/models/{model_dir_name}",
            "huggingface_link": self.get_huggingface_link(model_name),
            "extra_notes": None,
            "runnable": True
        } 
