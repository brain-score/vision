import os
import re
import sys
import yaml
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

    def _extract_data_id(self, plugin) -> Optional[str]:
        """Extract data ID from benchmark plugin."""
        try:
            # For multi-region assemblies, check benchmark identifier first to get region-specific ID
            if hasattr(plugin, 'identifier'):
                identifier = plugin.identifier
                
                # Handle multi-region data assemblies like Coggan
                if 'tong.Coggan2024_fMRI' in identifier:
                    # tong.Coggan2024_fMRI.V1-rdm -> tong.Coggan2024_fMRI.V1
                    parts = identifier.split('.')
                    if len(parts) >= 3:
                        region_part = parts[2].split('-')[0]  # V1 from V1-rdm
                        return f"tong.Coggan2024_fMRI.{region_part}"
                    return "tong.Coggan2024_fMRI"
                
                # Handle patterns like "Geirhos2021colour-top1" -> "Geirhos2021_colour"
                elif 'Geirhos2021' in identifier and '-' in identifier:
                    parts = identifier.split('-')
                    if len(parts) >= 2:
                        return parts[0].replace('Geirhos2021', 'Geirhos2021_')
            
            # Try assembly-based identifiers for single-assembly cases
            if hasattr(plugin, '_assembly') and hasattr(plugin._assembly, 'identifier'):
                assembly_id = plugin._assembly.identifier
                # If this is a multi-region case but we have a region, append it
                if hasattr(plugin, 'region') and plugin.region:
                    return f"{assembly_id}.{plugin.region}"
                return assembly_id
            elif hasattr(plugin, '_stimulus_set') and hasattr(plugin._stimulus_set, 'identifier'):
                return plugin._stimulus_set.identifier
            elif hasattr(plugin, 'stimulus_set') and hasattr(plugin.stimulus_set, 'identifier'):
                return plugin.stimulus_set.identifier
            elif hasattr(plugin, '_data_identifier'):
                return plugin._data_identifier
            elif hasattr(plugin, 'data_identifier'):
                return plugin.data_identifier
            elif hasattr(plugin, 'identifier'):
                return plugin.identifier
            elif hasattr(plugin, '_identifier'):
                return plugin._identifier
            return None
        except:
            return None

    def _extract_data_plugin_name(self, data_id: str) -> Optional[str]:
        """Extract plugin name from data ID. E.g., Ferguson2024_circle_line -> ferguson2024"""
        if not data_id:
            return None
        
        # Handle different naming patterns
        if data_id.startswith('Ferguson2024'):
            return 'ferguson2024'
        elif data_id.startswith('MajajHong2015') or data_id.startswith('dicarlo.MajajHong2015'):
            return 'majajhong2015'
        elif data_id.startswith('Geirhos2021'):
            return 'geirhos2021'
        elif 'Coggan2024' in data_id:
            if 'fMRI' in data_id:
                return 'coggan2024_fmri'
            elif 'behavior' in data_id:
                return 'coggan2024_behavior'
            else:
                return 'coggan2024'
        elif data_id.startswith('tong.Coggan2024'):
            return 'coggan2024_fmri'  # tong.Coggan2024_fMRI format
        elif data_id.startswith('dicarlo.'):
            # Handle dicarlo.MajajHong2015.* format
            parts = data_id.split('.')
            if len(parts) >= 2 and 'MajajHong2015' in parts[1]:
                return 'majajhong2015'
        elif '_' in data_id:
            return data_id.split('_')[0].lower()
        elif '.' in data_id:
            return data_id.split('.')[0].lower()
        else:
            return data_id.lower()

    def _extract_metric_id(self, plugin) -> Optional[str]:
        """Extract metric ID from benchmark plugin."""
        try:
            if hasattr(plugin, '_metric') and hasattr(plugin._metric, '__class__'):
                metric_class_name = plugin._metric.__class__.__name__
                metric_module = plugin._metric.__class__.__module__
                
                # Map class names to metric IDs
                if 'ValueDelta' in metric_class_name:
                    return 'value_delta'
                elif 'PLS' in metric_class_name or 'Regression' in metric_class_name:
                    return 'pls'
                elif 'Accuracy' in metric_class_name:
                    return 'accuracy'
                elif 'ErrorConsistency' in metric_class_name:
                    return 'error_consistency'
                
                # Check metric module/function names for additional patterns
                if 'regression_correlation' in metric_module:
                    return 'pls'
                elif 'value_delta' in metric_module:
                    return 'value_delta'
                elif 'accuracy' in metric_module:
                    return 'accuracy'
                elif 'error_consistency' in metric_module:
                    return 'error_consistency'
            
            # Also check benchmark identifier for metric hints
            if hasattr(plugin, 'identifier'):
                identifier = plugin.identifier.lower()
                if 'pls' in identifier:
                    return 'pls'
                elif 'value_delta' in identifier:
                    return 'value_delta'
                elif 'top1' in identifier or 'accuracy' in identifier:
                    return 'accuracy'
                elif 'error_consistency' in identifier:
                    return 'error_consistency'
                elif 'rdm' in identifier:
                    return 'rdm'
                elif 'coggan' in identifier and 'fmri' in identifier:
                    return 'rdm'  # Coggan fMRI benchmarks use RDM
                elif 'coggan' in identifier and 'behavior' in identifier:
                    return 'custom'  # Coggan behavior benchmarks use custom metrics
                    
            return None
        except:
            return None

    def _load_metadata_template(self, plugin_type: str, plugin_name: str) -> Dict[str, Any]:
        """Load metadata template from plugin folder."""
        try:
            # Get the directory where this script is located
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up one level to brainscore_vision
            brainscore_vision_dir = os.path.dirname(current_dir)
            
            metadata_path = os.path.join(brainscore_vision_dir, plugin_type, plugin_name, 'metadata.yaml')
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    return yaml.safe_load(f) or {}
            return {}
        except Exception as e:
            print(f"Warning: Could not load {plugin_type}/{plugin_name}/metadata.yaml: {e}", file=sys.stderr)
            return {}

    def _resolve_data_metadata(self, data_id: str, data_plugin_name: str, metadata_type: str) -> Dict[str, Any]:
        """Resolve data metadata using inheritance system."""
        data_metadata = self._load_metadata_template('data', data_plugin_name)
        
        if not data_metadata:
            return {}
        
        # Get defaults for the metadata type
        defaults = data_metadata.get('defaults', {}).get(metadata_type, {})
        
        # Get dataset-specific overrides
        dataset_overrides = data_metadata.get('data', {}).get(data_id, {}).get(metadata_type, {})
        
        # Merge defaults with overrides (overrides take precedence)
        resolved_metadata = {**defaults, **dataset_overrides}
        
        return resolved_metadata

    def create_inheritance_metadata(self, plugin: Any, plugin_dir_name: str) -> Dict[str, Any]:
        """
        Create inheritance-based metadata (data_id/metric_id format) instead of expanded metadata.
        
        :param plugin: The benchmark plugin instance.
        :param plugin_dir_name: str, name of the plugin directory.
        :return: Dict[str, Any], inheritance metadata dictionary.
        """
        try:
            # Extract data_id and metric_id from the plugin
            data_id = self._extract_data_id(plugin)
            metric_id = self._extract_metric_id(plugin)
            
            if data_id and metric_id:
                print(f"Creating inheritance metadata: data_id={data_id}, metric_id={metric_id}", file=sys.stderr)
                return {
                    "data_id": data_id,
                    "metric_id": metric_id
                }
            else:
                print(f"Warning: Could not extract data_id ({data_id}) or metric_id ({metric_id}) for inheritance format", file=sys.stderr)
                # Fall back to expanded format
                return {
                    "stimulus_set": self.create_stimuli_metadata(plugin, plugin_dir_name),
                    "data": self.create_data_metadata(plugin, plugin_dir_name),
                    "metric": self.create_metric_metadata(plugin, plugin_dir_name),
                }
        except Exception as e:
            print(f"Error creating inheritance metadata: {e}", file=sys.stderr)
            # Fall back to expanded format
            return {
                "stimulus_set": self.create_stimuli_metadata(plugin, plugin_dir_name),
                "data": self.create_data_metadata(plugin, plugin_dir_name),
                "metric": self.create_metric_metadata(plugin, plugin_dir_name),
            }

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
        # NEW: Try to resolve from data plugin metadata first
        try:
            data_id = self._extract_data_id(plugin)
            data_plugin_name = self._extract_data_plugin_name(data_id)
            
            if data_id and data_plugin_name:
                resolved_metadata = self._resolve_data_metadata(data_id, data_plugin_name, 'stimulus_set')
                if resolved_metadata:
                    print(f"Using template metadata for stimulus_set: {data_plugin_name}/{data_id}", file=sys.stderr)
                    return resolved_metadata
        except Exception as e:
            print(f"Warning: Could not resolve stimulus metadata from template: {e}", file=sys.stderr)
        
        # FALLBACK: Use existing extraction method
        print(f"Falling back to extraction for stimulus_set: {plugin_dir_name}", file=sys.stderr)
        
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
        # NEW: Try to resolve from data plugin metadata first
        try:
            data_id = self._extract_data_id(benchmark)
            data_plugin_name = self._extract_data_plugin_name(data_id)
            
            if data_id and data_plugin_name:
                resolved_metadata = self._resolve_data_metadata(data_id, data_plugin_name, 'data_assembly')
                if resolved_metadata:
                    print(f"Using template metadata for data_assembly: {data_plugin_name}/{data_id}", file=sys.stderr)
                    return resolved_metadata
        except Exception as e:
            print(f"Warning: Could not resolve data metadata from template: {e}", file=sys.stderr)
        
        # FALLBACK: Use existing extraction method
        print(f"Falling back to extraction for data_assembly: {benchmark_dir_name}", file=sys.stderr)
        
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
        # NEW: Try to resolve from metric plugin metadata first
        try:
            metric_id = self._extract_metric_id(plugin)
            
            if metric_id:
                metric_metadata = self._load_metadata_template('metrics', metric_id)
                if metric_metadata:
                    print(f"Using template metadata for metric: {metric_id}", file=sys.stderr)
                    return metric_metadata
        except Exception as e:
            print(f"Warning: Could not resolve metric metadata from template: {e}", file=sys.stderr)
        
        # FALLBACK: Use existing extraction method
        print(f"Falling back to extraction for metric: {plugin_dir_name}", file=sys.stderr)
        
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
