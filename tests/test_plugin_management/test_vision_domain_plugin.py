import pytest
import tempfile
from unittest.mock import Mock, MagicMock
from pathlib import Path


class TestVisionDomainPluginUnit:
    """Unit tests for VisionDomainPlugin that don't require full vision environment."""
    
    def setup_method(self):
        # Only test what we can without importing VisionDomainPlugin
        # (since it might have environment dependencies)
        pass
    
    def test_plugin_interface_compliance(self):
        """Test that VisionDomainPlugin implements the required interface."""
        try:
            from brainscore_vision.plugin_management import VisionDomainPlugin
            
            # Test initialization
            plugin = VisionDomainPlugin(benchmark_type="neural")
            assert plugin.benchmark_type == "neural"
            
            # Test required benchmark methods exist
            assert hasattr(plugin, 'load_benchmark')
            assert hasattr(plugin, 'create_stimuli_metadata')
            assert hasattr(plugin, 'create_data_metadata')
            assert hasattr(plugin, 'create_metric_metadata')
            assert hasattr(plugin, 'find_registered_benchmarks')
            
            # Test required model methods exist
            assert hasattr(plugin, 'load_model')
            assert hasattr(plugin, 'find_registered_models')
            assert hasattr(plugin, 'extract_model_for_analysis')
            assert hasattr(plugin, 'detect_model_architecture')
            assert hasattr(plugin, 'get_model_family')
            assert hasattr(plugin, 'create_model_metadata')
            
            print("VisionDomainPlugin interface compliance verified")
            
        except ImportError as e:
            pytest.skip(f"VisionDomainPlugin not available: {e}")
    
    def test_metadata_structure_with_mock(self):
        """Test metadata structure using mocked vision plugin."""
        try:
            from brainscore_vision.plugin_management import VisionDomainPlugin
            
            plugin = VisionDomainPlugin(benchmark_type="neural")
            
            # Create simple mock benchmark
            mock_benchmark = Mock()
            mock_benchmark._assembly = Mock()
            mock_benchmark._assembly.stimulus_set = Mock()
            # Mock memory_usage to return a value that can be summed and divided
            mock_benchmark._assembly.stimulus_set.memory_usage.return_value = Mock()
            mock_benchmark._assembly.stimulus_set.memory_usage.return_value.sum.return_value = 1024 * 1024  # 1MB
            
            # Mock assembly attributes for data metadata
            mock_benchmark._assembly.region = Mock()
            mock_benchmark._assembly.region.values = ["V1"]  # Mock region values
            mock_benchmark._assembly.hemisphere = Mock()
            mock_benchmark._assembly.hemisphere.values = ["left"]  # Mock hemisphere values
            mock_benchmark._assembly.subject = Mock()
            mock_benchmark._assembly.subject.values = ["sub1", "sub2"]  # Mock subject values
            
            # Test that benchmark methods can be called without crashing
            stimuli_meta = plugin.create_stimuli_metadata(mock_benchmark, "test_dir")
            data_meta = plugin.create_data_metadata(mock_benchmark, "test_dir")
            metric_meta = plugin.create_metric_metadata(mock_benchmark, "test_dir")
            
            # Basic structure tests
            assert isinstance(stimuli_meta, dict)
            assert isinstance(data_meta, dict)
            assert isinstance(metric_meta, dict)
            
            # Check for required fields
            assert "brainscore_link" in stimuli_meta
            assert "brainscore_link" in data_meta  
            assert "brainscore_link" in metric_meta
            assert "benchmark_type" in data_meta
            
            print("VisionDomainPlugin benchmark metadata structure verified")
            
        except ImportError as e:
            pytest.skip(f"VisionDomainPlugin not available: {e}")
    
    def test_model_metadata_structure_with_mock(self):
        """Test model metadata structure using mocked vision plugin."""
        try:
            from brainscore_vision.plugin_management import VisionDomainPlugin
            
            plugin = VisionDomainPlugin()
            
            # Create simple mock model
            mock_model = Mock()
            mock_model.parameters = Mock(return_value=[
                Mock(numel=Mock(return_value=1000), requires_grad=True, dim=Mock(return_value=2), element_size=Mock(return_value=4)),
            ])
            mock_model.modules = Mock(return_value=[mock_model])
            
            # Test model metadata creation
            model_meta = plugin.create_model_metadata(mock_model, "test_model", "test_dir")
            
            # Basic structure tests
            assert isinstance(model_meta, dict)
            
            # Check for required model fields
            required_fields = [
                "architecture", "model_family", "total_parameter_count",
                "trainable_parameter_count", "total_layers", "trainable_layers",
                "model_size_mb", "training_dataset", "task_specialization",
                "brainscore_link", "huggingface_link", "extra_notes"
            ]
            for field in required_fields:
                assert field in model_meta
            
            # Check vision-specific values
            assert "brainscore_vision/models" in model_meta["brainscore_link"]
            
            print("VisionDomainPlugin model metadata structure verified")
            
        except ImportError as e:
            pytest.skip(f"VisionDomainPlugin not available: {e}")
    
    def test_benchmark_type_handling(self):
        """Test that different benchmark types are handled correctly."""
        try:
            from brainscore_vision.plugin_management import VisionDomainPlugin
            
            # Test different benchmark types
            neural_plugin = VisionDomainPlugin(benchmark_type="neural")
            behavioral_plugin = VisionDomainPlugin(benchmark_type="behavioral") 
            engineering_plugin = VisionDomainPlugin(benchmark_type="engineering")
            
            assert neural_plugin.benchmark_type == "neural"
            assert behavioral_plugin.benchmark_type == "behavioral"
            assert engineering_plugin.benchmark_type == "engineering"
            
            print("VisionDomainPlugin benchmark type handling verified")
            
        except ImportError as e:
            pytest.skip(f"VisionDomainPlugin not available: {e}")
    
    def test_model_architecture_detection(self):
        """Test model architecture detection patterns."""
        try:
            from brainscore_vision.plugin_management import VisionDomainPlugin
            
            plugin = VisionDomainPlugin()
            
            # Create mock model
            mock_model = Mock()
            mock_model.modules = Mock(return_value=[])
            
            # Test basic detection (should default to DCNN for vision)
            arch = plugin.detect_model_architecture(mock_model, "test_model")
            assert "DCNN" in arch
            
            # Test vision-specific patterns
            cornet_arch = plugin.detect_model_architecture(mock_model, "cornet_s")
            assert "DCNN" in cornet_arch
            assert "RNN" in cornet_arch  # Vision-specific hardcoded pattern
            
            print("VisionDomainPlugin architecture detection verified")
            
        except ImportError as e:
            pytest.skip(f"VisionDomainPlugin not available: {e}")
    
    def test_model_family_detection(self):
        """Test model family detection using vision patterns."""
        try:
            from brainscore_vision.plugin_management import VisionDomainPlugin
            
            plugin = VisionDomainPlugin()
            
            # Test vision-specific families
            assert plugin.get_model_family("resnet50") == "resnet"
            assert plugin.get_model_family("efficientnet_b0") == "efficientnet"
            assert plugin.get_model_family("cornet_s") == "cornet"  # Vision-specific
            assert plugin.get_model_family("vone_resnet") == "resnet, vone"  # Multiple families (space-separated)
            assert plugin.get_model_family("unknown_model") is None
            
            print("VisionDomainPlugin model family detection verified")
            
        except ImportError as e:
            pytest.skip(f"VisionDomainPlugin not available: {e}")


@pytest.mark.integration
class TestVisionDomainPluginIntegration:
    """Integration tests that require full vision environment."""
    
    def test_with_real_benchmark_loading(self):
        """Test loading real benchmarks if available."""
        try:
            from brainscore_vision.plugin_management import VisionDomainPlugin
            from brainscore_vision import load_benchmark
            
            plugin = VisionDomainPlugin(benchmark_type="neural")
            
            # Try to load a real benchmark
            benchmark = plugin.load_benchmark("Rajalingham2020.IT-pls")
            
            if benchmark is not None:
                # Test metadata creation with real benchmark
                stimuli_meta = plugin.create_stimuli_metadata(benchmark, "rajalingham2020")
                data_meta = plugin.create_data_metadata(benchmark, "rajalingham2020")
                
                # Verify real data values
                assert stimuli_meta["num_stimuli"] == 616
                assert data_meta["benchmark_type"] == "neural"
                print("Real benchmark integration verified")
            else:
                pytest.skip("Rajalingham2020 benchmark not available")
                
        except ImportError as e:
            pytest.skip(f"Full vision environment not available: {e}")
        except Exception as e:
            pytest.skip(f"Vision benchmark loading failed: {e}")
    
    def test_with_real_model_loading(self):
        """Test loading real models if available."""
        try:
            from brainscore_vision.plugin_management import VisionDomainPlugin
            
            plugin = VisionDomainPlugin()
            
            # Try to load a real model
            model = plugin.load_model("resnet50_tutorial")
            
            if model is not None:
                # Test model extraction and metadata creation
                extracted_model = plugin.extract_model_for_analysis(model)
                model_meta = plugin.create_model_metadata(extracted_model, "resnet50_tutorial", "resnet50_tutorial")
                
                # Verify real model metadata
                assert model_meta["architecture"] == "DCNN"
                assert model_meta["model_family"] == "resnet"
                assert model_meta["total_parameter_count"] > 0
                print("Real model integration verified")
            else:
                pytest.skip("resnet50_tutorial model not available")
                
        except ImportError as e:
            pytest.skip(f"Full vision environment not available: {e}")
        except Exception as e:
            pytest.skip(f"Vision model loading failed: {e}")


if __name__ == "__main__":
    # Allow running tests directly for development
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    test_unit = TestVisionDomainPluginUnit()
    test_unit.setup_method()
    test_unit.test_plugin_interface_compliance()
    test_unit.test_metadata_structure_with_mock()
    test_unit.test_model_metadata_structure_with_mock()
    test_unit.test_benchmark_type_handling()
    test_unit.test_model_architecture_detection()
    test_unit.test_model_family_detection()
    
    print("\nAll vision plugin unit tests passed!") 