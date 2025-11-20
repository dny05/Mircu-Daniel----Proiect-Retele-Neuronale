"""
Installation Test Script
Verify that all components are properly installed and working
"""

import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


class TestRunner:
    """Run installation tests"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []
    
    def test(self, name, func):
        """Run a single test"""
        try:
            func()
            logger.info(f"✓ {name}")
            self.passed += 1
            self.tests.append((name, True, None))
        except Exception as e:
            logger.error(f"✗ {name}: {str(e)}")
            self.failed += 1
            self.tests.append((name, False, str(e)))
    
    def summary(self):
        """Print test summary"""
        total = self.passed + self.failed
        logger.info("\n" + "=" * 60)
        logger.info(f"Test Results: {self.passed}/{total} passed")
        logger.info("=" * 60)
        
        if self.failed > 0:
            logger.info("\nFailed tests:")
            for name, success, error in self.tests:
                if not success:
                    logger.info(f"  - {name}: {error}")
        
        return self.failed == 0


def test_python_version():
    """Test Python version"""
    version = sys.version_info
    assert version.major == 3 and version.minor >= 8, \
        f"Python 3.8+ required, got {version.major}.{version.minor}"


def test_dependencies():
    """Test that all required packages are installed"""
    required = [
        'numpy',
        'pandas',
        'scipy',
        'torch',
        'sklearn',
        'matplotlib',
        'yaml',
        'tqdm'
    ]
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            raise ImportError(f"Package '{package}' not installed")


def test_directory_structure():
    """Test that directory structure exists"""
    required_dirs = [
        'data/raw',
        'data/processed',
        'data/models',
        'data/logs',
        'config',
        'src',
        'gui',
        'scripts'
    ]
    
    for directory in required_dirs:
        path = Path(directory)
        assert path.exists(), f"Directory '{directory}' not found"


def test_config_files():
    """Test that configuration files exist"""
    config_files = [
        'config/config.yaml',
        'config/model_config.yaml'
    ]
    
    for config_file in config_files:
        path = Path(config_file)
        assert path.exists(), f"Config file '{config_file}' not found"


def test_module_imports():
    """Test that core modules can be imported"""
    sys.path.insert(0, 'src')
    
    modules = [
        'data_loader',
        'preprocessing',
        'feature_extraction',
        'models',
        'trainer',
        'evaluator',
        'utils'
    ]
    
    for module in modules:
        try:
            __import__(module)
        except ImportError as e:
            raise ImportError(f"Cannot import module '{module}': {e}")


def test_data_loader():
    """Test TelemetryDataLoader"""
    sys.path.insert(0, 'src')
    from data_loader import TelemetryDataLoader
    
    loader = TelemetryDataLoader()
    assert loader is not None
    assert hasattr(loader, 'load_csv')
    assert hasattr(loader, 'validate_data_quality')


def test_preprocessor():
    """Test TelemetryPreprocessor"""
    sys.path.insert(0, 'src')
    from preprocessing import TelemetryPreprocessor
    import numpy as np
    
    preprocessor = TelemetryPreprocessor()
    
    # Test filtering
    test_signal = np.random.randn(1000)
    filtered = preprocessor.apply_butterworth_filter(test_signal)
    assert filtered.shape == test_signal.shape


def test_feature_extractor():
    """Test FeatureExtractor"""
    sys.path.insert(0, 'src')
    from feature_extraction import FeatureExtractor
    import numpy as np
    
    extractor = FeatureExtractor()
    
    # Test feature extraction
    windows = np.random.randn(10, 200, 4)
    features = extractor.extract_features_from_windows(windows)
    assert features.shape[0] == 10  # Same number of windows
    assert features.shape[1] > 0    # Has features


def test_model_creation():
    """Test model creation"""
    sys.path.insert(0, 'src')
    from models import create_model
    
    # Test classifier
    classifier = create_model('classifier')
    assert classifier is not None
    assert hasattr(classifier, 'forward')
    
    # Test regressor
    regressor = create_model('regressor')
    assert regressor is not None
    assert hasattr(regressor, 'forward')


def test_model_forward_pass():
    """Test model forward pass"""
    sys.path.insert(0, 'src')
    from models import create_model
    import torch
    
    model = create_model('classifier')
    
    # Test forward pass
    batch_size = 4
    input_size = 30
    x = torch.randn(batch_size, input_size)
    
    output = model(x)
    assert output.shape == (batch_size, 2)


def test_trainer_initialization():
    """Test ModelTrainer"""
    sys.path.insert(0, 'src')
    from trainer import ModelTrainer
    from models import create_model
    
    model = create_model('classifier')
    trainer = ModelTrainer(model)
    
    assert trainer is not None
    assert hasattr(trainer, 'train')


def test_evaluator_initialization():
    """Test SetupEvaluator"""
    sys.path.insert(0, 'src')
    from evaluator import SetupEvaluator
    from models import create_model
    
    model = create_model('classifier')
    evaluator = SetupEvaluator(model)
    
    assert evaluator is not None
    assert hasattr(evaluator, 'evaluate_windows')


def test_cuda_availability():
    """Test CUDA availability (info only)"""
    import torch
    
    if torch.cuda.is_available():
        logger.info(f"  CUDA is available: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("  CUDA is not available (will use CPU)")


def test_sample_data_generation():
    """Test sample data generation"""
    sys.path.insert(0, 'scripts')
    
    # Just import - don't run the generation
    try:
        import generate_sample_data
        assert hasattr(generate_sample_data, 'generate_telemetry_data')
    except Exception as e:
        raise ImportError(f"Cannot import generate_sample_data: {e}")


def main():
    """Run all tests"""
    logger.info("=" * 60)
    logger.info("SUSPENSION SETUP EVALUATOR - INSTALLATION TEST")
    logger.info("=" * 60)
    logger.info("")
    
    runner = TestRunner()
    
    # Core tests
    logger.info("1. Testing Python Environment...")
    runner.test("Python version 3.8+", test_python_version)
    runner.test("Required dependencies", test_dependencies)
    runner.test("Directory structure", test_directory_structure)
    runner.test("Configuration files", test_config_files)
    
    # Module tests
    logger.info("\n2. Testing Module Imports...")
    runner.test("Core module imports", test_module_imports)
    runner.test("TelemetryDataLoader", test_data_loader)
    runner.test("TelemetryPreprocessor", test_preprocessor)
    runner.test("FeatureExtractor", test_feature_extractor)
    
    # Model tests
    logger.info("\n3. Testing Neural Network Models...")
    runner.test("Model creation", test_model_creation)
    runner.test("Model forward pass", test_model_forward_pass)
    runner.test("ModelTrainer initialization", test_trainer_initialization)
    runner.test("SetupEvaluator initialization", test_evaluator_initialization)
    
    # Additional tests
    logger.info("\n4. Additional Checks...")
    runner.test("CUDA availability check", test_cuda_availability)
    runner.test("Sample data generator", test_sample_data_generation)
    
    # Summary
    logger.info("")
    success = runner.summary()
    
    if success:
        logger.info("\n✓ All tests passed! Installation is complete.")
        logger.info("\nNext steps:")
        logger.info("  1. Generate sample data: python scripts/generate_sample_data.py")
        logger.info("  2. Train a model: python scripts/train_model.py --synthetic")
        logger.info("  3. Run the GUI: python main.py")
    else:
        logger.info("\n✗ Some tests failed. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()