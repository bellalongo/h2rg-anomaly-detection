import logging
from typing import List, Dict
import os

from .loaders.cache_manager import CacheManager
from .loaders.data_storage import OptimizedDataStorage
from .loaders.training_data_loader import TrainingDataLoader

from .preprocessing.cleaners.reference_pixel_corrector import ReferencePixelCorrector
from .preprocessing.transformers.frame_difference import FrameDifferencer
from .preprocessing.transformers.patch_extractor import PatchExtractor
from .preprocessing.transformers.temporal_analyzer import TemporalAnalyzer
from .preprocessing.dataset_processors import EuclidProcessor
from .preprocessing.dataset_processors import CaseProcessor

from .validation.integrity_validator import DataIntegrityValidator


class DataProcessingOrchestrator:
    """
        * main orchestrator that coordinates all preprocessing components
    """
    def __init__(self, root_dir: str = 'training_set', 
                 data_root_dir: str = '/proj/case/2025-06-05'):
        """

        """
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.cache_manager = CacheManager(root_dir)
        self.validator = DataIntegrityValidator()
        self.storage = OptimizedDataStorage(self.cache_manager.root_dir)
        
        # Processing components
        self.reference_corrector = ReferencePixelCorrector()
        self.frame_differencer = FrameDifferencer(self.reference_corrector)
        self.temporal_analyzer = TemporalAnalyzer()
        self.patch_extractor = PatchExtractor()
        
        # Dataset processors (Euclid)
        self.euclid_processor = EuclidProcessor(
            self.frame_differencer, self.temporal_analyzer, self.patch_extractor,
            self.storage, self.cache_manager, self.validator
        )
        
        # Dataset processor (Case)
        self.case_processor = CaseProcessor(
            self.frame_differencer, self.temporal_analyzer, self.patch_extractor,
            self.storage, self.cache_manager, self.validator
        )
        
        # Data loader
        self.data_loader = TrainingDataLoader(self.cache_manager)

        # Test mode settings
        self.test_mode = False
        self.test_frames = 10
        
        # Data directories
        self.data_root_dir = data_root_dir
        self._initialize_data_directories()

    def enable_test_mode(self, test_frames: int = 10):
        """
            * enable test mode with limited data processing
        """
        self.test_mode = True
        self.test_frames = test_frames
        self.logger.info(f"🧪 TEST MODE ENABLED: Processing only {test_frames} frames per file")
        
        # Create test subdirectory structure
        original_root = self.cache_manager.root_dir
        test_root = original_root / "test"
        
        # Update cache manager for test mode
        self.cache_manager = CacheManager(str(test_root))
        self.storage = OptimizedDataStorage(self.cache_manager.root_dir)
        self.data_loader = TrainingDataLoader(self.cache_manager)
        
        # Propagate test mode to all processors
        self.frame_differencer.set_test_mode(test_frames)
        self.euclid_processor.set_test_mode(test_frames)
        self.case_processor.set_test_mode(test_frames)
        
        # Update cache manager references in processors
        self.euclid_processor.cache_manager = self.cache_manager
        self.euclid_processor.storage = self.storage
        self.case_processor.cache_manager = self.cache_manager
        self.case_processor.storage = self.storage
        
        # Re-initialize data directories with test mode filtering
        self._initialize_data_directories()

    def _initialize_data_directories(self):
        """
            * find and categorize data directories (with test mode filtering)
        """
        try:
            # Get all directories
            all_dirs = os.listdir(self.data_root_dir)
            
            euclid_dirs = [d for d in all_dirs if 'Euclid' in d]
            
            case_dirs = []
            for d in all_dirs:
                if 'FPM' in d:
                    nested_dirs = os.listdir(f'{self.data_root_dir}/{d}')
                    if nested_dirs:
                        case_dirs.append(f'{d}/{nested_dirs[0]}')
            
            # Apply test mode filtering
            if self.test_mode:
                self.euclid_dirs = euclid_dirs[:1]  # Only first directory
                self.case_dirs = case_dirs[:1]      # Only first directory
                self.logger.info(f"TEST MODE: Limited to {len(self.euclid_dirs)} EUCLID and {len(self.case_dirs)} CASE directories")
            else:
                self.euclid_dirs = euclid_dirs
                self.case_dirs = case_dirs
        
        except FileNotFoundError:
            self.logger.error(f'Data root directory not found: {self.data_root_dir}')
            self.euclid_dirs = []
            self.case_dirs = []

    def apply_config(self, config: Dict):
        """
            * apply configuration to all components
        """
        self.logger.info("Applying configuration to processing components...")
        
        # Apply preprocessing config
        if 'preprocessing' in config:
            preprocessing_config = config['preprocessing']
            
            # Update patch extractor
            if 'patch_sizes' in preprocessing_config:
                self.patch_extractor.patch_sizes = preprocessing_config['patch_sizes']
                self.logger.info(f"Set patch sizes: {preprocessing_config['patch_sizes']}")
            
            if 'overlap_ratio' in preprocessing_config:
                self.patch_extractor.overlap_ratio = preprocessing_config['overlap_ratio']
                self.logger.info(f"Set overlap ratio: {preprocessing_config['overlap_ratio']}")
            
            # Update temporal analyzer
            if 'sigma_threshold' in preprocessing_config:
                self.temporal_analyzer.sigma_threshold = preprocessing_config['sigma_threshold']
                self.logger.info(f"Set sigma threshold: {preprocessing_config['sigma_threshold']}")
        
        # Apply EUCLID-specific config
        if 'euclid' in config:
            euclid_config = config['euclid']
            
            if 'optimal_x' in euclid_config:
                self.reference_corrector.x_opt = euclid_config['optimal_x']
                self.logger.info(f"Set EUCLID optimal x: {euclid_config['optimal_x']}")
            
            if 'optimal_y' in euclid_config:
                self.reference_corrector.y_opt = euclid_config['optimal_y']
                self.logger.info(f"Set EUCLID optimal y: {euclid_config['optimal_y']}")
        
        # Apply CASE-specific config
        if 'case' in config:
            case_config = config['case']

            if 'optimal_x' in case_config:
                self.reference_corrector.x_opt = case_config['optimal_x']
                self.logger.info(f"Set CASE optimal x: {case_config['optimal_x']}")
            
            if 'optimal_y' in case_config:
                self.reference_corrector.y_opt = case_config['optimal_y']
                self.logger.info(f"Set CASE optimal y: {case_config['optimal_y']}")
            
            if 'total_frames' in case_config:
                self.case_processor.total_frames = case_config['total_frames']
                self.logger.info(f"Set CASE total frames: {case_config['total_frames']}")
        
        # Apply storage config (if needed in the future)
        if 'storage' in config:
            storage_config = config['storage']
            self.logger.info(f"Storage config available: {storage_config}")
        
        # Apply cache config (already handled by CacheManager initialization)
        if 'cache' in config:
            cache_config = config['cache']
            self.logger.info(f"Cache config: {cache_config}")
        
        self.logger.info("Configuration applied successfully")
    
    def _initialize_data_directories(self):
        """
            * find and categorize data directories
        """
        try:
            # Get all directories
            all_dirs = os.listdir(self.data_root_dir)
            
            self.euclid_dirs = [d for d in all_dirs if 'Euclid' in d]
            
            self.case_dirs = []
            for d in all_dirs:
                if 'FPM' in d:
                    nested_dirs = os.listdir(f'{self.data_root_dir}/{d}')
                    if nested_dirs:
                        self.case_dirs.append(f'{d}/{nested_dirs[0]}')
        
        except FileNotFoundError:
            self.logger.error(f'Data root directory not found: {self.data_root_dir}')
            self.euclid_dirs = []
            self.case_dirs = []
    
    def preprocess(self, dataset_name: str) -> List[str]:
        """
            * main preprocessing dispatcher
        """
        dataset_name = dataset_name.upper()
        
        if dataset_name == 'EUCLID':
            return self.euclid_processor.process_directory(self.data_root_dir, self.euclid_dirs)
        elif dataset_name == 'CASE':
            return self.case_processor.process_directory(self.data_root_dir, self.case_dirs)
        else:
            raise ValueError(f'Invalid dataset name: {dataset_name}. Must be EUCLID or CASE')
    
    def load_training_dataset(self, **kwargs) -> Dict:
        """
            * load training dataset with filtering options
        """
        return self.data_loader.load_training_dataset(**kwargs)
    
    def get_statistics(self) -> Dict:
        """
            * get processing statistics
        """
        return self.data_loader.get_statistics()
    
    def save_registry(self) -> str:
        """
            * save processing registry
        """
        return self.cache_manager.save_registry()

    def get_test_summary(self) -> Dict:
        """
            * get summary of what will be processed in test mode
        """
        if not self.test_mode:
            return {"test_mode": False}
        
        summary = {
            "test_mode": True,
            "test_frames": self.test_frames,
            "euclid_dirs": self.euclid_dirs,
            "case_dirs": self.case_dirs,
            "expected_output": {
                "euclid_exposures": len(self.euclid_dirs),  # 1 file per dir
                "case_exposures": len(self.case_dirs) * 2,  # 2 detectors per dir
                "total_difference_frames": len(self.euclid_dirs) * self.test_frames + len(self.case_dirs) * 2 * self.test_frames,
                "cache_location": str(self.cache_manager.root_dir)
            }
        }
        
        return summary