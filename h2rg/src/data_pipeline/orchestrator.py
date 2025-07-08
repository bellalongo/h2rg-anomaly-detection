import logging
from typing import List, Dict
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import psutil
import time
from pathlib import Path

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
    def __init__(self, root_dir: str = '/projects/JWST_planets/ilongo/processed_data', 
                 data_root_dir: str = '/projects/JWST_planets/ilongo/raw_data'):
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

    def enable_production_mode(self, max_parallel_exposures: int = None):
        """
        Enable production mode with parallel processing
        """
        if max_parallel_exposures is None:
            # Auto-detect optimal number based on system resources
            cpu_count = psutil.cpu_count()
            memory_gb = psutil.virtual_memory().total / (1024**3)
            
            # Conservative estimate: each exposure needs ~4GB RAM during processing
            memory_limited = int(memory_gb / 4)
            cpu_limited = min(cpu_count, 8)  # Don't go too high to avoid I/O bottlenecks
            
            max_parallel_exposures = min(memory_limited, cpu_limited, 6)
        
        self.max_parallel_exposures = max_parallel_exposures
        self.production_mode = True
        
        self.logger.info(f"PRODUCTION MODE: {max_parallel_exposures} parallel exposures")
        self.logger.info(f"System: {psutil.cpu_count()} CPUs, {psutil.virtual_memory().total/(1024**3):.1f}GB RAM")

    def enable_test_mode(self, test_frames: int = 10, root_dir = 'data/test'):
        """
            * enable test mode with limited data processing
        """
        self.test_mode = True
        self.test_frames = test_frames
        self.logger.info(f"TEST MODE ENABLED: Processing only {test_frames} frames per file")
        
        # # Create test subdirectory structure
        # test_root = f'{self.cache_manager.root_dir}/test'
        
        # Update cache manager for test mode
        self.cache_manager = CacheManager(root_dir)
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
            
            case_dirs = [d for d in all_dirs if 'noise' in d]
            
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
            
            self.case_dirs = [d for d in all_dirs if 'noise' in d]
        
        except FileNotFoundError:
            self.logger.error(f'Data root directory not found: {self.data_root_dir}')
            self.euclid_dirs = []
            self.case_dirs = []

    def _process_euclid_parallel(self) -> List[str]:
        """Process ALL EUCLID data with parallel execution"""
        self.logger.info("Processing EUCLID dataset in PARALLEL mode")
        
        # Collect all EUCLID exposures
        exposure_list = []
        for dir_name in self.euclid_dirs:
            dir_path = Path(self.data_root_dir) / dir_name
            fits_files = sorted(dir_path.glob("*.fits"))
            
            for fits_file in fits_files:
                exposure_id = f'euclid_{dir_name}_{fits_file.stem}'
                
                # Check if already cached
                if not self.cache_manager.is_exposure_cached(exposure_id, [str(fits_file)], self.validator):
                    exposure_list.append((exposure_id, str(fits_file), dir_name, fits_file.name))
        
        self.logger.info(f"Processing {len(exposure_list)} uncached EUCLID exposures")
        
        if not exposure_list:
            self.logger.info("All EUCLID exposures already cached")
            return []
        
        # Process in parallel batches
        return self._process_exposures_in_batches(exposure_list, 'EUCLID')
    
    def _process_case_parallel(self) -> List[str]:
        """Process ALL CASE data with parallel execution"""
        self.logger.info("Processing CASE dataset in PARALLEL mode")
        
        # Collect all CASE exposures
        exposure_list = []
        for dir_name in self.case_dirs:
            dir_path = Path(self.data_root_dir) / dir_name
            tif_files = sorted(dir_path.glob("*.tif"))
            
            # Group into exposures of 450 files
            total_frames = 450
            for exposure_idx, i in enumerate(range(0, len(tif_files), total_frames)):
                group = tif_files[i:i + total_frames]
                
                if len(group) == total_frames:  # Only process complete exposures
                    exposure_id = f'case_{dir_name.replace("/", "__")}_exp{exposure_idx:03d}'
                    file_paths = [str(f) for f in group]
                    
                    # Check if both detectors are cached
                    det1_cached = self.cache_manager.is_exposure_cached(
                        f'{exposure_id}_det1', file_paths, self.validator
                    )
                    det2_cached = self.cache_manager.is_exposure_cached(
                        f'{exposure_id}_det2', file_paths, self.validator
                    )
                    
                    if not (det1_cached and det2_cached):
                        exposure_list.append((exposure_id, file_paths, dir_name, exposure_idx))
        
        self.logger.info(f"Processing {len(exposure_list)} uncached CASE exposures")
        
        if not exposure_list:
            self.logger.info("All CASE exposures already cached")
            return []
        
        # Process in parallel batches
        return self._process_exposures_in_batches(exposure_list, 'CASE')
    
    def preprocess(self, dataset_name: str) -> List[str]:
        """
        Main preprocessing dispatcher - enhanced to use parallel processing when available
        """
        dataset_name = dataset_name.upper()
        
        # Use parallel processing if production mode is enabled
        if hasattr(self, 'production_mode') and self.production_mode:
            return self.preprocess_parallel(dataset_name)
        else:
            # Standard processing
            if dataset_name == 'EUCLID':
                return self.euclid_processor.process_directory(self.data_root_dir, self.euclid_dirs)
            elif dataset_name == 'CASE':
                return self.case_processor.process_directory(self.data_root_dir, self.case_dirs)
            else:
                raise ValueError(f'Invalid dataset name: {dataset_name}. Must be EUCLID or CASE')

    def preprocess_parallel(self, dataset_name: str) -> List[str]:
        """
        Parallel preprocessing for production workloads
        Processes ALL data with maximum speed but NO shortcuts
        """
        if not hasattr(self, 'production_mode'):
            self.logger.warning("Production mode not enabled. Using standard processing.")
            return self.preprocess(dataset_name)
        
        dataset_name = dataset_name.upper()
        start_time = time.time()
        
        self.logger.info(f"Starting PARALLEL processing for {dataset_name}")
        self._log_system_resources()
        
        if dataset_name == 'EUCLID':
            processed_exposures = self._process_euclid_parallel()
        elif dataset_name == 'CASE':
            processed_exposures = self._process_case_parallel()
        elif dataset_name == 'ALL':
            euclid_exposures = self._process_euclid_parallel()
            case_exposures = self._process_case_parallel()
            processed_exposures = euclid_exposures + case_exposures
        else:
            raise ValueError(f'Invalid dataset name: {dataset_name}')
        
        total_time = time.time() - start_time
        self._log_completion_stats(processed_exposures, total_time, dataset_name)
        
        return processed_exposures
    
    def _log_system_resources(self):
        """Log current system resource usage"""
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        self.logger.info(f"System Resources:")
        self.logger.info(f"  CPU Usage: {cpu_percent:.1f}%")
        self.logger.info(f"  Memory: {memory.used/1e9:.1f}GB/{memory.total/1e9:.1f}GB ({memory.percent:.1f}%)")
        self.logger.info(f"  Available Memory: {memory.available/1e9:.1f}GB")

    def _log_completion_stats(self, processed_exposures: List[str], total_time: float, dataset_name: str):
        """Log completion statistics"""
        self.logger.info("="*60)
        self.logger.info(f"PARALLEL PROCESSING COMPLETE - {dataset_name}")
        self.logger.info(f"Total exposures processed: {len(processed_exposures)}")
        self.logger.info(f"Total processing time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
        
        if len(processed_exposures) > 0:
            avg_time_per_exposure = total_time / len(processed_exposures)
            self.logger.info(f"Average time per exposure: {avg_time_per_exposure:.2f}s")
            
            # Estimate total dataset processing time
            if dataset_name == 'EUCLID':
                total_dirs = len(self.euclid_dirs)
                estimated_exposures = total_dirs * 10  # Rough estimate
            elif dataset_name == 'CASE':
                total_dirs = len(self.case_dirs)
                estimated_exposures = total_dirs * 2  # 2 detectors per directory
            else:
                estimated_exposures = len(processed_exposures)
            
            estimated_full_time = avg_time_per_exposure * estimated_exposures
            self.logger.info(f"Estimated full dataset time: {estimated_full_time/3600:.1f} hours")
        
        self.logger.info("="*60)
    
    
    # def preprocess(self, dataset_name: str) -> List[str]:
    #     """
    #         * main preprocessing dispatcher
    #     """
    #     dataset_name = dataset_name.upper()
        
    #     if dataset_name == 'EUCLID':
    #         return self.euclid_processor.process_directory(self.data_root_dir, self.euclid_dirs)
    #     elif dataset_name == 'CASE':
    #         return self.case_processor.process_directory(self.data_root_dir, self.case_dirs)
    #     else:
    #         raise ValueError(f'Invalid dataset name: {dataset_name}. Must be EUCLID or CASE')
    
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