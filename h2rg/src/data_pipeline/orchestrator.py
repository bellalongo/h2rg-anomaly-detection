import logging
from typing import List, Dict
import os

from loaders.cache_manager import CacheManager
from loaders.data_storage import OptimizedDataStorage
from loaders.training_data_loader import TrainingDataLoader

from preprocessing.cleaners.reference_pixel_corrector import ReferencePixelCorrector
from preprocessing.transformers.frame_difference import FrameDifferencer
from preprocessing.transformers.patch_extractor import PatchExtractor
from preprocessing.transformers.temporal_analyzer import TemporalAnalyzer
from preprocessing.dataset_processors import EuclidProcessor
from preprocessing.dataset_processors import CaseProcessor

from validation.integrity_validator import DataIntegrityValidator


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
        
        # Data directories
        self.data_root_dir = data_root_dir
        self._initialize_data_directories()
    
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