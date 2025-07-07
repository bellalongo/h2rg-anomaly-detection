import os
import re
from typing import List, Optional
from pathlib import Path
import numpy as np
import logging
from datetime import datetime
import warnings

# Supress tiffile warnings
warnings.filterwarnings("ignore")
logging.getLogger('tifffile').setLevel(logging.ERROR)

import tifffile

from .transformers.frame_difference import FrameDifferencer
from .transformers.temporal_analyzer import TemporalAnalyzer
from .transformers.patch_extractor import PatchExtractor
from ..loaders.data_storage import OptimizedDataStorage
from ..loaders.cache_manager import CacheManager
from ..validation.integrity_validator import DataIntegrityValidator


class EuclidProcessor:
    """
        * processes EUCLID FITS files for training data
    """
    def __init__(self, frame_differencer: FrameDifferencer, 
                 temporal_analyzer: TemporalAnalyzer,
                 patch_extractor: PatchExtractor,
                 storage: OptimizedDataStorage,
                 cache_manager: CacheManager,
                 validator: DataIntegrityValidator):
        
        self.frame_differencer = frame_differencer
        self.temporal_analyzer = temporal_analyzer
        self.patch_extractor = patch_extractor
        self.storage = storage
        self.cache_manager = cache_manager
        self.validator = validator

        # Test mode details
        self.test_mode = False
        self.test_frames = 10

        self.logger = logging.getLogger(__name__)

    def set_test_mode(self, test_frames: int = 10):
        """
            * enable test mode for limited processing
        """
        self.test_mode = test_frames
        self.test_frames = test_frames
        self.frame_differencer.set_test_mode(test_frames)
        self.logger.info(f"EUCLID processor: TEST MODE enabled ({test_frames} frames)")
    
    def process_directory(self, data_root_dir: str, euclid_dirs: List[str]) -> List[str]:
        """
            * process all EUCLID directories
        """
        processed_exposures = []
        
        # Process all Euclid directories
        for dir_name in euclid_dirs:
            path = f'{data_root_dir}/{dir_name}'
            filenames = self._get_filenames(path)

            # Just grab the fits files
            filenames = [f for f in filenames if f.endswith('.fits')]
            
            # Make sure there are actual filenames
            if not filenames:
                self.logger.error(
                    '"filenames" array is empty, please make sure that "EUCLID_DIRS" has been loaded')
                continue
            
            # Sort the filename by exposure (18283_Euclid_SCA IS DIFFERENT BRUH)
            try:
                sorted_filenames = sorted(filenames,
                                          key=lambda x: int(re.search(r'E(\d+)', x).group(1))
                )
            except AttributeError as e:
                sorted_filenames = sorted(filenames,
                                          key=lambda x: int(re.search(r'_(\d+)\.fits$', x).group(1))
                )
            
            #  Test mode details
            if self.test_mode:
                sorted_filenames = sorted_filenames[:1]
                self.logger.info(f"TEST MODE: Processing only first file: {sorted_filenames[0]}")
            
            # Iterate through all filenames in order
            for filename in sorted_filenames:
                # Create exposure id and current path
                exposure_id = f'euclid_{dir_name}_{Path(filename).stem}'
                curr_path = f'{path}/{filename}'
                
                # Check if already cached
                if self.cache_manager.is_exposure_cached(exposure_id, [curr_path], self.validator):
                    self.logger.debug(f'Exposure {exposure_id} already cached')
                    continue
                
                # Process exposure
                if self._process_single_exposure(exposure_id, curr_path, dir_name, filename):
                    processed_exposures.append(exposure_id)
        
        return processed_exposures
    
    def _process_single_exposure(self, exposure_id: str, file_path: str,
                                 dir_name: str, filename: str) -> bool:
        """
            * process a single EUCLID exposure
        """
        try:
            self.logger.info(f'Processing exposure {exposure_id}...')

            # Compute differences
            diff_data = self.frame_differencer.compute_fits_differences(file_path)
            
            # Analyze temporal patterns
            temporal_data = self.temporal_analyzer.analyze_temporal_patterns(diff_data['differences'])
            
            # Extract patches at multiple scales
            patches_data = {}
            for patch_size in self.patch_extractor.patch_sizes:
                patches_data[f'patches_{patch_size}'] = self.patch_extractor.extract_patches(
                    diff_data['differences'], patch_size
                )
            
            # Prepare detector info
            detector_info = {
                'dataset_type': 'EUCLID',
                'directory': dir_name,
                'filename': filename,
                'file_path': file_path,
                'detector_id': 'single',
                'exposure_hash': self.validator.get_exposure_hash([file_path])
            }
            
            # Save everything
            self.storage.save_processed_exposure(exposure_id, diff_data, temporal_data, 
                                               patches_data, detector_info)
            
            # Update registry
            self.cache_manager.registry['processed_exposures'][exposure_id] = {
                'exposure_hash': detector_info['exposure_hash'],
                'processed_time': datetime.now().isoformat(timespec='microseconds'),
                'dataset_type': 'EUCLID',
                'detector_info': detector_info
            }
            self.cache_manager.save_registry()

            self.logger.info(f'Successfully processed exposure {exposure_id}')
            
            return True
            
        except Exception as e:
            self.logger.error(f'Error processing {exposure_id}: {e}')
            return False
    
    def _get_filenames(self, path: str) -> Optional[List[str]]:
        """
            * get all filenames in directory
        """
        try:
            entries = os.listdir(path)
            return [entry for entry in entries if os.path.isfile(os.path.join(path, entry))]
        except FileNotFoundError:
            self.logger.error(f'Directory not found: {path}')
            return None


class CaseProcessor:
    """
        * processes CASE TIF files with dual detectors
    """
    def __init__(self, frame_differencer: FrameDifferencer,
                 temporal_analyzer: TemporalAnalyzer,
                 patch_extractor: PatchExtractor,
                 storage: OptimizedDataStorage,
                 cache_manager: CacheManager,
                 validator: DataIntegrityValidator):
        """

        """
        self.frame_differencer = frame_differencer
        self.temporal_analyzer = temporal_analyzer  
        self.patch_extractor = patch_extractor
        self.storage = storage
        self.cache_manager = cache_manager
        self.validator = validator
        self.logger = logging.getLogger(__name__)

        # Test mode details
        self.test_mode = False
        self.test_frames = 10
        
        # Constants
        self.total_frames = 450
        self.img_size = (2048, 2048)

    def set_test_mode(self, test_frames: int = 10):
        """
            * enable test mode for limited processing
        """
        self.test_mode = True
        self.test_frames = test_frames
        self.frame_differencer.set_test_mode(test_frames)
        self.logger.info(f"CASE processor: TEST MODE enabled ({test_frames} frames)")
    
    def process_directory(self, data_root_dir: str, case_dirs: List[str]) -> List[str]:
        """
            * process all CASE directories
        """
        processed_exposures = []
        
        # Process all case directories
        for dir_name in case_dirs:
            # Make sure to grab all files in the nested directory
            path = f'{data_root_dir}/{dir_name}'

            filenames = self._get_filenames(path)

            # Just grab tif filenames
            filenames = [f for f in filenames if f.lower().endswith(('.tif', '.tiff'))]
            
            if not filenames:
                continue
            
            # Sort the filenames by their index number
            sorted_filenames = sorted(
                filenames,
                key=lambda x: (
                    int(re.search(r'_E(\d+)_', x).group(1)),  # First: sort by E####
                    int(re.search(r'_N(\d+)\.tif$$', x).group(1))   # Then: sort by N####
                )
            )
            
            # Test mode: Only take first N files for testing
            if self.test_mode:
                frames_needed = self.test_frames + 1  # +1 for reference frame
                sorted_filenames = sorted_filenames[:frames_needed]
                self.logger.info(f"TEST MODE: Using only {len(sorted_filenames)} files for {self.test_frames} difference frames")

            # Process in groups (modified for test mode)
            frames_per_group = len(sorted_filenames) if self.test_mode else self.total_frames
            
            # Process in groups of 450 files
            for exposure_idx, i in enumerate(range(0, len(sorted_filenames), self.total_frames)):
                group = sorted_filenames[i:i + self.total_frames]
                
                # # Makse sure that the length is 450 frames
                # if len(group) != self.total_frames:
                #     self.logger.warning(f'Incomplete exposure: {len(group)} files')
                #     continue
                
                # Create a expossure id
                exp_id_dir = dir_name.replace('/', '__')
                exposure_id = f'case_{exp_id_dir}_exp{exposure_idx:03d}'
                # if self.test_mode:
                #     exposure_id += f'_test{self.test_frames}frames'

                # Group filepaths for the current exposure
                group_paths = [f'{path}/{filename}' for filename in group]
                
                # Process both detectors
                detector_exposures = self._process_dual_detectors(
                    exposure_id, group, group_paths, path, dir_name, exposure_idx
                )
                
                processed_exposures.extend(detector_exposures)

                # Break if test mode
                if self.test_mode:
                    break
        
        return processed_exposures
    
    def _process_dual_detectors(self, exposure_id: str, group: List[str], 
                              group_paths: List[str], path: str, dir_name: str, 
                              exposure_idx: int) -> List[str]:
        """
            * process both detectors for a CASE exposure
        """
        processed = []
        
        # Check cache status for both detectors
        d1_cached = self.cache_manager.is_exposure_cached(
            f'{exposure_id}_det1', group_paths, self.validator
        )
        d2_cached = self.cache_manager.is_exposure_cached(
            f'{exposure_id}_det2', group_paths, self.validator
        )
        
        # If they both have been cached already, dont return anything
        if d1_cached and d2_cached:
            self.logger.debug(f'Both detectors for {exposure_id} cached')
            return []
        
        # Load and split detector data
        d1_stack, d2_stack = self._load_and_split_detectors(group, path)
        
        # Process detector 1 and 2
        for detector_idx, frame_stack in [(1, d1_stack), (2, d2_stack)]:
            # See of one of the detecors is not already cached
            cached = d1_cached if detector_idx == 1 else d2_cached
            
            # If both have not been cached yet
            if not cached:
                detector_exposure_id = f'{exposure_id}_det{detector_idx}'
                
                # Process the current detector exposure
                if self._process_detector_exposure(
                    detector_exposure_id, frame_stack, dir_name, 
                    exposure_idx, detector_idx, group, group_paths
                ):
                    processed.append(detector_exposure_id)
        
        return processed
    
    def _load_and_split_detectors(self, group: List[str], path: str):
        """
            * load TIF files and split into two detectors
        """
        d1_frames, d2_frames = [], []
        goal_width = 2048
        
        # Iterate through each filename in the group
        for filename in group:
            tif_data = tifffile.imread(f'{path}/{filename}')
            
            # Find split point and how many cols to cut
            split_index = tif_data.shape[1] // 2
            cols_to_cut = (split_index - goal_width) // 2

            d1 = tif_data[:, :split_index][:, cols_to_cut:-cols_to_cut]
            d2 = tif_data[:, split_index:][:, cols_to_cut:-cols_to_cut]
            
            # Make sure that the shape is still consitent
            if d1.shape != self.img_size or d2.shape != self.img_size:
                raise ValueError(f'Invalid detector shapes: {d1.shape}, {d2.shape}')
            
            d1_frames.append(d1)
            d2_frames.append(d2)

        if self.test_mode:
            self.logger.info(f"TEST MODE: Loaded {len(d1_frames)} frames per detector")
        
        return np.array(d1_frames), np.array(d2_frames)
    
    def _process_detector_exposure(self, detector_exposure_id: str, frame_stack: np.ndarray,
                                 dir_name: str, exposure_idx: int, detector_idx: int,
                                 group: List[str], group_paths: List[str]) -> bool:
        """
            * process single detector exposure
        """
        try:
            # Compute differences
            diff_data = self.frame_differencer.compute_tif_differences(frame_stack)
            
            # Analyze temporal patterns
            temporal_data = self.temporal_analyzer.analyze_temporal_patterns(diff_data['differences'])
            
            # Extract patches
            patches_data = {}
            for patch_size in self.patch_extractor.patch_sizes:
                patches_data[f'patches_{patch_size}'] = self.patch_extractor.extract_patches(
                    diff_data['differences'], patch_size
                )
            
            # Prepare detector info
            detector_info = {
                'dataset_type': 'CASE',
                'directory': dir_name,
                'exposure_index': exposure_idx,
                'detector_id': f'detector_{detector_idx}',
                'total_files': len(group),
                'file_range': f'{group[0]} to {group[-1]}',
                'exposure_hash': self.validator.get_exposure_hash(group_paths)
            }
            
            # Save everything
            self.storage.save_processed_exposure(detector_exposure_id, diff_data, temporal_data,
                                               patches_data, detector_info)
            
            # Update registry
            self.cache_manager.registry['processed_exposures'][detector_exposure_id] = {
                'exposure_hash': detector_info['exposure_hash'],
                'processed_time': datetime.now().isoformat(timespec='microseconds'),
                'dataset_type': 'CASE',
                'detector_info': detector_info
            }
            
            return True
            
        except Exception as e:
            self.logger.error(f'Error processing {detector_exposure_id}: {e}')
            return False
    
    def _get_filenames(self, path: str) -> Optional[List[str]]:
        """
            * get all filenames in directory
        """
        try:
            entries = os.listdir(path)
            return [entry for entry in entries if os.path.isfile(os.path.join(path, entry))]
        except FileNotFoundError:
            self.logger.error(f'Directory not found: {path}')
            return None