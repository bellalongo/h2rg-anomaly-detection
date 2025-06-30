import h5py
import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm
import logging

from ..loaders.cache_manager import CacheManager


class TrainingDataLoader:
    """
        * loads processed data for training with filtering and batching capabilities
    """
    def __init__(self, cache_manager: CacheManager):
        """

        """
        self.cache_manager = cache_manager
        self.logger = logging.getLogger(__name__)
    
    def load_training_dataset(self, dataset_type: Optional[str] = None,
                              detector_id: Optional[str] = None,
                              patch_size: int = 512,
                              min_anomaly_score: float = 1.0) -> Dict:
        """
            * load processed data for training with filtering
        """
        # Filter exposures by criteria
        exposures_to_load = self._filter_exposures(dataset_type, detector_id)
        
        self.logger.info(f'Loading {len(exposures_to_load)} exposures for training...')
        
        # Load and combine data
        combined_data = self._load_and_combine_patches(
            exposures_to_load, patch_size, min_anomaly_score
        )
        
        combined_data.update({
            'patch_size': patch_size,
            'dataset_type': dataset_type,
            'detector_id': detector_id,
            'min_anomaly_score': min_anomaly_score
        })
        
        self.logger.info(f'Loaded {combined_data["total_patches"]} patches for training')

        return combined_data
    
    def _filter_exposures(self, dataset_type: Optional[str], 
                          detector_id: Optional[str]) -> List[str]:
        """
            * filter exposures based on dataset type and detector
        """
        exposures = []
        
        # Iterate through all exposures in the registry
        for exposure_id, info in self.cache_manager.registry['processed_exposures'].items():
            # Filter by dataset type
            if dataset_type and info.get('dataset_type') != dataset_type:
                continue
            
            # Filter by detector ID
            if detector_id and info['detector_info'].get('detector_id') != detector_id:
                continue
            
            exposures.append(exposure_id)
        
        return exposures
    
    def _load_and_combine_patches(self, exposures: List[str], patch_size: int,
                                  min_anomaly_score: float) -> Dict:
        """
            * load and combine patch data from multiple exposures
        """
        all_patches = []
        all_positions = []
        all_frame_indices = []
        all_anomaly_scores = []
        all_exposure_ids = []
        
        # Iterate through all exposures
        for exposure_id in tqdm(exposures, desc='Loading training data'):
            # Grab the current patch file
            patch_file = self.cache_manager.root_dir / 'patches' / f'{exposure_id}_patches_{patch_size}.h5'
            
            # Make sure that the file exists
            if patch_file.exists():
                # Open the file and read the data inside of it
                with h5py.File(patch_file, 'r') as f:
                    patches = f['patches'][:]
                    positions = f['positions'][:]
                    frame_indices = f['frame_indices'][:]
                    anomaly_scores = f['anomaly_scores'][:]
                    
                    # Filter by anomaly score
                    mask = anomaly_scores >= min_anomaly_score
                    
                    # Append the anomaly data
                    if np.any(mask):
                        all_patches.append(patches[mask])
                        all_positions.append(positions[mask])
                        all_frame_indices.append(frame_indices[mask])
                        all_anomaly_scores.append(anomaly_scores[mask])
                        all_exposure_ids.extend([exposure_id] * np.sum(mask))
        
        # Combine all data
        return {
            'patches': np.concatenate(all_patches) if all_patches else np.array([]),
            'positions': np.concatenate(all_positions) if all_positions else np.array([]),
            'frame_indices': np.concatenate(all_frame_indices) if all_frame_indices else np.array([]),
            'anomaly_scores': np.concatenate(all_anomaly_scores) if all_anomaly_scores else np.array([]),
            'exposure_ids': all_exposure_ids,
            'total_patches': len(all_exposure_ids)
        }
    
    def get_statistics(self) -> Dict:
        """
            * get comprehensive statistics about processed data
        """
        stats = {
            'total_exposures': len(self.cache_manager.registry['processed_exposures']),
            'euclid_exposures': 0,
            'case_exposures': 0,
            'case_detector_1': 0,
            'case_detector_2': 0,
            'patch_sizes_available': set(),
            'processing_times': []
        }
        
        # Iterate through all exposures
        for exposure_id, info in self.cache_manager.registry['processed_exposures'].items():
            # Grab the dataset type
            dataset_type = info.get('dataset_type', 'unknown')
            detector_id = info['detector_info'].get('detector_id', 'unknown')
            
            # Make sure the dataset type is Euclid or Case
            if dataset_type == 'EUCLID':
                stats['euclid_exposures'] += 1
            elif dataset_type == 'CASE':
                stats['case_exposures'] += 1

                # Grab the specific detector for case
                if detector_id == 'detector_1':
                    stats['case_detector_1'] += 1
                elif detector_id == 'detector_2':
                    stats['case_detector_2'] += 1
            
            # Add the processing time 
            stats['processing_times'].append(info.get('processed_time', ''))
            
            # Check available patch sizes
            for patch_file in (self.cache_manager.root_dir / 'patches').glob(f'{exposure_id}_patches_*.h5'):
                size = int(patch_file.stem.split('_')[-1])
                stats['patch_sizes_available'].add(size)
        
        stats['patch_sizes_available'] = sorted(list(stats['patch_sizes_available']))
        
        if stats['processing_times']:
            valid_times = [t for t in stats['processing_times'] if t]
            if valid_times:
                stats['first_processed'] = min(valid_times)
                stats['last_processed'] = max(valid_times)
        
        return stats