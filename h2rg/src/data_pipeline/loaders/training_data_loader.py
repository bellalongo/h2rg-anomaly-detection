"""
Updated Training Data Loader for ViT-VAE
Works with your current processed data structure in job_outputs
"""

import h5py
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm

from .cache_manager import CacheManager


class TrainingDataLoader:
    """
    Updated training data loader that works with your job_outputs structure
    Loads processed data for ViT-VAE training with filtering and batching capabilities
    """
    
    def __init__(self, cache_manager: CacheManager = None, job_outputs_dir: str = None):
        """
        Initialize with either cache_manager (legacy) or job_outputs_dir (new)
        """
        self.cache_manager = cache_manager
        self.job_outputs_dir = Path(job_outputs_dir) if job_outputs_dir else None
        self.logger = logging.getLogger(__name__)
        
        # Use job_outputs structure if provided, otherwise fall back to cache_manager
        if self.job_outputs_dir and self.job_outputs_dir.exists():
            self.use_job_structure = True
            self.logger.info(f"Using job outputs structure: {self.job_outputs_dir}")
        elif self.cache_manager:
            self.use_job_structure = False
            self.logger.info("Using legacy cache manager structure")
        else:
            raise ValueError("Must provide either job_outputs_dir or cache_manager")
    
    def load_training_dataset(self, 
                              dataset_type: Optional[str] = None,
                              detector_id: Optional[str] = None,
                              patch_size: int = 512,
                              min_anomaly_score: float = 1.0,
                              max_samples: Optional[int] = None,
                              job_folders: Optional[List[str]] = None) -> Dict:
        """
        Load processed data for training with filtering
        
        Args:
            dataset_type: Filter by dataset type ('EUCLID', 'CASE', or None for all)
            detector_id: Filter by detector ID (for CASE datasets)
            patch_size: Patch size to load (128, 256, or 512)
            min_anomaly_score: Minimum anomaly score threshold
            max_samples: Maximum number of samples to load
            job_folders: Specific job folders to load from (if None, uses all)
        
        Returns:
            Dictionary containing loaded data
        """
        if self.use_job_structure:
            return self._load_from_job_structure(
                dataset_type, detector_id, patch_size, min_anomaly_score, 
                max_samples, job_folders
            )
        else:
            return self._load_from_cache_manager(
                dataset_type, detector_id, patch_size, min_anomaly_score
            )
    
    def _load_from_job_structure(self,
                                dataset_type: Optional[str],
                                detector_id: Optional[str], 
                                patch_size: int,
                                min_anomaly_score: float,
                                max_samples: Optional[int],
                                job_folders: Optional[List[str]]) -> Dict:
        """Load data from job_outputs directory structure"""
        
        # Get available job folders
        if job_folders is None:
            available_folders = [f.name for f in self.job_outputs_dir.iterdir() if f.is_dir()]
        else:
            available_folders = job_folders
        
        self.logger.info(f"Loading from {len(available_folders)} job folders...")
        
        # Initialize data containers
        all_patches = []
        all_anomaly_scores = []
        all_temporal_labels = []
        all_temporal_features = []
        all_positions = []
        all_frame_indices = []
        all_metadata = []
        
        processed_folders = 0
        total_samples = 0
        
        for folder_name in tqdm(available_folders, desc="Loading job folders"):
            folder_path = self.job_outputs_dir / folder_name
            
            if not folder_path.exists():
                continue
            
            # Apply dataset type filter
            if dataset_type:
                if dataset_type.upper() == 'EUCLID' and 'Euclid_SCA' not in folder_name:
                    continue
                elif dataset_type.upper() == 'CASE' and 'noise' not in folder_name:
                    continue
            
            # Apply detector filter for CASE datasets
            if detector_id and 'noise' in folder_name:
                # For CASE datasets, detector info might be in the file names
                pass  # Will check at file level
            
            try:
                folder_data = self._load_job_folder(
                    folder_path, patch_size, min_anomaly_score, detector_id
                )
                
                if folder_data['num_samples'] > 0:
                    all_patches.extend(folder_data['patches'])
                    all_anomaly_scores.extend(folder_data['anomaly_scores'])
                    all_temporal_labels.extend(folder_data['temporal_labels'])
                    all_temporal_features.extend(folder_data['temporal_features'])
                    all_positions.extend(folder_data['positions'])
                    all_frame_indices.extend(folder_data['frame_indices'])
                    all_metadata.extend(folder_data['metadata'])
                    
                    total_samples += folder_data['num_samples']
                    processed_folders += 1
                    
                    # Check if we've reached max samples
                    if max_samples and total_samples >= max_samples:
                        break
                        
            except Exception as e:
                self.logger.warning(f"Error loading folder {folder_name}: {e}")
                continue
        
        # Truncate to max_samples if specified
        if max_samples and total_samples > max_samples:
            all_patches = all_patches[:max_samples]
            all_anomaly_scores = all_anomaly_scores[:max_samples]
            all_temporal_labels = all_temporal_labels[:max_samples]
            all_temporal_features = all_temporal_features[:max_samples]
            all_positions = all_positions[:max_samples]
            all_frame_indices = all_frame_indices[:max_samples]
            all_metadata = all_metadata[:max_samples]
            total_samples = max_samples
        
        # Convert to numpy arrays
        result = {
            'patches': np.array(all_patches) if all_patches else np.array([]),
            'anomaly_scores': np.array(all_anomaly_scores) if all_anomaly_scores else np.array([]),
            'temporal_labels': np.array(all_temporal_labels) if all_temporal_labels else np.array([]),
            'temporal_features': np.array(all_temporal_features) if all_temporal_features else np.array([]),
            'positions': np.array(all_positions) if all_positions else np.array([]),
            'frame_indices': np.array(all_frame_indices) if all_frame_indices else np.array([]),
            'metadata': all_metadata,
            'total_patches': total_samples,
            'processed_folders': processed_folders,
            'patch_size': patch_size,
            'dataset_type': dataset_type,
            'detector_id': detector_id,
            'min_anomaly_score': min_anomaly_score
        }
        
        self.logger.info(f"Loaded {total_samples} samples from {processed_folders} folders")
        return result
    
    def _load_job_folder(self, 
                        folder_path: Path, 
                        patch_size: int, 
                        min_anomaly_score: float,
                        detector_id: Optional[str]) -> Dict:
        """Load data from a single job folder"""
        
        # Find patch files for the requested patch size
        patch_files = list(folder_path.glob(f"*_patches_{patch_size}.h5"))
        
        patches = []
        anomaly_scores = []
        temporal_labels = []
        temporal_features = []
        positions = []
        frame_indices = []
        metadata = []
        
        for patch_file in patch_files:
            # Extract base name
            base_name = patch_file.stem.replace(f"_patches_{patch_size}", "")
            
            # Apply detector filter
            if detector_id:
                if detector_id.lower() not in base_name.lower():
                    continue
            
            # Find corresponding files
            temporal_file = folder_path / f"{base_name}_temporal.h5"
            metadata_file = folder_path / f"{base_name}_metadata.json"
            
            if not temporal_file.exists():
                self.logger.warning(f"Missing temporal file for {base_name}")
                continue
            
            try:
                # Load patch data
                with h5py.File(patch_file, 'r') as f:
                    patch_data = f['patches'][:]
                    anomaly_data = f['anomaly_scores'][:]
                    
                    # Optional arrays that might not exist
                    position_data = f['positions'][:] if 'positions' in f else None
                    frame_data = f['frame_indices'][:] if 'frame_indices' in f else None
                
                # Load temporal data
                with h5py.File(temporal_file, 'r') as f:
                    first_appearance = f['first_appearance'][:]
                    persistence_count = f['persistence_count'][:]
                
                # Load metadata if available
                file_metadata = {}
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        file_metadata = json.load(f)
                
                # Filter by anomaly score
                valid_mask = anomaly_data >= min_anomaly_score
                valid_indices = np.where(valid_mask)[0]
                
                if len(valid_indices) == 0:
                    continue
                
                # Extract valid samples
                valid_patches = patch_data[valid_indices]
                valid_anomaly_scores = anomaly_data[valid_indices]
                
                # Generate temporal labels and features for valid samples
                valid_temporal_labels = []
                valid_temporal_features = []
                valid_positions = []
                valid_frame_indices = []
                
                for idx in valid_indices:
                    # Get position for temporal analysis
                    if position_data is not None:
                        pos = position_data[idx]
                    else:
                        pos = (0, 0)  # Default position
                    
                    # Get frame index
                    frame_idx = frame_data[idx] if frame_data is not None else 0
                    
                    # Classify temporal pattern
                    temporal_label = self._classify_temporal_pattern(
                        first_appearance, persistence_count, pos, valid_anomaly_scores[len(valid_temporal_labels)]
                    )
                    
                    # Extract temporal features
                    temporal_feat = self._extract_temporal_features(
                        first_appearance, persistence_count, pos, valid_anomaly_scores[len(valid_temporal_labels)]
                    )
                    
                    valid_temporal_labels.append(temporal_label)
                    valid_temporal_features.append(temporal_feat)
                    valid_positions.append(pos)
                    valid_frame_indices.append(frame_idx)
                
                # Add to collections
                patches.extend(valid_patches)
                anomaly_scores.extend(valid_anomaly_scores)
                temporal_labels.extend(valid_temporal_labels)
                temporal_features.extend(valid_temporal_features)
                positions.extend(valid_positions)
                frame_indices.extend(valid_frame_indices)
                
                # Add metadata for each sample
                for _ in range(len(valid_patches)):
                    sample_metadata = file_metadata.copy()
                    sample_metadata.update({
                        'base_name': base_name,
                        'folder_name': folder_path.name,
                        'patch_file': str(patch_file),
                        'temporal_file': str(temporal_file)
                    })
                    metadata.append(sample_metadata)
                
            except Exception as e:
                self.logger.warning(f"Error loading {patch_file}: {e}")
                continue
        
        return {
            'patches': patches,
            'anomaly_scores': anomaly_scores,
            'temporal_labels': temporal_labels,
            'temporal_features': temporal_features,
            'positions': positions,
            'frame_indices': frame_indices,
            'metadata': metadata,
            'num_samples': len(patches)
        }
    
    def _classify_temporal_pattern(self, 
                                 first_appearance: np.ndarray,
                                 persistence_count: np.ndarray, 
                                 position: Tuple[int, int],
                                 anomaly_score: float) -> int:
        """
        Classify temporal pattern based on your analysis
        Returns: 0=normal, 1=snowball, 2=cosmic_ray, 3=telegraph, 4=hot_pixel
        """
        # Get local temporal info around the position
        y, x = position
        patch_size = 32  # Approximate patch size in temporal map coordinates
        
        # Extract local region
        y_start = max(0, y - patch_size//2)
        y_end = min(first_appearance.shape[0], y + patch_size//2)
        x_start = max(0, x - patch_size//2)  
        x_end = min(first_appearance.shape[1], x + patch_size//2)
        
        if y_end <= y_start or x_end <= x_start:
            return 0  # Normal if no valid region
        
        local_first = first_appearance[y_start:y_end, x_start:x_end]
        local_persist = persistence_count[y_start:y_end, x_start:x_end]
        
        # Compute average temporal characteristics
        first_valid = local_first[local_first >= 0]
        avg_first = np.mean(first_valid) if len(first_valid) > 0 else -1
        avg_persist = np.mean(local_persist)
        
        # Classification logic from your temporal analysis
        appears_suddenly = avg_first > 0
        high_persistence = avg_persist > 100  # Adjust threshold based on your data
        
        if anomaly_score < 1.0:  # Below anomaly threshold
            return 0  # Normal
        
        if appears_suddenly and high_persistence:
            # Medium size, circular
            if 5.0 <= anomaly_score <= 50.0:
                return 1  # Snowball
            # High intensity, elongated
            elif anomaly_score > 50.0:
                return 2  # Cosmic ray  
            # Small, isolated
            else:
                return 3  # Telegraph noise
        elif not appears_suddenly and high_persistence:
            return 4  # Hot pixel (present from start)
        
        return 0  # Normal/unknown
    
    def _extract_temporal_features(self,
                                 first_appearance: np.ndarray,
                                 persistence_count: np.ndarray,
                                 position: Tuple[int, int], 
                                 anomaly_score: float) -> np.ndarray:
        """Extract numerical temporal features"""
        y, x = position
        patch_size = 32
        
        # Extract local region
        y_start = max(0, y - patch_size//2)
        y_end = min(first_appearance.shape[0], y + patch_size//2)
        x_start = max(0, x - patch_size//2)
        x_end = min(first_appearance.shape[1], x + patch_size//2)
        
        if y_end <= y_start or x_end <= x_start:
            return np.zeros(8, dtype=np.float32)
        
        local_first = first_appearance[y_start:y_end, x_start:x_end]
        local_persist = persistence_count[y_start:y_end, x_start:x_end]
        
        first_valid = local_first[local_first >= 0]
        
        features = [
            np.mean(first_valid) if len(first_valid) > 0 else 0,  # Mean first appearance
            np.std(first_valid) if len(first_valid) > 0 else 0,   # Std first appearance  
            np.mean(local_persist),                               # Mean persistence
            np.std(local_persist),                                # Std persistence
            np.max(local_persist),                                # Max persistence
            float(len(first_valid) > 0),                         # Has sudden appearance
            float(np.mean(local_persist) > 100),                 # High persistence
            anomaly_score                                         # Anomaly intensity
        ]
        
        return np.array(features, dtype=np.float32)
    
    def _load_from_cache_manager(self,
                               dataset_type: Optional[str],
                               detector_id: Optional[str],
                               patch_size: int, 
                               min_anomaly_score: float) -> Dict:
        """Legacy method using cache manager (if needed)"""
        
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
        """Filter exposures by criteria (legacy method)"""
        if not self.cache_manager:
            return []
            
        registry = self.cache_manager.registry
        
        exposures_to_load = []
        for exposure_id in registry.get("processed_exposures", {}):
            exposure_info = registry["processed_exposures"][exposure_id]
            
            # Filter by dataset type
            if dataset_type:
                exposure_dataset_type = exposure_info.get("detector_info", {}).get("dataset_type", "")
                if exposure_dataset_type.upper() != dataset_type.upper():
                    continue
            
            # Filter by detector ID  
            if detector_id:
                exposure_detector_id = exposure_info.get("detector_info", {}).get("detector_id", "")
                if exposure_detector_id != detector_id:
                    continue
            
            exposures_to_load.append(exposure_id)
        
        return exposures_to_load
    
    def _load_and_combine_patches(self,
                                exposures: List[str],
                                patch_size: int,
                                min_anomaly_score: float) -> Dict:
        """Load and combine patches from multiple exposures (legacy method)"""
        # This would use your existing cache manager logic
        # Implementation depends on your specific cache structure
        pass
    
    def get_data_statistics(self) -> Dict:
        """Get statistics about available data"""
        if not self.use_job_structure:
            return {}
        
        stats = {
            'total_folders': 0,
            'dataset_types': {},
            'patch_sizes': set(),
            'total_files': 0
        }
        
        for folder in self.job_outputs_dir.iterdir():
            if not folder.is_dir():
                continue
                
            stats['total_folders'] += 1
            
            # Determine dataset type
            if 'Euclid_SCA' in folder.name:
                dataset_type = 'EUCLID'
            elif 'noise' in folder.name:
                dataset_type = 'CASE'
            else:
                dataset_type = 'UNKNOWN'
            
            if dataset_type not in stats['dataset_types']:
                stats['dataset_types'][dataset_type] = 0
            stats['dataset_types'][dataset_type] += 1
            
            # Check available patch sizes
            patch_files = list(folder.glob("*_patches_*.h5"))
            stats['total_files'] += len(patch_files)
            
            for patch_file in patch_files:
                # Extract patch size from filename
                parts = patch_file.stem.split('_')
                for i, part in enumerate(parts):
                    if part == 'patches' and i + 1 < len(parts):
                        try:
                            patch_size = int(parts[i + 1])
                            stats['patch_sizes'].add(patch_size)
                        except ValueError:
                            pass
        
        stats['patch_sizes'] = sorted(list(stats['patch_sizes']))
        return stats


# For testing the updated data loader
if __name__ == "__main__":
    # Test with your job_outputs structure
    job_outputs_dir = "/projects/ilongo/processed_data/job_outputs"
    
    loader = TrainingDataLoader(job_outputs_dir=job_outputs_dir)
    
    # Test getting statistics
    print("Data Statistics:")
    stats = loader.get_data_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test loading a small subset
    print("\nTesting data loading...")
    data = loader.load_training_dataset(
        patch_size=512,
        min_anomaly_score=1.0,
        max_samples=100  # Small test
    )
    
    print(f"Loaded data:")
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {value}")