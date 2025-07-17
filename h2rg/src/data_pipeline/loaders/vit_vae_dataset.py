"""
PyTorch Dataset classes for ViT-VAE training
Optimized for your job_outputs directory structure
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import warnings


class JobBasedAnomalyDataset(Dataset):
    """
    PyTorch Dataset that loads from your job_outputs directory structure
    Handles multiple patch sizes and temporal information efficiently
    """
    
    def __init__(self, 
                 job_outputs_dir: str,
                 patch_size: int = 512,
                 min_anomaly_score: float = 1.0,
                 max_samples_per_job: Optional[int] = None,
                 temporal_classification: bool = True,
                 cache_in_memory: bool = False,
                 data_augmentation: bool = True,
                 job_folders: Optional[List[str]] = None,
                 normalize_patches: bool = True):
        
        self.job_outputs_dir = Path(job_outputs_dir)
        self.patch_size = patch_size
        self.min_anomaly_score = min_anomaly_score
        self.max_samples_per_job = max_samples_per_job
        self.temporal_classification = temporal_classification
        self.cache_in_memory = cache_in_memory
        self.data_augmentation = data_augmentation
        self.job_folders = job_folders
        self.normalize_patches = normalize_patches
        
        self.logger = logging.getLogger(__name__)
        
        # Build dataset index
        self.data_index = self._build_dataset_index()
        self.logger.info(f"Dataset initialized with {len(self.data_index)} samples")
        
        # Preload data if caching enabled and dataset is small
        if self.cache_in_memory and len(self.data_index) < 10000:
            self.logger.info("Preloading data into memory...")
            self._preload_data()
    
    def _build_dataset_index(self) -> List[Dict]:
        """Build index of all available data samples from job outputs"""
        index = []
        
        if not self.job_outputs_dir.exists():
            self.logger.error(f"Job outputs directory not found: {self.job_outputs_dir}")
            return index
        
        # Get job folders to process
        if self.job_folders is not None:
            job_folders = [self.job_outputs_dir / name for name in self.job_folders 
                          if (self.job_outputs_dir / name).exists()]
        else:
            job_folders = [f for f in self.job_outputs_dir.iterdir() if f.is_dir()]
        
        self.logger.info(f"Indexing {len(job_folders)} job folders...")
        
        for job_folder in tqdm(job_folders, desc="Indexing job folders"):
            try:
                job_samples = self._index_job_folder(job_folder)
                index.extend(job_samples)
            except Exception as e:
                self.logger.warning(f"Error indexing job folder {job_folder}: {e}")
                continue
        
        self.logger.info(f"Indexed {len(job_folders)} job folders, found {len(index)} samples")
        return index
    
    def _index_job_folder(self, job_folder: Path) -> List[Dict]:
        """Index a single job folder for available data"""
        samples = []
        
        # Look in the patches subdirectory
        patches_dir = job_folder / "patches"
        if not patches_dir.exists():
            return samples
        
        # Find patch files for the requested patch size
        patch_files = list(patches_dir.glob(f"*_patches_{self.patch_size}.h5"))
        
        for patch_file in patch_files:
            # Extract base name to find corresponding files
            base_name = patch_file.stem.replace(f"_patches_{self.patch_size}", "")
            
            # Find corresponding temporal and metadata files in their subdirectories
            temporal_file = job_folder / "temporal_analysis" / f"{base_name}_temporal.h5"
            metadata_file = job_folder / "metadata" / f"{base_name}_metadata.json"
            
            # Validate required files exist
            if not temporal_file.exists():
                self.logger.warning(f"Missing temporal file for {base_name}")
                continue
            
            try:
                # Quick check of patch file to get valid sample indices
                with h5py.File(patch_file, 'r') as f:
                    if 'patches' not in f or 'anomaly_scores' not in f:
                        continue
                        
                    num_patches = f['patches'].shape[0]
                    anomaly_scores = f['anomaly_scores'][:]
                    
                    # Filter by anomaly score
                    valid_indices = np.where(anomaly_scores >= self.min_anomaly_score)[0]
                    
                    if len(valid_indices) == 0:
                        continue
                    
                    # Limit samples per job if specified
                    if self.max_samples_per_job and len(valid_indices) > self.max_samples_per_job:
                        valid_indices = np.random.choice(
                            valid_indices, 
                            self.max_samples_per_job, 
                            replace=False
                        )
                    
                    # Add samples to index
                    for patch_idx in valid_indices:
                        samples.append({
                            'job_folder': job_folder.name,
                            'base_name': base_name,
                            'patch_file': patch_file,
                            'temporal_file': temporal_file,
                            'metadata_file': metadata_file if metadata_file.exists() else None,
                            'patch_idx': int(patch_idx),
                            'anomaly_score': float(anomaly_scores[patch_idx])
                        })
            
            except Exception as e:
                self.logger.warning(f"Error reading patch file {patch_file}: {e}")
                continue
        
        return samples
    
    def _preload_data(self):
        """Preload all data into memory for faster access"""
        self.memory_cache = {}
        
        # Use ThreadPoolExecutor for parallel loading
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            for idx, sample in enumerate(self.data_index):
                future = executor.submit(self._load_sample_from_disk, sample)
                futures[future] = idx
            
            for future in tqdm(futures, desc="Preloading data"):
                try:
                    idx = futures[future]
                    data = future.result()
                    self.memory_cache[idx] = data
                except Exception as e:
                    self.logger.warning(f"Error preloading sample: {e}")
                    continue
    
    def _load_sample_from_disk(self, sample_info: Dict) -> Dict:
        """Load a single sample from disk"""
        try:
            # Load patch data
            with h5py.File(sample_info['patch_file'], 'r') as f:
                patch = f['patches'][sample_info['patch_idx']].copy()
                position = f['positions'][sample_info['patch_idx']].copy() if 'positions' in f else np.array([0, 0])
                frame_idx = f['frame_indices'][sample_info['patch_idx']] if 'frame_indices' in f else 0
            
            # Load temporal data if needed
            temporal_features = None
            temporal_label = 0
            
            if self.temporal_classification:
                try:
                    with h5py.File(sample_info['temporal_file'], 'r') as f:
                        first_appearance = f['first_appearance'][:]
                        persistence_count = f['persistence_count'][:]
                        
                        # Extract temporal features around patch position
                        temporal_features = self._extract_temporal_features(
                            first_appearance, persistence_count, position, sample_info['anomaly_score']
                        )
                        
                        # Classify temporal pattern
                        temporal_label = self._classify_temporal_pattern(
                            first_appearance, persistence_count, position, sample_info['anomaly_score']
                        )
                
                except Exception as e:
                    self.logger.warning(f"Error loading temporal data: {e}")
                    temporal_features = np.zeros(8, dtype=np.float32)
                    temporal_label = 0
            
            # Load metadata if available
            metadata = {}
            if sample_info['metadata_file'] and sample_info['metadata_file'].exists():
                try:
                    with open(sample_info['metadata_file'], 'r') as f:
                        metadata = json.load(f)
                except Exception as e:
                    self.logger.warning(f"Error loading metadata: {e}")
            
            return {
                'patch': patch.astype(np.float32),
                'temporal_features': temporal_features,
                'temporal_label': temporal_label,
                'anomaly_score': sample_info['anomaly_score'],
                'position': position,
                'frame_idx': frame_idx,
                'metadata': metadata,
                'job_folder': sample_info['job_folder'],
                'base_name': sample_info['base_name']
            }
            
        except Exception as e:
            self.logger.error(f"Error loading sample {sample_info}: {e}")
            # Return dummy data to prevent crashes
            return {
                'patch': np.zeros((self.patch_size, self.patch_size), dtype=np.float32),
                'temporal_features': np.zeros(8, dtype=np.float32),
                'temporal_label': 0,
                'anomaly_score': 0.0,
                'position': np.array([0, 0]),
                'frame_idx': 0,
                'metadata': {},
                'job_folder': sample_info.get('job_folder', ''),
                'base_name': sample_info.get('base_name', '')
            }
    
    def _extract_temporal_features(self, first_appearance: np.ndarray, 
                                 persistence_count: np.ndarray, 
                                 position: np.ndarray, 
                                 anomaly_score: float) -> np.ndarray:
        """Extract numerical temporal features for the model"""
        y, x = position.astype(int)
        
        # Calculate patch size in temporal map coordinates
        temporal_patch_size = max(1, self.patch_size // 16)
        
        # Extract local region around patch position
        y_start = max(0, y - temporal_patch_size//2)
        y_end = min(first_appearance.shape[0], y + temporal_patch_size//2)
        x_start = max(0, x - temporal_patch_size//2)
        x_end = min(first_appearance.shape[1], x + temporal_patch_size//2)
        
        if y_end <= y_start or x_end <= x_start:
            return np.zeros(8, dtype=np.float32)
        
        local_first = first_appearance[y_start:y_end, x_start:x_end]
        local_persist = persistence_count[y_start:y_end, x_start:x_end]
        
        # Compute statistical features from temporal maps
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
    
    def _classify_temporal_pattern(self, first_appearance: np.ndarray, 
                                 persistence_count: np.ndarray, 
                                 position: np.ndarray, 
                                 anomaly_score: float) -> int:
        """
        Classify temporal pattern based on your analysis
        Returns: 0=normal, 1=snowball, 2=cosmic_ray, 3=telegraph, 4=hot_pixel
        """
        y, x = position.astype(int)
        temporal_patch_size = max(1, self.patch_size // 16)
        
        # Extract local region
        y_start = max(0, y - temporal_patch_size//2)
        y_end = min(first_appearance.shape[0], y + temporal_patch_size//2)
        x_start = max(0, x - temporal_patch_size//2)
        x_end = min(first_appearance.shape[1], x + temporal_patch_size//2)
        
        if y_end <= y_start or x_end <= x_start:
            return 0
        
        local_first = first_appearance[y_start:y_end, x_start:x_end]
        local_persist = persistence_count[y_start:y_end, x_start:x_end]
        
        first_valid = local_first[local_first >= 0]
        avg_first = np.mean(first_valid) if len(first_valid) > 0 else -1
        avg_persist = np.mean(local_persist)
        
        appears_suddenly = avg_first > 0
        high_persistence = avg_persist > 100  # Adjust threshold based on your data
        
        if anomaly_score < self.min_anomaly_score:
            return 0  # Normal
        
        if appears_suddenly and high_persistence:
            if 5.0 <= anomaly_score <= 50.0:  # Medium size, circular
                return 1  # Snowball
            elif anomaly_score > 50.0:        # High intensity, elongated
                return 2  # Cosmic ray
            else:                             # Small, isolated
                return 3  # Telegraph noise
        elif not appears_suddenly and high_persistence:
            return 4  # Hot pixel (present from start)
        
        return 0  # Normal/unknown
    
    def _apply_augmentation(self, patch: np.ndarray) -> np.ndarray:
        """Apply data augmentation to patch"""
        if not self.data_augmentation:
            return patch
        
        # Random rotation (90, 180, 270 degrees)
        if np.random.random() < 0.5:
            patch = np.rot90(patch, k=np.random.randint(1, 4))
        
        # Random flip
        if np.random.random() < 0.5:
            patch = np.flip(patch, axis=0).copy()
        if np.random.random() < 0.5:
            patch = np.flip(patch, axis=1).copy()
        
        # Small random noise (preserve anomalies)
        if np.random.random() < 0.2:
            noise_scale = 0.01 * np.std(patch)
            if noise_scale > 0:
                patch = patch + np.random.normal(0, noise_scale, patch.shape).astype(np.float32)
        
        return patch
    
    def _normalize_patch(self, patch: np.ndarray) -> np.ndarray:
        """Normalize patch for training"""
        if not self.normalize_patches:
            return patch
        
        # Robust normalization to handle outliers
        patch_mean = np.mean(patch)
        patch_std = np.std(patch)
        
        if patch_std > 0:
            patch = (patch - patch_mean) / patch_std
        
        # Clip extreme values to prevent gradient issues
        patch = np.clip(patch, -5, 5)
        
        return patch
    
    def __len__(self) -> int:
        return len(self.data_index)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample"""
        # Load from memory cache if available
        if hasattr(self, 'memory_cache') and idx in self.memory_cache:
            data = self.memory_cache[idx]
        else:
            sample_info = self.data_index[idx]
            data = self._load_sample_from_disk(sample_info)
        
        # Process patch
        patch = data['patch'].copy()
        
        # Apply augmentation
        patch = self._apply_augmentation(patch)
        
        # Normalize patch
        patch = self._normalize_patch(patch)
        
        # Convert to torch tensors
        result = {
            'patch': torch.from_numpy(patch).float(),
            'anomaly_score': torch.tensor(data['anomaly_score'], dtype=torch.float32),
            'job_folder': data['job_folder'],
            'base_name': data['base_name']
        }
        
        # Add temporal information if available
        if self.temporal_classification:
            result['temporal_label'] = torch.tensor(data['temporal_label'], dtype=torch.long)
            if data['temporal_features'] is not None:
                result['temporal_features'] = torch.from_numpy(data['temporal_features']).float()
            else:
                result['temporal_features'] = torch.zeros(8, dtype=torch.float32)
        
        # Add position information
        result['position'] = torch.from_numpy(data['position']).long()
        result['frame_idx'] = torch.tensor(data['frame_idx'], dtype=torch.long)
        
        return result
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Get distribution of temporal classes in the dataset"""
        if not self.temporal_classification:
            return {}
        
        class_counts = {}
        for idx in tqdm(range(len(self)), desc="Computing class distribution"):
            sample = self[idx]
            label = sample['temporal_label'].item()
            class_counts[label] = class_counts.get(label, 0) + 1
        
        return class_counts
    
    def get_anomaly_score_statistics(self) -> Dict[str, float]:
        """Get statistics about anomaly scores in the dataset"""
        scores = [sample['anomaly_score'] for sample in self.data_index]
        
        return {
            'min': np.min(scores),
            'max': np.max(scores),
            'mean': np.mean(scores),
            'std': np.std(scores),
            'median': np.median(scores),
            'count': len(scores)
        }


class JobDataLoaderFactory:
    """Factory class to create optimized data loaders for your job structure"""
    
    @staticmethod
    def create_train_val_loaders(job_outputs_dir: str,
                               config: Dict,
                               val_split: float = 0.2) -> Tuple[DataLoader, DataLoader]:
        """Create train and validation data loaders"""
        
        # Get all job folders
        job_path = Path(job_outputs_dir)
        job_folders = [f.name for f in job_path.iterdir() if f.is_dir()]
        
        if len(job_folders) == 0:
            raise ValueError(f"No job folders found in {job_outputs_dir}")
        
        # Split job folders into train and validation
        np.random.seed(42)  # For reproducible splits
        np.random.shuffle(job_folders)
        
        val_size = max(1, int(len(job_folders) * val_split))
        train_folders = job_folders[:-val_size] if val_size < len(job_folders) else job_folders
        val_folders = job_folders[-val_size:]
        
        print(f"Train folders: {len(train_folders)}, Val folders: {len(val_folders)}")
        
        # Create datasets
        train_dataset = JobBasedAnomalyDataset(
            job_outputs_dir=job_outputs_dir,
            patch_size=config.get('patch_size', 512),
            min_anomaly_score=config.get('min_anomaly_score', 1.0),
            max_samples_per_job=config.get('max_samples_per_job', None),
            temporal_classification=config.get('temporal_classification', True),
            cache_in_memory=config.get('cache_in_memory', False),
            data_augmentation=True,
            job_folders=train_folders,
            normalize_patches=config.get('normalize_patches', True)
        )
        
        val_dataset = JobBasedAnomalyDataset(
            job_outputs_dir=job_outputs_dir,
            patch_size=config.get('patch_size', 512),
            min_anomaly_score=config.get('min_anomaly_score', 1.0),
            max_samples_per_job=config.get('max_val_samples_per_job', None),
            temporal_classification=config.get('temporal_classification', True),
            cache_in_memory=config.get('cache_in_memory', False),
            data_augmentation=False,  # No augmentation for validation
            job_folders=val_folders,
            normalize_patches=config.get('normalize_patches', True)
        )
        
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.get('batch_size', 8),
            shuffle=True,
            num_workers=config.get('num_workers', 4),
            pin_memory=True,
            drop_last=True,
            persistent_workers=config.get('num_workers', 4) > 0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.get('batch_size', 8),
            shuffle=False,
            num_workers=config.get('num_workers', 4),
            pin_memory=True,
            persistent_workers=config.get('num_workers', 4) > 0
        )
        
        return train_loader, val_loader
    
    @staticmethod
    def create_inference_loader(job_outputs_dir: str,
                              config: Dict,
                              job_folders: Optional[List[str]] = None) -> DataLoader:
        """Create data loader for inference on specific job folders"""
        
        dataset = JobBasedAnomalyDataset(
            job_outputs_dir=job_outputs_dir,
            patch_size=config.get('patch_size', 512),
            min_anomaly_score=config.get('min_anomaly_score', 0.0),  # Include all for inference
            max_samples_per_job=None,
            temporal_classification=config.get('temporal_classification', True),
            cache_in_memory=False,
            data_augmentation=False,
            job_folders=job_folders,
            normalize_patches=config.get('normalize_patches', True)
        )
        
        loader = DataLoader(
            dataset,
            batch_size=config.get('inference_batch_size', 16),
            shuffle=False,
            num_workers=config.get('num_workers', 4),
            pin_memory=True
        )
        
        return loader


def test_dataset():
    """Test the dataset with your job structure"""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Test configuration
    job_outputs_dir = "/projects/ilongo/processed_data/job_outputs"
    
    try:
        # Create dataset
        dataset = JobBasedAnomalyDataset(
            job_outputs_dir=job_outputs_dir,
            patch_size=512,
            min_anomaly_score=1.0,
            max_samples_per_job=50,  # Small test
            temporal_classification=True,
            data_augmentation=True
        )
        
        print(f"Dataset size: {len(dataset)}")
        
        if len(dataset) > 0:
            # Test loading a sample
            sample = dataset[0]
            print(f"Sample keys: {list(sample.keys())}")
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape} ({value.dtype})")
                else:
                    print(f"  {key}: {type(value)}")
            
            # Test class distribution
            if len(dataset) < 1000:  # Only for small datasets
                class_dist = dataset.get_class_distribution()
                print(f"Class distribution: {class_dist}")
            
            # Test anomaly score statistics
            score_stats = dataset.get_anomaly_score_statistics()
            print(f"Anomaly score statistics: {score_stats}")
            
            print("Dataset test successful!")
        else:
            print("Dataset is empty - check your data path and filters")
            
    except Exception as e:
        print(f"Dataset test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_dataset()