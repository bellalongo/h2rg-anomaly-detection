
import h5py
import json
import numpy as np
from typing import Dict
from pathlib import Path
import logging
import time
from datetime import datetime


class OptimizedDataStorage:
    """
        * handles optimized storage and retrieval of processed astronomical data
    """
    def __init__(self, root_dir: Path):
        """

        """
        self.root_dir = root_dir
        self.logger = logging.getLogger(__name__)
    
    def save_processed_exposure(self, exposure_id: str, diff_data: Dict, 
                              temporal_data: Dict, patches_data: Dict, 
                              detector_info: Dict):
        """
            * save all processed data with optimized compression
        """
        try:
            # Save difference data
            self._save_differences(exposure_id, diff_data, detector_info)
            
            # Save temporal analysis
            self._save_temporal_data(exposure_id, temporal_data)
            
            # Save patches at multiple scales
            self._save_patches_data(exposure_id, patches_data)
            
            # Save metadata
            self._save_metadata(exposure_id, diff_data, detector_info)
            
        except Exception as e:
            self.logger.error(f"Error saving exposure {exposure_id}: {e}")
            self._cleanup_partial_files(exposure_id)
            raise
    
    def _save_differences(self, exposure_id: str, diff_data: Dict, detector_info: Dict):
        """
            * save difference arrays with LZF compression
        """
        difference_start = time.time()
        self.logger.info(f"Saving difference data for {exposure_id}...")

        # Save difference data with optimized settings
        diff_file = self.root_dir / 'raw_differences' / f'{exposure_id}_differences.h5'
        
        # Open an h5py instance
        with h5py.File(diff_file, 'w') as f:
            # Use chunking and less aggressive compression or large arrays (saves faster!)
            chunk_shape = (1, diff_data['differences'].shape[1], diff_data['differences'].shape[2])
            
            # Create differences dataset
            f.create_dataset('differences', 
                           data=diff_data['differences'],
                           compression='lzf',
                           chunks=chunk_shape,
                           shuffle=True)
            
            # Save the frame times
            f.create_dataset('frame_times', data=diff_data['frame_times'])
            f.create_dataset('reference_frame', 
                           data=diff_data['reference_frame'],
                           compression='lzf',
                           shuffle=True)
            
            # Store attributes
            f.attrs['total_frames'] = diff_data['total_frames']
            f.attrs['exposure_id'] = exposure_id
            
            # Save detector information
            for key, value in detector_info.items():
                f.attrs[key] = str(value) if not isinstance(value, str) else value

            self.logger.info(f"Difference data saved in {time.time() - difference_start:.1f}s")
    
    def _save_temporal_data(self, exposure_id: str, temporal_data: Dict):
        """
            * save temporal analysis with chunked compression
        """
        temporal_start = time.time()
        self.logger.info(f"Saving temporal analysis for {exposure_id}...")

        # Grab temporal filename
        temporal_file = self.root_dir / 'temporal_analysis' / f'{exposure_id}_temporal.h5'
        
        # Open h5py instance
        with h5py.File(temporal_file, 'w') as f:
            # Use chunking to save faster
            chunk_2d = (512, 512)
            
            for key in ['first_appearance', 'persistence_count', 'max_intensity']:
                f.create_dataset(key,
                               data=temporal_data[key],
                               chunks=chunk_2d,
                               compression='lzf')
            
            f.attrs['threshold_used'] = temporal_data['threshold_used']
            
            # Save temporal evolution as structured array
            evolution_dtype = [('frame', 'i4'), ('n_anomalies', 'i4'),
                             ('mean_intensity', 'f4'), ('max_intensity', 'f4')]
            
            # Create the evolutional dataset
            evolution_array = np.array(
                [(e['frame'], e['n_anomalies'], e['mean_intensity'], e['max_intensity'])
                 for e in temporal_data['temporal_evolution']], 
                dtype=evolution_dtype
            )
            
            f.create_dataset('temporal_evolution', data=evolution_array)

            self.logger.info(f"Temporal analysis saved in {time.time() - temporal_start:.1f}s")
    
    def _save_patches_data(self, exposure_id: str, patches_data: Dict):
        """
            * save patches with optimized chunking
        """
        patch_start = time.time()
        self.logger.info(f"Saving {len(patch_data['patches'])} patches of size {patch_size}x{patch_size}...")

        # Iterate through all patch data
        for patch_key, patch_data in patches_data.items():
            # Grab patch size and filename
            patch_size = patch_data['patch_size']
            patch_file = self.root_dir / 'patches' / f'{exposure_id}_patches_{patch_size}.h5'
            
            # Open h5py instance
            with h5py.File(patch_file, 'w') as f:
                # Optimize the chunk size for the current patch
                n_patches = patch_data['patches'].shape[0]
                chunk_size = min(100, n_patches)
                patch_chunk = (chunk_size, patch_size, patch_size)
                
                # Use faster compression for the patchs (save faster!!)
                f.create_dataset('patches',
                               data=patch_data['patches'],
                               compression='lzf',
                               chunks=patch_chunk,
                               shuffle=True)
                
                # Store metadata
                for key in ['positions', 'frame_indices', 'anomaly_scores']:
                    f.create_dataset(key, data=patch_data[key])
                
                # Store attributes
                for attr in ['patch_size', 'overlap', 'stride', 'grid_shape']:
                    f.attrs[attr] = patch_data[attr]

            self.logger.info(f"Patches {patch_size}x{patch_size} saved")

        self.logger.info(f"All patches saved in {time.time() - patch_start:.1f}s")
    
    def _save_metadata(self, exposure_id: str, diff_data: Dict, detector_info: Dict):
        """
            * save processing metadata as JSON
        """
        metadata_start = time.time()

        # Create metadata dictionary w/ current data
        metadata = {
            'exposure_id': exposure_id,
            'processing_time': datetime.now().isoformat(timespec='microseconds'),
            'detector_info': detector_info,
            'total_frames': diff_data['total_frames']
        }
        
        # Create metadata file and write the metadata to it
        metadata_file = self.root_dir / 'metadata' / f'{exposure_id}_metadata.json'

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Metadata saved in {time.time() - metadata_start:.1f}s")
    
    def _cleanup_partial_files(self, exposure_id: str):
        """
            * remove partially written files on error
        """
        patterns = [
            f"{exposure_id}_differences.h5",
            f"{exposure_id}_temporal.h5",
            f"{exposure_id}_patches_*.h5", 
            f"{exposure_id}_metadata.json"
        ]
        
        # Iterate through every file type and remove the incomplete ones
        for pattern in patterns:
            for file_path in self.root_dir.rglob(pattern):
                if file_path.exists():
                    try:
                        file_path.unlink()
                        self.logger.info(f"Cleaned up partial file: {file_path}")
                    except Exception as e:
                        self.logger.warning(f"Could not clean up {file_path}: {e}")
