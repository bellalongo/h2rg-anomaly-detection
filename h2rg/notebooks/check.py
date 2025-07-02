from tqdm import tqdm
import os
from os.path import exists
from pathlib import Path
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.animation as animation
from PIL import Image
import re
import tifffile
import h5py
import json
import hashlib
import time
import logging
from typing import Dict, List, Tuple, Optional

# Your existing constants
DATA_ROOT_DIR = '/proj/case/2025-06-05'
ALL_DIRS = os.listdir(DATA_ROOT_DIR)
EUCLID_DIRS = []
CASE_DIRS = []

FPM_NUM_FILES = 22500
TOTAL_FRAMES = 450
IMG_SIZE = (2048, 2048)
DETECTOR_SIZE = (450, 2048, 2048)

for dir in ALL_DIRS:
    if 'Euclid' in dir:
        EUCLID_DIRS.append(dir)
    elif 'FPM' in dir:
        CASE_DIRS.append(dir)

print(f'Euclid test data directories: {EUCLID_DIRS}\n')
print(f'Case test data directories: {CASE_DIRS}')


class TrainingData:
    """
    Enhanced training data class with caching for EUCLID and CASE datasets
    """
    
    def __init__(self, root_dir='training_set'):
        """
        Initialize training data processor with caching capabilities
        """
        self.root_dir = Path(root_dir)
        self.create_directory()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Cache registry
        self.registry_file = self.root_dir / "processing_registry.json"
        self.registry = self.load_registry()
        
        # Processing parameters
        self.patch_sizes = [512, 256, 128]
        self.overlap_ratio = 0.25  # 25% overlap

    def create_directory(self):
        """
        Create training set directory structure for preprocessed data storage as h5py
        """
        # Make the training set directory
        self.root_dir.mkdir(parents=True, exist_ok=True)

        # Make difference array directory
        (self.root_dir / 'raw_differences').mkdir(parents=True, exist_ok=True)

        # Make patches directory
        (self.root_dir / 'patches').mkdir(parents=True, exist_ok=True)

        # Make temporal analysis directory
        (self.root_dir / 'temporal_analysis').mkdir(parents=True, exist_ok=True)
        
        # Make metadata directory
        (self.root_dir / 'metadata').mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Created directory structure at {self.root_dir}")

    def load_registry(self) -> Dict:
        """Load processing registry or create new one"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {"processed_files": {}, "processed_exposures": {}, "last_updated": time.time()}
    
    def save_registry(self):
        """Save processing registry"""
        self.registry["last_updated"] = time.time()
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def get_file_hash(self, file_path: str) -> str:
        """Generate hash of file for cache validation"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def get_exposure_hash(self, file_paths: List[str]) -> str:
        """Generate hash for a group of files (exposure)"""
        hasher = hashlib.md5()
        for file_path in sorted(file_paths):  # Sort for consistent hashing
            hasher.update(file_path.encode())
            if os.path.exists(file_path):
                hasher.update(str(os.path.getmtime(file_path)).encode())
        return hasher.hexdigest()
    
    def is_exposure_cached(self, exposure_id: str, file_paths: List[str]) -> bool:
        """Check if exposure has been processed and cached"""
        if exposure_id not in self.registry["processed_exposures"]:
            return False
        
        # Verify files haven't changed
        current_hash = self.get_exposure_hash(file_paths)
        cached_hash = self.registry["processed_exposures"][exposure_id]["exposure_hash"]
        
        return current_hash == cached_hash

    def _grab_filenames(self, path):
        """
        Grab all filenames for the current path
        """
        try:
            entries = os.listdir(path)
            filenames = [entry for entry in entries if os.path.isfile(os.path.join(path, entry))]
        
        except FileNotFoundError as e:
            self.logger.error(f'Folder not found at {path}')
            return None
        
        return filenames

    def _subtract_ref_pixels(self, frame):
        """
        Perform ref pixel subtraction as documented in paper:
        Reference Pixel Subtraction Method for EUCLID SCS Noise Specification
        By Bogna Kubik
        """
        # Make sure shape is 2048 x 2048
        if frame.shape != (2048, 2048):
            raise ValueError(f'Invalid frame shape of {frame.shape}, must be (2048, 2048)')
        
        corrected_frame = frame.copy().astype(np.float64)

        # Optimal window size via paper
        x_opt = 64
        y_opt = 4

        # Reference pixel regions
        up_ref = frame[0:4, :] # Top 4 rows
        down_ref = frame[2044:2048, :] # Bottom 4 rows
        left_ref = frame[:, 0:4] # Left 4 cols
        right_ref = frame[:, 2044:2048]

        # Process each of the 32 channels
        for ch in range(32):
            # Skip left ref pixels
            if ch == 0:
                col_start, col_end = 4, 64
            
            # Skip right ref pixels
            elif ch == 31:
                col_start, col_end = ch * 64, 2044

            # Inner channels
            else:
                col_start, col_end = ch * 64, (ch + 1) * 64

            # Up/down correction w/ extract channel specific ref pixels 
            for col in range(col_end - col_start):
                global_col = col_start + col

                # Sliding window for up/down
                window_start = max(0, global_col - x_opt)
                window_end = min(2048, global_col + x_opt + 1)

                # Average the up and down ref pixels
                up_avg = np.mean(up_ref[:, window_start:window_end])
                down_avg = np.mean(down_ref[:, window_start:window_end])

                # Interpolate correction
                slope = (up_avg - down_avg) / 2044

                # Apply correction to each row in the column, skipping refs
                for row in range(4, 2044):
                    ref_correction = down_avg + (row - 1.5) * slope
                    corrected_frame[row, global_col] -= ref_correction

        # Correct left ref pixel
        left_ref_corrected = left_ref.copy()
        right_ref_corrected = right_ref.copy()

        # Subtract the up/down correction from the left/right pixels
        up_avg_full = np.mean(up_ref)
        down_avg_full = np.mean(down_ref)
        slope_full = (up_avg_full - down_avg_full) / 2044

        # Correct the left/right ref pixels with proper loop
        for row in range(4, 2044):
            ref_correction = down_avg_full + (row - 1.5) * slope_full
            left_ref_corrected[row, :] -= ref_correction
            right_ref_corrected[row, :] -= ref_correction

        # Apply the correction using sliding window
        for row in range(4, 2044):
            # Sliding window for left/right
            window_start = max(4, row - y_opt)
            window_end = min(2044, row + y_opt + 1)

            # Average corrected left/right pixels
            left_avg = np.mean(left_ref_corrected[window_start:window_end, :])
            right_avg = np.mean(right_ref_corrected[window_start:window_end, :])
            lr_correction = (left_avg + right_avg) / 2

            corrected_frame[row, 4:2044] -= lr_correction

        return corrected_frame

    def _compute_difference_fits(self, file_path):
        """
        Subtract frame 0 from all subsequent frames in the FITS file
        """
        frame_difs = []
        frame_times = []

        # Open the current fits file
        with fits.open(file_path) as hdul:
            # Grab frame 0 and apply reference pixel correction
            frame_0 = hdul[1].data.astype(np.float64)
            frame_0 = self._subtract_ref_pixels(frame_0)

            # Iterate through all frames, skipping the first
            for i in tqdm(range(2, min(TOTAL_FRAMES, len(hdul))), 
                         desc=f"Processing {Path(file_path).name}"):
                # Grab the current frame and apply reference pixel correction
                curr_frame = hdul[i].data.astype(np.float64)
                curr_frame = self._subtract_ref_pixels(curr_frame)

                # Calculate difference from frame 0
                dif = np.abs(curr_frame - frame_0)
                frame_difs.append(dif)
                frame_times.append(i)

        return {
            "differences": np.array(frame_difs),
            "frame_times": np.array(frame_times),
            "reference_frame": frame_0,
            "total_frames": len(frame_difs)
        }
    
    def _compute_difference_tif(self, detector_frames):
        """
        Subtract frame 0 from all subsequent frames for TIF detector data
        """
        detector_frames = np.array(detector_frames)
        
        # Grab the 0th frame for the current detector
        frame_0 = detector_frames[0].astype(np.float64)
        frame_difs = []
        frame_times = []

        # Iterate through all frames
        for i, frame in enumerate(detector_frames[1:], start=1):  # Skip frame 0
            curr_frame = frame.astype(np.float64)

            # Perform the frame difference
            dif = np.abs(curr_frame - frame_0)
            frame_difs.append(dif)
            frame_times.append(i)

        return {
            "differences": np.array(frame_difs),
            "frame_times": np.array(frame_times),
            "reference_frame": frame_0,
            "total_frames": len(frame_difs)
        }

    def _analyze_temporal_patterns(self, diff_stack: np.ndarray, threshold: float = 5.0):
        """
        Analyze temporal patterns in difference images for snowball/cosmic ray detection
        """
        n_frames, h, w = diff_stack.shape
        
        # Initialize temporal maps
        first_appearance = np.full((h, w), -1, dtype=np.int32)
        persistence_count = np.zeros((h, w), dtype=np.int32)
        max_intensity = np.zeros((h, w), dtype=np.float32)
        
        # Track temporal evolution
        temporal_evolution = []
        
        for frame_idx in tqdm(range(n_frames), desc="Temporal analysis"):
            diff_frame = diff_stack[frame_idx]
            
            # Find anomalies above threshold
            anomaly_mask = diff_frame > threshold
            
            # Update first appearance
            new_anomalies = anomaly_mask & (first_appearance == -1)
            first_appearance[new_anomalies] = frame_idx
            
            # Update persistence count
            persistence_count[anomaly_mask] += 1
            
            # Update max intensity
            max_intensity = np.maximum(max_intensity, diff_frame)
            
            # Track frame-level statistics
            temporal_evolution.append({
                "frame": frame_idx,
                "n_anomalies": np.sum(anomaly_mask),
                "mean_intensity": np.mean(diff_frame[anomaly_mask]) if np.any(anomaly_mask) else 0,
                "max_intensity": np.max(diff_frame)
            })
        
        return {
            "first_appearance": first_appearance,
            "persistence_count": persistence_count,
            "max_intensity": max_intensity,
            "temporal_evolution": temporal_evolution,
            "threshold_used": threshold
        }
    
    def _extract_patches(self, diff_stack: np.ndarray, patch_size: int):
        """
        Extract overlapping patches from difference images for ViT-VAE training
        """
        n_frames, h, w = diff_stack.shape
        overlap = int(patch_size * self.overlap_ratio)
        stride = patch_size - overlap
        
        patches = []
        positions = []
        frame_indices = []
        anomaly_scores = []
        
        # Calculate patch grid
        n_patches_h = (h - patch_size) // stride + 1
        n_patches_w = (w - patch_size) // stride + 1
        
        self.logger.info(f"Extracting {n_patches_h}x{n_patches_w} patches of size {patch_size}x{patch_size}")
        
        for frame_idx in tqdm(range(n_frames), desc=f"Extracting {patch_size}x{patch_size} patches"):
            diff_frame = diff_stack[frame_idx]
            
            for i in range(0, h - patch_size + 1, stride):
                for j in range(0, w - patch_size + 1, stride):
                    patch = diff_frame[i:i+patch_size, j:j+patch_size]
                    
                    # Calculate anomaly score for this patch
                    anomaly_score = np.max(patch)  # Simple max intensity
                    
                    patches.append(patch)
                    positions.append((i, j))
                    frame_indices.append(frame_idx)
                    anomaly_scores.append(anomaly_score)
        
        return {
            "patches": np.array(patches),
            "positions": np.array(positions),
            "frame_indices": np.array(frame_indices),
            "anomaly_scores": np.array(anomaly_scores),
            "patch_size": patch_size,
            "overlap": overlap,
            "stride": stride,
            "grid_shape": (n_patches_h, n_patches_w)
        }

    def _save_processed_exposure(self, exposure_id: str, diff_data: Dict, 
                               temporal_data: Dict, patches_data: Dict,
                               detector_info: Dict):
        """
        Save all processed data for an exposure to H5 files
        """
        # Save difference data
        diff_file = self.root_dir / 'raw_differences' / f"{exposure_id}_differences.h5"
        with h5py.File(diff_file, 'w') as f:
            f.create_dataset("differences", data=diff_data["differences"], 
                           compression='gzip', compression_opts=9)
            f.create_dataset("frame_times", data=diff_data["frame_times"])
            f.create_dataset("reference_frame", data=diff_data["reference_frame"],
                           compression='gzip', compression_opts=9)
            f.attrs["total_frames"] = diff_data["total_frames"]
            f.attrs["exposure_id"] = exposure_id
            
            # Save detector info
            for key, value in detector_info.items():
                f.attrs[key] = value
        
        # Save temporal analysis
        temporal_file = self.root_dir / 'temporal_analysis' / f"{exposure_id}_temporal.h5"
        with h5py.File(temporal_file, 'w') as f:
            f.create_dataset("first_appearance", data=temporal_data["first_appearance"])
            f.create_dataset("persistence_count", data=temporal_data["persistence_count"])
            f.create_dataset("max_intensity", data=temporal_data["max_intensity"])
            f.attrs["threshold_used"] = temporal_data["threshold_used"]
            
            # Save temporal evolution
            evolution_dtype = [('frame', 'i4'), ('n_anomalies', 'i4'), 
                             ('mean_intensity', 'f4'), ('max_intensity', 'f4')]
            evolution_array = np.array([(e["frame"], e["n_anomalies"], 
                                       e["mean_intensity"], e["max_intensity"]) 
                                      for e in temporal_data["temporal_evolution"]], 
                                     dtype=evolution_dtype)
            f.create_dataset("temporal_evolution", data=evolution_array)
        
        # Save patches at different scales
        for patch_key, patch_data in patches_data.items():
            patch_size = patch_data['patch_size']
            patch_file = self.root_dir / 'patches' / f"{exposure_id}_patches_{patch_size}.h5"
            
            with h5py.File(patch_file, 'w') as f:
                f.create_dataset("patches", data=patch_data["patches"], 
                               compression='gzip', compression_opts=9)
                f.create_dataset("positions", data=patch_data["positions"])
                f.create_dataset("frame_indices", data=patch_data["frame_indices"])
                f.create_dataset("anomaly_scores", data=patch_data["anomaly_scores"])
                f.attrs["patch_size"] = patch_data["patch_size"]
                f.attrs["overlap"] = patch_data["overlap"]
                f.attrs["stride"] = patch_data["stride"]
                f.attrs["grid_shape"] = patch_data["grid_shape"]
        
        # Save metadata
        metadata = {
            "exposure_id": exposure_id,
            "processing_time": time.time(),
            "patch_sizes": list(patches_data.keys()),
            "detector_info": detector_info,
            "total_frames": diff_data["total_frames"]
        }
        
        metadata_file = self.root_dir / 'metadata' / f"{exposure_id}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _process_single_exposure(self, exposure_id: str, diff_data: Dict, 
                               dataset_type: str, detector_info: Dict):
        """
        Complete preprocessing pipeline for a single exposure
        """
        try:
            self.logger.info(f"Processing exposure {exposure_id}...")
            
            # Step 1: Temporal analysis
            temporal_data = self._analyze_temporal_patterns(diff_data["differences"])
            
            # Step 2: Extract patches at multiple scales
            patches_data = {}
            for patch_size in self.patch_sizes:
                patches_data[f"patches_{patch_size}"] = self._extract_patches(
                    diff_data["differences"], patch_size
                )
            
            # Step 3: Save everything
            self._save_processed_exposure(exposure_id, diff_data, temporal_data, 
                                        patches_data, detector_info)
            
            # Step 4: Update registry
            self.registry["processed_exposures"][exposure_id] = {
                "exposure_hash": detector_info.get("exposure_hash", ""),
                "processed_time": time.time(),
                "dataset_type": dataset_type,
                "detector_info": detector_info
            }
            self.save_registry()
            
            self.logger.info(f"Successfully processed exposure {exposure_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing exposure {exposure_id}: {e}")
            return False

    def _preprocess(self, dataset_name):
        """
        Main preprocessing dispatcher
        """
        dataset_name = dataset_name.upper()
        
        if dataset_name == 'EUCLID':
            return self._euclid_preprocess()
        elif dataset_name == 'CASE':
            return self._case_preprocess()
        else:
            raise ValueError(f'Invalid dataset name of {dataset_name}, must be EUCLID or CASE')

    def _euclid_preprocess(self):
        """
        Preprocessing for Euclid testing data
        """ 
        processed_exposures = []
        
        # Process all Euclid directories
        for dir_name in EUCLID_DIRS:
            path = f'{DATA_ROOT_DIR}/{dir_name}'
            filenames = self._grab_filenames(path)
            
            if filenames is None:
                continue
                
            for filename in filenames:
                if filename.endswith('.fits'):
                    curr_path = f'{path}/{filename}'
                    exposure_id = f"euclid_{dir_name}_{Path(filename).stem}"
                    
                    # Check if already cached
                    if self.is_exposure_cached(exposure_id, [curr_path]):
                        self.logger.info(f"Exposure {exposure_id} already cached, skipping...")
                        continue
                    
                    # Calculate frame differences
                    diff_data = self._compute_difference_fits(curr_path)
                    
                    # Detector info
                    detector_info = {
                        "dataset_type": "EUCLID",
                        "directory": dir_name,
                        "filename": filename,
                        "file_path": curr_path,
                        "detector_id": "single",
                        "exposure_hash": self.get_exposure_hash([curr_path])
                    }
                    
                    # Process the exposure
                    if self._process_single_exposure(exposure_id, diff_data, "EUCLID", detector_info):
                        processed_exposures.append(exposure_id)
        
        self.logger.info(f"Processed {len(processed_exposures)} Euclid exposures")
        return processed_exposures

    def _case_preprocess(self):
        """
        Preprocessing for CASE test data with proper 450-frame grouping
        """
        processed_exposures = []
        
        # Process all CASE directories
        for dir_name in CASE_DIRS:
            # Navigate to nested directory structure
            path = f'{DATA_ROOT_DIR}/{dir_name}'
            nested_dirs = os.listdir(path)
            if not nested_dirs:
                continue
                
            path = f'{path}/{nested_dirs[0]}'
            filenames = self._grab_filenames(path)
            
            if filenames is None:
                continue

            # Sort filenames in ascending order
            sorted_filenames = sorted(filenames, key=lambda x: int(re.search(r'N(\d+)', x).group(1)))
            
            self.logger.info(f"Found {len(sorted_filenames)} files in {dir_name}")

            # Group every 450 files together (1-450, 451-900, etc)
            for exposure_idx, i in enumerate(range(0, len(sorted_filenames), TOTAL_FRAMES)):
                group = sorted_filenames[i : i + TOTAL_FRAMES]

                # Make sure that the group has 450 files in it
                if len(group) != TOTAL_FRAMES:
                    self.logger.warning(f"Incomplete exposure group: {len(group)} files instead of {TOTAL_FRAMES}")
                    continue

                exposure_id = f"case_{dir_name}_exp{exposure_idx:03d}"
                
                # Create file paths for this exposure
                group_paths = [f'{path}/{filename}' for filename in group]
                
                # Check if already cached
                if self.is_exposure_cached(exposure_id, group_paths):
                    self.logger.info(f"Exposure {exposure_id} already cached, skipping...")
                    continue

                detector_1_frames = []
                detector_2_frames = []
                
                # Process each frame in the exposure
                for curr_file in tqdm(group, desc=f"Loading {exposure_id}"):
                    curr_path = f'{path}/{curr_file}'

                    # Read TIF file
                    tif_data = tifffile.imread(curr_path)

                    # Split the data in half for two detectors
                    split_index = tif_data.shape[1] // 2
                    detector_1 = tif_data[:, :split_index]
                    detector_2 = tif_data[:, split_index:]

                    # Remove the extra 8 columns from each side
                    detector_1 = detector_1[:, 8:-8]
                    detector_2 = detector_2[:, 8:-8]

                    # Verify correct size
                    if detector_1.shape != IMG_SIZE or detector_2.shape != IMG_SIZE:
                        self.logger.error(f"Invalid detector shapes: {detector_1.shape}, {detector_2.shape}")
                        continue
                    
                    detector_1_frames.append(detector_1)
                    detector_2_frames.append(detector_2)

                # Convert to numpy arrays
                detector_1_frames = np.array(detector_1_frames)
                detector_2_frames = np.array(detector_2_frames)
                
                # Verify final shapes
                if detector_1_frames.shape != DETECTOR_SIZE or detector_2_frames.shape != DETECTOR_SIZE:
                    self.logger.error(f"Invalid detector array shapes: {detector_1_frames.shape}, {detector_2_frames.shape}")
                    continue

                # Process both detectors
                for detector_idx, detector_frames in enumerate([detector_1_frames, detector_2_frames], 1):
                    detector_exposure_id = f"{exposure_id}_det{detector_idx}"
                    
                    # Calculate frame differences
                    diff_data = self._compute_difference_tif(detector_frames)
                    
                    # Detector info
                    detector_info = {
                        "dataset_type": "CASE",
                        "directory": dir_name,
                        "exposure_index": exposure_idx,
                        "detector_id": f"detector_{detector_idx}",
                        "total_files": len(group),
                        "file_range": f"{group[0]} to {group[-1]}",
                        "exposure_hash": self.get_exposure_hash(group_paths)
                    }
                    
                    # Process the detector exposure
                    if self._process_single_exposure(detector_exposure_id, diff_data, "CASE", detector_info):
                        processed_exposures.append(detector_exposure_id)

                # Break after first exposure for testing
                # Remove this break to process all exposures
                break
        
        self.logger.info(f"Processed {len(processed_exposures)} CASE detector exposures")
        return processed_exposures

    def load_training_dataset(self, dataset_type: str = None, 
                            patch_size: int = 512,
                            min_anomaly_score: float = 0.0,
                            detector_id: str = None) -> Dict:
        """
        Load processed data for ViT-VAE training
        
        Args:
            dataset_type: 'EUCLID', 'CASE', or None for both
            patch_size: Size of patches to load
            min_anomaly_score: Minimum anomaly score threshold
            detector_id: For CASE data, specify detector ('detector_1', 'detector_2')
        """
        all_patches = []
        all_positions = []
        all_frame_indices = []
        all_anomaly_scores = []
        all_exposure_ids = []
        
        # Filter exposures by dataset type and detector
        exposures_to_load = []
        for exposure_id, info in self.registry["processed_exposures"].items():
            if dataset_type is None or info.get("dataset_type") == dataset_type:
                if detector_id is None or info["detector_info"].get("detector_id") == detector_id:
                    exposures_to_load.append(exposure_id)
        
        self.logger.info(f"Loading {len(exposures_to_load)} exposures for training...")
        
        for exposure_id in tqdm(exposures_to_load, desc="Loading training data"):
            patch_file = self.root_dir / 'patches' / f"{exposure_id}_patches_{patch_size}.h5"
            
            if patch_file.exists():
                with h5py.File(patch_file, 'r') as f:
                    patches = f["patches"][:]
                    positions = f["positions"][:]
                    frame_indices = f["frame_indices"][:]
                    anomaly_scores = f["anomaly_scores"][:]
                    
                    # Filter by anomaly score
                    mask = anomaly_scores >= min_anomaly_score
                    
                    all_patches.append(patches[mask])
                    all_positions.append(positions[mask])
                    all_frame_indices.append(frame_indices[mask])
                    all_anomaly_scores.append(anomaly_scores[mask])
                    all_exposure_ids.extend([exposure_id] * np.sum(mask))
        
        # Combine all data
        combined_data = {
            "patches": np.concatenate(all_patches) if all_patches else np.array([]),
            "positions": np.concatenate(all_positions) if all_positions else np.array([]),
            "frame_indices": np.concatenate(all_frame_indices) if all_frame_indices else np.array([]),
            "anomaly_scores": np.concatenate(all_anomaly_scores) if all_anomaly_scores else np.array([]),
            "exposure_ids": all_exposure_ids,
            "patch_size": patch_size,
            "dataset_type": dataset_type,
            "detector_id": detector_id,
            "total_patches": len(all_exposure_ids)
        }
        
        self.logger.info(f"Loaded {combined_data['total_patches']} patches for training")
        return combined_data

    def get_statistics(self):
        """
        Get statistics about processed data
        """
        stats = {
            "total_exposures": len(self.registry["processed_exposures"]),
            "euclid_exposures": 0,
            "case_exposures": 0,
            "case_detector_1": 0,
            "case_detector_2": 0,
            "patch_sizes_available": set(),
            "processing_times": []
        }
        
        for exposure_id, info in self.registry["processed_exposures"].items():
            dataset_type = info.get("dataset_type", "unknown")
            detector_id = info["detector_info"].get("detector_id", "unknown")
            
            if dataset_type == "EUCLID":
                stats["euclid_exposures"] += 1
            elif dataset_type == "CASE":
                stats["case_exposures"] += 1
                if detector_id == "detector_1":
                    stats["case_detector_1"] += 1
                elif detector_id == "detector_2":
                    stats["case_detector_2"] += 1
            
            stats["processing_times"].append(info.get("processed_time", 0))
            
            # Check available patch sizes
            for patch_file in (self.root_dir / 'patches').glob(f"{exposure_id}_patches_*.h5"):
                size = int(patch_file.stem.split('_')[-1])
                stats["patch_sizes_available"].add(size)
        
        stats["patch_sizes_available"] = sorted(list(stats["patch_sizes_available"]))
        return stats


# Usage Example
if __name__ == "__main__":
    # Initialize training data processor
    trainer = TrainingData(root_dir='training_set')
    
    # Process Euclid data
    print("Processing Euclid data...")
    euclid_exposures = trainer._preprocess('EUCLID')
    
    # Process CASE data (properly grouped by 450 frames)
    print("Processing CASE data...")
    case_exposures = trainer._preprocess('CASE')
    
    # Load training dataset examples
    print("\nLoading training datasets...")
    
    # Load Euclid data
    euclid_data = trainer.load_training_dataset(
        dataset_type='EUCLID',
        patch_size=512,
        min_anomaly_score=1.0
    )
    
    # Load CASE detector 1 data
    case_det1_data = trainer.load_training_dataset(
        dataset_type='CASE',
        detector_id='detector_1',
        patch_size=512,
        min_anomaly_score=1.0
    )
    
    # Load CASE detector 2 data
    case_det2_data = trainer.load_training_dataset(
        dataset_type='CASE',
        detector_id='detector_2',
        patch_size=512,
        min_anomaly_score=1.0
    )
    
    print(f"Euclid training data: {euclid_data['total_patches']} patches")
    print(f"CASE detector 1 training data: {case_det1_data['total_patches']} patches")
    print(f"CASE detector 2 training data: {case_det2_data['total_patches']} patches")
    
    # Get statistics
    stats = trainer.get_statistics()
    print(f"\nDataset statistics: {stats}")