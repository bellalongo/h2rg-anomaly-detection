import numpy as np
from typing import Dict, List
from tqdm import tqdm
import logging
import time
from numba import jit, prange
from concurrent.futures import ThreadPoolExecutor


class PatchExtractor:
    """
        * extracts overlapping patches from difference images for ViT-VAE training
    """
    def __init__(self, patch_sizes: List[int] = [512, 256, 128]):
        """

        """
        self.patch_sizes = patch_sizes
        # self.overlap_ratio = overlap_ratio
        self.logger = logging.getLogger(__name__)

    def extract_patches(self, diff_stack: np.ndarray, patch_size: int) -> Dict:
        """
        Memory-efficient patch extraction - processes patches in chunks
        """
        start_time = time.time()
        
        # Initialize detector data
        num_frames, height, width = diff_stack.shape
        # overlap = int(patch_size * self.overlap_ratio)
        stride = patch_size
        
        # Calculate patch grid
        num_patches_height = (height - patch_size) // stride + 1
        num_patches_width = (width - patch_size) // stride + 1
        total_patches_per_frame = num_patches_height * num_patches_width
        total_patches = num_frames * total_patches_per_frame
        
        self.logger.info(
            f'Extracting {num_patches_height}x{num_patches_width} patches of {patch_size}x{patch_size} '
            f'from {num_frames} frames (total: {total_patches} patches)'
        )
        
        # Memory check - if too many patches, use chunked processing
        memory_per_patch = patch_size * patch_size * 4  # 4 bytes per float32
        total_memory_mb = (total_patches * memory_per_patch) / (1024 * 1024)
        
        if total_memory_mb > 100:  # If more than 100MB, use chunked processing
            self.logger.warning(f"Large memory requirement ({total_memory_mb:.1f}MB), using chunked processing")
            return self._extract_patches_chunked(diff_stack, patch_size, stride, 
                                               num_patches_height, num_patches_width)
        else:
            return self._extract_patches_optimized(diff_stack, patch_size, stride,
                                                 num_patches_height, num_patches_width)
    
    def _extract_patches_optimized(self, diff_stack: np.ndarray, patch_size: int, stride: int,
                                 num_patches_height: int, num_patches_width: int) -> Dict:
        """
        Optimized patch extraction for smaller datasets
        """
        num_frames = diff_stack.shape[0]
        total_patches = num_frames * num_patches_height * num_patches_width
        
        # Pre-allocate arrays
        patches = np.empty((total_patches, patch_size, patch_size), dtype=np.float32)
        positions = np.empty((total_patches, 2), dtype=np.int32)
        frame_indices = np.empty(total_patches, dtype=np.int32)
        anomaly_scores = np.empty(total_patches, dtype=np.float32)
        
        patch_idx = 0
        for frame_idx in range(num_frames):
            diff_frame = diff_stack[frame_idx]
            
            for i in range(0, diff_frame.shape[0] - patch_size + 1, stride):
                for j in range(0, diff_frame.shape[1] - patch_size + 1, stride):
                    # Extract patch
                    patch = diff_frame[i:i+patch_size, j:j+patch_size]
                    
                    # Store data
                    patches[patch_idx] = patch
                    positions[patch_idx] = [i, j]
                    frame_indices[patch_idx] = frame_idx
                    anomaly_scores[patch_idx] = np.max(patch)
                    
                    patch_idx += 1
        
        return {
            'patches': patches,
            'positions': positions,
            'frame_indices': frame_indices,
            'anomaly_scores': anomaly_scores,
            'patch_size': patch_size,
            # 'overlap': int(patch_size * self.overlap_ratio),
            'stride': stride,
            'grid_shape': (num_patches_height, num_patches_width)
        }
    
    def _extract_patches_chunked(self, diff_stack: np.ndarray, patch_size: int, stride: int,
                               num_patches_height: int, num_patches_width: int) -> Dict:
        """
        Chunked patch extraction for large datasets to avoid memory issues
        """
        num_frames = diff_stack.shape[0]
        total_patches = num_frames * num_patches_height * num_patches_width
        
        self.logger.info(f"Using chunked processing for {total_patches} patches")
        
        # Process in smaller chunks
        chunk_size = 1000  # Process 1000 patches at a time
        all_patches = []
        all_positions = []
        all_frame_indices = []
        all_anomaly_scores = []
        
        patch_idx = 0
        for frame_idx in tqdm(range(num_frames), desc=f'Extracting {patch_size}x{patch_size} patches'):
            diff_frame = diff_stack[frame_idx]
            
            frame_patches = []
            frame_positions = []
            frame_indices = []
            frame_scores = []
            
            for i in range(0, diff_frame.shape[0] - patch_size + 1, stride):
                for j in range(0, diff_frame.shape[1] - patch_size + 1, stride):
                    # Extract patch
                    patch = diff_frame[i:i+patch_size, j:j+patch_size].copy()
                    
                    # Store data
                    frame_patches.append(patch)
                    frame_positions.append([i, j])
                    frame_indices.append(frame_idx)
                    frame_scores.append(np.max(patch))
                    
                    patch_idx += 1
                    
                    # If chunk is full, add to main arrays
                    if len(frame_patches) >= chunk_size:
                        all_patches.extend(frame_patches)
                        all_positions.extend(frame_positions)
                        all_frame_indices.extend(frame_indices)
                        all_anomaly_scores.extend(frame_scores)
                        
                        # Clear temporary arrays
                        frame_patches.clear()
                        frame_positions.clear()
                        frame_indices.clear()
                        frame_scores.clear()
            
            # Add remaining patches from this frame
            if frame_patches:
                all_patches.extend(frame_patches)
                all_positions.extend(frame_positions)
                all_frame_indices.extend(frame_indices)
                all_anomaly_scores.extend(frame_scores)
        
        # Convert to numpy arrays
        patches = np.array(all_patches, dtype=np.float32)
        positions = np.array(all_positions, dtype=np.int32)
        frame_indices = np.array(all_frame_indices, dtype=np.int32)
        anomaly_scores = np.array(all_anomaly_scores, dtype=np.float32)
        
        return {
            'patches': patches,
            'positions': positions,
            'frame_indices': frame_indices,
            'anomaly_scores': anomaly_scores,
            'patch_size': patch_size,
            # 'overlap': int(patch_size * self.overlap_ratio),
            'stride': stride,
            'grid_shape': (num_patches_height, num_patches_width)
        }
    
    def extract_all_patches_parallel(self, diff_stack: np.ndarray) -> Dict[str, Dict]:
        """
        Extract ALL patch sizes in parallel - memory-efficient version
        """
        self.logger.info(f"Extracting patches at {len(self.patch_sizes)} scales in parallel")
        patches_data = {}
        
        # Process all patch sizes in parallel using threads
        with ThreadPoolExecutor(max_workers=min(len(self.patch_sizes), 4)) as executor:
            # Submit all patch extraction tasks
            future_to_size = {
                executor.submit(self.extract_patches, diff_stack, size): size 
                for size in self.patch_sizes
            }
            
            # Collect results as they complete
            for future in future_to_size:
                patch_size = future_to_size[future]
                try:
                    result = future.result()
                    patches_data[f'patches_{patch_size}'] = result
                    self.logger.info(f"Completed {patch_size}x{patch_size}: {result['patches'].shape[0]} patches")
                except Exception as e:
                    self.logger.error(f"Failed to extract {patch_size}x{patch_size} patches: {e}")
        
        # Log summary
        total_patches = sum(data['patches'].shape[0] for data in patches_data.values())
        self.logger.info(f"Total patches extracted: {total_patches}")
        
        return patches_data


    
    # def extract_patches(self, diff_stack: np.ndarray, patch_size: int) -> Dict:
    #     """
    #         * extract overlapping patches from difference images
    #     """
    #     # Initialize detector data
    #     num_frames, height, width = diff_stack.shape
    #     overlap = int(patch_size * self.overlap_ratio)
    #     stride = patch_size - overlap
        
    #     # Initialize patch info
    #     patches = []
    #     positions = []
    #     frame_indices = []
    #     anomaly_scores = []
        
    #     # Calculate patch grid
    #     num_patches_height = (height - patch_size) // stride + 1
    #     num_patches_width = (width - patch_size) // stride + 1
        
    #     self.logger.info(
    #         f'Extracting {num_patches_height}x{num_patches_width} patches of {patch_size}x{patch_size}'
    #         )
        
    #     # Iterate through every frame
    #     for frame_idx in tqdm(range(num_frames), desc=f'Extracting {patch_size}x{patch_size} patches'):
    #         diff_frame = diff_stack[frame_idx]
            
    #         # Iterate through each pixel in that current patch
    #         for i in range(0, height - patch_size + 1, stride):
    #             for j in range(0, width - patch_size + 1, stride):
    #                 patch = diff_frame[i:i+patch_size, j:j+patch_size]

    #                 # Calculate the anomally score (just the max intensity)
    #                 anomaly_score = np.max(patch)
                    
    #                 patches.append(patch)
    #                 positions.append((i, j))
    #                 frame_indices.append(frame_idx)
    #                 anomaly_scores.append(anomaly_score)
        
    #     return {
    #         'patches': np.array(patches),
    #         'positions': np.array(positions),
    #         'frame_indices': np.array(frame_indices),
    #         'anomaly_scores': np.array(anomaly_scores),
    #         'patch_size': patch_size,
    #         'overlap': overlap,
    #         'stride': stride,
    #         'grid_shape': (num_patches_height, num_patches_width)
    #     }