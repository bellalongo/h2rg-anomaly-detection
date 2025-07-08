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
    def __init__(self, patch_sizes: List[int] = [512, 256, 128], overlap_ratio: float = 0.25):
        """

        """
        self.patch_sizes = patch_sizes
        self.overlap_ratio = overlap_ratio
        self.logger = logging.getLogger(__name__)

    @staticmethod
    @jit(nopython=True, parallel=True, cache=True)
    def _extract_patches_numba(diff_stack, patch_size, stride):
        """
        Numba-optimized patch extraction - processes ALL patches with maximum speed
        """
        num_frames, height, width = diff_stack.shape
        
        # Calculate exact number of patches
        num_patches_h = (height - patch_size) // stride + 1
        num_patches_w = (width - patch_size) // stride + 1
        total_patches = num_frames * num_patches_h * num_patches_w
        
        # Pre-allocate for ALL patches
        patches = np.empty((total_patches, patch_size, patch_size), dtype=np.float32)
        positions = np.empty((total_patches, 2), dtype=np.int32)
        frame_indices = np.empty(total_patches, dtype=np.int32)
        anomaly_scores = np.empty(total_patches, dtype=np.float32)
        
        # Parallel extraction across all patches
        patch_idx = 0
        for frame_idx in range(num_frames):
            for i in range(0, height - patch_size + 1, stride):
                for j in range(0, width - patch_size + 1, stride):
                    # Extract full patch
                    for pi in range(patch_size):
                        for pj in range(patch_size):
                            patches[patch_idx, pi, pj] = diff_stack[frame_idx, i + pi, j + pj]
                    
                    # Calculate anomaly score (max intensity)
                    max_val = patches[patch_idx, 0, 0]
                    for pi in range(patch_size):
                        for pj in range(patch_size):
                            if patches[patch_idx, pi, pj] > max_val:
                                max_val = patches[patch_idx, pi, pj]
                    
                    # Store metadata
                    positions[patch_idx, 0] = i
                    positions[patch_idx, 1] = j
                    frame_indices[patch_idx] = frame_idx
                    anomaly_scores[patch_idx] = max_val
                    
                    patch_idx += 1
        
        return patches, positions, frame_indices, anomaly_scores
    
    def extract_patches(self, diff_stack: np.ndarray, patch_size: int) -> Dict:
        """
        Enhanced patch extraction with Numba optimization
        Extracts ALL patches from difference images
        """
        start_time = time.time()
        
        # Initialize detector data
        num_frames, height, width = diff_stack.shape
        overlap = int(patch_size * self.overlap_ratio)
        stride = patch_size - overlap
        
        # Calculate patch grid
        num_patches_height = (height - patch_size) // stride + 1
        num_patches_width = (width - patch_size) // stride + 1
        
        self.logger.info(
            f'Extracting {num_patches_height}x{num_patches_width} patches of {patch_size}x{patch_size} '
            f'from {num_frames} frames'
        )
        
        # Use Numba for fast extraction
        patches, positions, frame_indices, anomaly_scores = self._extract_patches_numba(
            diff_stack, patch_size, stride
        )
        
        extraction_time = time.time() - start_time
        self.logger.info(f'Extracted {len(patches)} patches of {patch_size}x{patch_size} in {extraction_time:.2f}s')
        
        return {
            'patches': patches,
            'positions': positions,
            'frame_indices': frame_indices,
            'anomaly_scores': anomaly_scores,
            'patch_size': patch_size,
            'overlap': overlap,
            'stride': stride,
            'grid_shape': (num_patches_height, num_patches_width),
            'extraction_time': extraction_time
        }
    
    def extract_all_patches_parallel(self, diff_stack: np.ndarray) -> Dict[str, Dict]:
        """
        Extract ALL patch sizes in parallel - no data loss
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