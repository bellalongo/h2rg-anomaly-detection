import numpy as np
from typing import Dict, List
from tqdm import tqdm
import logging


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
    
    def extract_patches(self, diff_stack: np.ndarray, patch_size: int) -> Dict:
        """
            * extract overlapping patches from difference images
        """
        # Initialize detector data
        num_frames, height, width = diff_stack.shape
        overlap = int(patch_size * self.overlap_ratio)
        stride = patch_size - overlap
        
        # Initialize patch info
        patches = []
        positions = []
        frame_indices = []
        anomaly_scores = []
        
        # Calculate patch grid
        num_patches_height = (height - patch_size) // stride + 1
        num_patches_heigth = (width - patch_size) // stride + 1
        
        self.logger.info(
            f'Extracting {num_patches_height}x{num_patches_heigth} patches of {patch_size}x{patch_size}'
            )
        
        # Iterate through every frame
        for frame_idx in tqdm(range(num_frames), desc=f'Extracting {patch_size}x{patch_size} patches'):
            diff_frame = diff_stack[frame_idx]
            
            # Iterate through each pixel in that current patch
            for i in range(0, height - patch_size + 1, stride):
                for j in range(0, width - patch_size + 1, stride):
                    patch = diff_frame[i:i+patch_size, j:j+patch_size]

                    # Calculate the anomally score (just the max intensity)
                    anomaly_score = np.max(patch)
                    
                    patches.append(patch)
                    positions.append((i, j))
                    frame_indices.append(frame_idx)
                    anomaly_scores.append(anomaly_score)
        
        return {
            'patches': np.array(patches),
            'positions': np.array(positions),
            'frame_indices': np.array(frame_indices),
            'anomaly_scores': np.array(anomaly_scores),
            'patch_size': patch_size,
            'overlap': overlap,
            'stride': stride,
            'grid_shape': (num_patches_height, num_patches_heigth)
        }