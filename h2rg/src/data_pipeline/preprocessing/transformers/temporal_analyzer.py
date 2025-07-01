import numpy as np
from typing import Dict
from tqdm import tqdm
import logging


class TemporalAnalyzer:
    """
        * analyzes temporal patterns in difference images for anomaly detection
    """
    def __init__(self, sigma_threshold: float = 3.0, use_robust_stats: bool = True):
        """
            * threshold: how many std above mean to define events
        """
        # Update to be > AVG
        self.sigma_threshold = sigma_threshold
        self.use_robust_stats = use_robust_stats
        self.logger = logging.getLogger(__name__)
    
    def analyze_temporal_patterns(self, diff_stack: np.ndarray) -> Dict:
        """
            * analyze when anomalies first appear and their persistence
        """
        # Calculate background statistics and adaptive threshold
        self.background_stats = self._calculate_background_statistics(diff_stack)
        self.calculated_threshold = self.background_stats['threshold_intensity']
        
        # Initialize detector size
        num_frames, height, width = diff_stack.shape
        
        # Feature 1: When did it first appear?
        first_appearance = np.full((height, width), -1, dtype=np.int32)
        
        # Feature 2: How long does it persist?
        persistence_count = np.zeros((height, width), dtype=np.int32)
        
        # Feature 3: What is the maximum intensity over time?
        max_intensity = np.zeros((height, width), dtype=np.float32)
        
        # Feature 4: Track temporal evolution
        temporal_evolution = []
        
        # Feature 5: Track significance level of each pixel
        max_significance = np.zeros((height, width), dtype=np.float32)
        
        self.logger.info(f"Processing {num_frames} frames with {self.sigma_threshold}σ threshold...")
        
        # Iterate through every frame
        for frame_idx in tqdm(range(num_frames), desc='Temporal analysis'):
            # Grab the current diff frame
            diff_frame = diff_stack[frame_idx]
            
            # Find anomalies above adaptive threshold
            anomaly_mask = diff_frame > self.calculated_threshold
            
            # Calculate significance level for this frame
            if self.use_robust_stats:
                significance_frame = (diff_frame - self.background_stats['center']) / self.background_stats['scale']
            else:
                significance_frame = (diff_frame - self.background_stats['center']) / self.background_stats['scale']
            
            # Update first appearance
            new_anomalies = anomaly_mask & (first_appearance == -1)
            first_appearance[new_anomalies] = frame_idx
            
            # Update the persistence counts
            persistence_count[anomaly_mask] += 1
            
            # Update the max intensity
            max_intensity = np.maximum(max_intensity, diff_frame)
            
            # Update max significance level
            max_significance = np.maximum(max_significance, significance_frame)
            
            # Track frame-level statistics
            if np.any(anomaly_mask):
                mean_anomaly_intensity = np.mean(diff_frame[anomaly_mask])
                mean_anomaly_significance = np.mean(significance_frame[anomaly_mask])
                max_frame_intensity = np.max(diff_frame)
                max_frame_significance = np.max(significance_frame)
            else:
                mean_anomaly_intensity = 0
                mean_anomaly_significance = 0
                max_frame_intensity = np.max(diff_frame)
                max_frame_significance = np.max(significance_frame)
            
            temporal_evolution.append({
                'frame': frame_idx,
                'n_anomalies': np.sum(anomaly_mask),
                'mean_intensity': mean_anomaly_intensity,
                'max_intensity': max_frame_intensity,
                'mean_significance': mean_anomaly_significance,
                'max_significance': max_frame_significance,
                'anomaly_fraction': np.sum(anomaly_mask) / (height * width)
            })

        # Calculate summary statistics
        total_anomaly_pixels = np.sum(first_appearance >= 0)
        total_pixels = height * width

        return {
            'first_appearance': first_appearance,
            'persistence_count': persistence_count,
            'max_intensity': max_intensity,
            'max_significance': max_significance,  
            'temporal_evolution': temporal_evolution,
            'threshold_used': self.calculated_threshold,  
            'sigma_threshold': self.sigma_threshold,      
            'background_stats': self.background_stats,    
            'summary_stats': {
                'total_anomaly_pixels': total_anomaly_pixels,
                'anomaly_fraction': total_anomaly_pixels / total_pixels,
                'max_significance_sigma': np.max(max_significance)
            }
        }
    
    def get_anomaly_significance(self, pixel_intensity: float) -> float:
        """
            * convert pixel intensity to significance level in sigma units
        """
        if self.background_stats is None:
            raise ValueError("Must run analyze_temporal_patterns first")
        
        return (pixel_intensity - self.background_stats['center']) / self.background_stats['scale']
    
    def classify_anomaly_strength(self, pixel_intensity: float) -> str:
        """
            * classify anomaly strength based on significance level
        """
        sigma_level = self.get_anomaly_significance(pixel_intensity)
        
        if sigma_level < 3:
            return "background"
        elif sigma_level < 5:
            return "weak_anomaly"
        elif sigma_level < 10:
            return "moderate_anomaly"
        elif sigma_level < 20:
            return "strong_anomaly"
        else:
            return "extreme_anomaly"

    def _calculate_background_statistics(self, diff_stack: np.ndarray) -> Dict:
        """
            * calculate background statistics from the difference stack
        """
        # Flatten all frames to get population statistics
        all_pixels = diff_stack.flatten()
        
        # Remove extreme outliers to get clean background statistics (anomalies)
        p1, p99 = np.percentile(all_pixels, [1, 99])
        background_pixels = all_pixels[(all_pixels >= p1) & (all_pixels <= p99)]
        
        # Robust statistics (less sensitive to outliers)
        if self.use_robust_stats:
            background_center = np.median(background_pixels)
            mad = np.median(np.abs(background_pixels - background_center))
            background_scale = mad * 1.4826  # Convert MAD to sigma equivalent
            stats_type = "robust (median ± MAD)"
         # Classical statistics
        else:
            background_center = np.mean(background_pixels)
            background_scale = np.std(background_pixels)
            stats_type = "classical (mean ± std)"
        
        # Calculate the actual threshold in intensity units
        calculated_threshold = background_center + self.sigma_threshold * background_scale
        
        stats = {
            'center': background_center,
            'scale': background_scale,
            'threshold_intensity': calculated_threshold,
            'sigma_level': self.sigma_threshold,
            'stats_type': stats_type,
            'n_background_pixels': len(background_pixels),
            'total_pixels': len(all_pixels),
            'background_fraction': len(background_pixels) / len(all_pixels)
        }
        
        self.logger.info(f"Background statistics ({stats_type}):")
        self.logger.info(f"  Center: {background_center:.2f} electrons")
        self.logger.info(f"  Scale: {background_scale:.2f} electrons") 
        self.logger.info(f"  {self.sigma_threshold}σ threshold: {calculated_threshold:.2f} electrons")
        
        return stats