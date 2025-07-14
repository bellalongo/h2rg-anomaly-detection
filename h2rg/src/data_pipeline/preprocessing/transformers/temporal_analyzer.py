import numpy as np
from typing import Dict
from tqdm import tqdm
import logging
from skimage.measure import label, regionprops


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

        # Feature 6: Track the size of the anomally
        anomaly_sizes_per_frame = [] 
        anomaly_groups_per_frame = []
        unique_anomaly_tracker = {}

        
        self.logger.info(f"Processing {num_frames} frames with {self.sigma_threshold}σ threshold...")
        
        # Iterate through every frame
        for frame_idx in tqdm(range(num_frames), desc='Temporal analysis'):
            # Grab the current diff frame
            diff_frame = diff_stack[frame_idx]
            
            # Find anomalies above adaptive threshold
            anomaly_mask = diff_frame > self.calculated_threshold

            # Group adjacent anomalies w/ connected components
            labeled_anomalies, num_anomalies = label(anomaly_mask, return_num=True)
            frame_anomaly_groups = []

            # Analyze each connected component
            if num_anomalies > 0:
                regions = regionprops(labeled_anomalies)

                # Iterate through each region
                for region in regions:
                    anomaly_group = {
                        'area': region.area,                            # Size of the grouped anomaly
                        'centroid': region.centroid,                    # Center position
                        'bbox': region.bbox,                            # Bounding box
                        'frame': frame_idx,                             # Frame index
                        'eccentricity': region.eccentricity,            # Shape measure
                        'major_axis_length': region.major_axis_length,
                        'minor_axis_length': region.minor_axis_length,
                        'mean_intensity': np.mean(diff_frame[labeled_anomalies == region.label])
                    }
                    frame_anomaly_groups.append(anomaly_group)

                    # Track individual anomaly evolution across frames
                    anomaly_id = self._assign_anomaly_id(region.centroid, frame_idx, unique_anomaly_tracker)
                    if anomaly_id not in unique_anomaly_tracker:
                        unique_anomaly_tracker[anomaly_id] = []
                    
                    unique_anomaly_tracker[anomaly_id].append(anomaly_group)
                
            anomaly_groups_per_frame.append(frame_anomaly_groups)
            
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
                total_anomaly_area = np.sum(anomaly_mask)
                mean_grouped_anomaly_size = np.mean([g['area'] for g in frame_anomaly_groups]) if frame_anomaly_groups else 0
                max_grouped_anomaly_size = np.max([g['area'] for g in frame_anomaly_groups]) if frame_anomaly_groups else 0

                mean_anomaly_intensity = np.mean(diff_frame[anomaly_mask])
                mean_anomaly_significance = np.mean(significance_frame[anomaly_mask])
                max_frame_intensity = np.max(diff_frame)
                max_frame_significance = np.max(significance_frame)
            else:
                total_anomaly_area = 0
                mean_grouped_anomaly_size = 0
                max_grouped_anomaly_size = 0
                mean_anomaly_intensity = 0
                mean_anomaly_significance = 0
                max_frame_intensity = np.max(diff_frame)
                max_frame_significance = np.max(significance_frame)
            
            temporal_evolution.append({
                'frame': frame_idx,
                'n_anomalies': np.sum(anomaly_mask),                    # Total anomalous pixels
                'n_grouped_anomalies': num_anomalies,                   # NEW: Number of grouped anomalies
                'total_anomaly_area': total_anomaly_area,               # NEW: Total area covered
                'mean_grouped_anomaly_size': mean_grouped_anomaly_size, # NEW: Average size of groups
                'max_grouped_anomaly_size': max_grouped_anomaly_size,   # NEW: Largest group size
                'mean_intensity': mean_anomaly_intensity,
                'max_intensity': max_frame_intensity,
                'mean_significance': mean_anomaly_significance,
                'max_significance': max_frame_significance,
                'anomaly_fraction': np.sum(anomaly_mask) / (height * width)
            })

        # Calculate summary statistics
        total_anomaly_pixels = np.sum(first_appearance >= 0)
        total_pixels = height * width

        total_unique_anomalies = len(unique_anomaly_tracker)
        grouped_anomaly_stats = self._calculate_grouped_anomaly_statistics(unique_anomaly_tracker)

        return {
            'first_appearance': first_appearance,
            'persistence_count': persistence_count,
            'max_intensity': max_intensity,
            'max_significance': max_significance,  
            'temporal_evolution': temporal_evolution,
            'threshold_used': self.calculated_threshold,  
            'sigma_threshold': self.sigma_threshold,      
            'background_stats': self.background_stats,
            
            'anomaly_groups_per_frame': anomaly_groups_per_frame,
            'unique_anomaly_tracker': unique_anomaly_tracker,
            'grouped_anomaly_stats': grouped_anomaly_stats,
            
            'summary_stats': {
                'total_anomaly_pixels': total_anomaly_pixels,
                'anomaly_fraction': total_anomaly_pixels / total_pixels,
                'max_significance_sigma': np.max(max_significance),
                'total_unique_grouped_anomalies': total_unique_anomalies,  # NEW
                'grouped_size_distribution': grouped_anomaly_stats         # NEW
            }
        }
    
    def _assign_anomaly_id(self, centroid, frame_idx, anomaly_tracker, max_distance=50.0):
        """
            * assign consistent ID to anomalies based on proximity to previous frames
        """
        if frame_idx == 0 or not anomaly_tracker:
            return len(anomaly_tracker)
        
        # Find closest existing anomaly
        min_distance = float('inf')
        closest_id = -1
        
        for anomaly_id, history in anomaly_tracker.items():
            if history:  # If anomaly has previous detections
                last_centroid = history[-1]['centroid']
                distance = np.sqrt((centroid[0] - last_centroid[0])**2 + 
                                 (centroid[1] - last_centroid[1])**2)
                
                if distance < min_distance and distance < max_distance:
                    min_distance = distance
                    closest_id = anomaly_id
        
        # If no close anomaly found, create new ID
        if closest_id == -1:
            return len(anomaly_tracker)
        
        return closest_id
    
    def _calculate_grouped_anomaly_statistics(self, anomaly_tracker):
        """
            * calculate statistics for grouped anomalies across all frames
        """
        all_areas = []
        all_persistence = []
        
        for anomaly_id, history in anomaly_tracker.items():
            if history:
                areas = [detection['area'] for detection in history]
                all_areas.extend(areas)
                all_persistence.append(len(history))
        
        if not all_areas:
            return {}
        
        return {
            'mean_area': np.mean(all_areas),
            'median_area': np.median(all_areas),
            'std_area': np.std(all_areas),
            'mean_persistence': np.mean(all_persistence),
            'max_persistence': np.max(all_persistence),
            'area_percentiles': {
                '25th': np.percentile(all_areas, 25),
                '75th': np.percentile(all_areas, 75),
                '90th': np.percentile(all_areas, 90),
                '95th': np.percentile(all_areas, 95)
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