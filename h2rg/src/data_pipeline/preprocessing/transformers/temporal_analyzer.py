import numpy as np
from typing import Dict
from tqdm import tqdm
import logging
from scipy.ndimage import label as scipy_label, binary_dilation
from typing import Dict, List


class TemporalAnalyzer:
    """
        * analyzes temporal patterns in difference images for anomaly detection
    """
    def __init__(self, sigma_threshold: float = 3.0, use_robust_stats: bool = True,
                 dilation_size = 1.0):
        """
            * threshold: how many std above mean to define events
        """
        # Update to be > AVG
        self.sigma_threshold = sigma_threshold
        self.use_robust_stats = use_robust_stats
        self.dilation_size = dilation_size

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

        # Feature 6: Grouping features
        anomaly_groups_per_frame = []
        unique_anomaly_tracker = {}
        
        self.logger.info(f"Processing {num_frames} frames with {self.sigma_threshold}σ threshold...")
        
        # Iterate through every frame
        for frame_idx in tqdm(range(num_frames), desc='Temporal analysis'):
            # Grab the current diff frame
            diff_frame = diff_stack[frame_idx]
            
            # Find anomalies above adaptive threshold
            anomaly_mask = diff_frame > self.calculated_threshold

            # Groups anomalies per frame
            frame_anomaly_groups = self._efficient_morphological_grouping(
                diff_frame, anomaly_mask, frame_idx, unique_anomaly_tracker
            )
            anomaly_groups_per_frame.append(frame_anomaly_groups)
            num_grouped_anomalies = len(frame_anomaly_groups)
            
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
                # Check if there are anomaly groups
                if frame_anomaly_groups:
                    mean_group_size = np.mean([g['area'] for g in frame_anomaly_groups])
                    max_group_size = np.max([g['area'] for g in frame_anomaly_groups])
                else:
                    mean_group_size = 0
                    max_group_size = 0

                mean_anomaly_intensity = np.mean(diff_frame[anomaly_mask])
                mean_anomaly_significance = np.mean(significance_frame[anomaly_mask])
                max_frame_intensity = np.max(diff_frame)
                max_frame_significance = np.max(significance_frame)
            else:
                mean_group_size = 0
                max_group_size = 0
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
                'anomaly_fraction': np.sum(anomaly_mask) / (height * width),

                'n_grouped_anomalies': num_grouped_anomalies,
                'mean_group_size': mean_group_size,
                'max_group_size': max_group_size
            })

        # Calculate summary statistics
        total_anomaly_pixels = np.sum(first_appearance >= 0)
        total_pixels = height * width

        grouped_stats = self._calculate_grouped_statistics(unique_anomaly_tracker)

        return {
            'first_appearance': first_appearance,
            'persistence_count': persistence_count,
            'max_intensity': max_intensity,
            'max_significance': max_significance,  
            'temporal_evolution': temporal_evolution,
            'threshold_used': self.calculated_threshold,  
            'sigma_threshold': self.sigma_threshold,      
            'background_stats': self.background_stats,
            
            # Move these to main level (not in summary_stats)
            'anomaly_groups_per_frame': anomaly_groups_per_frame,
            'unique_anomaly_tracker': unique_anomaly_tracker,
            'grouped_anomaly_stats': grouped_stats,
            
            'summary_stats': {
                'total_anomaly_pixels': total_anomaly_pixels,
                'anomaly_fraction': total_anomaly_pixels / total_pixels,
                'max_significance_sigma': np.max(max_significance),
                'total_unique_grouped_anomalies': len(unique_anomaly_tracker)  # ADD THIS
            }
        }
    
    def _efficient_morphological_grouping(self, diff_frame: np.ndarray, anomaly_mask: np.ndarray, 
                                        frame_idx: int, unique_anomaly_tracker: Dict) -> List[Dict]:
        """
        NEW: Efficient morphological grouping using scipy operations
        Much faster than skimage.regionprops while maintaining accuracy
        """
        if not np.any(anomaly_mask):
            return []
        
        # Step 1: Optional dilation to connect nearby anomalies
        if self.dilation_size > 0:
            structure = np.ones((2*self.dilation_size + 1, 2*self.dilation_size + 1))
            dilated_mask = binary_dilation(anomaly_mask, structure=structure)
        else:
            dilated_mask = anomaly_mask
        
        # Step 2: Fast connected components using scipy
        labeled_regions, num_regions = scipy_label(dilated_mask)
        
        if num_regions == 0:
            return []
        
        # Step 3: Extract properties efficiently
        frame_anomaly_groups = []
        
        for region_id in range(1, num_regions + 1):
            region_mask = labeled_regions == region_id
            
            # Only include regions that contain original anomalous pixels
            original_pixels_in_region = np.sum(anomaly_mask & region_mask)
            if original_pixels_in_region == 0:
                continue
            
            # Fast coordinate extraction
            coords = np.where(region_mask)
            original_coords = np.where(anomaly_mask & region_mask)
            
            # Calculate properties efficiently
            centroid_y = np.mean(original_coords[0])
            centroid_x = np.mean(original_coords[1])
            
            min_y, max_y = np.min(coords[0]), np.max(coords[0])
            min_x, max_x = np.min(coords[1]), np.max(coords[1])
            
            # Mean intensity of original anomalous pixels only
            mean_intensity = np.mean(diff_frame[original_coords])
            
            # Create group info
            anomaly_group = {
                'area': original_pixels_in_region,
                'centroid': (centroid_y, centroid_x),
                'bbox': (min_y, min_x, max_y + 1, max_x + 1),
                'frame': frame_idx,
                'width': max_x - min_x + 1,
                'height': max_y - min_y + 1,
                'aspect_ratio': (max_y - min_y + 1) / (max_x - min_x + 1),
                'mean_intensity': mean_intensity
            }
            
            frame_anomaly_groups.append(anomaly_group)
            
            # Track across frames for persistence analysis
            anomaly_id = self._assign_anomaly_id(
                (centroid_y, centroid_x), frame_idx, unique_anomaly_tracker
            )
            if anomaly_id not in unique_anomaly_tracker:
                unique_anomaly_tracker[anomaly_id] = []
            
            unique_anomaly_tracker[anomaly_id].append(anomaly_group)
        
        return frame_anomaly_groups
    
    def _assign_anomaly_id(self, centroid, frame_idx, anomaly_tracker, max_distance=50.0):
        """
        Assign consistent ID to anomalies based on proximity across frames
        Optimized for speed with limited lookback
        """
        if frame_idx == 0 or not anomaly_tracker:
            return len(anomaly_tracker)
        
        # Only check recent frames for performance
        recent_frame_threshold = max(0, frame_idx - 3)
        
        min_distance_sq = float('inf')
        closest_id = -1
        max_distance_sq = max_distance ** 2
        
        for anomaly_id, history in anomaly_tracker.items():
            if history and history[-1]['frame'] >= recent_frame_threshold:
                last_centroid = history[-1]['centroid']
                distance_sq = ((centroid[0] - last_centroid[0])**2 + 
                              (centroid[1] - last_centroid[1])**2)
                
                if distance_sq < min_distance_sq and distance_sq < max_distance_sq:
                    min_distance_sq = distance_sq
                    closest_id = anomaly_id
        
        return closest_id if closest_id != -1 else len(anomaly_tracker)
    
    def _calculate_grouped_statistics(self, anomaly_tracker: Dict) -> Dict:
        """
        Calculate statistics for grouped anomalies across all frames
        """
        if not anomaly_tracker:
            return {}
        
        all_areas = []
        all_persistence = []
        all_aspect_ratios = []
        
        for anomaly_id, history in anomaly_tracker.items():
            if history:
                areas = [detection['area'] for detection in history]
                aspect_ratios = [detection.get('aspect_ratio', 1.0) for detection in history]
                
                all_areas.extend(areas)
                all_aspect_ratios.extend(aspect_ratios)
                all_persistence.append(len(history))
        
        return {
            'mean_area': np.mean(all_areas),
            'median_area': np.median(all_areas),
            'std_area': np.std(all_areas),
            'mean_aspect_ratio': np.mean(all_aspect_ratios),
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