import numpy as np
import logging
from numba import jit, prange


class ReferencePixelCorrector:
    """
        * handles reference pixel subtraction for EUCLID SCS detectors
    """
    def __init__(self):
        """

        """
        self.x_opt = 64  # From paper
        self.y_opt = 4   # From paper

        self.logger = logging.getLogger(__name__)
    
    def subtract_reference_pixels(self, frame: np.ndarray) -> np.ndarray:
        """
            * apply reference pixel subtraction as per EUCLID specification
        """
        if frame.shape != (2048, 2048):
            raise ValueError(f'Invalid frame shape {frame.shape}, must be (2048, 2048)')
        
        return ReferencePixelCorrector._subtract_reference_pixels_numba(frame, self.x_opt, self.y_opt)
    
    @staticmethod
    @jit(nopython=True, cache=True)
    def _subtract_reference_pixels_numba(frame, x_opt, y_opt):
        """
        Numba-optimized reference pixel subtraction
        Maintains ALL the correction logic but with maximum speed
        """
        corrected_frame = frame.copy().astype(np.float64)
        
        # Extract reference pixel regions
        up_ref = frame[0:4, :]
        down_ref = frame[2044:2048, :]
        left_ref = frame[:, 0:4]
        right_ref = frame[:, 2044:2048]
        
        # Process each of the 32 channels
        for ch in range(32):
            corrected_frame = _process_channel_numba(
                corrected_frame, ch, up_ref, down_ref, x_opt
            )
        
        # Perform left/right correction
        corrected_frame = _perform_lr_correction_numba(
            corrected_frame, up_ref, down_ref, left_ref, right_ref, y_opt
        )
        
        return corrected_frame

@jit(nopython=True, cache=True)
def _process_channel_numba(corrected_frame, ch, up_ref, down_ref, x_opt):
    """
    Numba-optimized channel processing
    """
    # Define column ranges for each channel
    if ch == 0:
        col_start, col_end = 4, 64
    elif ch == 31:
        col_start, col_end = ch * 64, 2044
    else:
        col_start, col_end = ch * 64, (ch + 1) * 64

    # Up/down correction with sliding window
    for col in range(col_start, min(col_end, 2044)):
        # Sliding window for up/down
        window_start = max(0, col - x_opt)
        window_end = min(2048, col + x_opt + 1)

        # Calculate averages
        up_sum = 0.0
        down_sum = 0.0
        count = 0
        
        for i in range(4):  # 4 rows of reference pixels
            for j in range(window_start, window_end):
                up_sum += up_ref[i, j]
                down_sum += down_ref[i, j]
                count += 1
        
        up_avg = up_sum / count
        down_avg = down_sum / count

        # Interpolate correction
        slope = (up_avg - down_avg) / 2044

        # Apply correction to each row in the column, skipping refs
        for row in range(4, 2044):
            ref_correction = down_avg + (row - 1.5) * slope
            corrected_frame[row, col] -= ref_correction

    return corrected_frame

@jit(nopython=True, cache=True)
def _perform_lr_correction_numba(corrected_frame, up_ref, down_ref, left_ref, right_ref, y_opt):
    """
    Numba-optimized left/right correction
    """
    # Correct left ref pixels
    left_ref_corrected = left_ref.copy()
    right_ref_corrected = right_ref.copy()

    # Calculate full average for up/down correction
    up_sum = 0.0
    down_sum = 0.0
    up_count = 0
    down_count = 0
    
    for i in range(4):
        for j in range(2048):
            up_sum += up_ref[i, j]
            down_sum += down_ref[i, j]
            up_count += 1
            down_count += 1
    
    up_avg_full = up_sum / up_count
    down_avg_full = down_sum / down_count
    slope_full = (up_avg_full - down_avg_full) / 2044

    # Correct the left/right ref pixels
    for row in range(4, 2044):
        ref_correction = down_avg_full + (row - 1.5) * slope_full
        for col in range(4):
            left_ref_corrected[row, col] -= ref_correction
            right_ref_corrected[row, col] -= ref_correction

    # Apply left/right correction using sliding window
    for row in range(4, 2044):
        # Sliding window for left/right
        window_start = max(4, row - y_opt)
        window_end = min(2044, row + y_opt + 1)

        # Calculate averages from corrected reference pixels
        left_sum = 0.0
        right_sum = 0.0
        count = 0
        
        for i in range(window_start, window_end):
            for j in range(4):
                left_sum += left_ref_corrected[i, j]
                right_sum += right_ref_corrected[i, j]
                count += 1
        
        left_avg = left_sum / count
        right_avg = right_sum / count
        lr_correction = (left_avg + right_avg) / 2

        # Apply correction to active pixels
        for col in range(4, 2044):
            corrected_frame[row, col] -= lr_correction

    return corrected_frame

# Batch processing function for multiple frames
@jit(nopython=True, parallel=True, cache=True)
def batch_reference_correction(frame_stack, x_opt=64, y_opt=4):
    """
    Batch reference pixel correction for multiple frames using Numba
    Processes ALL frames with maximum speed
    """
    n_frames, height, width = frame_stack.shape
    corrected_stack = np.empty_like(frame_stack)
    
    # Process each frame in parallel
    for frame_idx in prange(n_frames):
        frame = frame_stack[frame_idx]
        corrected_stack[frame_idx] = ReferencePixelCorrector._subtract_reference_pixels_numba(
            frame, x_opt, y_opt
        )
    
    return corrected_stack


        # corrected_frame = frame.copy().astype(np.float64)
        
        # # Extract reference pixel regions
        # up_ref = frame[0:4, :]
        # down_ref = frame[2044:2048, :]
        # left_ref = frame[:, 0:4]
        # right_ref = frame[:, 2044:2048]
        
        # # Process each of the 32 channels
        # for ch in range(32):
        #     corrected_frame = self._process_channel(
        #         corrected_frame, ch, up_ref, down_ref, self.x_opt
        #         )
        
        # # Perform left/right correction
        # corrected_frame = self._perform_lr_correction(
        #     corrected_frame, up_ref, down_ref, left_ref, right_ref, self.y_opt
        #     )
        
        # return corrected_frame 
    
    # def _process_channel(self, corrected_frame: np.ndarray, ch: int, 
    #                      up_ref: np.ndarray, down_ref: np.ndarray, 
    #                      x_opt: int) -> np.ndarray:
    #     """
            
    #     """
    #     # Skip left ref pixels
    #     if ch == 0:
    #         col_start, col_end = 4, 64
        
    #     # Skip right ref pixels
    #     elif ch == 31:
    #         col_start, col_end = ch * 64, 2044  # Fixed: clearer intent

    #     # Inner channels
    #     else:
    #         col_start, col_end = ch * 64, (ch + 1) * 64

    #     # Up/down correction w/ extract channel specific ref pixels 
    #     for col in range(col_end - col_start):
    #         global_col = col_start + col

    #         # Sliding window for up/down
    #         window_start = max(0, global_col -  x_opt)
    #         window_end = min(2048, global_col + x_opt + 1)

    #         # Average the up and down ref pixels
    #         up_avg = np.mean(up_ref[:, window_start:window_end])
    #         down_avg = np.mean(down_ref[:, window_start:window_end])

    #         # Interpolate correction
    #         slope = (up_avg - down_avg) / 2044

    #         # Apply correction to each row in the column, skipping refs
    #         for row in range(4, 2044):
    #             ref_correction = down_avg + (row - 1.5) * slope
    #             corrected_frame[row, global_col] -= ref_correction

    #     return corrected_frame
        
    # def _perform_lr_correction(self, corrected_frame: np.ndarray, up_ref: np.ndarray,
    #                            down_ref: np.ndarray, left_ref: np.ndarray,
    #                            right_ref: np.ndarray, y_opt: int) -> np.ndarray:
    #     """

    #     """
    #     # Correct left ref pixel
    #     left_ref_corrected = left_ref.copy()
    #     right_ref_corrected = right_ref.copy()

    #     # Subtract the up/down correction from the left/right pixels
    #     up_avg_full = np.mean(up_ref)
    #     down_avg_full = np.mean(down_ref)
    #     slope_full = (up_avg_full - down_avg_full) / 2044

    #     # Fixed: Correct the left/right ref pixels with proper loop
    #     for row in range(4, 2044):
    #         ref_correction = down_avg_full + (row - 1.5) * slope_full
    #         left_ref_corrected[row, :] -= ref_correction
    #         right_ref_corrected[row, :] -= ref_correction

    #     # Apply the correction using sliding window
    #     for row in range(4, 2044):
    #         # Sliding window for left/right
    #         window_start = max(4, row - y_opt)
    #         window_end = min(2044, row + y_opt + 1)

    #         # Fixed: Average corrected left/right pixels (syntax error)
    #         left_avg = np.mean(left_ref_corrected[window_start:window_end, :])
    #         right_avg = np.mean(right_ref_corrected[window_start:window_end, :])
    #         lr_correction = (left_avg + right_avg) / 2

    #         corrected_frame[row, 4:2044] -= lr_correction

    #     return corrected_frame