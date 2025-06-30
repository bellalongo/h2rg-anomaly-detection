import numpy as np


class ReferencePixelCorrector:
    """
        * handles reference pixel subtraction for EUCLID SCS detectors
    """
    def __init__(self):
        """

        """
        self.x_opt = 64  # From paper
        self.y_opt = 4   # From paper
    
    def subtract_reference_pixels(self, frame: np.ndarray) -> np.ndarray:
        """
            * apply reference pixel subtraction as per EUCLID specification
        """
        if frame.shape != (2048, 2048):
            raise ValueError(f'Invalid frame shape {frame.shape}, must be (2048, 2048)')
        
        corrected_frame = frame.copy().astype(np.float64)
        
        # Extract reference pixel regions
        up_ref = frame[0:4, :]
        down_ref = frame[2044:2048, :]
        left_ref = frame[:, 0:4]
        right_ref = frame[:, 2044:2048]
        
        # Process each of the 32 channels
        for ch in range(32):
            corrected_frame = self._process_channel(
                corrected_frame, ch, up_ref, down_ref, self.x_opt
                )
        
        # Perform left/right correction
        corrected_frame = self._perform_lr_correction(
            corrected_frame, up_ref, down_ref, left_ref, right_ref, self.y_opt
            )
        
        return corrected_frame 
    
    def _process_channel(self, corrected_frame: np.ndarray, ch: int, 
                         up_ref: np.ndarray, down_ref: np.ndarray, 
                         x_opt: int) -> np.ndarray:
        """
            
        """
        # Skip left ref pixels
        if ch == 0:
            col_start, col_end = 4, 64
        
        # Skip right ref pixels
        elif ch == 31:
            col_start, col_end = ch * 64, 2044  # Fixed: clearer intent

        # Inner channels
        else:
            col_start, col_end = ch * 64, (ch + 1) * 64

        # Up/down correction w/ extract channel specific ref pixels 
        for col in range(col_end - col_start):
            global_col = col_start + col

            # Sliding window for up/down
            window_start = max(0, global_col -  x_opt)
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

        return corrected_frame
        
    def _perform_lr_correction(self, corrected_frame: np.ndarray, up_ref: np.ndarray,
                               down_ref: np.ndarray, left_ref: np.ndarray,
                               right_ref: np.ndarray, y_opt: int) -> np.ndarray:
        """

        """
        # Correct left ref pixel
        left_ref_corrected = left_ref.copy()
        right_ref_corrected = right_ref.copy()

        # Subtract the up/down correction from the left/right pixels
        up_avg_full = np.mean(up_ref)
        down_avg_full = np.mean(down_ref)
        slope_full = (up_avg_full - down_avg_full) / 2044

        # Fixed: Correct the left/right ref pixels with proper loop
        for row in range(4, 2044):
            ref_correction = down_avg_full + (row - 1.5) * slope_full
            left_ref_corrected[row, :] -= ref_correction
            right_ref_corrected[row, :] -= ref_correction

        # Apply the correction using sliding window
        for row in range(4, 2044):
            # Sliding window for left/right
            window_start = max(4, row - y_opt)
            window_end = min(2044, row + y_opt + 1)

            # Fixed: Average corrected left/right pixels (syntax error)
            left_avg = np.mean(left_ref_corrected[window_start:window_end, :])
            right_avg = np.mean(right_ref_corrected[window_start:window_end, :])  # Fixed the dot
            lr_correction = (left_avg + right_avg) / 2

            corrected_frame[row, 4:2044] -= lr_correction

        return corrected_frame