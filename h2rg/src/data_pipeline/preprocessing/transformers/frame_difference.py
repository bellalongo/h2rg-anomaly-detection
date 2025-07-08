import numpy as np
from typing import Dict
from tqdm import tqdm
from astropy.io import fits
import logging
import time
from numba import jit, prange

from ..cleaners.reference_pixel_corrector import ReferencePixelCorrector


class FrameDifferencer:
    """
        * handles frame differencing for both EUCLID and CASE datasets
    """
    def __init__(self, reference_corrector: ReferencePixelCorrector):
        """

        """
        self.reference_corrector = reference_corrector
        self.test_mode = False
        self.test_frames = 10
        self.logger = logging.getLogger(__name__)

    def set_test_mode(self, test_frames: int = 10):
        """
            * enable test mode for limited frame processing
        """
        self.test_mode = True
        self.test_frames = test_frames
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"FrameDifferencer: TEST MODE enabled ({test_frames} frames)")

    @staticmethod
    @jit(nopython=True, parallel=True, cache=True)
    def _numba_frame_differences(frame_stack, reference_frame):
        """
            * numba optimized frame differencer 
              (https://numba.pydata.org/numba-doc/dev/cuda/examples.html)
        """
        n_frames, height, width = frame_stack.shape
        diff_stack = np.empty((n_frames, height, width), dtype=np.float32)
        
        # Parallel processing across frames and spatial dimensions
        for i in prange(n_frames):
            for h in prange(height):
                for w in prange(width):
                    diff_stack[i, h, w] = frame_stack[i, h, w] - reference_frame[h, w]
        
        return diff_stack
    
    def compute_fits_differences(self, file_path: str, total_frames: int = 450) -> Dict:
        """
            * enhanced FITS difference computation with Numba optimization
        """
        self.logger.info(f"Processing {total_frames} frames from {file_path}")
        start_time = time.time()

        # Adjust frame count for test mode
        if self.test_mode:
            max_frames = min(self.test_frames + 1, total_frames)
            self.logger.info(f"TEST MODE: Processing only {max_frames} frames ({self.test_frames} differences)")
        else:
            max_frames = total_frames

        # Pre-allocate for ALL frames (or test frames)
        frame_stack = np.empty((max_frames - 1, 2048, 2048), dtype=np.float32)

        # Batch load all frames
        load_start = time.time()
        with fits.open(file_path) as hdul:
            # Load reference frame
            frame_0 = hdul[1].data.astype(np.float32)
            
            # Batch load all difference frames
            actual_frames = 0
            for i in range(1, max_frames):
                if i + 1 < len(hdul):
                    frame_stack[actual_frames] = hdul[i+1].data.astype(np.float32)
                    actual_frames += 1
                else:
                    # Handle incomplete exposures
                    break
            
            # Resize if we got fewer frames than expected
            if actual_frames < max_frames - 1:
                frame_stack = frame_stack[:actual_frames]

        load_time = time.time() - load_start
        self.logger.debug(f"Frame loading: {load_time:.2f}s")

        # Reference pixel correction
        correction_start = time.time()
        reference_frame = self.reference_corrector.subtract_reference_pixels(frame_0)
        
        # Apply reference correction to all frames
        corrected_stack = np.empty_like(frame_stack)
        for i in range(frame_stack.shape[0]):
            corrected_stack[i] = self.reference_corrector.subtract_reference_pixels(frame_stack[i])
        
        correction_time = time.time() - correction_start
        self.logger.debug(f"Reference correction: {correction_time:.2f}s")

        # Numba-optimized difference computation
        diff_start = time.time()
        diff_stack = self._numba_frame_differences(corrected_stack, reference_frame)
        diff_time = time.time() - diff_start
        self.logger.debug(f"Difference computation: {diff_time:.2f}s")

        total_time = time.time() - start_time
        self.logger.info(f"Total processing time: {total_time:.2f}s for {frame_stack.shape[0]} frames")

        return {
            'differences': diff_stack,
            'frame_times': np.arange(1, len(diff_stack) + 1),
            'reference_frame': reference_frame,
            'total_frames': len(diff_stack),
            'processing_stats': {
                'load_time': load_time,
                'correction_time': correction_time,
                'diff_time': diff_time,
                'total_time': total_time
            }
        }
    
    def compute_tif_differences(self, detector_frames: np.ndarray) -> Dict:
        """
            * enhanced tif difference computation with numba optimization
        """
        self.logger.info(f"Processing {len(detector_frames)} TIF frames")
        start_time = time.time()
        
        detector_frames = np.array(detector_frames)
        
        # Adjust frame count for test mode
        if self.test_mode:
            max_frames = min(self.test_frames + 1, len(detector_frames))
            detector_frames = detector_frames[:max_frames]
            self.logger.info(f"TEST MODE: Processing only {max_frames} frames ({max_frames-1} differences)")
        
        # Reference pixel correction
        correction_start = time.time()
        
        # Get reference frame (frame 0) and apply correction
        frame_0 = detector_frames[0].astype(np.float32)
        reference_frame = self.reference_corrector.subtract_reference_pixels(frame_0)
        
        # Apply correction to all remaining frames
        remaining_frames = detector_frames[1:].astype(np.float32)
        corrected_frames = np.empty_like(remaining_frames)
        
        for i in range(len(remaining_frames)):
            corrected_frames[i] = self.reference_corrector.subtract_reference_pixels(remaining_frames[i])
        
        correction_time = time.time() - correction_start
        self.logger.debug(f"Reference correction: {correction_time:.2f}s")
        
        # Numba-optimized difference computation
        diff_start = time.time()
        diff_stack = self._numba_frame_differences(corrected_frames, reference_frame)
        diff_time = time.time() - diff_start
        self.logger.debug(f"Difference computation: {diff_time:.2f}s")
        
        total_time = time.time() - start_time
        self.logger.info(f"Total TIF processing time: {total_time:.2f}s for {len(diff_stack)} frames")
        
        return {
            'differences': diff_stack,
            'frame_times': np.arange(1, len(diff_stack) + 1),
            'reference_frame': reference_frame,
            'total_frames': len(diff_stack),
            'processing_stats': {
                'correction_time': correction_time,
                'diff_time': diff_time,
                'total_time': total_time
            }
        }
    
    # def compute_fits_differences(self, file_path: str, total_frames: int = 450) -> Dict:
    #     """
    #         * compute frame differences for EUCLID FITS files
    #     """
    #     dif_stack = []
    #     time_stack = []

    #     # Adjust frame count for test mode
    #     if self.test_mode:
    #         max_frames = min(self.test_frames + 1, total_frames)  # +1 for reference frame
    #         self.logger.info(f"TEST MODE: Processing only {max_frames} frames ({self.test_frames} differences)")
    #     else:
    #         max_frames = total_frames
        
    #     # Open current fits file
    #     with fits.open(file_path) as hdul:
    #         # Get reference frame (frame 0) and perform correction
    #         frame_0 = hdul[1].data.astype(np.float64)
    #         frame_0 = self.reference_corrector.subtract_reference_pixels(frame_0)

    #         # Process remaining frames (besides 0th frame)
    #         end_frame = min(max_frames, len(hdul) - 1)
            
    #         # Process remaining frames (besides 0th frame)
    #         for i in tqdm(range(2, end_frame + 1), desc='EUCLID frame differences'):
    #             # Grab current frame and perform the ref px correction
    #             curr_frame = hdul[i].data.astype(np.float64)
    #             curr_frame = self.reference_corrector.subtract_reference_pixels(curr_frame)
                
    #             # Calculate the difference
    #             dif = curr_frame - frame_0
    #             dif_stack.append(dif)
    #             time_stack.append(i - 1)
        
    #     result = {
    #         'differences': np.array(dif_stack),
    #         'frame_times': np.array(time_stack),
    #         'reference_frame': frame_0,
    #         'total_frames': len(dif_stack)
    #     }
    
    #     if self.test_mode:
    #         self.logger.info(f"TEST MODE: Generated {len(dif_stack)} difference frames")

    #     return result
    
    # def compute_tif_differences(self, detector_frames: np.ndarray) -> Dict:
    #     """
    #         * compute frame differences for CASE TIF detector arrays
    #     """
    #     detector_frames = np.array(detector_frames)
    #     dif_stack = []
    #     time_stack = []
        
    #     # Get reference frame (frame 0)
    #     frame_0 = detector_frames[0].astype(np.float64)
    #     frame_0 = self.reference_corrector.subtract_reference_pixels(frame_0)
        
    #     # Process remaining frames (besides the 0th frame)
    #     for i, frame in tqdm(enumerate(detector_frames[1:], start=1), 
    #                        desc='CASE frame differences'):
    #         # Grab the curret frame and perform the 0th ref px correction
    #         curr_frame = frame.astype(np.float64)
    #         curr_frame = self.reference_corrector.subtract_reference_pixels(curr_frame)
            
    #         # Calculate the difference
    #         dif = curr_frame - frame_0
    #         dif_stack.append(dif)
    #         time_stack.append(i)
        
    #     return {
    #         'differences': np.array(dif_stack),
    #         'frame_times': np.array(time_stack),
    #         'reference_frame': frame_0,
    #         'total_frames': len(dif_stack)
    #     }