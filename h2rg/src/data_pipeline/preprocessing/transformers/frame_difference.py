import numpy as np
from typing import Dict
from tqdm import tqdm
from astropy.io import fits
import logging

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

    def set_test_mode(self, test_frames: int = 10):
        """
            * enable test mode for limited frame processing
        """
        self.test_mode = True
        self.test_frames = test_frames
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"FrameDifferencer: TEST MODE enabled ({test_frames} frames)")
    
    def compute_fits_differences(self, file_path: str, total_frames: int = 450) -> Dict:
        """
            * compute frame differences for EUCLID FITS files
        """
        dif_stack = []
        time_stack = []

        # Adjust frame count for test mode
        if self.test_mode:
            max_frames = min(self.test_frames + 1, total_frames)  # +1 for reference frame
            self.logger.info(f"TEST MODE: Processing only {max_frames} frames ({self.test_frames} differences)")
        else:
            max_frames = total_frames
        
        # Open current fits file
        with fits.open(file_path) as hdul:
            # Get reference frame (frame 0) and perform correction
            frame_0 = hdul[1].data.astype(np.float64)
            frame_0 = self.reference_corrector.subtract_reference_pixels(frame_0)

            # Process remaining frames (besides 0th frame)
            end_frame = min(max_frames, len(hdul) - 1)
            
            # Process remaining frames (besides 0th frame)
            for i in tqdm(range(2, end_frame + 1), desc='EUCLID frame differences'):
                # Grab current frame and perform the ref px correction
                curr_frame = hdul[i].data.astype(np.float64)
                curr_frame = self.reference_corrector.subtract_reference_pixels(curr_frame)
                
                # Calculate the difference
                dif = curr_frame - frame_0
                dif_stack.append(dif)
                time_stack.append(i - 1)
        
        result = {
            'differences': np.array(dif_stack),
            'frame_times': np.array(time_stack),
            'reference_frame': frame_0,
            'total_frames': len(dif_stack)
        }
    
        if self.test_mode:
            self.logger.info(f"TEST MODE: Generated {len(dif_stack)} difference frames")

        return result
    
    def compute_tif_differences(self, detector_frames: np.ndarray) -> Dict:
        """
            * compute frame differences for CASE TIF detector arrays
        """
        detector_frames = np.array(detector_frames)
        dif_stack = []
        time_stack = []
        
        # Get reference frame (frame 0)
        frame_0 = detector_frames[0].astype(np.float64)
        frame_0 = self.reference_corrector.subtract_reference_pixels(frame_0)
        
        # Process remaining frames (besides the 0th frame)
        for i, frame in tqdm(enumerate(detector_frames[1:], start=1), 
                           desc='CASE frame differences'):
            # Grab the curret frame and perform the 0th ref px correction
            curr_frame = frame.astype(np.float64)
            curr_frame = self.reference_corrector.subtract_reference_pixels(curr_frame)
            
            # Calculate the difference
            dif = curr_frame - frame_0
            dif_stack.append(dif)
            time_stack.append(i)
        
        return {
            'differences': np.array(dif_stack),
            'frame_times': np.array(time_stack),
            'reference_frame': frame_0,
            'total_frames': len(dif_stack)
        }