import hashlib
import os
import json
from typing import List
from pathlib import Path
import logging
import h5py


class DataIntegrityValidator:
    """
        * handles hash-based validation for cache integrity
    """
    def __init__(self):
        """

        """
        self.logger = logging.getLogger(__name__)
    
    def get_file_hash(self, file_path: str) -> str:
        """
            * generate MD5 hash for single file
        """
        hasher = hashlib.md5()

        # Open the current file
        with open(file_path, 'rb') as f:
            # Iterate through every chunk in the file
            for chunk in iter(lambda: f.read(4096), b""):
                # Update the hasher with the chunk info
                hasher.update(chunk)

        return hasher.hexdigest()
    
    def get_exposure_hash(self, file_paths: List[str]) -> str:
        """
            * generate hash for group of files (exposure)
        """
        hasher = hashlib.md5()

        # Iterae through every file (frame)
        for file_path in sorted(file_paths):
            # Update the hasher wit the encoded current file
            hasher.update(file_path.encode())

            # Check if the file path exists in the root directory
            if os.path.exists(file_path):
                hasher.update(str(os.path.getmtime(file_path)).encode())

        return hasher.hexdigest()
    
    def verify_cache_integrity(self, expected_files: List[Path]) -> bool:
        """
            * verify all cache files exist and are readable
        """
        # Go through every file
        for file_path in expected_files:

            # Make sure that the file exists
            if not file_path.exists():
                self.logger.error(f"Missing cache file: {file_path}")
                return False
            
            # Make sure that the cache file was properly saved
            try:
                # Check if the file was a h5
                if file_path.suffix == '.h5':
                    with h5py.File(file_path, 'r') as f:
                        list(f.keys())

                # Check if the file was a json
                elif file_path.suffix == '.json':
                    with open(file_path, 'r') as f:
                        json.load(f)
                        
            except Exception as e:
                self.logger.error(f"Corrupted cache file {file_path}: {e}")
                return False
        
        return True