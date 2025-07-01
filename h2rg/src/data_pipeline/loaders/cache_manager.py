
import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import logging
import hashlib

from ..validation.integrity_validator import DataIntegrityValidator


class CacheManager:
    """
        * manages the preprocessing cache registry and storage
    """
    def __init__(self, root_dir: str = 'data/processed'):
        """

        """
        # Define root directory
        self.root_dir = Path(root_dir)

        # Establish registry 
        self.registry_file = self.root_dir / 'processing_registry.json'
        self.registry = self.load_registry()

        # Establish logger instance
        self.logger = logging.getLogger(__name__)
        
        # Create directory structure
        self._create_directories()
    
    def _create_directories(self):
        """
            * create cache directory structure
        """
        # Define needed directories
        directories = [
            'raw_differences',
            'patches', 
            'temporal_analysis',
            'metadata'
        ]
        
        # Create all directories in the root directory
        for dir_name in directories:
            (self.root_dir / dir_name).mkdir(parents=True, exist_ok=True)
    
    def load_registry(self) -> Dict:
        """
            * load existing registry or create new one
        """
        # Check if the already registry exists
        if self.registry_file.exists():
            # Load registy data
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        
        # Return blank registry if not
        return {
            'processed_files': {},
            'processed_exposures': {},
            'last_updated': datetime.now().isoformat(timespec='microseconds')
        }
    
    def save_registry(self) -> str:
        """
            * save registry with hash validation
        """
        # Update the registry upadte time with the current time
        self.registry['last_updated'] = datetime.now().isoformat(timespec='microseconds')
        
        # Save the registry data
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
        
        # Hash the saved file
        hasher = hashlib.md5()
        with open(self.registry_file, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hasher.update(chunk)
        
        registry_hash = hasher.hexdigest()
        self.logger.info(f"Registry saved with hash: {registry_hash}")

        return registry_hash
    
    def is_exposure_cached(self, exposure_id: str, file_paths: List[str],
                           validator: 'DataIntegrityValidator') -> bool:
        """
            * check if exposure is cached and up-to-date
        """
        # Check if the exposure id is in the processed exposures
        if exposure_id not in self.registry['processed_exposures']:
            return False
        
        # Make sure the file has not been changed
        current_hash = validator.get_exposure_hash(file_paths)
        cached_hash = self.registry['processed_exposures'][exposure_id]['exposure_hash']
        
        return current_hash == cached_hash