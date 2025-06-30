import logging
import yaml
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import sys

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from data_pipeline.orchestrator import DataProcessingOrchestrator

@dataclass
class ProcessingConfig:
    """
        * configuration for training set creation
    """
    config_path: str
    dataset: str
    data_root: str
    output_dir: Optional[str]
    log_level: str
    log_file: Optional[str]
    dry_run: bool

class TrainingSetCreator:
    """
        * class responsible for creating training datasets from raw astronomical data
    """
    def __init__(self, config: ProcessingConfig):
        """
            * initialize the training set creator
        """
        self.config = config
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.logger = None
        self.processor = None
        self.yaml_config = None
        
        # Initialize components
        self._setup_logging()
        self._load_configuration()
        self._initialize_processor()
    
    def _setup_logging(self):
        """
            * setup logging configuration
        """
        handlers = [logging.StreamHandler()]
        
        # Check to seee if the config file is in the log files
        if self.config.log_file:
            log_path = self.project_root / self.config.log_file
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(log_path))
        
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=handlers
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Training set creator initialized")
    
    def _load_configuration(self):
        """
            * load YAML configuration file
        """
        config_path = self.project_root / self.config.config_path
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self.yaml_config = yaml.safe_load(f)
        
        self.logger.info(f"Configuration loaded from {config_path}")
    
    def _initialize_processor(self):
        """
            * initialize the data processing orchestrator
        """
        # Determine output directory
        if self.config.output_dir:
            output_dir = self.config.output_dir
        else:
            output_dir = self.yaml_config['cache']['root_dir']
        
        # Convert to absolute path
        if not Path(output_dir).is_absolute():
            output_dir = self.project_root / output_dir
        
        self.logger.info(f"Output directory: {output_dir}")
        
        # Initialize orchestrator
        self.processor = DataProcessingOrchestrator(
            root_dir=str(output_dir),
            data_root_dir=self.config.data_root
        )
        
        # Apply configuration
        self.processor.apply_config(self.yaml_config)
        self.logger.info("Processor initialized and configured")
    
    def validate_configuration(self) -> bool:
        """
            * validate the configuration and setup
        """
        try:
            # Check data root exists
            data_root = Path(self.config.data_root)
            if not data_root.exists():
                self.logger.error(f"Data root directory not found: {data_root}")
                return False
            
            # Check required config sections
            required_sections = ['preprocessing', 'cache']
            for section in required_sections:
                if section not in self.yaml_config:
                    self.logger.error(f"Missing required config section: {section}")
                    return False
            
            # Validate dataset choice
            valid_datasets = ['EUCLID', 'CASE', 'ALL']
            if self.config.dataset not in valid_datasets:
                self.logger.error(f"Invalid dataset: {self.config.dataset}")
                return False
            
            self.logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False
    
    def run_dry_run(self) -> Dict[str, int]:
        """
            * perform a dry run to show what would be processed
        """
        self.logger.info("Running dry run analysis...")
        
        self.processor._initialize_data_directories()
        
        dry_run_results = {
            'euclid_directories': len(self.processor.euclid_dirs),
            'case_directories': len(self.processor.case_dirs),
            'total_directories': len(self.processor.euclid_dirs) + len(self.processor.case_dirs)
        }
        
        self.logger.info("Dry run results:")
        self.logger.info(f"  EUCLID directories: {dry_run_results['euclid_directories']}")
        self.logger.info(f"  CASE directories: {dry_run_results['case_directories']}")
        self.logger.info(f"  Total directories: {dry_run_results['total_directories']}")
        
        return dry_run_results
    
    def process_euclid_data(self) -> List[str]:
        """
            * process EUCLID dataset
        """
        self.logger.info("Processing EUCLID data...")
        
        try:
            exposures = self.processor.preprocess('EUCLID')
            self.logger.info(f"Successfully processed {len(exposures)} EUCLID exposures")
            return exposures
        except Exception as e:
            self.logger.error(f"EUCLID processing failed: {e}")
            raise
    
    def process_case_data(self) -> List[str]:
        """
            * process CASE dataset
        """
        self.logger.info("Processing CASE data...")
        
        try:
            exposures = self.processor.preprocess('CASE')
            self.logger.info(f"Successfully processed {len(exposures)} CASE exposures")
            return exposures
        except Exception as e:
            self.logger.error(f"CASE processing failed: {e}")
            raise
    
    def generate_final_report(self, processed_exposures: List[str]) -> Dict:
        """
            * generate final processing report
        """
        stats = self.processor.get_statistics()
        
        report = {
            'processing_summary': {
                'total_processed': len(processed_exposures),
                'euclid_processed': len([e for e in processed_exposures if 'euclid' in e]),
                'case_processed': len([e for e in processed_exposures if 'case' in e])
            },
            'dataset_statistics': stats,
            'configuration_used': self.yaml_config,
            'output_directory': str(self.processor.cache_manager.root_dir)
        }
        
        # Save registry and get hash
        registry_hash = self.processor.save_registry()
        report['registry_hash'] = registry_hash
        
        # Log final report
        self.logger.info("=== PROCESSING COMPLETE ===")
        self.logger.info(f"Total exposures processed: {report['processing_summary']['total_processed']}")
        self.logger.info(f"EUCLID exposures: {report['processing_summary']['euclid_processed']}")
        self.logger.info(f"CASE exposures: {report['processing_summary']['case_processed']}")
        self.logger.info(f"Registry hash: {registry_hash}")
        self.logger.info(f"Training data ready in: {report['output_directory']}")
        
        return report
    
    def run(self) -> Dict:
        """
            * main execution method
        """
        self.logger.info("Starting training set creation pipeline")
        
        # Validate configuration
        if not self.validate_configuration():
            raise ValueError("Configuration validation failed")
        
        # Handle dry run
        if self.config.dry_run:
            dry_run_results = self.run_dry_run()
            return {'dry_run': True, 'results': dry_run_results}
        
        # Process datasets based on selection
        processed_exposures = []
        
        try:
            if self.config.dataset in ['EUCLID', 'ALL']:
                euclid_exposures = self.process_euclid_data()
                processed_exposures.extend(euclid_exposures)
            
            if self.config.dataset in ['CASE', 'ALL']:
                case_exposures = self.process_case_data()
                processed_exposures.extend(case_exposures)
            
            # Generate final report
            final_report = self.generate_final_report(processed_exposures)
            
            return {
                'dry_run': False,
                'success': True,
                'processed_exposures': processed_exposures,
                'report': final_report
            }
            
        except Exception as e:
            self.logger.error(f"Processing pipeline failed: {e}")
            self.logger.exception("Full traceback:")
            
            return {
                'dry_run': False,
                'success': False,
                'error': str(e),
                'processed_exposures': processed_exposures
            }