import sys
import argparse
import logging
from pathlib import Path
import yaml

# Add src directory to path for imports
project_root = Path(__file__).parent.parent.parent.parent  # Go up to project root
sys.path.insert(0, str(project_root / 'src'))

from data_pipeline.orchestrator import DataProcessingOrchestrator

def load_config(config_path: str):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_logging(log_level: str = 'INFO', log_file: str = None):
    """Setup logging configuration with automatic directory creation"""
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        # Create parent directories if they don't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Create training dataset for astronomical anomaly detection'
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/preprocessing_config.yml',
        help='Path to configuration file (relative to project root)'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['EUCLID', 'CASE', 'ALL'],
        default='ALL',
        help='Which dataset to process'
    )
    
    parser.add_argument(
        '--data-root',
        type=str,
        default='/proj/case/2025-06-05',
        help='Root directory containing raw data'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,  # Will use config default
        help='Output directory for processed data'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        default='results/preprocessing.log',
        help='Log file path (relative to project root)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be processed without actually processing'
    )

    parser.add_argument(
        '--test-run',
        action='store_true',
        help='Process only first file from each dataset with 10 frames each (30 total frames)'
    )

    parser.add_argument(
        '--test-frames',
        type=int,
        default=10,
        help='Number of frames to process in test run mode (default: 10)'
    )
    
    return parser.parse_args()

def main():
    """Main processing pipeline"""
    args = parse_arguments()
    
    # Determine paths relative to project root
    project_root = Path(__file__).parent.parent.parent.parent
    config_path = project_root / args.config
    
    # Setup logging with absolute path
    log_file = project_root / args.log_file if args.log_file else None
    setup_logging(args.log_level, str(log_file) if log_file else None)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting astronomical data preprocessing pipeline")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Load configuration
        if config_path.exists():
            config = load_config(config_path)
            logger.info(f"Loaded configuration from {config_path}")
        else:
            logger.error(f"Config file {config_path} not found!")
            sys.exit(1)
        
        # Determine output directory
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = config['cache']['root_dir']

        # Check for test run
        if args.test_run:
            base_dir = Path(output_dir)
            output_dir = str(base_dir.parent / "test")
            logger.info(f"TEST RUN MODE: Using test output directory: {output_dir}")
        
        # Convert to absolute path
        if not Path(output_dir).is_absolute():
            output_dir = project_root / output_dir
        
        logger.info(f"Output directory: {output_dir}")
        
        # Initialize orchestrator
        processor = DataProcessingOrchestrator(
            root_dir=str(output_dir),
            data_root_dir=args.data_root
        )
        
        # Apply configuration
        processor.apply_config(config)
        logger.info(f"Applied configuration: {config}")

        processor.apply_config(config)
        if args.test_run:
            processor.enable_test_mode(args.test_frames, output_dir)
        
        if args.dry_run:
            logger.info("DRY RUN MODE - showing what would be processed")
            processor._initialize_data_directories()
            logger.info(f"Would process {len(processor.euclid_dirs)} EUCLID directories")
            logger.info(f"Would process {len(processor.case_dirs)} CASE directories")
            return
        
        # Process datasets
        total_processed = 0
        
        if args.dataset in ['EUCLID', 'ALL']:
            logger.info("Processing EUCLID data...")
            euclid_exposures = processor.preprocess('EUCLID')
            logger.info(f"Processed {len(euclid_exposures)} EUCLID exposures")
            total_processed += len(euclid_exposures)
        
        if args.dataset in ['CASE', 'ALL']:
            logger.info("Processing CASE data...")
            case_exposures = processor.preprocess('CASE')
            logger.info(f"Processed {len(case_exposures)} CASE exposures")
            total_processed += len(case_exposures)
        
        # Get final statistics
        stats = processor.get_statistics()
        logger.info("=== PROCESSING COMPLETE ===")
        logger.info(f"Total exposures processed: {total_processed}")
        logger.info(f"Dataset statistics:")
        logger.info(f"  Total exposures: {stats['total_exposures']}")
        logger.info(f"  EUCLID: {stats['euclid_exposures']}")
        logger.info(f"  CASE: {stats['case_exposures']}")
        logger.info(f"  Available patch sizes: {stats['patch_sizes_available']}")
        
        # Save registry
        hash_val = processor.save_registry()
        logger.info(f"Registry saved with hash: {hash_val}")
        
        logger.info(f"Training data ready in: {output_dir}")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)

if __name__ == "__main__":
    main()