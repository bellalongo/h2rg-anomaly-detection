import os
import sys
import argparse
import logging
from pathlib import Path
import time
import math

# Add src to path
project_root = Path(__file__).parent.parent.parent.parent  # Go up to project root
sys.path.insert(0, str(project_root / 'src'))

from data_pipeline.orchestrator import DataOrchestrator
import yaml

def load_config(config_path: str):
    """

    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_logging(log_file: str, log_level: str = "INFO"):
    """

    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def get_exposure_list(data_root: str, dataset_type: str = "ALL"):
    """

    """
    exposure_list = []
    data_root_path = Path(data_root)
    
    if dataset_type in ["ALL", "EUCLID"]:
        # Get EUCLID exposures - look for directories with 'Euclid_SCA' pattern
        for dir_path in data_root_path.iterdir():
            if dir_path.is_dir() and 'Euclid_SCA' in dir_path.name:
                for file_path in dir_path.glob("*.fits"):
                    exposure_id = f"EUCLID_{dir_path.name}_{file_path.stem}"
                    exposure_list.append(('EUCLID', exposure_id, str(file_path)))
    
    if dataset_type in ["ALL", "CASE"]:
        # Get CASE exposures - look for directories with 'noise' pattern  
        for dir_path in data_root_path.iterdir():
            if dir_path.is_dir() and 'noise' in dir_path.name:
                for file_path in dir_path.glob("*.fits"):
                    exposure_id = f"CASE_{dir_path.name}_{file_path.stem}"
                    exposure_list.append(('CASE', exposure_id, str(file_path)))
    
    return exposure_list

def distribute_exposures(exposure_list, array_task_id, total_array_tasks):
    """

    """
    total_exposures = len(exposure_list)
    exposures_per_task = math.ceil(total_exposures / total_array_tasks)
    
    start_idx = (array_task_id - 1) * exposures_per_task
    end_idx = min(start_idx + exposures_per_task, total_exposures)
    
    task_exposures = exposure_list[start_idx:end_idx]
    
    return task_exposures, start_idx, end_idx

def process_exposure_subset(orchestrator, exposure_subset, logger):
    """

    """
    processed_count = 0
    failed_count = 0
    
    for dataset_type, exposure_id, file_path in exposure_subset:
        try:
            logger.info(f"Processing {dataset_type} exposure: {exposure_id}")
            
            # Process based on dataset type
            if dataset_type == "EUCLID":
                # Use existing EUCLID processing logic
                success = orchestrator.euclid_processor._process_single_exposure(
                    processed_count, exposure_id, file_path, 
                    Path(file_path).parent.name, Path(file_path).name
                )
            elif dataset_type == "CASE":
                # Use existing CASE processing logic  
                success = orchestrator.case_processor._process_single_exposure(
                    processed_count, exposure_id, file_path,
                    Path(file_path).parent.name, Path(file_path).name
                )
            else:
                logger.error(f"Unknown dataset type: {dataset_type}")
                success = False
            
            if success:
                processed_count += 1
                logger.info(f"Successfully processed: {exposure_id}")
            else:
                failed_count += 1
                logger.error(f"Failed to process: {exposure_id}")
                
        except Exception as e:
            failed_count += 1
            logger.error(f"Exception processing {exposure_id}: {str(e)}")
    
    return processed_count, failed_count

def main():
    parser = argparse.ArgumentParser(description='Array-based H2RG training set creation')
    parser.add_argument('--array-task-id', type=int, required=True,
                        help='SLURM array task ID')
    parser.add_argument('--total-array-tasks', type=int, required=True,
                        help='Total number of array tasks')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--data-root', type=str, required=True,
                        help='Root directory for raw data')
    parser.add_argument('--output-dir', type=str, default='array_output',
                        help='Output directory for processed data')
    parser.add_argument('--log-file', type=str, default='array_processing.log',
                        help='Log file path')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    parser.add_argument('--dataset-type', type=str, default='ALL',
                        choices=['ALL', 'EUCLID', 'CASE'],
                        help='Dataset type to process')
    
    args = parser.parse_args()
    
    # Setup logging
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    logger = setup_logging(args.log_file, args.log_level)
    
    logger.info(f"Starting array task {args.array_task_id} of {args.total_array_tasks}")
    logger.info(f"Data root: {args.data_root}")
    logger.info(f"Output dir: {args.output_dir}")
    
    try:
        # Load configuration
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
        
        # Get complete exposure list
        logger.info("Getting complete exposure list...")
        all_exposures = get_exposure_list(args.data_root, args.dataset_type)
        logger.info(f"Found {len(all_exposures)} total exposures")
        
        # Distribute exposures for this array task
        task_exposures, start_idx, end_idx = distribute_exposures(
            all_exposures, args.array_task_id, args.total_array_tasks
        )
        
        logger.info(f"Array task {args.array_task_id} processing exposures {start_idx}-{end_idx-1}")
        logger.info(f"Task will process {len(task_exposures)} exposures")
        
        if not task_exposures:
            logger.info("No exposures assigned to this array task")
            return 0
        
        # Create data orchestrator
        orchestrator = DataOrchestrator(
            data_root_dir=args.data_root,
            root_dir=args.output_dir
        )
        
        # Apply configuration
        orchestrator.apply_config(config)
        
        # Enable production mode with limited parallelism per task
        orchestrator.enable_production_mode(max_parallel_exposures=4)
        
        # Process assigned exposures
        start_time = time.time()
        processed_count, failed_count = process_exposure_subset(
            orchestrator, task_exposures, logger
        )
        end_time = time.time()
        
        # Log completion statistics
        total_time = end_time - start_time
        logger.info("="*60)
        logger.info(f"ARRAY TASK {args.array_task_id} COMPLETED")
        logger.info(f"Processed: {processed_count} exposures")
        logger.info(f"Failed: {failed_count} exposures")
        logger.info(f"Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
        
        if processed_count > 0:
            avg_time = total_time / processed_count
            logger.info(f"Average time per exposure: {avg_time:.2f}s")
        
        # Return appropriate exit code
        if failed_count == 0:
            logger.info("Array task completed successfully!")
            return 0
        elif processed_count > 0:
            logger.warning(f"Array task completed with {failed_count} failures")
            return 1
        else:
            logger.error("Array task failed - no exposures processed")
            return 2
            
    except Exception as e:
        logger.error(f"Array task {args.array_task_id} failed with exception: {str(e)}")
        return 3

if __name__ == "__main__":
    sys.exit(main())