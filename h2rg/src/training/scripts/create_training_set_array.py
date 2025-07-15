import os
import sys
import argparse
import logging
from pathlib import Path
import time
import math
import yaml

# Add src directory to path for imports
project_root = Path(__file__).parent.parent.parent.parent  # Go up to project root
sys.path.insert(0, str(project_root / 'src'))

from data_pipeline.orchestrator import DataProcessingOrchestrator

def load_config(config_path: str):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_logging(log_file: str, log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def get_files_in_folder(data_root: str, folder_name: str, dataset_type: str):
    """Get all files in a specific folder based on dataset type"""
    folder_path = Path(data_root) / folder_name
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Choose file extension based on dataset type
    if dataset_type == "EUCLID":
        # EUCLID datasets use FITS files
        files = list(folder_path.glob("*.fits"))
    elif dataset_type == "CASE":
        # CASE datasets use TIF/TIFF files
        tif_files = list(folder_path.glob("*.tif"))
        tiff_files = list(folder_path.glob("*.tiff"))
        files = tif_files + tiff_files
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    files.sort()  # Ensure consistent ordering
    
    return [(str(f), f.stem) for f in files]

def distribute_files_for_job(file_list, job_within_folder, total_jobs_per_folder):
    """Distribute files among jobs within a folder"""
    total_files = len(file_list)
    files_per_job = math.ceil(total_files / total_jobs_per_folder)
    
    start_idx = (job_within_folder - 1) * files_per_job
    end_idx = min(start_idx + files_per_job, total_files)
    
    job_files = file_list[start_idx:end_idx]
    
    return job_files, start_idx, end_idx

def process_folder_files(orchestrator, folder_name, file_subset, dataset_type, logger):
    """Process a subset of files from a specific folder"""
    processed_count = 0
    failed_count = 0
    
    logger.info(f"Processing {len(file_subset)} files from folder: {folder_name}")
    
    for file_path, file_stem in file_subset:
        try:
            exposure_id = f"{dataset_type}_{folder_name}_{file_stem}"
            logger.info(f"Processing: {exposure_id}")
            
            # Process based on dataset type
            if dataset_type == "EUCLID":
                success = orchestrator.euclid_processor._process_single_exposure(
                    processed_count, exposure_id, file_path, 
                    folder_name, Path(file_path).name
                )
            elif dataset_type == "CASE":
                success = orchestrator.case_processor._process_single_exposure(
                    processed_count, exposure_id, file_path,
                    folder_name, Path(file_path).name
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
            logger.error(f"Exception processing {file_path}: {str(e)}")
    
    return processed_count, failed_count

def main():
    parser = argparse.ArgumentParser(description='Folder-based H2RG training set creation')
    parser.add_argument('--folder-name', type=str, required=True,
                        help='Name of the folder to process')
    parser.add_argument('--job-within-folder', type=int, required=True,
                        help='Job number within the folder (1-based)')
    parser.add_argument('--total-jobs-per-folder', type=int, required=True,
                        help='Total number of jobs per folder')
    parser.add_argument('--dataset-type', type=str, required=True,
                        choices=['EUCLID', 'CASE'],
                        help='Dataset type being processed')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--data-root', type=str, required=True,
                        help='Root directory for raw data')
    parser.add_argument('--output-dir', type=str, default='folder_output',
                        help='Output directory for processed data')
    parser.add_argument('--log-file', type=str, default='folder_processing.log',
                        help='Log file path')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    logger = setup_logging(args.log_file, args.log_level)
    
    logger.info(f"=== Folder-based processing started ===")
    logger.info(f"Folder: {args.folder_name}")
    logger.info(f"Job: {args.job_within_folder} of {args.total_jobs_per_folder}")
    logger.info(f"Dataset type: {args.dataset_type}")
    logger.info(f"Data root: {args.data_root}")
    logger.info(f"Output dir: {args.output_dir}")
    
    try:
        # Load configuration
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
        
        # Get all files in the folder
        logger.info(f"Getting files from folder: {args.folder_name}")
        all_files = get_files_in_folder(args.data_root, args.folder_name, args.dataset_type)
        logger.info(f"Found {len(all_files)} FITS files in folder")
        
        # Distribute files for this specific job
        job_files, start_idx, end_idx = distribute_files_for_job(
            all_files, args.job_within_folder, args.total_jobs_per_folder
        )
        
        logger.info(f"Job {args.job_within_folder} processing files {start_idx}-{end_idx-1}")
        logger.info(f"Processing {len(job_files)} files in this job")
        
        if not job_files:
            logger.info("No files assigned to this job")
            return 0
        
        # Create orchestrator
        orchestrator = DataProcessingOrchestrator(
            data_root_dir=args.data_root,
            root_dir=args.output_dir
        )
        
        # Apply configuration
        orchestrator.apply_config(config)
        
        # Enable production mode with limited parallelism per task
        orchestrator.enable_production_mode(max_parallel_exposures=4)
        
        # Process assigned files
        start_time = time.time()
        processed_count, failed_count = process_folder_files(
            orchestrator, args.folder_name, job_files, args.dataset_type, logger
        )
        end_time = time.time()
        
        # Log completion statistics
        total_time = end_time - start_time
        logger.info("="*60)
        logger.info(f"FOLDER JOB COMPLETED")
        logger.info(f"Folder: {args.folder_name}")
        logger.info(f"Job: {args.job_within_folder} of {args.total_jobs_per_folder}")
        logger.info(f"Processed: {processed_count} files")
        logger.info(f"Failed: {failed_count} files")
        logger.info(f"Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
        
        if processed_count > 0:
            avg_time = total_time / processed_count
            logger.info(f"Average time per file: {avg_time:.2f}s")
        
        # Return appropriate exit code
        if failed_count == 0:
            logger.info("Folder job completed successfully!")
            return 0
        elif processed_count > 0:
            logger.warning(f"Folder job completed with {failed_count} failures")
            return 1
        else:
            logger.error("Folder job failed - no files processed")
            return 2
            
    except Exception as e:
        logger.error(f"Folder job failed with exception: {str(e)}")
        return 3

if __name__ == "__main__":
    sys.exit(main())