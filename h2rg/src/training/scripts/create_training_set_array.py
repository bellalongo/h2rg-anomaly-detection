import os
import sys
import argparse
import logging
from pathlib import Path
import time
import math
import yaml
import json
import shutil
import re
from datetime import datetime

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

def create_job_specific_output_dir(base_output_dir: str, folder_name: str, job_id: int):
    """Create job-specific output directory to avoid registry conflicts"""
    job_output_dir = Path(base_output_dir) / "job_outputs" / f"{folder_name}_job_{job_id:03d}"
    job_output_dir.mkdir(parents=True, exist_ok=True)
    return str(job_output_dir)

def get_files_in_folder(data_root: str, folder_name: str, dataset_type: str):
    """Get all files in a specific folder based on dataset type"""
    folder_path = Path(data_root) / folder_name
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Choose file extension based on dataset type
    if dataset_type == "EUCLID":
        # EUCLID datasets use FITS files
        files = list(folder_path.glob("*.fits"))
        files.sort()  # Ensure consistent ordering
        
    elif dataset_type == "CASE":
        # CASE datasets use TIF/TIFF files
        tif_files = list(folder_path.glob("*.tif"))
        tiff_files = list(folder_path.glob("*.tiff"))
        files = tif_files + tiff_files
        
        # CRITICAL: Sort CASE files properly by E#### and N#### indices
        if files:
            files = sorted(files, key=lambda x: (
                int(re.search(r'_E(\d+)_', x.name).group(1)) if re.search(r'_E(\d+)_', x.name) else 0,
                int(re.search(r'_N(\d+)\.tif', x.name).group(1)) if re.search(r'_N(\d+)\.tif', x.name) else 0
            ))
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
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
    """
        
    """
    processed_count = 0
    failed_count = 0
    
    if dataset_type == "EUCLID":
        # EUCLID: Process individual files
        logger.info(f"Processing {len(file_subset)} EUCLID files from folder: {folder_name}")
        
        for file_path, file_stem in file_subset:
            try:
                exposure_id = f"{dataset_type}_{folder_name}_{file_stem}"
                logger.info(f"Processing: {exposure_id}")
                
                success = orchestrator.euclid_processor._process_single_exposure(
                    processed_count, exposure_id, file_path, 
                    folder_name, Path(file_path).name
                )
                
                if success:
                    processed_count += 1
                    logger.info(f"Successfully processed: {exposure_id}")
                else:
                    failed_count += 1
                    logger.error(f"Failed to process: {exposure_id}")
                    
            except Exception as e:
                failed_count += 1
                logger.error(f"Exception processing {file_path}: {str(e)}")
    
    elif dataset_type == "CASE":
        # CASE: Process entire folder (not individual files!)
        logger.info(f"Processing CASE folder: {folder_name} (entire folder)")
        
        try:
            # Process the entire folder using the correct method
            processed_exposures = orchestrator.case_processor.process_directory(
                orchestrator.data_root_dir, [folder_name]
            )
            
            processed_count = len(processed_exposures)
            failed_count = 0
            
            logger.info(f"Successfully processed {processed_count} CASE exposures from folder {folder_name}")
            
            # Log each exposure
            for exposure_id in processed_exposures:
                logger.info(f"Successfully processed CASE exposure: {exposure_id}")
                
        except Exception as e:
            failed_count = 1  # One folder failed
            logger.error(f"Exception processing CASE folder {folder_name}: {str(e)}")
    
    else:
        logger.error(f"Unknown dataset type: {dataset_type}")
    
    return processed_count, failed_count

def merge_job_registries(base_output_dir: str, folder_name: str, total_jobs: int, logger):
    """
        
    """
    main_output_dir = Path(base_output_dir)
    main_registry_file = main_output_dir / 'processing_registry.json'
    
    # Load existing main registry if it exists
    if main_registry_file.exists():
        with open(main_registry_file, 'r') as f:
            main_registry = json.load(f)
    else:
        main_registry = {
            'processed_exposures': {},
            'last_updated': datetime.now().isoformat(timespec='microseconds')
        }
    
    # Merge all job registries
    merged_count = 0
    for job_id in range(1, total_jobs + 1):
        job_output_dir = main_output_dir / "job_outputs" / f"{folder_name}_job_{job_id:03d}"
        job_registry_file = job_output_dir / 'processing_registry.json'
        
        if job_registry_file.exists():
            try:
                with open(job_registry_file, 'r') as f:
                    job_registry = json.load(f)
                
                # Merge processed exposures
                for exposure_id, exposure_data in job_registry.get('processed_exposures', {}).items():
                    main_registry['processed_exposures'][exposure_id] = exposure_data
                    merged_count += 1
                
                logger.info(f"Merged registry from job {job_id}: {len(job_registry.get('processed_exposures', {}))} exposures")
                
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load registry from job {job_id}: {e}")
    
    # Update timestamp and save merged registry
    main_registry['last_updated'] = datetime.now().isoformat(timespec='microseconds')
    
    with open(main_registry_file, 'w') as f:
        json.dump(main_registry, f, indent=2)
    
    logger.info(f"Merged {merged_count} exposures from {total_jobs} jobs into main registry")
    return merged_count

def copy_processed_data_to_main(base_output_dir: str, folder_name: str, total_jobs: int, logger):
    """Copy processed data from job directories to main output directory"""
    main_output_dir = Path(base_output_dir)
    
    # Data directories to merge
    data_dirs = ['raw_differences', 'patches', 'temporal_analysis', 'metadata']
    
    copied_totals = {}
    for data_dir in data_dirs:
        main_data_dir = main_output_dir / data_dir
        main_data_dir.mkdir(parents=True, exist_ok=True)
        
        copied_files = 0
        for job_id in range(1, total_jobs + 1):
            job_output_dir = main_output_dir / "job_outputs" / f"{folder_name}_job_{job_id:03d}"
            job_data_dir = job_output_dir / data_dir
            
            if job_data_dir.exists():
                for file_path in job_data_dir.glob('*'):
                    if file_path.is_file():
                        dest_path = main_data_dir / file_path.name
                        if not dest_path.exists():
                            shutil.copy2(file_path, dest_path)
                            copied_files += 1
                        else:
                            logger.debug(f"File already exists, skipping: {file_path.name}")
        
        copied_totals[data_dir] = copied_files
        logger.info(f"Copied {copied_files} files to {data_dir}/")
    
    return copied_totals

def cleanup_job_directories(base_output_dir: str, folder_name: str, total_jobs: int, logger):
    """Clean up job-specific directories after successful merge"""
    main_output_dir = Path(base_output_dir)
    job_outputs_dir = main_output_dir / "job_outputs"
    
    removed_count = 0
    for job_id in range(1, total_jobs + 1):
        job_output_dir = job_outputs_dir / f"{folder_name}_job_{job_id:03d}"
        
        if job_output_dir.exists():
            try:
                shutil.rmtree(job_output_dir)
                removed_count += 1
                logger.debug(f"Removed job directory: {job_output_dir}")
            except OSError as e:
                logger.warning(f"Failed to remove job directory {job_output_dir}: {e}")
    
    # Remove job_outputs directory if empty
    if job_outputs_dir.exists() and not any(job_outputs_dir.iterdir()):
        try:
            job_outputs_dir.rmdir()
            logger.info("Removed empty job_outputs directory")
        except OSError:
            pass
    
    logger.info(f"Cleaned up {removed_count} job directories")

def main():
    parser = argparse.ArgumentParser(description='Folder-based H2RG training set creation with isolated registries')
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
                        help='Base output directory for processed data')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    
    # Special mode for merging results
    parser.add_argument('--merge-mode', action='store_true',
                        help='Merge job registries and data into main output (run after all jobs complete)')
    parser.add_argument('--cleanup-jobs', action='store_true',
                        help='Clean up job directories after merging (use with --merge-mode)')
    
    args = parser.parse_args()
    
    # Handle merge mode
    if args.merge_mode:
        log_file = Path(args.output_dir) / f'merge_{args.folder_name}.log'
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logger = setup_logging(log_file, args.log_level)
        
        logger.info(f"=== Starting merge mode for folder: {args.folder_name} ===")
        
        try:
            # Merge registries
            merged_count = merge_job_registries(
                args.output_dir, args.folder_name, args.total_jobs_per_folder, logger
            )
            
            # Copy processed data
            copied_totals = copy_processed_data_to_main(
                args.output_dir, args.folder_name, args.total_jobs_per_folder, logger
            )
            
            # Clean up job directories if requested
            if args.cleanup_jobs:
                cleanup_job_directories(
                    args.output_dir, args.folder_name, args.total_jobs_per_folder, logger
                )
            
            logger.info("="*60)
            logger.info("MERGE COMPLETED SUCCESSFULLY")
            logger.info(f"Merged {merged_count} exposure records")
            for data_dir, count in copied_totals.items():
                logger.info(f"Copied {count} files from {data_dir}/")
            logger.info("="*60)
            
            return 0
            
        except Exception as e:
            logger.error(f"Merge failed with exception: {str(e)}")
            return 1
    
    # Regular processing mode with job isolation
    
    # Create job-specific output directory
    job_output_dir = create_job_specific_output_dir(
        args.output_dir, args.folder_name, args.job_within_folder
    )
    
    # Setup logging with job-specific log file
    log_file = Path(job_output_dir) / f'job_{args.job_within_folder:03d}_processing.log'
    logger = setup_logging(log_file, args.log_level)
    
    logger.info(f"=== Folder-based processing started (JOB ISOLATED) ===")
    logger.info(f"Folder: {args.folder_name}")
    logger.info(f"Job: {args.job_within_folder} of {args.total_jobs_per_folder}")
    logger.info(f"Dataset type: {args.dataset_type}")
    logger.info(f"Data root: {args.data_root}")
    logger.info(f"Job output dir: {job_output_dir}")
    
    try:
        # Load configuration
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
        
        # Create orchestrator with job-specific output directory
        orchestrator = DataProcessingOrchestrator(
            data_root_dir=args.data_root,
            root_dir=job_output_dir  # This prevents registry conflicts!
        )
        
        # Apply configuration
        orchestrator.apply_config(config)
        
        # Enable production mode with limited parallelism per task
        orchestrator.enable_production_mode(max_parallel_exposures=4)
        
        # Handle CASE and EUCLID differently
        if args.dataset_type == "CASE":
            # CASE: Process entire folder at once (optimal)
            logger.info("CASE dataset: Processing entire folder")
            
            start_time = time.time()
            processed_count, failed_count = process_folder_files(
                orchestrator, args.folder_name, [], args.dataset_type, logger
            )
            end_time = time.time()
            
        else:
            # EUCLID: Get files and distribute among jobs
            logger.info(f"Getting files from folder: {args.folder_name}")
            all_files = get_files_in_folder(args.data_root, args.folder_name, args.dataset_type)
            
            file_type = "FITS" if args.dataset_type == "EUCLID" else "TIF"
            logger.info(f"Found {len(all_files)} {file_type} files in folder")
            
            # Distribute files for this specific job
            job_files, start_idx, end_idx = distribute_files_for_job(
                all_files, args.job_within_folder, args.total_jobs_per_folder
            )
            
            logger.info(f"Job {args.job_within_folder} processing files {start_idx}-{end_idx-1}")
            logger.info(f"Processing {len(job_files)} files in this job")
            
            if not job_files:
                logger.info("No files assigned to this job")
                return 0
            
            # Process assigned files
            start_time = time.time()
            processed_count, failed_count = process_folder_files(
                orchestrator, args.folder_name, job_files, args.dataset_type, logger
            )
            end_time = time.time()
        
        # Log completion statistics
        total_time = end_time - start_time
        logger.info("="*60)
        logger.info(f"JOB COMPLETED (ISOLATED)")
        logger.info(f"Folder: {args.folder_name}")
        logger.info(f"Job: {args.job_within_folder} of {args.total_jobs_per_folder}")
        logger.info(f"Dataset type: {args.dataset_type}")
        logger.info(f"Processed: {processed_count} items")
        logger.info(f"Failed: {failed_count} items")
        logger.info(f"Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
        logger.info(f"Job output saved to: {job_output_dir}")
        
        if processed_count > 0:
            avg_time = total_time / processed_count
            logger.info(f"Average time per item: {avg_time:.2f}s")
        
        logger.info("="*60)
        logger.info("NEXT STEPS:")
        logger.info("1. Wait for all jobs to complete")
        logger.info("2. Run with --merge-mode to combine results:")
        logger.info(f"   python {Path(__file__).name} --folder-name {args.folder_name} "
                   f"--job-within-folder 1 --total-jobs-per-folder {args.total_jobs_per_folder} "
                   f"--dataset-type {args.dataset_type} --config {args.config} "
                   f"--data-root {args.data_root} --output-dir {args.output_dir} --merge-mode")
        logger.info("="*60)
        
        # Return appropriate exit code
        if failed_count == 0:
            logger.info("Job completed successfully!")
            return 0
        elif processed_count > 0:
            logger.warning(f"Job completed with {failed_count} failures")
            return 1
        else:
            logger.error("Job failed - no items processed")
            return 2
            
    except Exception as e:
        logger.error(f"Job failed with exception: {str(e)}")
        return 3

if __name__ == "__main__":
    sys.exit(main())