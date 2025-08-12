#!/usr/bin/env python3
"""
Main pipeline orchestrator for the β-lactam adjuvant discovery system.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
import yaml
import subprocess
import pandas as pd

def setup_logging(config):
    """Setup logging configuration."""
    log_dir = Path(config['logging']['file']).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, config['logging']['level']),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config['logging']['file']),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_command(cmd: str, logger: logging.Logger, description: str) -> bool:
    """Run a shell command and log the result."""
    logger.info(f"Running: {description}")
    logger.info(f"Command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"✅ {description} completed successfully")
        if result.stdout:
            logger.debug(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed with exit code {e.returncode}")
        logger.error(f"Error output: {e.stderr}")
        return False

def run_data_ingestion(config: dict, logger: logging.Logger) -> bool:
    """Run data ingestion steps."""
    logger.info("Starting data ingestion...")
    
    # Check if DrugBank data already exists
    if not Path(config['data_sources']['drugbank_edges']).exists():
        logger.info("Processing DrugBank database...")
        cmd = f"python scripts/drugbank_parser.py --xml {config['data_sources']['drugbank']}"
        if not run_command(cmd, logger, "DrugBank parsing"):
            return False
    else:
        logger.info("DrugBank data already exists, skipping...")
    
    # Check if STRING data exists
    if not Path(config['data_sources']['string']).exists():
        logger.info("Generating sample STRING interactions...")
        # For now, we'll use the sample data we created
        logger.info("Using sample STRING interactions data")
    else:
        logger.info("STRING data already exists, skipping...")
    
    # Check if ChEMBL data exists
    if not Path(config['data_sources']['chembl']).exists():
        logger.info("Generating sample ChEMBL targets...")
        # For now, we'll use the sample data we created
        logger.info("Using sample ChEMBL targets data")
    else:
        logger.info("ChEMBL data already exists, skipping...")
    
    logger.info("Data ingestion completed successfully")
    return True

def run_kg_assembly(config: dict, logger: logging.Logger) -> bool:
    """Run knowledge graph assembly."""
    logger.info("Starting knowledge graph assembly...")
    
    cmd = f"python scripts/assemble_kg.py --config {config['config_file']}"
    if not run_command(cmd, logger, "Knowledge graph assembly"):
        return False
    
    logger.info("Knowledge graph assembly completed successfully")
    return True

def run_training(config: dict, logger: logging.Logger) -> bool:
    """Run model training."""
    logger.info("Starting model training...")
    
    cmd = f"python scripts/train_mini_transe.py --config {config['config_file']} --epochs 50 --embedding-dim 128"
    if not run_command(cmd, logger, "Model training"):
        return False
    
    logger.info("Model training completed successfully")
    return True

def run_prediction(config: dict, logger: logging.Logger) -> bool:
    """Run adjuvant prediction."""
    logger.info("Starting adjuvant prediction...")
    
    cmd = f"python scripts/predict_links.py --config {config['config_file']} --top-k 100"
    if not run_command(cmd, logger, "Adjuvant prediction"):
        return False
    
    logger.info("Adjuvant prediction completed successfully")
    return True

def run_validation(config: dict, logger: logging.Logger) -> bool:
    """Run computational validation."""
    logger.info("Starting computational validation...")
    
    cmd = f"python scripts/validate_predictions.py --config {config['config_file']} --top-k 50"
    if not run_command(cmd, logger, "Computational validation"):
        return False
    
    logger.info("Computational validation completed successfully")
    return True

def generate_report(config: dict, logger: logging.Logger) -> bool:
    """Generate final report and summary."""
    logger.info("Generating final report...")
    
    # Check if all required files exist
    required_files = [
        config['outputs']['metrics'],
        config['outputs']['predictions'],
        config['outputs']['validation']['summary']
    ]
    
    missing_files = [f for f in required_files if not Path(f).exists()]
    if missing_files:
        logger.warning(f"Missing required files: {missing_files}")
        return False
    
    # Load and display results
    try:
        with open(config['outputs']['metrics'], 'r') as f:
            metrics = yaml.safe_load(f)
        
        predictions_df = pd.read_csv(config['outputs']['predictions'], sep='\t')
        validation_df = pd.read_csv(config['outputs']['validation']['summary'], sep='\t')
        
        logger.info("=== FINAL RESULTS SUMMARY ===")
        logger.info(f"Model Performance:")
        logger.info(f"  Test MRR: {metrics['test_metrics']['mrr']:.4f}")
        logger.info(f"  Test Hits@10: {metrics['test_metrics']['hits_at_10']:.4f}")
        logger.info(f"  Test AUROC: {metrics['test_metrics']['auroc']:.4f}")
        
        logger.info(f"Predictions Generated: {len(predictions_df)}")
        logger.info(f"Validations Completed: {len(validation_df)}")
        
        # Show top predictions
        top_predictions = predictions_df.head(10)
        logger.info("Top 10 Predicted Adjuvants:")
        for _, row in top_predictions.iterrows():
            logger.info(f"  {row['head']} --[{row['relation']}]--> {row['tail']} (score: {row['score']:.4f})")
        
        logger.info("Report generation completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='β-lactam adjuvant discovery pipeline')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--steps', nargs='+', 
                       choices=['ingest', 'build_kg', 'train', 'predict', 'validate', 'report', 'all'],
                       default=['all'], help='Pipeline steps to run')
    parser.add_argument('--skip-existing', action='store_true', 
                       help='Skip steps if output files already exist')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    config['config_file'] = args.config
    
    # Setup logging
    logger = setup_logging(config)
    logger.info("Starting β-lactam adjuvant discovery pipeline")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Steps: {args.steps}")
    
    start_time = time.time()
    
    # Run pipeline steps
    success = True
    
    if 'all' in args.steps or 'ingest' in args.steps:
        if not run_data_ingestion(config, logger):
            success = False
    
    if success and ('all' in args.steps or 'build_kg' in args.steps):
        if not run_kg_assembly(config, logger):
            success = False
    
    if success and ('all' in args.steps or 'train' in args.steps):
        if not run_training(config, logger):
            success = False
    
    if success and ('all' in args.steps or 'predict' in args.steps):
        if not run_prediction(config, logger):
            success = False
    
    if success and ('all' in args.steps or 'validate' in args.steps):
        if not run_validation(config, logger):
            success = False
    
    if success and ('all' in args.steps or 'report' in args.steps):
        if not generate_report(config, logger):
            success = False
    
    # Final summary
    elapsed_time = time.time() - start_time
    if success:
        logger.info(f"🎉 Pipeline completed successfully in {elapsed_time:.2f} seconds")
        logger.info("Check the results/ directory for outputs")
    else:
        logger.error(f"❌ Pipeline failed after {elapsed_time:.2f} seconds")
        sys.exit(1)

if __name__ == "__main__":
    main()