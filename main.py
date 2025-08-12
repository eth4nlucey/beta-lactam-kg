#!/usr/bin/env python3
"""
β-lactam adjuvant discovery pipeline - Main orchestrator.
"""

import argparse
import subprocess
import sys
import time
import yaml
from pathlib import Path
import logging

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_logging(config: dict) -> logging.Logger:
    """Setup logging configuration."""
    log_level = getattr(logging, config['logging']['level'])
    log_file = config['logging']['file']
    
    # Ensure log directory exists
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def run_data_ingestion(config: dict, logger: logging.Logger) -> bool:
    """Run data ingestion steps."""
    logger.info("Starting data ingestion...")
    
    try:
        # Parse DrugBank XML
        logger.info("Parsing DrugBank XML...")
        result = subprocess.run([
            'python', 'scripts/drugbank_parser.py',
            '--xml', config['paths']['drugbank_xml'],
            '--out-drugs', 'data/drugbank_drugs.tsv',
            '--out-proteins', 'data/drugbank_proteins.tsv',
            '--out-edges', 'data/drugbank_edges.tsv'
        ], capture_output=True, text=True, check=True)
        logger.info("DrugBank parsing completed")
        
        # Fetch STRING interactions
        logger.info("Fetching STRING interactions...")
        result = subprocess.run([
            'python', 'scripts/string_api.py',
            '--config', config['config_file']
        ], capture_output=True, text=True, check=True)
        logger.info("STRING API completed")
        
        # Fetch ChEMBL targets
        logger.info("Fetching ChEMBL targets...")
        result = subprocess.run([
            'python', 'scripts/chembl_api.py',
            '--config', config['config_file']
        ], capture_output=True, text=True, check=True)
        logger.info("ChEMBL API completed")
        
        # Import DrugComb synergies
        logger.info("Importing DrugComb synergies...")
        result = subprocess.run([
            'python', 'scripts/drugcomb_import.py',
            '--config', config['config_file']
        ], capture_output=True, text=True, check=True)
        logger.info("DrugComb import completed")
        
        # Import CARD resistance
        logger.info("Importing CARD resistance data...")
        result = subprocess.run([
            'python', 'scripts/card_import.py',
            '--config', config['config_file']
        ], capture_output=True, text=True, check=True)
        logger.info("CARD import completed")
        
        logger.info("Data ingestion completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Data ingestion failed: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Data ingestion failed with error: {e}")
        return False

def run_kg_assembly(config: dict, logger: logging.Logger) -> bool:
    """Run knowledge graph assembly."""
    logger.info("Starting knowledge graph assembly...")
    
    try:
        result = subprocess.run([
            'python', 'scripts/assemble_kg.py',
            '--config', config['config_file']
        ], capture_output=True, text=True, check=True)
        
        logger.info("Knowledge graph assembly completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"KG assembly failed: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"KG assembly failed with error: {e}")
        return False

def run_training(config: dict, logger: logging.Logger) -> bool:
    """Run model training and evaluation."""
    logger.info("Starting model training...")
    
    try:
        result = subprocess.run([
            'python', 'scripts/train_mini_transe.py',
            '--config', config['config_file']
        ], capture_output=True, text=True, check=True)
        
        logger.info("Model training completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        return False

def run_prediction(config: dict, logger: logging.Logger) -> bool:
    """Run adjuvant prediction."""
    logger.info("Starting adjuvant prediction...")
    
    try:
        result = subprocess.run([
            'python', 'scripts/predict_links.py',
            '--config', config['config_file']
        ], capture_output=True, text=True, check=True)
        
        logger.info("Adjuvant prediction completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Prediction failed: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Prediction failed with error: {e}")
        return False

def run_validation(config: dict, logger: logging.Logger) -> bool:
    """Run computational validation."""
    logger.info("Starting computational validation...")
    
    try:
        result = subprocess.run([
            'python', 'scripts/validate_predictions.py',
            '--config', config['config_file']
        ], capture_output=True, text=True, check=True)
        
        logger.info("Computational validation completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Validation failed: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        return False

def generate_report(config: dict, logger: logging.Logger) -> bool:
    """Generate final report."""
    logger.info("Generating final report...")
    
    try:
        # Check artifacts integrity
        result = subprocess.run([
            'python', 'scripts/check_artifacts.py',
            '--config', config['config_file']
        ], capture_output=True, text=True, check=True)
        
        logger.info("Artifact integrity check completed")
        
        # Generate summary report
        report_path = Path(config['paths']['results_dir']) / 'REPORT.md'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write("# β-Lactam Adjuvant Discovery Pipeline Report\n\n")
            f.write("## Pipeline Summary\n\n")
            f.write("This report summarizes the execution of the β-lactam adjuvant discovery pipeline.\n\n")
            f.write("## Artifacts Generated\n\n")
            f.write("- Knowledge Graph: Drug-protein, drug-drug, and resistance relationships\n")
            f.write("- Trained TransE Model: Link prediction for adjuvant discovery\n")
            f.write("- Adjuvant Predictions: Ranked candidates for β-lactam combinations\n")
            f.write("- Computational Validation: Literature and database evidence\n\n")
            f.write("## Next Steps\n\n")
            f.write("1. Review top adjuvant predictions\n")
            f.write("2. Validate promising candidates experimentally\n")
            f.write("3. Extend knowledge graph with additional data sources\n")
            f.write("4. Optimize model hyperparameters\n\n")
            f.write("## Configuration\n\n")
            f.write(f"- Config file: {config['config_file']}\n")
            f.write(f"- Results directory: {config['paths']['results_dir']}\n")
            f.write(f"- Knowledge graph: {config['paths']['kg_dir']}\n")
        
        logger.info(f"Report generated: {report_path}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Report generation failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Report generation failed with error: {e}")
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
    
    config = load_config(args.config)
    config['config_file'] = args.config
    
    logger = setup_logging(config)
    logger.info("Starting β-lactam adjuvant discovery pipeline")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Steps: {args.steps}")
    
    start_time = time.time()
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
    
    elapsed_time = time.time() - start_time
    if success:
        logger.info(f"Pipeline completed successfully in {elapsed_time:.2f} seconds")
        logger.info("Check the results/ directory for outputs")
    else:
        logger.error(f"Pipeline failed after {elapsed_time:.2f} seconds")
        sys.exit(1)

if __name__ == "__main__":
    main()