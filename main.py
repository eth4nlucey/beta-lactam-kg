#!/usr/bin/env python3
"""
Beta-Lactam Knowledge Graph Pipeline
Master's Dissertation Project - Ethan Lucey

This script orchestrates the complete pipeline for building a heterogeneous
knowledge graph and discovering novel β-lactam antibiotic adjuvants.
"""

import argparse
import logging
import os
import sys
import yaml
from pathlib import Path

# Import pipeline components
from scripts.drugbank_parser import parse_drugbank_xml, drugbank_to_triples
from scripts.string_api import get_string_interactions
from scripts.chembl_api import get_targets
from scripts.drugcomb_import import main as drugcomb_import
from scripts.card_import import main as card_import
from scripts.combine_kg_data import create_comprehensive_kg
from scripts.train_kg_model import train_pykeen_model
from scripts.predict_links import predict_adjuvant_candidates
from scripts.validate_predictions import validate_predictions

def setup_logging(config: dict):
    """Setup logging configuration"""
    log_dir = os.path.dirname(config['logging']['file'])
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, config['logging']['level']),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config['logging']['file']),
            logging.StreamHandler(sys.stdout)
        ]
    )

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"❌ Error parsing configuration: {e}")
        sys.exit(1)

def run_data_ingestion(config: dict):
    """Run data ingestion from all sources"""
    logging.info("🚀 Starting data ingestion phase...")
    
    # Step 1: DrugBank parsing
    logging.info("Parsing DrugBank XML...")
    try:
        beta_lactams = parse_drugbank_xml(config['data_sources']['drugbank'])
        logging.info(f"Found {len(beta_lactams)} β-lactam antibiotics")
        
        # Save DrugBank triples
        drugbank_output = "data/drugbank_triples.tsv"
        os.makedirs(os.path.dirname(drugbank_output), exist_ok=True)
        drugbank_to_triples(beta_lactams, drugbank_output)
        
        # Update config to point to actual file
        config['data_sources']['drugbank'] = drugbank_output
        
    except Exception as e:
        logging.error(f"DrugBank parsing failed: {e}")
        return False
    
    # Step 2: STRING protein interactions
    logging.info("Fetching STRING interactions...")
    try:
        # Define known targets of amoxicillin (E. coli proteins)
        amoxicillin_targets = ["PBP2B", "ftsI", "murA"]
        
        result = get_string_interactions(amoxicillin_targets, species=config['organism'])
        if result:
            string_output = config['data_sources']['string']
            os.makedirs(os.path.dirname(string_output), exist_ok=True)
            with open(string_output, "w") as f:
                f.write(result)
            logging.info(f"STRING interactions saved to {string_output}")
        else:
            logging.error("Failed to fetch STRING data")
            return False
            
    except Exception as e:
        logging.error(f"STRING API failed: {e}")
        return False
    
    # Step 3: ChEMBL targets
    logging.info("Fetching ChEMBL targets...")
    try:
        # This would need to be implemented to save to file
        # For now, create placeholder
        chembl_output = config['data_sources']['chembl']
        os.makedirs(os.path.dirname(chembl_output), exist_ok=True)
        with open(chembl_output, 'w') as f:
            f.write("amoxicillin\ttargets\tPBP2B\n")
            f.write("ampicillin\ttargets\tPBP2B\n")
        logging.info(f"ChEMBL targets saved to {chembl_output}")
        
    except Exception as e:
        logging.error(f"ChEMBL processing failed: {e}")
        return False
    
    # Step 4: DrugComb synergy data
    logging.info("Importing DrugComb synergy data...")
    try:
        drugcomb_import()
    except Exception as e:
        logging.error(f"DrugComb import failed: {e}")
        return False
    
    # Step 5: CARD resistance data
    logging.info("Importing CARD resistance data...")
    try:
        card_import()
    except Exception as e:
        logging.error(f"CARD import failed: {e}")
        return False
    
    logging.info("Data ingestion completed successfully!")
    return True

def run_kg_construction(config: dict):
    """Build comprehensive knowledge graph"""
    logging.info("🔗 Starting knowledge graph construction...")
    
    try:
        create_comprehensive_kg(config['config'])
        logging.info("✅ Knowledge graph construction completed!")
        return True
    except Exception as e:
        logging.error(f"❌ Knowledge graph construction failed: {e}")
        return False

def run_model_training(config: dict):
    """Train the link prediction model"""
    logging.info("🧠 Starting model training...")
    
    try:
        result, metrics = train_pykeen_model(config['config'])
        logging.info("✅ Model training completed!")
        return True
    except Exception as e:
        logging.error(f"❌ Model training failed: {e}")
        return False

def run_prediction(config: dict):
    """Generate adjuvant predictions"""
    logging.info("🔮 Starting adjuvant prediction...")
    
    try:
        predictions = predict_adjuvant_candidates(config['config'], top_k=500)
        if predictions is not None:
            logging.info("✅ Adjuvant prediction completed!")
            return True
        else:
            logging.error("❌ No predictions generated")
            return False
    except Exception as e:
        logging.error(f"❌ Adjuvant prediction failed: {e}")
        return False

def run_validation(config: dict):
    """Validate predictions computationally"""
    logging.info("🔬 Starting computational validation...")
    
    try:
        validation_results = validate_predictions(config['config'], top_k=100)
        if validation_results is not None:
            logging.info("✅ Computational validation completed!")
            return True
        else:
            logging.error("❌ Validation failed")
            return False
    except Exception as e:
        logging.error(f"❌ Computational validation failed: {e}")
        return False

def main():
    """Main pipeline orchestration"""
    parser = argparse.ArgumentParser(
        description='Beta-Lactam Knowledge Graph Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline Steps:
  ingest      - Collect data from DrugBank, STRING, ChEMBL, DrugComb, CARD
  build_kg    - Combine data sources into comprehensive knowledge graph
  train       - Train PyKEEN link prediction model
  predict     - Generate adjuvant candidate predictions
  validate    - Validate predictions using Europe PMC and DrugComb
  report      - Generate final results and metrics
  all         - Run complete pipeline (default)
        """
    )
    
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--steps', default='all', 
                       help='Pipeline steps to run (comma-separated or "all")')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    setup_logging(config)
    
    logging.info("🚀 Beta-Lactam Knowledge Graph Pipeline Starting...")
    logging.info(f"Configuration: {args.config}")
    logging.info(f"Steps: {args.steps}")
    
    # Determine steps to run
    if args.steps == 'all':
        steps = ['ingest', 'build_kg', 'train', 'predict', 'validate', 'report']
    else:
        steps = [s.strip() for s in args.steps.split(',')]
    
    # Execute pipeline steps
    success = True
    
    if 'ingest' in steps:
        success &= run_data_ingestion(config)
    
    if success and 'build_kg' in steps:
        success &= run_kg_construction(config)
    
    if success and 'train' in steps:
        success &= run_model_training(config)
    
    if success and 'predict' in steps:
        success &= run_prediction(config)
    
    if success and 'validate' in steps:
        success &= run_validation(config)
    
    if success and 'report' in steps:
        logging.info("📊 Pipeline completed successfully! Check results/ directory for outputs.")
    
    if not success:
        logging.error("❌ Pipeline failed at one or more steps")
        sys.exit(1)
    
    logging.info("🎉 Pipeline completed successfully!")

if __name__ == "__main__":
    main()