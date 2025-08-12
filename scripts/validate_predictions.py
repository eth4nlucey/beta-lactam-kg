#!/usr/bin/env python3
"""
Validate predicted adjuvant candidates against Europe PMC literature and DrugComb database.
"""

import argparse
import pandas as pd
import requests
import time
import json
from pathlib import Path
import yaml
from typing import Dict, List, Tuple

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def search_europe_pmc(drug: str, adjuvant: str) -> Dict:
    """
    Search Europe PMC for literature evidence of drug-adjuvant combinations.
    """
    # Create search query for synergy/adjuvant effects
    query = f'"{drug}" AND "{adjuvant}" AND (synergy OR potentiation OR adjuvant OR "drug combination")'
    
    # Europe PMC REST API
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    params = {
        'query': query,
        'format': 'json',
        'pageSize': 25,
        'resultType': 'core'
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        hit_count = data.get('hitCount', 0)
        
        # Extract PMIDs from results
        pmids = []
        if 'resultList' in data and 'result' in data['resultList']:
            for result in data['resultList']['result']:
                if 'pmid' in result:
                    pmids.append(str(result['pmid']))
        
        return {
            'hit_count': hit_count,
            'pmids': pmids[:5],  # Top 5 PMIDs
            'query': query
        }
        
    except Exception as e:
        print(f"Error searching Europe PMC for {drug}-{adjuvant}: {e}")
        return {
            'hit_count': 0,
            'pmids': [],
            'query': query,
            'error': str(e)
        }

def check_drugcomb_synergy(drug: str, adjuvant: str) -> Dict:
    """
    Check DrugComb database for existing synergy data.
    For now, we'll use our sample data, but this can be extended to use the real DrugComb API.
    """
    # Load our DrugComb sample data
    try:
        drugcomb_data = pd.read_csv('data/drugcomb_synergy.tsv', sep='\t')
        
        # Check if the drug-adjuvant pair exists
        # Note: We need to handle different naming conventions
        found = False
        synergy_score = None
        cell_line = None
        
        # Check both directions (drug1-drug2 and drug2-drug1)
        for _, row in drugcomb_data.iterrows():
            if ((row['drug1'].lower() == drug.lower() and row['drug2'].lower() == adjuvant.lower()) or
                (row['drug2'].lower() == drug.lower() and row['drug1'].lower() == adjuvant.lower())):
                found = True
                synergy_score = row['synergy_score']
                cell_line = row['cell_line']
                break
        
        return {
            'found': found,
            'synergy_score': synergy_score,
            'cell_line': cell_line
        }
        
    except Exception as e:
        print(f"Error checking DrugComb for {drug}-{adjuvant}: {e}")
        return {
            'found': False,
            'synergy_score': None,
            'cell_line': None,
            'error': str(e)
        }

def validate_predictions(config: dict, top_k: int = 100) -> pd.DataFrame:
    """
    Validate top predicted adjuvant candidates against literature and databases.
    """
    # Load predictions
    predictions_path = config['outputs']['predictions']
    if not Path(predictions_path).exists():
        print(f"Predictions file not found: {predictions_path}")
        return pd.DataFrame()
    
    predictions_df = pd.read_csv(predictions_path, sep='\t')
    print(f"Loaded {len(predictions_df)} predictions")
    
    # Take top K predictions
    top_predictions = predictions_df.head(top_k)
    
    validation_results = []
    
    print(f"Validating top {len(top_predictions)} predictions...")
    
    for idx, row in top_predictions.iterrows():
        drug = row['head']
        adjuvant = row['tail']
        score = row['score']
        
        print(f"Validating {idx+1}/{len(top_predictions)}: {drug} + {adjuvant}")
        
        # Search Europe PMC
        epmc_results = search_europe_pmc(drug, adjuvant)
        
        # Check DrugComb
        drugcomb_results = check_drugcomb_synergy(drug, adjuvant)
        
        # Compile validation result
        validation_result = {
            'drug': drug,
            'adjuvant': adjuvant,
            'prediction_score': score,
            'epmc_hit_count': epmc_results['hit_count'],
            'epmc_top_pmids': ';'.join(epmc_results['pmids'][:3]),
            'drugcomb_found': drugcomb_results['found'],
            'drugcomb_synergy_score': drugcomb_results['synergy_score'],
            'drugcomb_cell_line': drugcomb_results['cell_line']
        }
        
        validation_results.append(validation_result)
        
        # Rate limiting for Europe PMC
        time.sleep(0.5)
    
    # Create validation summary
    validation_df = pd.DataFrame(validation_results)
    
    # Save validation results
    output_dir = Path(config['outputs']['validation']['summary']).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    validation_df.to_csv(config['outputs']['validation']['summary'], sep='\t', index=False)
    
    print(f"Validation completed. Results saved to {config['outputs']['validation']['summary']}")
    
    # Print summary statistics
    print("\nValidation Summary:")
    print(f"Total predictions validated: {len(validation_df)}")
    print(f"Europe PMC hits: {validation_df['epmc_hit_count'].sum()}")
    print(f"DrugComb matches: {validation_df['drugcomb_found'].sum()}")
    
    # Show top hits
    print("\nTop 5 literature-supported predictions:")
    top_hits = validation_df.nlargest(5, 'epmc_hit_count')
    for _, hit in top_hits.iterrows():
        print(f"  {hit['drug']} + {hit['adjuvant']}: {hit['epmc_hit_count']} PMIDs")
    
    return validation_df

def main():
    parser = argparse.ArgumentParser(description='Validate predicted adjuvant candidates')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--top-k', type=int, default=100, help='Number of top predictions to validate')
    parser.add_argument('--output', help='Output file path (optional)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override output path if specified
    if args.output:
        config['outputs']['validation']['summary'] = args.output
    
    # Run validation
    validation_results = validate_predictions(config, args.top_k)
    
    if not validation_results.empty:
        print(f"\nValidation completed successfully!")
        print(f"Results saved to: {config['outputs']['validation']['summary']}")
    else:
        print("Validation failed - no results generated")

if __name__ == "__main__":
    main()
