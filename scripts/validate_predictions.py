import pandas as pd
import requests
import yaml
import json
import os
from typing import List, Tuple, Dict
import time

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def search_europe_pmc(drug1: str, drug2: str, max_results: int = 10) -> Dict:
    """Search Europe PMC for literature evidence of drug combination"""
    
    # Construct search query
    query = f'"{drug1}" AND "{drug2}" AND (synergy OR potentiation OR adjuvant OR combination)'
    
    # Europe PMC API endpoint
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    params = {
        "query": query,
        "format": "json",
        "resultType": "core",
        "pageSize": max_results
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            
            # Extract results
            results = {
                'hit_count': data.get('hitCount', 0),
                'pmids': [],
                'titles': [],
                'abstracts': []
            }
            
            for article in data.get('resultList', {}).get('result', []):
                pmid = article.get('pmid', '')
                if pmid:
                    results['pmids'].append(pmid)
                    results['titles'].append(article.get('title', ''))
                    results['abstracts'].append(article.get('abstractText', ''))
            
            return results
        else:
                    print(f"Europe PMC API error: {response.status_code}")
        return {'hit_count': 0, 'pmids': [], 'titles': [], 'abstracts': []}
            
    except Exception as e:
        print(f"Europe PMC search failed: {e}")
        return {'hit_count': 0, 'pmids': [], 'titles': [], 'abstracts': []}

def check_drugcomb_synergy(drug1: str, drug2: str) -> Dict:
    """Check if drug combination exists in DrugComb data"""
    
    # Load DrugComb data
    config = load_config()
    drugcomb_path = config['data_sources']['drugcomb']
    
    if not os.path.exists(drugcomb_path):
        return {'found': False, 'synergy_score': None, 'cell_line': None}
    
    try:
        drugcomb_df = pd.read_csv(drugcomb_path, sep='\t', names=['head', 'relation', 'tail'])
        
        # Look for synergy relationship
        synergy_mask = (
            ((drugcomb_df['head'] == drug1) & (drugcomb_df['tail'] == drug2)) |
            ((drugcomb_df['head'] == drug2) & (drugcomb_df['tail'] == drug1))
        ) & (drugcomb_df['relation'] == 'synergizes_with')
        
        if synergy_mask.any():
            # Get synergy score if available
            score_mask = (
                ((drugcomb_df['head'] == drug1) | (drugcomb_df['head'] == drug2)) &
                (drugcomb_df['relation'] == 'synergy_score')
            )
            
            synergy_score = None
            if score_mask.any():
                synergy_score = drugcomb_df.loc[score_mask, 'tail'].iloc[0]
            
            # Get cell line info
            cell_line = None
            cell_mask = (
                ((drugcomb_df['head'] == drug1) | (drugcomb_df['head'] == drug2)) &
                (drugcomb_df['relation'] == 'tested_in')
            )
            if cell_mask.any():
                cell_line = drugcomb_df.loc[cell_mask, 'tail'].iloc[0]
            
            return {
                'found': True,
                'synergy_score': synergy_score,
                'cell_line': cell_line
            }
        else:
            return {'found': False, 'synergy_score': None, 'cell_line': None}
            
    except Exception as e:
        print(f"⚠️ DrugComb check failed: {e}")
        return {'found': False, 'synergy_score': None, 'cell_line': None}

def validate_predictions(config_path: str = "config.yaml", top_k: int = 100):
    """Validate top predictions using Europe PMC and DrugComb"""
    
    config = load_config(config_path)
    
    # Load predictions
    predictions_path = config['outputs']['predictions']
    if not os.path.exists(predictions_path):
        print(f"❌ Predictions file not found: {predictions_path}")
        return
    
    predictions_df = pd.read_csv(predictions_path, sep='\t')
    print(f"📊 Validating top {min(top_k, len(predictions_df))} predictions...")
    
    # Initialize validation results
    validation_results = []
    
    # Process top predictions
    for idx, row in predictions_df.head(top_k).iterrows():
        drug = row['head']
        adjuvant = row['tail']
        score = row['score']
        
        print(f"🔍 Validating: {drug} + {adjuvant} (score: {score:.4f})")
        
        # Europe PMC validation
        epmc_results = search_europe_pmc(drug, adjuvant)
        
        # DrugComb validation
        drugcomb_results = check_drugcomb_synergy(drug, adjuvant)
        
        # Store results
        validation_result = {
            'drug': drug,
            'adjuvant': adjuvant,
            'prediction_score': score,
            'epmc_hit_count': epmc_results['hit_count'],
            'epmc_top_pmids': ';'.join(epmc_results['pmids'][:3]),  # Top 3 PMIDs
            'drugcomb_found': drugcomb_results['found'],
            'drugcomb_synergy_score': drugcomb_results['synergy_score'],
            'drugcomb_cell_line': drugcomb_results['cell_line']
        }
        
        validation_results.append(validation_result)
        
        # Rate limiting for API calls
        time.sleep(0.5)
    
    # Convert to DataFrame
    validation_df = pd.DataFrame(validation_results)
    
    # Save validation results
    os.makedirs(os.path.dirname(config['outputs']['validation']['summary']), exist_ok=True)
    
    # Save detailed results
    validation_df.to_csv(config['outputs']['validation']['summary'], sep='\t', index=False)
    print(f"✅ Validation summary saved to {config['outputs']['validation']['summary']}")
    
    # Print summary statistics
    print("\n📊 Validation Summary:")
    print(f"  Total predictions validated: {len(validation_df)}")
    print(f"  Europe PMC hits: {validation_df['epmc_hit_count'].sum()}")
    print(f"  DrugComb found: {validation_df['drugcomb_found'].sum()}")
    
    # Top validated predictions
    top_validated = validation_df[
        (validation_df['epmc_hit_count'] > 0) | 
        (validation_df['drugcomb_found'])
    ].sort_values('prediction_score', ascending=False)
    
    if len(top_validated) > 0:
        print(f"\n🏆 Top validated predictions:")
        for _, row in top_validated.head(5).iterrows():
            print(f"  {row['drug']} + {row['adjuvant']}: Score={row['prediction_score']:.4f}, "
                  f"EPMC={row['epmc_hit_count']}, DrugComb={row['drugcomb_found']}")
    
    return validation_df

def main():
    """Main function to run validation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate predictions using Europe PMC and DrugComb')
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--top_k', type=int, default=100, help='Number of top predictions to validate')
    
    args = parser.parse_args()
    
    print("🔬 Starting computational validation of predictions...")
    validation_results = validate_predictions(args.config, args.top_k)
    print("✅ Validation completed successfully!")

if __name__ == "__main__":
    main()
