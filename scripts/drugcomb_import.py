#!/usr/bin/env python3
"""
Import real drug-drug synergy data from DrugComb database.
"""

import argparse
import pandas as pd
import requests
import json
import time
from pathlib import Path
import yaml
from typing import Dict, List, Optional
import hashlib
import os

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_cache_path(cache_dir: str, query_hash: str) -> str:
    """Get cache file path for a query."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir / f"{query_hash}.json")

def cache_response(cache_path: str, data: dict):
    """Cache API response to file."""
    with open(cache_path, 'w') as f:
        json.dump(data, f, indent=2)

def load_cached_response(cache_path: str) -> Optional[dict]:
    """Load cached API response if it exists."""
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except:
            pass
    return None

def fetch_drugcomb_synergy(drug_name: str, cache_dir: str, offline: bool = False) -> Optional[dict]:
    """
    Fetch drug synergy data from DrugComb API.
    """
    # Create query hash for caching
    query_hash = hashlib.md5(f"drugcomb_synergy_{drug_name}".encode()).hexdigest()
    cache_path = get_cache_path(cache_dir, query_hash)
    
    # Check cache first
    if not offline:
        cached = load_cached_response(cache_path)
        if cached:
            print(f"  Using cached data for {drug_name}")
            return cached
    
    if offline:
        print(f"  Offline mode: skipping {drug_name}")
        return None
    
    # DrugComb API endpoint (using their public API)
    # Note: This is a simplified version - in production you'd use their full API
    url = "https://drugcomb.fimm.fi/api/v1/synergies"
    
    # Search parameters
    params = {
        'drug_name': drug_name,
        'limit': 100
    }
    
    try:
        print(f"  Fetching DrugComb data for {drug_name}...")
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            # Cache the response
            cache_response(cache_path, data)
            
            return data
        else:
            print(f"  API error {response.status_code} for {drug_name}")
            return None
            
    except Exception as e:
        print(f"  Error fetching {drug_name}: {e}")
        return None

def normalize_drug_name(drug_name: str) -> str:
    """Normalize drug name for matching."""
    return drug_name.lower().strip()

def map_drug_to_drugbank(drug_name: str, drugbank_mapping: Dict[str, str]) -> Optional[str]:
    """Map drug name to DrugBank ID using existing mapping."""
    normalized = normalize_drug_name(drug_name)
    
    # Direct match
    if normalized in drugbank_mapping:
        return drugbank_mapping[normalized]
    
    # Partial matches
    for db_name, db_id in drugbank_mapping.items():
        if normalized in db_name or db_name in normalized:
            return db_id
    
    return None

def process_drugcomb_data(drugcomb_data: dict, drugbank_mapping: Dict[str, str]) -> List[Dict]:
    """
    Process DrugComb API response and extract synergy relationships.
    """
    synergies = []
    
    if 'data' not in drugcomb_data:
        return synergies
    
    for item in drugcomb_data['data']:
        try:
            # Extract drug names and synergy metrics
            drug1_name = item.get('drug1_name', '')
            drug2_name = item.get('drug2_name', '')
            synergy_score = item.get('synergy_score', 0.0)
            metric_type = item.get('metric_type', 'unknown')
            cell_line = item.get('cell_line', 'unknown')
            record_id = item.get('id', '')
            
            # Map to DrugBank IDs
            drug1_id = map_drug_to_drugbank(drug1_name, drugbank_mapping)
            drug2_id = map_drug_to_drugbank(drug2_name, drugbank_mapping)
            
            if drug1_id and drug2_id:
                synergy = {
                    'head_id': drug1_id,
                    'relation': 'synergizes_with',
                    'tail_id': drug2_id,
                    'weight': synergy_score,
                    'evidence_source': 'DrugComb',
                    'evidence_id': f"DC:{record_id}",
                    'metric_type': metric_type,
                    'cell_line': cell_line
                }
                synergies.append(synergy)
                
        except Exception as e:
            print(f"  Error processing DrugComb item: {e}")
            continue
    
    return synergies

def main():
    parser = argparse.ArgumentParser(description='Import real DrugComb synergy data')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--offline', action='store_true', help='Use cached data only')
    parser.add_argument('--output', help='Output file path')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Get paths
    cache_dir = config['paths']['cache_dir']
    output_path = args.output or config['data_sources']['drugcomb']
    
    # Load existing DrugBank mapping
    drugbank_mapping_path = 'data/drugbank_drugs.tsv'
    if not Path(drugbank_mapping_path).exists():
        print(f"❌ DrugBank mapping not found: {drugbank_mapping_path}")
        print("Please run DrugBank parser first")
        return
    
    drugbank_df = pd.read_csv(drugbank_mapping_path, sep='\t')
    drugbank_mapping = dict(zip(drugbank_df['name'].str.lower(), drugbank_df['drugbank_id']))
    
    print(f"✅ Loaded {len(drugbank_mapping)} DrugBank drug mappings")
    
    # Define β-lactam antibiotics to search for
    beta_lactams = [
        "amoxicillin", "ampicillin", "ceftriaxone", "ceftazidime", 
        "meropenem", "piperacillin", "cephalexin", "cefazolin",
        "cefuroxime", "cefotaxime", "cefepime", "ertapenem",
        "clavulanic acid", "sulbactam", "tazobactam", "avibactam"
    ]
    
    print(f"🔍 Searching DrugComb for {len(beta_lactams)} β-lactam antibiotics...")
    
    all_synergies = []
    
    for drug in beta_lactams:
        print(f"Processing {drug}...")
        
        # Fetch DrugComb data
        drugcomb_data = fetch_drugcomb_synergy(drug, cache_dir, args.offline)
        
        if drugcomb_data:
            # Process the data
            synergies = process_drugcomb_data(drugcomb_data, drugbank_mapping)
            all_synergies.extend(synergies)
            print(f"  Found {len(synergies)} synergy relationships")
        else:
            print(f"  No data found")
        
        # Rate limiting
        if not args.offline:
            time.sleep(1)
    
    # Create output DataFrame
    if all_synergies:
        df = pd.DataFrame(all_synergies)
        
        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to TSV
        df.to_csv(output_path, sep='\t', index=False)
        
        print(f"\n✅ Saved {len(df)} DrugComb synergy relationships to {output_path}")
        print(f"   Unique drugs: {df['head_id'].nunique()}")
        print(f"   Synergy relationships: {len(df)}")
        
        # Show sample
        print("\n📊 Sample synergies:")
        for _, row in df.head(5).iterrows():
            print(f"   {row['head_id']} synergizes_with {row['tail_id']} (score: {row['weight']:.3f})")
    
    else:
        print("\n⚠️  No DrugComb synergies found")
        
        # Create empty file with correct schema
        empty_df = pd.DataFrame(columns=[
            'head_id', 'relation', 'tail_id', 'weight', 
            'evidence_source', 'evidence_id', 'metric_type', 'cell_line'
        ])
        
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        empty_df.to_csv(output_path, sep='\t', index=False)
        print(f"Created empty file with schema: {output_path}")

if __name__ == "__main__":
    main()
