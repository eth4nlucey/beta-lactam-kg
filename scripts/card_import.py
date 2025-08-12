#!/usr/bin/env python3
"""
Import antimicrobial resistance data from CARD (Comprehensive Antibiotic Resistance Database).
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

def fetch_card_resistance(aro_id: str, cache_dir: str, offline: bool = False) -> Optional[dict]:
    """
    Fetch resistance data from CARD API.
    """
    # Create query hash for caching
    query_hash = hashlib.md5(f"card_resistance_{aro_id}".encode()).hexdigest()
    cache_path = get_cache_path(cache_dir, query_hash)
    
    # Check cache first
    if not offline:
        cached = load_cached_response(cache_path)
        if cached:
            return cached
    
    if offline:
        return None
    
    # CARD API endpoint
    url = f"https://card.mcmaster.ca/api/aro/{aro_id}"
    
    try:
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            # Cache the response
            cache_response(cache_path, data)
            
            return data
        else:
            print(f"  API error {response.status_code} for ARO:{aro_id}")
            return None
            
    except Exception as e:
        print(f"  Error fetching ARO:{aro_id}: {e}")
        return None

def fetch_card_aro_list(cache_dir: str, offline: bool = False) -> Optional[dict]:
    """
    Fetch list of ARO IDs from CARD.
    """
    query_hash = hashlib.md5("card_aro_list".encode()).hexdigest()
    cache_path = get_cache_path(cache_dir, query_hash)
    
    if not offline:
        cached = load_cached_response(cache_path)
        if cached:
            return cached
    
    if offline:
        return None
    
    # CARD ARO list endpoint
    url = "https://card.mcmaster.ca/api/aro"
    
    try:
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            cache_response(cache_path, data)
            return data
        else:
            print(f"  API error {response.status_code} for ARO list")
            return None
            
    except Exception as e:
        print(f"  Error fetching ARO list: {e}")
        return None

def map_drug_to_drugbank(drug_name: str, drugbank_mapping: Dict[str, str]) -> Optional[str]:
    """Map drug name to DrugBank ID using existing mapping."""
    normalized = drug_name.lower().strip()
    
    # Direct match
    if normalized in drugbank_mapping:
        return drugbank_mapping[normalized]
    
    # Partial matches
    for db_name, db_id in drugbank_mapping.items():
        if normalized in db_name or db_name in normalized:
            return db_id
    
    return None

def process_card_data(card_data: dict, drugbank_mapping: Dict[str, str]) -> List[Dict]:
    """
    Process CARD API response and extract resistance relationships.
    """
    resistances = []
    
    try:
        # Extract resistance information
        aro_id = card_data.get('aro_id', '')
        aro_name = card_data.get('aro_name', '')
        aro_description = card_data.get('aro_description', '')
        
        # Get drug resistance information
        drug_resistances = card_data.get('drug_resistances', [])
        
        for drug_resistance in drug_resistances:
            drug_name = drug_resistance.get('drug_name', '')
            resistance_type = drug_resistance.get('resistance_type', 'resistance')
            
            # Map drug to DrugBank ID
            drugbank_id = map_drug_to_drugbank(drug_name, drugbank_mapping)
            
            if drugbank_id:
                resistance = {
                    'head_id': drugbank_id,
                    'relation': 'resisted_by',
                    'tail_id': f"ARO:{aro_id}",
                    'weight': 1.0,
                    'evidence_source': 'CARD',
                    'evidence_id': f"ARO:{aro_id}",
                    'resistance_type': resistance_type,
                    'aro_name': aro_name,
                    'aro_description': aro_description
                }
                resistances.append(resistance)
        
        # Get protein information if available
        proteins = card_data.get('proteins', [])
        for protein in proteins:
            uniprot_id = protein.get('uniprot_id', '')
            gene_name = protein.get('gene_name', '')
            
            if uniprot_id:
                # Add protein-resistance relationship
                protein_resistance = {
                    'head_id': f"ARO:{aro_id}",
                    'relation': 'encodes',
                    'tail_id': uniprot_id,
                    'weight': 1.0,
                    'evidence_source': 'CARD',
                    'evidence_id': f"ARO:{aro_id}",
                    'gene_name': gene_name
                }
                resistances.append(protein_resistance)
                
    except Exception as e:
        print(f"  Error processing CARD data: {e}")
    
    return resistances

def main():
    parser = argparse.ArgumentParser(description='Import CARD antimicrobial resistance data')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--offline', action='store_true', help='Use cached data only')
    parser.add_argument('--output', help='Output file path')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Get paths
    cache_dir = config['paths']['cache_dir']
    output_path = args.output or config['data_sources']['card']
    
    # Load existing DrugBank mapping
    drugbank_mapping_path = 'data/drugbank_drugs.tsv'
    if not Path(drugbank_mapping_path).exists():
        print(f"❌ DrugBank mapping not found: {drugbank_mapping_path}")
        print("Please run DrugBank parser first")
        return
    
    drugbank_df = pd.read_csv(drugbank_mapping_path, sep='\t')
    drugbank_mapping = dict(zip(drugbank_df['name'].str.lower(), drugbank_df['drugbank_id']))
    
    print(f"✅ Loaded {len(drugbank_mapping)} DrugBank drug mappings")
    
    # Define key β-lactamase families to search for
    beta_lactamase_families = [
        "TEM", "SHV", "CTX-M", "KPC", "NDM", "OXA", 
        "VIM", "IMP", "GES", "PER", "VEB", "CMY"
    ]
    
    print(f"🔍 Searching CARD for {len(beta_lactamase_families)} β-lactamase families...")
    
    all_resistances = []
    
    # First, get ARO list to find relevant entries
    aro_list = fetch_card_aro_list(cache_dir, args.offline)
    
    if aro_list and 'aro' in aro_list:
        aro_entries = aro_list['aro']
        
        for aro_entry in aro_entries:
            aro_id = aro_entry.get('aro_id', '')
            aro_name = aro_entry.get('aro_name', '')
            
            # Check if this ARO is relevant to β-lactam resistance
            is_relevant = any(family.lower() in aro_name.lower() for family in beta_lactamase_families)
            
            if is_relevant:
                print(f"Processing ARO:{aro_id} - {aro_name}")
                
                # Fetch detailed resistance data
                card_data = fetch_card_resistance(aro_id, cache_dir, args.offline)
                
                if card_data:
                    # Process the data
                    resistances = process_card_data(card_data, drugbank_mapping)
                    all_resistances.extend(resistances)
                    print(f"  Found {len(resistances)} resistance relationships")
                
                # Rate limiting
                if not args.offline:
                    time.sleep(1)
    
    # Create output DataFrame
    if all_resistances:
        df = pd.DataFrame(all_resistances)
        
        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to TSV
        df.to_csv(output_path, sep='\t', index=False)
        
        print(f"\n✅ Saved {len(df)} CARD resistance relationships to {output_path}")
        print(f"   Unique drugs: {df[df['relation'] == 'resisted_by']['head_id'].nunique()}")
        print(f"   Resistance mechanisms: {df[df['relation'] == 'resisted_by']['tail_id'].nunique()}")
        
        # Show sample
        print("\n📊 Sample resistances:")
        resistance_df = df[df['relation'] == 'resisted_by'].head(5)
        for _, row in resistance_df.iterrows():
            print(f"   {row['head_id']} resisted_by {row['tail_id']} ({row['aro_name']})")
    
    else:
        print("\n⚠️  No CARD resistances found")
        
        # Create empty file with correct schema
        empty_df = pd.DataFrame(columns=[
            'head_id', 'relation', 'tail_id', 'weight', 
            'evidence_source', 'evidence_id', 'resistance_type', 'aro_name', 'aro_description'
        ])
        
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        empty_df.to_csv(output_path, sep='\t', index=False)
        print(f"Created empty file with schema: {output_path}")

if __name__ == "__main__":
    main()
