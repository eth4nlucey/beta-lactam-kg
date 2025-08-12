import pandas as pd
import yaml
import os
from typing import List

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def combine_knowledge_sources(config: dict, output_path: str):
    """Combine all knowledge sources into one comprehensive knowledge graph"""
    
    all_dfs = []
    
    # Read DrugBank triples
    if os.path.exists(config['data_sources']['drugbank']):
        drugbank_df = pd.read_csv(config['data_sources']['drugbank'], sep='\t', names=['head', 'relation', 'tail'])
        print(f"DrugBank: {len(drugbank_df)} triples")
        all_dfs.append(drugbank_df)
    
    # Read STRING triples  
    if os.path.exists(config['data_sources']['string']):
        string_df = pd.read_csv(config['data_sources']['string'], sep='\t', names=['head', 'relation', 'tail'])
        print(f"STRING: {len(string_df)} triples")
        all_dfs.append(string_df)
    
    # Read ChEMBL triples
    if os.path.exists(config['data_sources']['chembl']):
        chembl_df = pd.read_csv(config['data_sources']['chembl'], sep='\t', names=['head', 'relation', 'tail'])
        print(f"ChEMBL: {len(chembl_df)} triples")
        all_dfs.append(chembl_df)
    
    # Read DrugComb triples
    if os.path.exists(config['data_sources']['drugcomb']):
        drugcomb_df = pd.read_csv(config['data_sources']['drugcomb'], sep='\t', names=['head', 'relation', 'tail'])
        print(f"DrugComb: {len(drugcomb_df)} triples")
        all_dfs.append(drugcomb_df)
    
    # Read CARD resistance triples
    if os.path.exists(config['data_sources']['card']):
        card_df = pd.read_csv(config['data_sources']['card'], sep='\t', names=['head', 'relation', 'tail'])
        print(f"CARD: {len(card_df)} triples")
        all_dfs.append(card_df)
    
    if not all_dfs:
        print("No data sources found!")
        return
    
    # Combine all datasets
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Remove duplicates
    combined_df = combined_df.drop_duplicates()
    print(f"Combined: {len(combined_df)} unique triples")
    
    # Save combined knowledge graph
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined_df.to_csv(output_path, sep='\t', index=False, header=False)
    print(f"Saved combined KG to {output_path}")
    
    # Print relation statistics
    relation_counts = combined_df['relation'].value_counts()
    print("\nRelation distribution:")
    for relation, count in relation_counts.items():
        print(f"  {relation}: {count}")

def create_comprehensive_kg(config_path: str = "config.yaml"):
    """Create a comprehensive knowledge graph from all sources"""
    config = load_config(config_path)
    
    print("Creating comprehensive knowledge graph...")
    combine_knowledge_sources(config, config['kg']['edges'])
    
    # Also save nodes for reference
    nodes_path = config['kg']['nodes']
    os.makedirs(os.path.dirname(nodes_path), exist_ok=True)
    
    # Extract unique entities
    combined_df = pd.read_csv(config['kg']['edges'], sep='\t', names=['head', 'relation', 'tail'])
    all_entities = pd.concat([combined_df['head'], combined_df['tail']]).unique()
    
    nodes_df = pd.DataFrame({'entity': all_entities})
    nodes_df.to_csv(nodes_path, sep='\t', index=False, header=False)
    print(f"Saved {len(nodes_df)} unique entities to {nodes_path}")

if __name__ == "__main__":
    create_comprehensive_kg()