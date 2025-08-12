#!/usr/bin/env python3
"""
Knowledge Graph Assembly Script
Combines all data sources into a unified, deduplicated knowledge graph
"""

import pandas as pd
import os
import yaml
from typing import Dict, Set, List, Tuple
import argparse

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_data_sources(config: dict) -> Dict[str, pd.DataFrame]:
    """Load all data sources into DataFrames"""
    
    data_sources = {}
    
    # Load DrugBank data
    drugbank_edges = config['data_sources']['drugbank_edges']
    if os.path.exists(drugbank_edges):
        print(f"Loading DrugBank edges: {drugbank_edges}")
        df = pd.read_csv(drugbank_edges, sep='\t')
        if not df.empty:
            data_sources['drugbank'] = df
            print(f"  Loaded {len(df)} DrugBank edges")
    
    # Load STRING interactions
    string_edges = config['data_sources']['string']
    if os.path.exists(string_edges):
        print(f"Loading STRING interactions: {string_edges}")
        df = pd.read_csv(string_edges, sep='\t')
        if not df.empty:
            data_sources['string'] = df
            print(f"  Loaded {len(df)} STRING edges")
    
    # Load ChEMBL targets
    chembl_edges = config['data_sources']['chembl']
    if os.path.exists(chembl_edges):
        print(f"Loading ChEMBL targets: {chembl_edges}")
        df = pd.read_csv(chembl_edges, sep='\t')
        if not df.empty:
            data_sources['chembl'] = df
            print(f"  Loaded {len(df)} ChEMBL edges")
    
    # Load DrugComb synergy data
    drugcomb_edges = config['data_sources']['drugcomb']
    if os.path.exists(drugcomb_edges):
        print(f"Loading DrugComb synergy: {drugcomb_edges}")
        df = pd.read_csv(drugcomb_edges, sep='\t')
        if not df.empty:
            data_sources['drugcomb'] = df
            print(f"  Loaded {len(df)} DrugComb edges")
    
    # Load CARD resistance data
    card_edges = config['data_sources']['card']
    if os.path.exists(card_edges):
        print(f"Loading CARD resistance: {card_edges}")
        df = pd.read_csv(card_edges, sep='\t')
        if not df.empty:
            data_sources['card'] = df
            print(f"  Loaded {len(df)} CARD edges")
    
    return data_sources

def normalize_entities(data_sources: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Normalize and deduplicate entities across all data sources
    
    Returns:
        - Normalized edges DataFrame
        - Entity mapping dictionary
    """
    
    print("Normalizing entities across data sources...")
    
    # Collect all unique entities
    all_entities = set()
    entity_sources = {}  # Track which source each entity came from
    
    for source_name, df in data_sources.items():
        if df.empty:
            continue
        
        print(f"  Processing source: {source_name}")
        print(f"    Columns: {list(df.columns)}")
        print(f"    Shape: {df.shape}")
        
        # Handle different column names
        if 'head' in df.columns and 'tail' in df.columns:
            heads = df['head'].dropna().unique()
            tails = df['tail'].dropna().unique()
            
            print(f"    Unique heads: {len(heads)}")
            print(f"    Unique tails: {len(tails)}")
            print(f"    Sample heads: {list(heads[:5])}")
            print(f"    Sample tails: {list(tails[:5])}")
            
            for entity in heads:
                entity_str = str(entity).strip()
                if entity_str:  # Only add non-empty entities
                    all_entities.add(entity_str)
                    entity_sources[entity_str] = source_name
            
            for entity in tails:
                entity_str = str(entity).strip()
                if entity_str:  # Only add non-empty entities
                    all_entities.add(entity_str)
                    entity_sources[entity_str] = source_name
    
    print(f"Found {len(all_entities)} unique entities")
    
    # Create entity mapping (entity -> normalized_id)
    entity_mapping = {entity: f"E{i:06d}" for i, entity in enumerate(sorted(all_entities))}
    
    # Create entity metadata DataFrame
    entity_metadata = []
    for entity, normalized_id in entity_mapping.items():
        entity_metadata.append({
            'entity_id': normalized_id,
            'original_name': entity,
            'source': entity_sources.get(entity, 'unknown'),
            'entity_type': 'drug' if any(keyword in entity.lower() for keyword in 
                                        ['penicillin', 'amoxicillin', 'ampicillin', 'cephalexin', 'cefazolin']) else 'protein'
        })
    
    entity_df = pd.DataFrame(entity_metadata)
    
    return entity_df, entity_mapping

def normalize_edges(data_sources: Dict[str, pd.DataFrame], entity_mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Normalize edges using the entity mapping
    
    Returns:
        - Normalized edges DataFrame
    """
    
    print("Normalizing edges...")
    
    all_edges = []
    
    for source_name, df in data_sources.items():
        if df.empty:
            continue
        
        print(f"  Processing {source_name} edges...")
        
        # Handle different column structures
        if 'head' in df.columns and 'tail' in df.columns and 'relation' in df.columns:
            for _, row in df.iterrows():
                head = str(row['head'])
                tail = str(row['tail'])
                relation = str(row['relation'])
                
                # Skip if entities not in mapping
                if head not in entity_mapping or tail not in entity_mapping:
                    continue
                
                edge = {
                    'head': entity_mapping[head],
                    'relation': relation,
                    'tail': entity_mapping[tail],
                    'source': source_name,
                    'confidence': row.get('confidence', 1.0),
                    'original_head': head,
                    'original_tail': tail
                }
                
                all_edges.append(edge)
    
    edges_df = pd.DataFrame(all_edges)
    
    if not edges_df.empty:
        # Remove duplicates
        edges_df = edges_df.drop_duplicates(subset=['head', 'relation', 'tail'])
        print(f"  Created {len(edges_df)} normalized edges")
    
    return edges_df

def create_relation_mapping(edges_df: pd.DataFrame) -> Dict[str, str]:
    """Create normalized relation IDs"""
    
    unique_relations = edges_df['relation'].unique()
    relation_mapping = {rel: f"R{i:03d}" for i, rel in enumerate(sorted(unique_relations))}
    
    return relation_mapping

def finalize_kg(edges_df: pd.DataFrame, entity_df: pd.DataFrame, relation_mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Create final knowledge graph with normalized IDs
    
    Returns:
        - Final KG edges DataFrame
    """
    
    print("Finalizing knowledge graph...")
    
    # Apply relation mapping
    edges_df['relation_id'] = edges_df['relation'].map(relation_mapping)
    
    # Create final KG format
    final_kg = edges_df[['head', 'relation_id', 'tail', 'source', 'confidence']].copy()
    final_kg.columns = ['head', 'relation', 'tail', 'source', 'confidence']
    
    # Remove duplicates
    final_kg = final_kg.drop_duplicates(subset=['head', 'relation', 'tail'])
    
    print(f"Final KG: {len(final_kg)} edges")
    
    return final_kg

def save_kg_components(kg_edges: pd.DataFrame, entity_df: pd.DataFrame, 
                       relation_mapping: Dict[str, str], config: dict):
    """Save all KG components to files"""
    
    print("Saving knowledge graph components...")
    
    # Save edges
    edges_path = config['kg']['edges']
    os.makedirs(os.path.dirname(edges_path), exist_ok=True)
    kg_edges.to_csv(edges_path, sep='\t', index=False)
    print(f"  Saved {len(kg_edges)} edges to {edges_path}")
    
    # Save nodes
    nodes_path = config['kg']['nodes']
    os.makedirs(os.path.dirname(nodes_path), exist_ok=True)
    entity_df.to_csv(nodes_path, sep='\t', index=False)
    print(f"  Saved {len(entity_df)} entities to {nodes_path}")
    
    # Save relation mapping
    relations_path = os.path.join(os.path.dirname(edges_path), 'relations.tsv')
    relations_df = pd.DataFrame([
        {'relation_id': rid, 'relation_name': rname} 
        for rname, rid in relation_mapping.items()
    ])
    relations_df.to_csv(relations_path, sep='\t', index=False)
    print(f"  Saved {len(relations_df)} relations to {relations_path}")
    
    # Save entity mapping for reference
    mapping_path = os.path.join(os.path.dirname(edges_path), 'entity_mapping.tsv')
    mapping_df = pd.DataFrame([
        {'normalized_id': eid, 'original_name': ename, 'source': esource, 'entity_type': etype}
        for _, row in entity_df.iterrows()
        for eid, ename, esource, etype in [(row['entity_id'], row['original_name'], row['source'], row['entity_type'])]
    ])
    mapping_df.to_csv(mapping_path, sep='\t', index=False)
    print(f"  Saved entity mapping to {mapping_path}")

def print_kg_statistics(kg_edges: pd.DataFrame, entity_df: pd.DataFrame, relation_mapping: Dict[str, str]):
    """Print comprehensive KG statistics"""
    
    print("\n" + "="*50)
    print("KNOWLEDGE GRAPH STATISTICS")
    print("="*50)
    
    print(f"Total Entities: {len(entity_df)}")
    print(f"Total Relations: {len(relation_mapping)}")
    print(f"Total Edges: {len(kg_edges)}")
    
    print(f"\nEntity Types:")
    entity_types = entity_df['entity_type'].value_counts()
    for etype, count in entity_types.items():
        print(f"  {etype}: {count}")
    
    print(f"\nData Sources:")
    sources = kg_edges['source'].value_counts()
    for source, count in sources.items():
        print(f"  {source}: {count}")
    
    print(f"\nRelation Types:")
    relations = kg_edges['relation'].value_counts()
    for rel, count in relations.head(10).items():
        print(f"  {rel}: {count}")
    
    print(f"\nConfidence Distribution:")
    confidence_stats = kg_edges['confidence'].describe()
    print(f"  Mean: {confidence_stats['mean']:.3f}")
    print(f"  Std: {confidence_stats['std']:.3f}")
    print(f"  Min: {confidence_stats['min']:.3f}")
    print(f"  Max: {confidence_stats['max']:.3f}")

def main():
    """Main function for knowledge graph assembly"""
    
    parser = argparse.ArgumentParser(description='Assemble knowledge graph from multiple data sources')
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    print("Starting knowledge graph assembly...")
    
    # Load all data sources
    data_sources = load_data_sources(config)
    
    if not data_sources:
        print("No data sources found. Please ensure data files exist.")
        return
    
    # Normalize entities
    entity_df, entity_mapping = normalize_entities(data_sources)
    
    # Normalize edges
    edges_df = normalize_edges(data_sources, entity_mapping)
    
    if edges_df.empty:
        print("No edges found. Cannot create knowledge graph.")
        return
    
    # Create relation mapping
    relation_mapping = create_relation_mapping(edges_df)
    
    # Finalize knowledge graph
    final_kg = finalize_kg(edges_df, entity_df, relation_mapping)
    
    # Save components
    save_kg_components(final_kg, entity_df, relation_mapping, config)
    
    # Print statistics
    print_kg_statistics(final_kg, entity_df, relation_mapping)
    
    print("\nKnowledge graph assembly completed successfully!")

if __name__ == "__main__":
    main()
