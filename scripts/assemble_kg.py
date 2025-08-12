#!/usr/bin/env python3
"""
Assemble unified knowledge graph from all data sources.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from typing import Dict, List, Tuple
import os

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_data_sources(config: dict) -> Dict[str, pd.DataFrame]:
    """Load all data sources specified in config."""
    data_sources = {}
    
    print("Loading data sources...")
    
    # Load DrugBank edges
    drugbank_path = config['data_sources']['drugbank']
    if Path(drugbank_path).exists():
        drugbank_df = pd.read_csv(drugbank_path, sep='\t')
        drugbank_df['source'] = 'DrugBank'
        drugbank_df['weight'] = 1.0
        drugbank_df['evidence'] = 'DrugBank'
        data_sources['drugbank'] = drugbank_df
        print(f"  DrugBank: {len(drugbank_df)} edges")
    else:
        print(f"  ⚠️  DrugBank file not found: {drugbank_path}")
        data_sources['drugbank'] = pd.DataFrame()
    
    # Load STRING interactions
    string_path = config['data_sources']['string']
    if Path(string_path).exists():
        string_df = pd.read_csv(string_path, sep='\t')
        string_df['source'] = 'STRING'
        string_df['weight'] = string_df.get('confidence', 1.0)
        string_df['evidence'] = 'STRING'
        data_sources['string'] = string_df
        print(f"  STRING: {len(string_df)} edges")
    else:
        print(f"  ⚠️  STRING file not found: {string_path}")
        data_sources['string'] = pd.DataFrame()
    
    # Load ChEMBL targets
    chembl_path = config['data_sources']['chembl']
    if Path(chembl_path).exists():
        chembl_df = pd.read_csv(chembl_path, sep='\t')
        chembl_df['source'] = 'ChEMBL'
        chembl_df['weight'] = 1.0
        chembl_df['evidence'] = 'ChEMBL'
        data_sources['chembl'] = chembl_df
        print(f"  ChEMBL: {len(chembl_df)} edges")
    else:
        print(f"  ⚠️  ChEMBL file not found: {chembl_path}")
        data_sources['chembl'] = pd.DataFrame()
    
    # Load DrugComb synergies
    drugcomb_path = config['data_sources']['drugcomb']
    if Path(drugcomb_path).exists():
        drugcomb_df = pd.read_csv(drugcomb_path, sep='\t')
        if not drugcomb_df.empty:
            # Rename columns to match our schema
            if 'head_id' in drugcomb_df.columns:
                drugcomb_df = drugcomb_df.rename(columns={
                    'head_id': 'head',
                    'tail_id': 'tail',
                    'evidence_source': 'source',
                    'evidence_id': 'evidence'
                })
            drugcomb_df['weight'] = drugcomb_df.get('weight', 1.0)
            data_sources['drugcomb'] = drugcomb_df
            print(f"  DrugComb: {len(drugcomb_df)} edges")
        else:
            print(f"  ⚠️  DrugComb file is empty: {drugcomb_path}")
            data_sources['drugcomb'] = pd.DataFrame()
    else:
        print(f"  ⚠️  DrugComb file not found: {drugcomb_path}")
        data_sources['drugcomb'] = pd.DataFrame()
    
    # Load CARD resistance
    card_path = config['data_sources']['card']
    if Path(card_path).exists():
        card_df = pd.read_csv(card_path, sep='\t')
        if not card_df.empty:
            # Rename columns to match our schema
            if 'head_id' in card_df.columns:
                card_df = card_df.rename(columns={
                    'head_id': 'head',
                    'tail_id': 'tail',
                    'evidence_source': 'source',
                    'evidence_id': 'evidence'
                })
            card_df['weight'] = card_df.get('weight', 1.0)
            data_sources['card'] = card_df
            print(f"  CARD: {len(card_df)} edges")
        else:
            print(f"  ⚠️  CARD file is empty: {card_path}")
            data_sources['card'] = pd.DataFrame()
    else:
        print(f"  ⚠️  CARD file not found: {card_path}")
        data_sources['card'] = pd.DataFrame()
    
    return data_sources

def normalize_entities(data_sources: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Normalize and deduplicate entities across all sources."""
    print("\nNormalizing entities...")
    
    all_entities = set()
    
    # Collect entities from all sources
    for source_name, df in data_sources.items():
        if not df.empty:
            if 'head' in df.columns:
                all_entities.update(df['head'].dropna().unique())
            if 'tail' in df.columns:
                all_entities.update(df['tail'].dropna().unique())
    
    # Create normalized entity DataFrame
    entities_list = []
    entity_mapping = {}
    
    for i, entity in enumerate(sorted(all_entities)):
        if pd.isna(entity) or entity == '':
            continue
            
        normalized_id = f"E{i:06d}"
        entity_mapping[entity] = normalized_id
        
        # Determine entity type
        if entity.startswith('DB'):
            entity_type = 'drug'
        elif entity.startswith('P') or entity.startswith('Q') or entity.startswith('O') or entity.startswith('A'):
            entity_type = 'protein'
        elif entity.startswith('ARO:'):
            entity_type = 'resistance_mechanism'
        else:
            entity_type = 'other'
        
        entities_list.append({
            'id': normalized_id,
            'original_name': entity,
            'type': entity_type
        })
    
    entities_df = pd.DataFrame(entities_list)
    
    print(f"  Normalized {len(entities_df)} unique entities")
    print(f"    Drugs: {len(entities_df[entities_df['type'] == 'drug'])}")
    print(f"    Proteins: {len(entities_df[entities_df['type'] == 'protein'])}")
    print(f"    Resistance mechanisms: {len(entities_df[entities_df['type'] == 'resistance_mechanism'])}")
    
    return entities_df, entity_mapping

def normalize_edges(data_sources: Dict[str, pd.DataFrame], entity_mapping: Dict[str, int]) -> pd.DataFrame:
    """Normalize and combine all edges."""
    print("\nNormalizing edges...")
    
    all_edges = []
    
    for source_name, df in data_sources.items():
        if df.empty:
            continue
            
        print(f"  Processing {source_name}...")
        
        # Ensure required columns exist
        required_cols = ['head', 'relation', 'tail']
        if not all(col in df.columns for col in required_cols):
            print(f"    ⚠️  Missing required columns, skipping")
            continue
        
        # Filter out rows with missing values
        df_clean = df.dropna(subset=['head', 'relation', 'tail'])
        
        for _, row in df_clean.iterrows():
            head = row['head']
            tail = row['tail']
            relation = row['relation']
            
            # Skip if entities not in mapping
            if head not in entity_mapping or tail not in entity_mapping:
                continue
            
            edge = {
                'head': entity_mapping[head],
                'relation': relation,
                'tail': entity_mapping[tail],
                'weight': row.get('weight', 1.0),
                'source': row.get('source', source_name),
                'evidence': row.get('evidence', '')
            }
            
            all_edges.append(edge)
    
    edges_df = pd.DataFrame(all_edges)
    
    # Remove duplicates
    edges_df = edges_df.drop_duplicates(subset=['head', 'relation', 'tail'])
    
    print(f"  Combined {len(edges_df)} unique edges")
    
    return edges_df

def create_relation_mapping(edges_df: pd.DataFrame) -> Dict[str, int]:
    """Create mapping for relation types."""
    print("\nCreating relation mapping...")
    
    unique_relations = sorted(edges_df['relation'].unique())
    relation_mapping = {}
    
    for i, relation in enumerate(unique_relations):
        relation_mapping[relation] = f"R{i:03d}"
    
    print(f"  Mapped {len(relation_mapping)} relation types:")
    for relation, rid in relation_mapping.items():
        count = len(edges_df[edges_df['relation'] == relation])
        print(f"    {relation} ({rid}): {count} edges")
    
    return relation_mapping

def finalize_kg(edges_df: pd.DataFrame, entities_df: pd.DataFrame, relation_mapping: Dict[str, int]) -> Dict:
    """Finalize knowledge graph with all components."""
    print("\nFinalizing knowledge graph...")
    
    # Create final KG structure
    kg = {
        'edges': edges_df,
        'entities': entities_df,
        'relations': relation_mapping
    }
    
    return kg

def save_kg_components(kg: Dict, entities_df: pd.DataFrame, relation_mapping: Dict[str, int], config: dict):
    """Save knowledge graph components to files."""
    print("\nSaving knowledge graph components...")
    
    kg_dir = Path(config['paths']['kg_dir'])
    kg_dir.mkdir(parents=True, exist_ok=True)
    
    # Save edges
    edges_path = kg_dir / 'edges.tsv'
    kg['edges'].to_csv(edges_path, sep='\t', index=False)
    print(f"  Edges: {edges_path}")
    
    # Save entities
    entities_path = kg_dir / 'entities.tsv'
    entities_df.to_csv(entities_path, sep='\t', index=False)
    print(f"  Entities: {entities_path}")
    
    # Save entity mapping
    entity_mapping_path = kg_dir / 'entity_mapping.tsv'
    entities_df[['id', 'original_name']].to_csv(entity_mapping_path, sep='\t', index=False)
    print(f"  Entity mapping: {entity_mapping_path}")
    
    # Save relations
    relations_list = [{'id': rid, 'name': name} for name, rid in relation_mapping.items()]
    relations_df = pd.DataFrame(relations_list)
    relations_path = kg_dir / 'relations.tsv'
    relations_df.to_csv(relations_path, sep='\t', index=False)
    print(f"  Relations: {relations_path}")

def print_kg_statistics(kg: Dict, entities_df: pd.DataFrame, relation_mapping: Dict[str, int]):
    """Print comprehensive KG statistics."""
    print("\n" + "="*60)
    print("KNOWLEDGE GRAPH STATISTICS")
    print("="*60)
    
    edges_df = kg['edges']
    
    print(f"📊 Overall Statistics:")
    print(f"   Total entities: {len(entities_df):,}")
    print(f"   Total edges: {len(edges_df):,}")
    print(f"   Total relations: {len(relation_mapping)}")
    
    print(f"\n🏷️  Entity Types:")
    entity_type_counts = entities_df['type'].value_counts()
    for entity_type, count in entity_type_counts.items():
        print(f"   {entity_type}: {count:,}")
    
    print(f"\n🔗 Relation Types:")
    for relation, rid in relation_mapping.items():
        count = len(edges_df[edges_df['relation'] == relation])
        print(f"   {relation} ({rid}): {count:,} edges")
    
    print(f"\n📈 Data Sources:")
    source_counts = edges_df['source'].value_counts()
    for source, count in source_counts.items():
        print(f"   {source}: {count:,} edges")
    
    print(f"\n⚖️  Weight Distribution:")
    weights = edges_df['weight'].dropna()
    if len(weights) > 0:
        print(f"   Min weight: {weights.min():.3f}")
        print(f"   Max weight: {weights.max():.3f}")
        print(f"   Mean weight: {weights.mean():.3f}")
        print(f"   Median weight: {weights.median():.3f}")
    
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Assemble unified knowledge graph from all data sources')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    
    args = parser.parse_args()
    
    print("🔧 Knowledge Graph Assembly")
    print("="*50)
    
    # Load configuration
    config = load_config(args.config)
    
    # Load all data sources
    data_sources = load_data_sources(config)
    
    # Normalize entities
    entity_df, entity_mapping = normalize_entities(data_sources)
    
    # Normalize edges
    edges_df = normalize_edges(data_sources, entity_mapping)
    
    # Create relation mapping
    relation_mapping = create_relation_mapping(edges_df)
    
    # Finalize KG
    final_kg = finalize_kg(edges_df, entity_df, relation_mapping)
    
    # Save components
    save_kg_components(final_kg, entity_df, relation_mapping, config)
    
    # Print statistics
    print_kg_statistics(final_kg, entity_df, relation_mapping)
    
    print("\n✅ Knowledge graph assembly completed successfully!")

if __name__ == "__main__":
    main()
