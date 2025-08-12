#!/usr/bin/env python3
"""
Predict novel adjuvant candidates for β-lactam antibiotics using trained knowledge graph embeddings.
"""

import argparse
import json
import pandas as pd
import pickle
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import yaml
from typing import List, Dict, Tuple

class TransE(nn.Module):
    """TransE knowledge graph embedding model."""
    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int = 128, margin: float = 1.0):
        super(TransE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
        self.entity_embeddings.weight.data = torch.renorm(self.entity_embeddings.weight.data, p=2, dim=1, maxnorm=1)
    
    def forward(self, heads: torch.Tensor, relations: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        h = self.entity_embeddings(heads)
        r = self.relation_embeddings(relations)
        t = self.entity_embeddings(tails)
        scores = -torch.norm(h + r - t, p=2, dim=1)
        return scores

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_trained_model_and_mappings(model_path: str, mappings_path: str):
    """Load the trained TransE model and mappings."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(mappings_path, 'rb') as f:
        mappings = pickle.load(f)
    
    return model, mappings['entity_mapping'], mappings['relation_mapping']

def load_entity_mapping_from_tsv(mapping_path: str) -> Dict[str, int]:
    """Load entity ID mapping from TSV file."""
    df = pd.read_csv(mapping_path, sep='\t')
    # Use original_name as key and normalized_id as value
    return dict(zip(df['original_name'], df['normalized_id']))

def load_relation_mapping_from_tsv(relations_path: str) -> Dict[str, int]:
    """Load relation ID mapping from TSV file."""
    df = pd.read_csv(relations_path, sep='\t')
    # Use relation_name as key and relation_id as value
    return dict(zip(df['relation_name'], df['relation_id']))

def create_reverse_mapping(entity_mapping: Dict[str, int]) -> Dict[str, str]:
    """Create a mapping from DrugBank IDs to normalized IDs."""
    # Load the TSV to get the reverse mapping
    df = pd.read_csv('data/kg/entity_mapping.tsv', sep='\t')
    return dict(zip(df['original_name'], df['normalized_id']))

def create_relation_name_mapping(relation_mapping: Dict[str, int]) -> Dict[str, int]:
    """Create a mapping from relation names to relation IDs."""
    # Load the TSV to get the relation name to ID mapping
    df = pd.read_csv('data/kg/relations.tsv', sep='\t')
    return dict(zip(df['relation_name'], df['relation_id']))

def predict_adjuvant_candidates(model, entity_mapping: Dict[str, int], 
                              relation_mapping: Dict[str, int], 
                              config: dict, top_k: int = 100, reverse_mapping: Dict[str, str] = None, relation_name_mapping: Dict[str, int] = None) -> List[Dict]:
    """
    Predict adjuvant candidates for β-lactam antibiotics.
    """
    # Define β-lactam antibiotics with their DrugBank IDs
    beta_lactam_mapping = {
        "Flucloxacillin": "DB00301", "Piperacillin": "DB00319", "Ampicillin": "DB00415", 
        "Phenoxymethylpenicillin": "DB00417", "Ceftazidime": "DB00438", "Dicloxacillin": "DB00485", 
        "Cefotaxime": "DB00493", "Cephalexin": "DB00567", "Nafcillin": "DB00607", 
        "Oxacillin": "DB00713", "Meropenem": "DB00760", "Cefaclor": "DB00833", 
        "Benzylpenicillin": "DB01053", "Amoxicillin": "DB01060", "Cefuroxime": "DB01112", 
        "Cefadroxil": "DB01140", "Cloxacillin": "DB01147", "Ceftriaxone": "DB01212", 
        "Cefazolin": "DB01327"
    }
    
    # Define adjuvant relations (using the actual relation names from our KG)
    adjuvant_relations = ["targets", "ppi", "interacts_with"]
    
    # Get all potential adjuvant entities (proteins - UniProt IDs)
    all_entities = list(entity_mapping.keys())
    print(f"Debug: Total entities in mapping: {len(all_entities)}")
    print(f"Debug: Sample entities: {all_entities[:10]}")
    
    # Load the entity mapping TSV to get protein entities by original_name
    df = pd.read_csv('data/kg/entity_mapping.tsv', sep='\t')
    protein_entities = df[df['original_name'].str.match(r'^[PQOA]', na=False)]['normalized_id'].tolist()
    print(f"Debug: Protein entities found: {len(protein_entities)}")
    print(f"Debug: Sample protein entities: {protein_entities[:10]}")
    
    print(f"Found {len(beta_lactam_mapping)} β-lactam antibiotics")
    print(f"Found {len(protein_entities)} potential protein adjuvants")
    
    all_predictions = []
    
    for drug_name, drugbank_id in beta_lactam_mapping.items():
        if drugbank_id not in reverse_mapping:
            print(f"Warning: {drug_name} ({drugbank_id}) not found in reverse mapping, skipping...")
            continue
            
        normalized_id = reverse_mapping[drugbank_id]
        if normalized_id not in entity_mapping:
            print(f"Warning: {drug_name} ({drugbank_id} -> {normalized_id}) not found in entity mapping, skipping...")
            continue
            
        beta_lactam_entity_id = entity_mapping[normalized_id]
        print(f"Processing {drug_name} ({drugbank_id} -> {normalized_id}) -> entity ID {beta_lactam_entity_id}")
        
        for relation in adjuvant_relations:
            if relation not in relation_name_mapping:
                print(f"Warning: {relation} not found in relation name mapping, skipping...")
                continue
                
            relation_id = relation_name_mapping[relation]
            if relation_id not in relation_mapping:
                print(f"Warning: {relation} ({relation_id}) not found in relation mapping, skipping...")
                continue
                
            relation_entity_id = relation_mapping[relation_id]
            
            # Score all potential adjuvant combinations
            predictions = []
            for adjuvant in protein_entities[:1000]:  # Limit to first 1000 for performance
                if adjuvant in entity_mapping:
                    adjuvant_id = entity_mapping[adjuvant]
                    
                    # Create tensors for prediction
                    head_tensor = torch.tensor([beta_lactam_entity_id], dtype=torch.long)
                    rel_tensor = torch.tensor([relation_entity_id], dtype=torch.long)
                    tail_tensor = torch.tensor([adjuvant_id], dtype=torch.long)
                    
                    # Get prediction score
                    with torch.no_grad():
                        score = model.forward(head_tensor, rel_tensor, tail_tensor).item()
                    
                    predictions.append({
                        'head': drug_name,
                        'head_id': drugbank_id,
                        'relation': relation,
                        'tail': adjuvant,
                        'score': score
                    })
            
            # Sort by score and take top candidates
            predictions.sort(key=lambda x: x['score'], reverse=True)
            top_predictions = predictions[:top_k//len(adjuvant_relations)]
            
            all_predictions.extend(top_predictions)
    
    # Sort all predictions by score
    all_predictions.sort(key=lambda x: x['score'], reverse=True)
    
    return all_predictions[:top_k]

def main():
    parser = argparse.ArgumentParser(description='Predict adjuvant candidates for β-lactam antibiotics')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--top-k', type=int, default=100, help='Number of top predictions to return')
    parser.add_argument('--output', default='results/predicted_links.tsv', help='Output file path')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Load trained model and mappings
    model_path = config['outputs']['model_dir'] + '/trained_model.pkl'
    mappings_path = config['outputs']['model_dir'] + '/mappings.pkl'
    
    print(f"Loading trained model from {model_path}")
    model, entity_mapping, relation_mapping = load_trained_model_and_mappings(model_path, mappings_path)
    model.eval()
    
    print(f"Loaded {len(entity_mapping)} entities and {len(relation_mapping)} relations")
    
    # Create reverse mapping from DrugBank IDs to normalized IDs
    reverse_mapping = create_reverse_mapping(entity_mapping)
    print(f"Created reverse mapping with {len(reverse_mapping)} DrugBank IDs")
    
    # Create relation name mapping
    relation_name_mapping = create_relation_name_mapping(relation_mapping)
    print(f"Created relation name mapping with {len(relation_name_mapping)} relations")
    
    # Predict adjuvant candidates
    print("Predicting adjuvant candidates...")
    predictions = predict_adjuvant_candidates(
        model, entity_mapping, relation_mapping, config, args.top_k, reverse_mapping, relation_name_mapping
    )
    
    # Save predictions
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(predictions)
    df.to_csv(output_path, sep='\t', index=False)
    
    print(f"Saved {len(predictions)} predictions to {output_path}")
    
    # Print top predictions
    print("\nTop 10 predicted adjuvant candidates:")
    for i, pred in enumerate(predictions[:10]):
        print(f"{i+1:2d}. {pred['head']} --[{pred['relation']}]--> {pred['tail']} (score: {pred['score']:.4f})")

if __name__ == "__main__":
    main()
