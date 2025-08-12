#!/usr/bin/env python3
"""
Mini TransE Implementation for Knowledge Graph Embeddings
Provides MRR, Hits@K, and AUROC metrics without PyKEEN dependency
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import json
import os
import random
from typing import Dict, List, Tuple, Set
import argparse
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from pathlib import Path
import pickle

class TransE(nn.Module):
    """TransE knowledge graph embedding model"""
    
    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int = 128, margin: float = 1.0):
        super(TransE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin
        
        # Initialize embeddings
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
        
        # Normalize entity embeddings
        self.entity_embeddings.weight.data = torch.renorm(
            self.entity_embeddings.weight.data, p=2, dim=1, maxnorm=1
        )
    
    def forward(self, heads: torch.Tensor, relations: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        """Forward pass - compute scores for triples"""
        h = self.entity_embeddings(heads)
        r = self.relation_embeddings(relations)
        t = self.entity_embeddings(tails)
        
        # TransE scoring: -||h + r - t||
        scores = -torch.norm(h + r - t, p=2, dim=1)
        return scores
    
    def score_triple(self, head: int, relation: int, tail: int) -> float:
        """Score a single triple"""
        with torch.no_grad():
            h = torch.tensor([head])
            r = torch.tensor([relation])
            t = torch.tensor([tail])
            score = self.forward(h, r, t)
            return score.item()
    
    def get_embeddings(self):
        """Get entity and relation embeddings"""
        return {
            'entities': self.entity_embeddings.weight.data.cpu().numpy(),
            'relations': self.relation_embeddings.weight.data.cpu().numpy()
        }

class KGDataLoader:
    """Data loader for knowledge graph triples"""
    
    def __init__(self, edges_file: str, nodes_file: str, relations_file: str):
        self.edges_df = pd.read_csv(edges_file, sep='\t')
        self.nodes_df = pd.read_csv(nodes_file, sep='\t')
        self.relations_df = pd.read_csv(relations_file, sep='\t')
        
        # Create mappings
        self.entity_to_id = {row['original_name']: row['entity_id'] for _, row in self.nodes_df.iterrows()}
        self.id_to_entity = {row['entity_id']: row['original_name'] for _, row in self.nodes_df.iterrows()}
        
        self.relation_to_id = {row['relation_name']: row['relation_id'] for _, row in self.relations_df.iterrows()}
        self.id_to_relation = {row['relation_id']: row['relation_name'] for _, row in self.relations_df.iterrows()}
        
        # Convert to numeric IDs
        self.entity_id_map = {entity: i for i, entity in enumerate(sorted(self.entity_to_id.values()))}
        self.relation_id_map = {rel: i for i, rel in enumerate(sorted(self.relation_to_id.values()))}
        
        # Create triples
        self.triples = self._create_triples()
        
        print(f"Loaded {len(self.triples)} triples")
        print(f"Entities: {len(self.entity_id_map)}")
        print(f"Relations: {len(self.relation_id_map)}")
    
    def _create_triples(self) -> List[Tuple[int, int, int]]:
        """Convert edges to numeric triples"""
        triples = []
        
        for _, row in self.edges_df.iterrows():
            head = row['head']
            relation = row['relation']
            tail = row['tail']
            
            if head in self.entity_id_map and relation in self.relation_id_map and tail in self.entity_id_map:
                h_id = self.entity_id_map[head]
                r_id = self.relation_id_map[relation]
                t_id = self.entity_id_map[tail]
                triples.append((h_id, r_id, t_id))
        
        return triples
    
    def split_data(self, train_ratio: float = 0.8, valid_ratio: float = 0.1, test_ratio: float = 0.1):
        """Split triples into train/validation/test sets"""
        
        assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        # Shuffle triples
        random.shuffle(self.triples)
        
        n_triples = len(self.triples)
        n_train = int(n_triples * train_ratio)
        n_valid = int(n_triples * valid_ratio)
        
        train_triples = self.triples[:n_train]
        valid_triples = self.triples[n_train:n_train + n_valid]
        test_triples = self.triples[n_train + n_valid:]
        
        return train_triples, valid_triples, test_triples

def negative_sampling(triples: List[Tuple[int, int, int]], num_entities: int, num_negatives: int = 1) -> List[Tuple[int, int, int]]:
    """Generate negative samples by corrupting tails"""
    
    triples_set = set(triples)
    negative_triples = []
    
    for h, r, t in triples:
        for _ in range(num_negatives):
            # Corrupt tail
            t_neg = random.randint(0, num_entities - 1)
            while (h, r, t_neg) in triples_set:
                t_neg = random.randint(0, num_entities - 1)
            negative_triples.append((h, r, t_neg))
    
    return negative_triples

def train_model(model: TransE, train_triples: List[Tuple[int, int, int]], 
                valid_triples: List[Tuple[int, int, int]], num_epochs: int = 100, 
                batch_size: int = 1024, learning_rate: float = 0.001) -> Dict:
    """Train the TransE model"""
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MarginRankingLoss(margin=model.margin)
    
    train_losses = []
    valid_scores = []
    
    print(f"Training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # Shuffle training data
        random.shuffle(train_triples)
        
        # Process in batches
        for i in range(0, len(train_triples), batch_size):
            batch = train_triples[i:i + batch_size]
            
            # Generate negative samples
            neg_batch = negative_sampling(batch, model.num_entities, num_negatives=1)
            
            # Prepare tensors
            pos_h = torch.tensor([t[0] for t in batch])
            pos_r = torch.tensor([t[1] for t in batch])
            pos_t = torch.tensor([t[2] for t in batch])
            
            neg_h = torch.tensor([t[0] for t in neg_batch])
            neg_r = torch.tensor([t[1] for t in neg_batch])
            neg_t = torch.tensor([t[2] for t in neg_batch])
            
            # Forward pass
            pos_scores = model(pos_h, pos_r, pos_t)
            neg_scores = model(neg_h, neg_r, neg_t)
            
            # Loss
            loss = criterion(pos_scores, neg_scores, torch.ones_like(pos_scores))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Normalize entity embeddings
            with torch.no_grad():
                model.entity_embeddings.weight.data = torch.renorm(
                    model.entity_embeddings.weight.data, p=2, dim=1, maxnorm=1
                )
            
            total_loss += loss.item()
        
        avg_loss = total_loss / (len(train_triples) // batch_size + 1)
        train_losses.append(avg_loss)
        
        # Validation
        if epoch % 10 == 0:
            model.eval()
            valid_metrics = evaluate_model(model, valid_triples, 'validation')
            valid_scores.append(valid_metrics['mrr'])
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Valid MRR = {valid_metrics['mrr']:.4f}")
    
    return {
        'train_losses': train_losses,
        'valid_scores': valid_scores
    }

def evaluate_model(model, test_triples, all_triples, entity_mapping, relation_mapping, k_list=[1, 3, 10]):
    """
    Evaluate model with filtered metrics (ignore other true tails when ranking).
    """
    model.eval()
    
    print("Evaluating model with filtered metrics...")
    
    # Create set of all true triples for filtering
    all_true_triples = set()
    for _, row in all_triples.iterrows():
        head = entity_mapping.get(row['head'], row['head'])
        rel = relation_mapping.get(row['relation'], row['relation'])
        tail = entity_mapping.get(row['tail'], row['tail'])
        all_true_triples.add((head, rel, tail))
    
    # Convert test triples to tensors
    test_heads = torch.tensor([entity_mapping.get(row['head'], row['head']) for _, row in test_triples.iterrows()], dtype=torch.long)
    test_rels = torch.tensor([relation_mapping.get(row['relation'], row['relation']) for _, row in test_triples.iterrows()], dtype=torch.long)
    test_tails = torch.tensor([entity_mapping.get(row['tail'], row['tail']) for _, row in test_triples.iterrows()], dtype=torch.long)
    
    num_entities = len(entity_mapping)
    num_relations = len(relation_mapping)
    
    mrr_scores = []
    hits_at_k = {k: 0 for k in k_list}
    
    print(f"Evaluating {len(test_triples)} test triples...")
    
    for i in range(len(test_triples)):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(test_triples)}")
        
        head = test_heads[i]
        rel = test_rels[i]
        tail = test_tails[i]
        
        # Score all possible tails for this (head, relation) pair
        head_tensor = head.unsqueeze(0).expand(num_entities, -1)
        rel_tensor = rel.unsqueeze(0).expand(num_entities, -1)
        tail_candidates = torch.arange(num_entities)
        
        with torch.no_grad():
            scores = model.forward(head_tensor, rel_tensor, tail_candidates)
        
        # Filter out other true tails for this (head, relation) pair
        filtered_scores = scores.clone()
        for j in range(num_entities):
            if (head.item(), rel.item(), j) in all_true_triples and j != tail.item():
                filtered_scores[j] = float('-inf')  # Set to lowest possible score
        
        # Sort by score (descending)
        sorted_scores, sorted_indices = torch.sort(filtered_scores, descending=True)
        
        # Find rank of true tail
        true_rank = (sorted_indices == tail).nonzero(as_tuple=True)[0].item() + 1
        
        # Calculate MRR
        mrr_scores.append(1.0 / true_rank)
        
        # Calculate Hits@K
        for k in k_list:
            if true_rank <= k:
                hits_at_k[k] += 1
    
    # Calculate final metrics
    mrr = np.mean(mrr_scores)
    hits_at_k_normalized = {k: hits_at_k[k] / len(test_triples) for k in k_list}
    
    print(f"  MRR: {mrr:.4f}")
    for k in k_list:
        print(f"  Hits@{k}: {hits_at_k_normalized[k]:.4f}")
    
    return {
        'mrr': mrr,
        'hits_at_k': hits_at_k_normalized,
        'mrr_scores': mrr_scores
    }

def calculate_auroc(model, test_triples, all_triples, entity_mapping, relation_mapping, negatives_per_pos=1):
    """
    Calculate AUROC by scoring positive vs corrupted negative triples.
    """
    print("Calculating AUROC...")
    
    model.eval()
    
    # Create set of all true triples for filtering
    all_true_triples = set()
    for _, row in all_triples.iterrows():
        head = entity_mapping.get(row['head'], row['head'])
        rel = relation_mapping.get(row['relation'], row['relation'])
        tail = entity_mapping.get(row['tail'], row['tail'])
        all_true_triples.add((head, rel, tail))
    
    # Convert test triples to tensors
    test_heads = torch.tensor([entity_mapping.get(row['head'], row['head']) for _, row in test_triples.iterrows()], dtype=torch.long)
    test_rels = torch.tensor([relation_mapping.get(row['relation'], row['relation']) for _, row in test_triples.iterrows()], dtype=torch.long)
    test_tails = torch.tensor([entity_mapping.get(row['tail'], row['tail']) for _, row in test_triples.iterrows()], dtype=torch.long)
    
    num_entities = len(entity_mapping)
    
    # Score positive triples
    positive_scores = []
    with torch.no_grad():
        for i in range(len(test_triples)):
            head = test_heads[i]
            rel = test_rels[i]
            tail = test_tails[i]
            
            score = model.forward(head.unsqueeze(0), rel.unsqueeze(0), tail.unsqueeze(0))
            positive_scores.append(score.item())
    
    # Generate and score negative triples
    negative_scores = []
    for i in range(len(test_triples)):
        head = test_heads[i]
        rel = test_rels[i]
        
        for _ in range(negatives_per_pos):
            # Corrupt tail
            corrupted_tail = torch.randint(0, num_entities, (1,))
            while (head.item(), rel.item(), corrupted_tail.item()) in all_true_triples:
                corrupted_tail = torch.randint(0, num_entities, (1,))
            
            with torch.no_grad():
                score = model.forward(head.unsqueeze(0), rel.unsqueeze(0), corrupted_tail)
                negative_scores.append(score.item())
    
    # Combine scores and labels
    all_scores = positive_scores + negative_scores
    all_labels = [1] * len(positive_scores) + [0] * len(negative_scores)
    
    # Calculate AUROC
    from sklearn.metrics import roc_auc_score
    auroc = roc_auc_score(all_labels, all_scores)
    
    print(f"  AUROC: {auroc:.4f}")
    print(f"  Positive samples: {len(positive_scores)}")
    print(f"  Negative samples: {len(negative_scores)}")
    
    return auroc

def save_model_and_mappings(model, entity_mapping, relation_mapping, config):
    """Save the trained model and mappings."""
    output_dir = Path(config['paths']['results_dir']) / 'trained_model'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_dir / 'trained_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Save mappings
    mappings_path = output_dir / 'mappings.pkl'
    mappings = {
        'entity_mapping': entity_mapping,
        'relation_mapping': relation_mapping
    }
    with open(mappings_path, 'wb') as f:
        pickle.dump(mappings, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Save entity and relation mappings as JSON for easy access
    entity_json = {k: v for k, v in entity_mapping.items()}
    relation_json = {k: v for k, v in relation_mapping.items()}
    
    with open(output_dir / 'entity2id.json', 'w') as f:
        json.dump(entity_json, f, indent=2)
    
    with open(output_dir / 'relation2id.json', 'w') as f:
        json.dump(relation_json, f, indent=2)
    
    print(f"Model saved to {model_path}")
    print(f"Mappings saved to {mappings_path}")
    print(f"Entity mapping saved to {output_dir / 'entity2id.json'}")
    print(f"Relation mapping saved to {output_dir / 'relation2id.json'}")

def main():
    """Main training function"""
    
    parser = argparse.ArgumentParser(description='Train TransE model on knowledge graph')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--epochs', type=int, help='Number of training epochs (overrides config)')
    parser.add_argument('--embedding-dim', type=int, help='Embedding dimension (overrides config)')
    parser.add_argument('--batch-size', type=int, help='Batch size for training (overrides config)')
    parser.add_argument('--learning-rate', type=float, help='Learning rate (overrides config)')
    parser.add_argument('--margin', type=float, help='Margin for loss function (overrides config)')
    
    args = parser.parse_args()
    
    # Load configuration
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seeds
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("Loading knowledge graph data...")
    
    # Load data
    data_loader = KGDataLoader(
        edges_file=config['paths']['kg_dir'] + '/edges.tsv',
        nodes_file=config['paths']['kg_dir'] + '/entities.tsv',
        relations_file=config['paths']['kg_dir'] + '/relations.tsv'
    )
    
    # Split data
    train_triples, valid_triples, test_triples = data_loader.split_data()
    
    print(f"Data split: Train={len(train_triples)}, Valid={len(valid_triples)}, Test={len(test_triples)}")
    
    # Initialize model
    model = TransE(
        num_entities=len(data_loader.entity_id_map),
        num_relations=len(data_loader.relation_id_map),
        embedding_dim=args.embedding_dim or config['train']['dim'],
        margin=args.margin or 1.0
    )
    
    print(f"Model initialized with {len(data_loader.entity_id_map)} entities and {len(data_loader.relation_id_map)} relations")
    
    # Train model
    training_history = train_model(
        model=model,
        train_triples=train_triples,
        valid_triples=valid_triples,
        num_epochs=args.epochs or config['train']['epochs'],
        batch_size=args.batch_size or config['train']['batch_size'],
        learning_rate=args.learning_rate or config['train']['lr']
    )
    
    # Evaluate on test set with filtered metrics
    test_metrics = evaluate_model(
        model, test_triples, 
        pd.concat([train_triples, valid_triples, test_triples]),  # all triples for filtering
        data_loader.entity_id_map, 
        data_loader.relation_id_map,
        k_list=config['eval']['k_list']
    )
    
    # Calculate AUROC
    auroc = calculate_auroc(
        model, test_triples,
        pd.concat([train_triples, valid_triples, test_triples]),
        data_loader.entity_id_map,
        data_loader.relation_id_map,
        negatives_per_pos=config['eval']['negatives_per_pos']
    )
    
    # Add AUROC to test metrics
    test_metrics['auroc'] = auroc
    
    # Save results
    results = {
        'training_history': training_history,
        'test_metrics': test_metrics,
        'model_config': {
            'embedding_dim': args.embedding_dim or config['train']['dim'],
            'num_epochs': args.epochs or config['train']['epochs'],
            'batch_size': args.batch_size or config['train']['batch_size'],
            'learning_rate': args.learning_rate or config['train']['lr'],
            'margin': args.margin or 1.0
        },
        'data_stats': {
            'num_entities': len(data_loader.entity_id_map),
            'num_relations': len(data_loader.relation_id_map),
            'train_size': len(train_triples),
            'valid_size': len(valid_triples),
            'test_size': len(test_triples)
        }
    }
    
    # Save metrics
    metrics_path = Path(config['paths']['results_dir']) / 'metrics.json'
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Metrics saved to {metrics_path}")
    
    # Save model and mappings
    save_model_and_mappings(model, data_loader.entity_id_map, data_loader.relation_id_map, config)
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
