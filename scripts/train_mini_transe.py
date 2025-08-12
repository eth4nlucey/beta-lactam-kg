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

def evaluate_model(model: TransE, triples: List[Tuple[int, int, int]], 
                  split_name: str = 'test') -> Dict:
    """Evaluate model performance"""
    
    print(f"Evaluating on {split_name} set...")
    
    model.eval()
    
    # Metrics
    mrr_scores = []
    hits_at_1 = []
    hits_at_3 = []
    hits_at_10 = []
    
    # For AUROC calculation
    positive_scores = []
    negative_scores = []
    
    with torch.no_grad():
        for h, r, t in triples:
            # Score the positive triple
            pos_score = model.score_triple(h, r, t)
            positive_scores.append(pos_score)
            
            # Score all possible tails
            all_scores = []
            for t_candidate in range(model.num_entities):
                score = model.score_triple(h, r, t_candidate)
                all_scores.append((t_candidate, score))
            
            # Sort by score (descending)
            all_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Find rank of true tail
            true_rank = None
            for rank, (t_candidate, _) in enumerate(all_scores):
                if t_candidate == t:
                    true_rank = rank + 1
                    break
            
            if true_rank is not None:
                # MRR
                mrr = 1.0 / true_rank
                mrr_scores.append(mrr)
                
                # Hits@K
                hits_at_1.append(1.0 if true_rank <= 1 else 0.0)
                hits_at_3.append(1.0 if true_rank <= 3 else 0.0)
                hits_at_10.append(1.0 if true_rank <= 10 else 0.0)
                
                # Sample negative scores for AUROC
                for _ in range(5):  # Sample 5 negatives
                    t_neg = random.randint(0, model.num_entities - 1)
                    if t_neg != t:
                        neg_score = model.score_triple(h, r, t_neg)
                        negative_scores.append(neg_score)
    
    # Calculate metrics
    mrr = np.mean(mrr_scores) if mrr_scores else 0.0
    hits_1 = np.mean(hits_at_1) if hits_at_1 else 0.0
    hits_3 = np.mean(hits_at_3) if hits_at_3 else 0.0
    hits_10 = np.mean(hits_at_10) if hits_at_10 else 0.0
    
    # Calculate AUROC
    auroc = 0.5
    if positive_scores and negative_scores:
        y_true = [1] * len(positive_scores) + [0] * len(negative_scores)
        y_scores = positive_scores + negative_scores
        
        try:
            auroc = roc_auc_score(y_true, y_scores)
        except ValueError:
            auroc = 0.5
    
    metrics = {
        'mrr': mrr,
        'hits_at_1': hits_1,
        'hits_at_3': hits_3,
        'hits_at_10': hits_10,
        'auroc': auroc,
        'num_triples': len(triples)
    }
    
    print(f"{split_name.capitalize()} Metrics:")
    print(f"  MRR: {mrr:.4f}")
    print(f"  Hits@1: {hits_1:.4f}")
    print(f"  Hits@3: {hits_3:.4f}")
    print(f"  Hits@10: {hits_10:.4f}")
    print(f"  AUROC: {auroc:.4f}")
    
    return metrics

def save_model_and_mappings(model, entity_mapping, relation_mapping, config):
    """Save the trained model and mappings."""
    output_dir = Path(config['outputs']['model_dir'])
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
    
    print(f"Model saved to {model_path}")
    print(f"Mappings saved to {mappings_path}")

def main():
    """Main training function"""
    
    parser = argparse.ArgumentParser(description='Train TransE model on knowledge graph')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--embedding-dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--margin', type=float, default=1.0, help='Margin for loss function')
    
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
        edges_file=config['kg']['edges'],
        nodes_file=config['kg']['nodes'],
        relations_file=os.path.join(os.path.dirname(config['kg']['edges']), 'relations.tsv')
    )
    
    # Split data
    train_triples, valid_triples, test_triples = data_loader.split_data()
    
    print(f"Data split: Train={len(train_triples)}, Valid={len(valid_triples)}, Test={len(test_triples)}")
    
    # Initialize model
    model = TransE(
        num_entities=len(data_loader.entity_id_map),
        num_relations=len(data_loader.relation_id_map),
        embedding_dim=args.embedding_dim,
        margin=args.margin
    )
    
    print(f"Model initialized with {len(data_loader.entity_id_map)} entities and {len(data_loader.relation_id_map)} relations")
    
    # Train model
    training_history = train_model(
        model=model,
        train_triples=train_triples,
        valid_triples=valid_triples,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    # Evaluate on test set
    test_metrics = evaluate_model(model, test_triples, 'test')
    
    # Save results
    results = {
        'training_history': training_history,
        'test_metrics': test_metrics,
        'model_config': {
            'embedding_dim': args.embedding_dim,
            'num_epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'margin': args.margin
        }
    }
    
    # Save metrics
    metrics_path = config['outputs']['metrics']
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {metrics_path}")
    
    # Save model
    save_model_and_mappings(model, data_loader.entity_id_map, data_loader.relation_id_map, config)
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()
