import yaml
import json
import os
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from pykeen.evaluation import RankBasedEvaluator
from sklearn.metrics import roc_auc_score
import numpy as np
import torch

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def compute_auroc(model, tf, test_triples, num_negative_samples=1000):
    """Compute AUROC using positive test triples vs. corrupted negatives"""
    
    # Get positive test triples
    positive_triples = test_triples.mapped_triples
    
    # Generate corrupted negatives (same head, random tail)
    num_positives = len(positive_triples)
    num_negatives = min(num_negative_samples, num_positives)
    
    negative_scores = []
    positive_scores = []
    
    # Score positive triples
    with torch.no_grad():
        for i in range(min(num_positives, num_negatives)):
            h, r, t = positive_triples[i]
            score = model.score_hrt(torch.tensor([[h, r, t]])).item()
            positive_scores.append(score)
    
    # Score corrupted negatives
    with torch.no_grad():
        for i in range(num_negatives):
            h, r, t = positive_triples[i]
            # Corrupt tail
            corrupted_t = torch.randint(0, tf.num_entities, (1,)).item()
            score = model.score_hrt(torch.tensor([[h, r, corrupted_t]])).item()
            negative_scores.append(score)
    
    # Combine scores for AUROC
    y_true = [1] * len(positive_scores) + [0] * len(negative_scores)
    y_scores = positive_scores + negative_scores
    
    try:
        auroc = roc_auc_score(y_true, y_scores)
    except ValueError:
        auroc = 0.5  # Default if AUROC can't be computed
    
    return auroc

def train_pykeen_model(config_path: str = "config.yaml", triple_path: str = None, output_path: str = None):
    """Train PyKEEN model with proper train/valid/test split and evaluation metrics"""
    
    config = load_config(config_path)
    
    if triple_path is None:
        triple_path = config['kg']['edges']
    if output_path is None:
        output_path = config['outputs']['model_dir']
    
    print(f"Training {config['training']['model']} model...")
    print(f"Loading triples from: {triple_path}")
    
    # Load triples from file
    tf = TriplesFactory.from_path(triple_path)
    print(f"Knowledge Graph: {tf.num_triples} triples, {tf.num_entities} entities, {tf.num_relations} relations")
    
    # Split triples into train/valid/test
    train_tf, test_tf = tf.split([1 - config['training']['test_ratio'], config['training']['test_ratio']])
    train_tf, valid_tf = train_tf.split([1 - config['training']['valid_ratio'], config['training']['valid_ratio']])
    
    print(f"Split: Train={len(train_tf.mapped_triples)}, Valid={len(valid_tf.mapped_triples)}, Test={len(test_tf.mapped_triples)}")
    
    # Run the pipeline with proper evaluation
    result = pipeline(
        training=train_tf,
        validation=valid_tf,
        testing=test_tf,
        model=config['training']['model'],
        model_kwargs={"embedding_dim": config['training']['embedding_dim']},
        training_kwargs={"num_epochs": config['training']['epochs']},
        random_seed=config['training']['random_seed'],
        evaluator=RankBasedEvaluator(),
        dataset_kwargs={"create_inverse_triples": False},
    )
    
    # Save model
    os.makedirs(output_path, exist_ok=True)
    result.save_to_directory(output_path)
    print(f"Model trained and saved to {output_path}")
    
    # Extract evaluation metrics
    metrics = {}
    
    # PyKEEN metrics
    metrics['mrr'] = result.metric_results.get('both.realistic.mr', 0.0)
    metrics['hits_at_1'] = result.metric_results.get('both.realistic.hits_at_1', 0.0)
    
    # Precision@K
    for k in config['eval']['k_list']:
        key = f'both.realistic.hits_at_{k}'
        if key in result.metric_results:
            metrics[f'precision_at_{k}'] = result.metric_results[key]
    
    # Compute AUROC
    auroc = compute_auroc(result.model, tf, test_tf)
    metrics['auroc'] = auroc
    
    # Save metrics
    metrics_path = config['outputs']['metrics']
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Metrics saved to {metrics_path}")
    print("\nEvaluation Results:")
    print(f"  MRR: {metrics['mrr']:.4f}")
    print(f"  AUROC: {metrics['auroc']:.4f}")
    print(f"  Hits@1: {metrics['hits_at_1']:.4f}")
    
    for k in config['eval']['k_list']:
        key = f'precision_at_{k}'
        if key in metrics:
            print(f"  Precision@{k}: {metrics[key]:.4f}")
    
    return result, metrics

if __name__ == "__main__":
    train_pykeen_model()
