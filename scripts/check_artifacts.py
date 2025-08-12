#!/usr/bin/env python3
"""
Check integrity of all pipeline artifacts and verify outputs.
"""

import argparse
import json
import pandas as pd
from pathlib import Path
import yaml
from typing import Dict, List

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def check_file_exists(file_path: str, description: str) -> bool:
    """Check if a file exists and is non-empty."""
    path = Path(file_path)
    if not path.exists():
        print(f"❌ {description}: {file_path} - FILE MISSING")
        return False
    
    if path.stat().st_size == 0:
        print(f"❌ {description}: {file_path} - FILE EMPTY")
        return False
    
    print(f"✅ {description}: {file_path}")
    return True

def check_kg_artifacts(config: dict) -> Dict[str, int]:
    """Check knowledge graph artifacts and return counts."""
    kg_dir = Path(config['paths']['kg_dir'])
    
    print("\n📊 Knowledge Graph Artifacts:")
    
    # Check edges
    edges_path = kg_dir / 'edges.tsv'
    if check_file_exists(str(edges_path), "KG Edges"):
        edges_df = pd.read_csv(edges_path, sep='\t')
        edge_count = len(edges_df)
        print(f"   Edges: {edge_count:,}")
    else:
        edge_count = 0
    
    # Check nodes (entities)
    nodes_path = kg_dir / 'nodes.tsv'
    if check_file_exists(str(nodes_path), "KG Nodes"):
        nodes_df = pd.read_csv(nodes_path, sep='\t')
        node_count = len(nodes_df)
        print(f"   Nodes: {node_count:,}")
    else:
        node_count = 0
    
    # Check entity mapping
    entity_mapping_path = kg_dir / 'entity_mapping.tsv'
    if check_file_exists(str(entity_mapping_path), "KG Entity Mapping"):
        entity_mapping_df = pd.read_csv(entity_mapping_path, sep='\t')
        entity_mapping_count = len(entity_mapping_df)
        print(f"   Entity Mappings: {entity_mapping_count:,}")
    else:
        entity_mapping_count = 0
    
    # Check relations
    relations_path = kg_dir / 'relations.tsv'
    if check_file_exists(str(relations_path), "KG Relations"):
        relations_df = pd.read_csv(relations_path, sep='\t')
        relation_count = len(relations_df)
        print(f"   Relations: {relation_count:,}")
    else:
        relation_count = 0
    
    return {
        'edges': edge_count,
        'nodes': node_count,
        'entity_mappings': entity_mapping_count,
        'relations': relation_count
    }

def check_results_artifacts(config: dict) -> Dict[str, any]:
    """Check results artifacts and return key metrics."""
    results_dir = Path(config['paths']['results_dir'])
    
    print("\n📈 Results Artifacts:")
    
    # Check metrics
    metrics_path = results_dir / 'metrics.json'
    if check_file_exists(str(metrics_path), "Model Metrics"):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        test_metrics = metrics.get('test_metrics', {})
        print(f"   Test MRR: {test_metrics.get('mrr', 'N/A'):.4f}")
        print(f"   Test Hits@10: {test_metrics.get('hits_at_10', 'N/A'):.4f}")
        print(f"   Test AUROC: {test_metrics.get('auroc', 'N/A'):.4f}")
        
        return test_metrics
    else:
        return {}
    
    # Check predictions
    predictions_path = results_dir / 'predicted_links.tsv'
    if check_file_exists(str(predictions_path), "Adjuvant Predictions"):
        predictions_df = pd.read_csv(predictions_path, sep='\t')
        prediction_count = len(predictions_df)
        print(f"   Predictions: {prediction_count:,}")
        
        # Show top 5 predictions
        print("\n🏆 Top 5 Predictions:")
        top_predictions = predictions_df.head(5)
        for _, row in top_predictions.iterrows():
            print(f"   {row['head']} --[{row['relation']}]--> {row['tail']} (score: {row['score']:.4f})")
        
        return {'predictions': prediction_count}
    else:
        return {'predictions': 0}
    
    # Check validation
    validation_path = results_dir / 'validation' / 'validation_summary.tsv'
    if check_file_exists(str(validation_path), "Validation Results"):
        validation_df = pd.read_csv(validation_path, sep='\t')
        validation_count = len(validation_df)
        print(f"   Validations: {validation_count:,}")
        
        # Show validation summary
        if 'epmc_hit_count' in validation_df.columns:
            total_hits = validation_df['epmc_hit_count'].sum()
            print(f"   Total Europe PMC hits: {total_hits}")
        
        if 'drugcomb_found' in validation_df.columns:
            drugcomb_matches = validation_df['drugcomb_found'].sum()
            print(f"   DrugComb matches: {drugcomb_matches}")
        
        return {'validations': validation_count}
    else:
        return {'validations': 0}

def main():
    parser = argparse.ArgumentParser(description='Check integrity of pipeline artifacts')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    
    args = parser.parse_args()
    
    print("🔍 Pipeline Artifact Integrity Check")
    print("=" * 50)
    
    # Load configuration
    try:
        config = load_config(args.config)
        print(f"✅ Configuration loaded: {args.config}")
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return
    
    # Check KG artifacts
    kg_stats = check_kg_artifacts(config)
    
    # Check results artifacts
    results_stats = check_results_artifacts(config)
    
    # Summary
    print("\n📋 Summary:")
    print(f"   Knowledge Graph: {kg_stats['nodes']:,} nodes, {kg_stats['edges']:,} edges, {kg_stats['relations']} relations")
    
    if 'predictions' in results_stats:
        print(f"   Predictions: {results_stats['predictions']:,} adjuvant candidates")
    
    if 'validations' in results_stats:
        print(f"   Validations: {results_stats['validations']:,} computational validations")
    
    # Overall status
    total_checks = len(kg_stats) + len(results_stats)
    passed_checks = sum(1 for v in kg_stats.values() if v > 0) + sum(1 for v in results_stats.values() if v > 0)
    
    if passed_checks == total_checks:
        print(f"\n🎉 All artifacts present and valid! ({passed_checks}/{total_checks} checks passed)")
    else:
        print(f"\n⚠️  Some artifacts missing or invalid ({passed_checks}/{total_checks} checks passed)")

if __name__ == "__main__":
    main()
