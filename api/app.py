#!/usr/bin/env python3
"""
FastAPI backend for β-lactam adjuvant discovery dashboard.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import yaml

app = FastAPI(
    title="β-Lactam Adjuvant Discovery API",
    description="API for exploring knowledge graph results and adjuvant predictions",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_config() -> dict:
    """Load configuration from YAML file."""
    config_path = Path("../config.yaml")
    if not config_path.exists():
        config_path = Path("config.yaml")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_results_dir() -> Path:
    """Get results directory path."""
    config = load_config()
    return Path(config['paths']['results_dir'])

def get_kg_dir() -> Path:
    """Get knowledge graph directory path."""
    config = load_config()
    return Path(config['paths']['kg_dir'])

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "β-Lactam Adjuvant Discovery API",
        "version": "1.0.0",
        "endpoints": [
            "/api/metrics",
            "/api/predictions",
            "/api/validation", 
            "/api/kg/stats",
            "/api/kg/sample"
        ]
    }

@app.get("/api/metrics")
async def get_metrics():
    """Get model performance metrics."""
    try:
        metrics_path = get_results_dir() / "metrics.json"
        if not metrics_path.exists():
            raise HTTPException(status_code=404, detail="Metrics file not found")
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Extract key metrics
        test_metrics = metrics.get('test_metrics', {})
        model_config = metrics.get('model_config', {})
        data_stats = metrics.get('data_stats', {})
        
        return {
            "mrr": test_metrics.get('mrr', 0.0),
            "hits_at_1": test_metrics.get('hits_at_k', {}).get('1', 0.0),
            "hits_at_3": test_metrics.get('hits_at_k', {}).get('3', 0.0),
            "hits_at_10": test_metrics.get('hits_at_k', {}).get('10', 0.0),
            "auroc": test_metrics.get('auroc', 0.0),
            "epochs": model_config.get('epochs', 0),
            "embedding_dim": model_config.get('embedding_dim', 0),
            "num_entities": data_stats.get('num_entities', 0),
            "num_relations": data_stats.get('num_relations', 0),
            "train_size": data_stats.get('train_size', 0),
            "valid_size": data_stats.get('valid_size', 0),
            "test_size": data_stats.get('test_size', 0)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading metrics: {str(e)}")

@app.get("/api/predictions")
async def get_predictions(limit: int = 100, drug: Optional[str] = None):
    """Get adjuvant predictions."""
    try:
        predictions_path = get_results_dir() / "predicted_links.tsv"
        if not predictions_path.exists():
            raise HTTPException(status_code=404, detail="Predictions file not found")
        
        df = pd.read_csv(predictions_path, sep='\t')
        
        # Filter by drug if specified
        if drug:
            df = df[df['head'].str.contains(drug, case=False, na=False)]
        
        # Sort by score and limit results
        df = df.sort_values('score', ascending=False).head(limit)
        
        # Convert to list of dictionaries
        predictions = []
        for _, row in df.iterrows():
            predictions.append({
                "drug": row['head'],
                "adjuvant": row['tail'],
                "relation": row['relation'],
                "score": float(row['score'])
            })
        
        return {
            "predictions": predictions,
            "total": len(predictions),
            "filtered_by": drug if drug else None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading predictions: {str(e)}")

@app.get("/api/validation")
async def get_validation(limit: int = 100):
    """Get validation results."""
    try:
        validation_path = get_results_dir() / "validation" / "validation_summary.tsv"
        if not validation_path.exists():
            raise HTTPException(status_code=404, detail="Validation file not found")
        
        df = pd.read_csv(validation_path, sep='\t')
        
        # Sort by prediction score and limit results
        df = df.sort_values('prediction_score', ascending=False).head(limit)
        
        # Convert to list of dictionaries
        validations = []
        for _, row in df.iterrows():
            validation = {
                "drug_a": row['drug_a'],
                "drug_b": row['drug_b'],
                "model_score": float(row['prediction_score']),
                "epmc_hits": int(row.get('epmc_hit_count', 0)),
                "top_pmids": row.get('top_pmids', '').split(';') if pd.notna(row.get('top_pmids')) else [],
                "drugcomb_found": bool(row.get('drugcomb_found', False)),
                "synergy_metric": row.get('synergy_metric', ''),
                "synergy_value": float(row.get('synergy_value', 0.0)) if pd.notna(row.get('synergy_value')) else None
            }
            validations.append(validation)
        
        return {
            "validations": validations,
            "total": len(validations)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading validation: {str(e)}")

@app.get("/api/kg/stats")
async def get_kg_stats():
    """Get knowledge graph statistics."""
    try:
        kg_dir = get_kg_dir()
        
        # Load edges
        edges_path = kg_dir / "edges.tsv"
        if not edges_path.exists():
            raise HTTPException(status_code=404, detail="KG edges file not found")
        
        edges_df = pd.read_csv(edges_path, sep='\t')
        
        # Load entities
        entities_path = kg_dir / "entities.tsv"
        if not entities_path.exists():
            raise HTTPException(status_code=404, detail="KG entities file not found")
        
        entities_df = pd.read_csv(entities_path, sep='\t')
        
        # Load relations
        relations_path = kg_dir / "relations.tsv"
        if not relations_path.exists():
            raise HTTPException(status_code=404, detail="KG relations file not found")
        
        relations_df = pd.read_csv(relations_path, sep='\t')
        
        # Calculate statistics
        entity_type_counts = entities_df['type'].value_counts().to_dict()
        relation_counts = edges_df['relation'].value_counts().to_dict()
        source_counts = edges_df['source'].value_counts().to_dict()
        
        return {
            "total_entities": len(entities_df),
            "total_edges": len(edges_df),
            "total_relations": len(relations_df),
            "entity_types": entity_type_counts,
            "relation_counts": relation_counts,
            "data_sources": source_counts
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading KG stats: {str(e)}")

@app.get("/api/kg/sample")
async def get_kg_sample(n: int = 500):
    """Get a random sample of the knowledge graph for visualization."""
    try:
        kg_dir = get_kg_dir()
        
        # Load edges
        edges_path = kg_dir / "edges.tsv"
        if not edges_path.exists():
            raise HTTPException(status_code=404, detail="KG edges file not found")
        
        edges_df = pd.read_csv(edges_path, sep='\t')
        
        # Load entities
        entities_path = kg_dir / "entities.tsv"
        if not entities_path.exists():
            raise HTTPException(status_code=404, detail="KG entities file not found")
        
        entities_df = pd.read_csv(entities_path, sep='\t')
        
        # Sample edges
        if len(edges_df) > n:
            sample_edges = edges_df.sample(n=n, random_state=42)
        else:
            sample_edges = edges_df
        
        # Get unique entities from sample
        unique_entities = set()
        for _, row in sample_edges.iterrows():
            unique_entities.add(row['head'])
            unique_entities.add(row['tail'])
        
        # Get entity details
        sample_entities = entities_df[entities_df['id'].isin(unique_entities)]
        
        # Convert to Cytoscape format
        nodes = []
        for _, row in sample_entities.iterrows():
            nodes.append({
                "id": row['id'],
                "label": row['original_name'],
                "type": row['type']
            })
        
        edges = []
        for _, row in sample_edges.iterrows():
            edges.append({
                "source": row['head'],
                "target": row['tail'],
                "relation": row['relation'],
                "weight": float(row.get('weight', 1.0)),
                "source_db": row.get('source', 'unknown')
            })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "total_nodes": len(nodes),
            "total_edges": len(edges)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading KG sample: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": pd.Timestamp.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
