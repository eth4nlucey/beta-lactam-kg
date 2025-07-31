import torch
import pandas as pd
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline

def train_kg_embeddings(kg_path, model_name="TransE", embedding_dim=64, epochs=200):
    """Train knowledge graph embeddings for adjuvant prediction"""
    
    print(f"🧠 Training {model_name} model with {embedding_dim}D embeddings...")
    
    # Load triples
    tf = TriplesFactory.from_path(kg_path)
    print(f"📊 Knowledge Graph: {tf.num_triples} triples, {tf.num_entities} entities, {tf.num_relations} relations")
    
    # Train model
    result = pipeline(
        training=tf,
        testing=tf,
        model=model_name,
        model_kwargs={"embedding_dim": embedding_dim},
        training_kwargs={"num_epochs": epochs},
        random_seed=42,
        dataset_kwargs={"create_inverse_triples": False}
    )
    
    return result

def predict_drug_combinations(model_result, beta_lactam_drug, top_k=10):
    """Use trained embeddings to predict drug combinations"""
    
    model = model_result.model
    tf = model_result.training
    
    # Get entity mappings
    entity_to_id = tf.entity_to_id
    
    # Find β-lactam drug ID
    if beta_lactam_drug not in entity_to_id:
        print(f"❌ {beta_lactam_drug} not found in knowledge graph")
        return []
    
    beta_lactam_id = entity_to_id[beta_lactam_drug]
    
    # Get all possible drug entities (exclude proteins)
    drug_entities = [entity for entity in entity_to_id.keys() 
                    if not any(protein_term in entity.lower() 
                             for protein_term in ['protein', 'binding', 'carrier', 'synthase', 'ase'])]
    
    print(f"🔍 Found {len(drug_entities)} potential drug entities")
    
    # Predict combinations using model embeddings
    predictions = []
    
    for drug in drug_entities:
        if drug != beta_lactam_drug:
            drug_id = entity_to_id[drug]
            
            # Create hypothetical triple: beta_lactam + targets + drug_target  
            targets_relation = "targets"
            if targets_relation in tf.relation_to_id:
                relation_id = tf.relation_to_id[targets_relation]
                
                # Score the potential combination
                triple_tensor = torch.tensor([[beta_lactam_id, relation_id, drug_id]])
                with torch.no_grad():
                    score = model.score_hrt(triple_tensor).item()
                
                predictions.append((drug, score))
    
    # Sort by score and return top predictions
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:top_k]