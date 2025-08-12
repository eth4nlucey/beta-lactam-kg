import torch
import pandas as pd
import yaml
import os
from pykeen.triples import TriplesFactory
from typing import List, Tuple, Dict

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def predict_tail_entities(model_path, triples_path, head, relation, candidates):
    """Predict tail entities for a given head and relation"""
    tf = TriplesFactory.from_path(triples_path)
    model = torch.load(model_path, weights_only=False).to("cpu").eval()

    # Get entity and relation ID mappings
    entity_to_id = tf.entity_to_id
    relation_to_id = tf.relation_to_id

    h_id = entity_to_id[head]
    r_id = relation_to_id[relation]

    scores = []
    for tail in candidates:
        t_id = entity_to_id[tail]
        hrt_tensor = torch.tensor([[h_id, r_id, t_id]])
        score = model.score_hrt(hrt_tensor).item()
        scores.append((tail, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

def predict_adjuvant_candidates(config_path: str = "config.yaml", top_k: int = 500):
    """Predict top adjuvant candidates for β-lactam antibiotics"""
    
    config = load_config(config_path)
    
    # Load trained model
    model_path = os.path.join(config['outputs']['model_dir'], 'trained_model.pkl')
    if not os.path.exists(model_path):
        print(f"❌ Trained model not found: {model_path}")
        print("Please run training first: python scripts/train_kg_model.py")
        return
    
    # Load knowledge graph
    kg_path = config['kg']['edges']
    if not os.path.exists(kg_path):
        print(f"❌ Knowledge graph not found: {kg_path}")
        return
    
    print("Predicting adjuvant candidates for β-lactam antibiotics...")
    
    # Load triples factory
    tf = TriplesFactory.from_path(kg_path)
    
    # Load model
    model = torch.load(model_path, weights_only=False).to("cpu").eval()
    
    # Define β-lactam antibiotics
    beta_lactams = [
        "amoxicillin", "ampicillin", "ceftriaxone", "ceftazidime", 
        "meropenem", "piperacillin", "cephalexin", "cefazolin"
    ]
    
    # Define potential adjuvant relations
    adjuvant_relations = ["synergizes_with", "inhibits", "targets"]
    
    all_predictions = []
    
    for beta_lactam in beta_lactams:
        if beta_lactam not in tf.entity_to_id:
            print(f"Warning: {beta_lactam} not found in knowledge graph, skipping...")
            continue
        
        print(f"Predicting adjuvants for {beta_lactam}...")
        
        # Get all potential adjuvant entities (exclude proteins and resistance mechanisms)
        potential_adjuvants = []
        for entity in tf.entity_to_id.keys():
            # Filter for drug-like entities
            if not any(term in entity.lower() for term in [
                'protein', 'binding', 'carrier', 'synthase', 'ase', 'pump', 
                'mut', 'tem', 'shv', 'ctx', 'ndm', 'kpc', 'oxa'
            ]):
                potential_adjuvants.append(entity)
        
        # Score potential combinations
        for relation in adjuvant_relations:
            if relation not in tf.relation_to_id:
                continue
                
            relation_id = tf.relation_to_id[relation]
            beta_lactam_id = tf.entity_to_id[beta_lactam]
            
            predictions = []
            
            for adjuvant in potential_adjuvants:
                if adjuvant == beta_lactam:
                    continue
                    
                adjuvant_id = tf.entity_to_id[adjuvant]
                
                # Score the potential combination
                triple_tensor = torch.tensor([[beta_lactam_id, relation_id, adjuvant_id]])
                with torch.no_grad():
                    score = model.score_hrt(triple_tensor).item()
                
                predictions.append({
                    'head': beta_lactam,
                    'relation': relation,
                    'tail': adjuvant,
                    'score': score
                })
            
            # Sort by score and take top predictions
            predictions.sort(key=lambda x: x['score'], reverse=True)
            top_predictions = predictions[:top_k//len(adjuvant_relations)]
            
            all_predictions.extend(top_predictions)
    
    # Combine all predictions and sort by score
    if all_predictions:
        predictions_df = pd.DataFrame(all_predictions)
        predictions_df = predictions_df.sort_values('score', ascending=False).head(top_k)
        
        # Save predictions
        output_path = config['outputs']['predictions']
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        predictions_df.to_csv(output_path, sep='\t', index=False)
        
        print(f"Saved {len(predictions_df)} top predictions to {output_path}")
        
        # Print top predictions
        print(f"\nTop 10 adjuvant predictions:")
        for _, row in predictions_df.head(10).iterrows():
            print(f"  {row['head']} {row['relation']} {row['tail']}: {row['score']:.4f}")
        
        return predictions_df
    else:
        print("No predictions generated")
        return None

def main():
    """Main function to run adjuvant prediction"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict adjuvant candidates for β-lactam antibiotics')
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--top_k', type=int, default=500, help='Number of top predictions to generate')
    
    args = parser.parse_args()
    
    print("Starting adjuvant prediction...")
    predictions = predict_adjuvant_candidates(args.config, args.top_k)
    print("Adjuvant prediction completed successfully!")

if __name__ == "__main__":
    main()
