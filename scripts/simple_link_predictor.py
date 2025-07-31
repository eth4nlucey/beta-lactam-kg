import pandas as pd
from collections import defaultdict, Counter
import random

def simple_link_prediction(kg_path, target_drug="Ampicillin", top_k=10):
    """Simple link prediction based on target protein similarity"""
    
    # Load knowledge graph
    df = pd.read_csv(kg_path, sep='\t', names=['head', 'relation', 'tail'])
    
    # Find targets of the input drug
    drug_targets = df[df['head'] == target_drug]['tail'].tolist()
    print(f"🎯 {target_drug} targets: {len(drug_targets)} proteins")
    
    # Find other drugs and their targets
    other_drugs = df[df['head'] != target_drug]['head'].unique()
    
    # Calculate similarity scores
    similarity_scores = []
    
    for other_drug in other_drugs:
        other_targets = df[df['head'] == other_drug]['tail'].tolist()
        
        # Calculate Jaccard similarity
        intersection = len(set(drug_targets) & set(other_targets))
        union = len(set(drug_targets) | set(other_targets))
        
        if union > 0:
            jaccard_score = intersection / union
            similarity_scores.append((other_drug, jaccard_score, len(other_targets)))
    
    # Sort by similarity score
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    
    return similarity_scores[:top_k]

def predict_adjuvant_combinations(kg_path, beta_lactam="Ampicillin", top_k=5):
    """Predict potential adjuvant combinations based on shared protein targets"""
    
    df = pd.read_csv(kg_path, sep='\t', names=['head', 'relation', 'tail'])
    
    # Get β-lactam targets
    beta_lactam_targets = set(df[df['head'] == beta_lactam]['tail'].tolist())
    print(f"🎯 {beta_lactam} targets {len(beta_lactam_targets)} proteins")
    
    # Find all drugs that share targets (potential adjuvants)
    all_drugs = df['head'].unique()
    adjuvant_scores = []
    
    for drug in all_drugs:
        if drug != beta_lactam:
            drug_targets = set(df[df['head'] == drug]['tail'].tolist())
            shared_targets = beta_lactam_targets & drug_targets
            
            if len(shared_targets) > 0:
                # Score based on shared targets and unique targets
                shared_score = len(shared_targets)
                unique_score = len(drug_targets - beta_lactam_targets)
                combined_score = shared_score + (unique_score * 0.5)
                
                adjuvant_scores.append((drug, combined_score, list(shared_targets)))
    
    # Sort by combined score
    adjuvant_scores.sort(key=lambda x: x[1], reverse=True)
    
    return adjuvant_scores[:top_k]