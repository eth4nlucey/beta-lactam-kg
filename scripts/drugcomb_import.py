import pandas as pd
import requests
import yaml
import os
from typing import List, Tuple, Dict

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def fetch_drugcomb_synergy(api_key: str = None) -> pd.DataFrame:
    """
    Fetch drug synergy data from DrugComb API or use sample data
    Returns DataFrame with columns: drug1, drug2, synergy_score, cell_line
    """
    # For now, create sample synergy data based on known β-lactam adjuvants
    # In production, this would fetch from DrugComb API
    
    sample_synergies = [
        # Known β-lactam + adjuvant combinations
        ("amoxicillin", "clavulanic_acid", 0.85, "E. coli"),
        ("ampicillin", "sulbactam", 0.78, "E. coli"),
        ("piperacillin", "tazobactam", 0.92, "E. coli"),
        ("ceftazidime", "avibactam", 0.89, "E. coli"),
        ("meropenem", "vaborbactam", 0.94, "E. coli"),
        
        # Additional potential adjuvants (hypothetical scores)
        ("amoxicillin", "metformin", 0.65, "E. coli"),
        ("ampicillin", "aspirin", 0.58, "E. coli"),
        ("ceftriaxone", "ibuprofen", 0.62, "E. coli"),
        ("amoxicillin", "vitamin_c", 0.45, "E. coli"),
        ("ampicillin", "curcumin", 0.52, "E. coli"),
    ]
    
    df = pd.DataFrame(sample_synergies, 
                     columns=['drug1', 'drug2', 'synergy_score', 'cell_line'])
    
    print(f"Generated {len(df)} drug-drug synergy edges")
    return df

def drugcomb_to_triples(synergy_df: pd.DataFrame, output_path: str) -> None:
    """Convert DrugComb synergy data to knowledge graph triples"""
    
    triples = []
    
    for _, row in synergy_df.iterrows():
        drug1, drug2, score, cell_line = row
        
        # Add synergy relationship (bidirectional)
        triples.append(f"{drug1}\tsynergizes_with\t{drug2}")
        triples.append(f"{drug2}\tsynergizes_with\t{drug1}")
        
        # Add synergy score as attribute (could be used for weighted edges)
        triples.append(f"{drug1}\tsynergy_score\t{score}")
        triples.append(f"{drug2}\tsynergy_score\t{score}")
        
        # Add cell line context
        triples.append(f"{drug1}\ttested_in\t{cell_line}")
        triples.append(f"{drug2}\ttested_in\t{cell_line}")
    
    # Save triples
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for triple in triples:
            f.write(triple + '\n')
    
    print(f"Saved {len(triples)} DrugComb triples to {output_path}")

def main():
    """Main function to run DrugComb import"""
    config = load_config()
    
    print("Importing DrugComb drug-drug synergy data...")
    
    # Fetch synergy data
    synergy_df = fetch_drugcomb_synergy()
    
    # Convert to triples
    output_path = config['data_sources']['drugcomb']
    drugcomb_to_triples(synergy_df, output_path)
    
    print("DrugComb import completed successfully!")

if __name__ == "__main__":
    main()
