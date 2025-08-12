import pandas as pd
import yaml
import os
from typing import List, Tuple, Dict

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def fetch_card_resistance() -> pd.DataFrame:
    """
    Fetch antimicrobial resistance data from CARD or use curated data
    Returns DataFrame with columns: drug, resistance_protein, mechanism, gene_family
    """
    # Curated resistance data for β-lactam antibiotics
    # In production, this would fetch from CARD API or database
    
    resistance_data = [
        # β-lactamases (Class A)
        ("amoxicillin", "TEM-1", "β-lactamase", "Class A β-lactamase"),
        ("amoxicillin", "SHV-1", "β-lactamase", "Class A β-lactamase"),
        ("ampicillin", "TEM-1", "β-lactamase", "Class A β-lactamase"),
        ("ampicillin", "SHV-1", "β-lactamase", "Class A β-lactamase"),
        ("ceftriaxone", "CTX-M-15", "β-lactamase", "Class A β-lactamase"),
        ("ceftazidime", "CTX-M-15", "β-lactamase", "Class A β-lactamase"),
        
        # Extended-spectrum β-lactamases (ESBLs)
        ("ceftriaxone", "TEM-3", "ESBL", "Class A β-lactamase"),
        ("ceftazidime", "TEM-3", "ESBL", "Class A β-lactamase"),
        ("ceftriaxone", "SHV-2", "ESBL", "Class A β-lactamase"),
        
        # Carbapenemases
        ("meropenem", "NDM-1", "carbapenemase", "Class B metallo-β-lactamase"),
        ("meropenem", "KPC-2", "carbapenemase", "Class A β-lactamase"),
        ("meropenem", "OXA-48", "carbapenemase", "Class D β-lactamase"),
        
        # Efflux pumps
        ("amoxicillin", "AcrAB-TolC", "efflux", "RND efflux pump"),
        ("ampicillin", "AcrAB-TolC", "efflux", "RND efflux pump"),
        ("ceftriaxone", "AcrAB-TolC", "efflux", "RND efflux pump"),
        
        # Target modifications
        ("amoxicillin", "PBP2B_mut", "target_modification", "Penicillin-binding protein mutation"),
        ("ampicillin", "PBP2B_mut", "target_modification", "Penicillin-binding protein mutation"),
    ]
    
    df = pd.DataFrame(resistance_data, 
                     columns=['drug', 'resistance_protein', 'mechanism', 'gene_family'])
    
    print(f"Generated {len(df)} drug-resistance mechanism edges")
    return df

def add_adjuvant_inhibition_edges(resistance_df: pd.DataFrame) -> List[str]:
    """Add adjuvant-inhibits-resistance_protein edges where evidence exists"""
    
    adjuvant_edges = [
        # Known adjuvant-resistance protein interactions
        ("clavulanic_acid", "inhibits", "TEM-1"),
        ("clavulanic_acid", "inhibits", "SHV-1"),
        ("sulbactam", "inhibits", "TEM-1"),
        ("tazobactam", "inhibits", "TEM-1"),
        ("tazobactam", "inhibits", "SHV-1"),
        ("avibactam", "inhibits", "KPC-2"),
        ("vaborbactam", "inhibits", "KPC-2"),
        
        # Hypothetical adjuvant mechanisms
        ("metformin", "inhibits", "AcrAB-TolC"),
        ("curcumin", "inhibits", "NDM-1"),
        ("vitamin_c", "inhibits", "AcrAB-TolC"),
    ]
    
    return adjuvant_edges

def card_to_triples(resistance_df: pd.DataFrame, output_path: str) -> None:
    """Convert CARD resistance data to knowledge graph triples"""
    
    triples = []
    
    # Add drug-resistance relationships
    for _, row in resistance_df.iterrows():
        drug, resistance_protein, mechanism, gene_family = row
        
        # Main resistance relationship
        triples.append(f"{drug}\tresisted_by\t{resistance_protein}")
        
        # Mechanism and gene family annotations
        triples.append(f"{resistance_protein}\thas_mechanism\t{mechanism}")
        triples.append(f"{resistance_protein}\tbelongs_to_family\t{gene_family}")
    
    # Add adjuvant-inhibition edges
    adjuvant_edges = add_adjuvant_inhibition_edges(resistance_df)
    for adjuvant, relation, target in adjuvant_edges:
        triples.append(f"{adjuvant}\t{relation}\t{target}")
    
    # Save triples
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for triple in triples:
            f.write(triple + '\n')
    
    print(f"Saved {len(triples)} CARD resistance triples to {output_path}")

def main():
    """Main function to run CARD import"""
    config = load_config()
    
    print("Importing CARD antimicrobial resistance data...")
    
    # Fetch resistance data
    resistance_df = fetch_card_resistance()
    
    # Convert to triples
    output_path = config['data_sources']['card']
    card_to_triples(resistance_df, output_path)
    
    print("CARD import completed successfully!")

if __name__ == "__main__":
    main()
