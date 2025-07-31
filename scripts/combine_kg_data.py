import pandas as pd

def combine_knowledge_sources(drugbank_path, string_path, output_path):
    """Combine DrugBank and STRING data into one knowledge graph"""
    
    # Read DrugBank triples
    drugbank_df = pd.read_csv(drugbank_path, sep='\t', names=['head', 'relation', 'tail'])
    print(f"📊 DrugBank: {len(drugbank_df)} triples")
    
    # Read STRING triples  
    string_df = pd.read_csv(string_path, sep='\t', names=['head', 'relation', 'tail'])
    print(f"📊 STRING: {len(string_df)} triples")
    
    # Combine datasets
    combined_df = pd.concat([drugbank_df, string_df], ignore_index=True)
    
    # Remove duplicates
    combined_df = combined_df.drop_duplicates()
    print(f"📊 Combined: {len(combined_df)} unique triples")
    
    # Save combined knowledge graph
    combined_df.to_csv(output_path, sep='\t', index=False, header=False)
    print(f"✅ Saved combined KG to {output_path}")

def create_comprehensive_kg():
    """Create a comprehensive knowledge graph from all sources"""
    combine_knowledge_sources(
        drugbank_path="data/drugbank_triples.tsv",
        string_path="data/kg_triples.tsv", 
        output_path="data/comprehensive_kg.tsv"
    )