#!/usr/bin/env python3
"""
STRING API Integration for Protein-Protein Interactions
Fetches real protein interaction data for E. coli K12 with confidence scores
"""

import requests
import pandas as pd
import time
import json
from typing import List, Dict, Optional
import argparse
import os

class STRINGAPI:
    """STRING API client for fetching protein-protein interactions"""
    
    def __init__(self, species_id: int = 511145):  # E. coli K12
        self.species_id = species_id
        self.base_url = "https://string-db.org/api"
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Beta-Lactam-KG/1.0 (Academic Research)"
        })
    
    def get_interactions_batch(self, protein_ids: List[str], required_score: int = 400) -> pd.DataFrame:
        """
        Fetch protein interactions for a batch of proteins
        
        Args:
            protein_ids: List of protein identifiers (UniProt, gene names, etc.)
            required_score: Minimum interaction score (0-1000)
        
        Returns:
            DataFrame with interactions
        """
        if not protein_ids:
            return pd.DataFrame()
        
        print(f"Fetching STRING interactions for {len(protein_ids)} proteins...")
        
        # STRING network API endpoint
        url = f"{self.base_url}/tsv/network"
        
        # Join protein IDs with newlines
        identifiers = "\n".join(protein_ids)
        
        params = {
            "identifiers": identifiers,
            "species": self.species_id,
            "required_score": required_score,
            "network_type": "physical",  # Physical interactions only
            "additional_network_nodes": 0,  # Don't add additional nodes
            "show_query_node_labels": 1
        }
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            if not response.text.strip():
                print("No interactions found")
                return pd.DataFrame()
            
            # Parse TSV response
            lines = response.text.strip().split('\n')
            if len(lines) <= 1:  # Only header or empty
                return pd.DataFrame()
            
            # Parse the TSV data
            interactions = []
            for line in lines[1:]:  # Skip header
                parts = line.split('\t')
                if len(parts) >= 12:
                    interaction = {
                        'stringId_A': parts[0],
                        'stringId_B': parts[1],
                        'preferredName_A': parts[2],
                        'preferredName_B': parts[3],
                        'ncbiTaxonId': parts[4],
                        'score': float(parts[5]),
                        'nscore': float(parts[6]),
                        'fscore': float(parts[7]),
                        'pscore': float(parts[8]),
                        'ascore': float(parts[9]),
                        'escore': float(parts[10]),
                        'dscore': float(parts[11]),
                        'tscore': float(parts[12]) if len(parts) > 12 else 0.0
                    }
                    interactions.append(interaction)
            
            df = pd.DataFrame(interactions)
            print(f"Retrieved {len(df)} interactions")
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching STRING data: {e}")
            return pd.DataFrame()
    
    def get_interactions_for_proteins(self, protein_list: List[str], batch_size: int = 100) -> pd.DataFrame:
        """
        Fetch interactions for a list of proteins, processing in batches
        
        Args:
            protein_list: List of protein identifiers
            batch_size: Number of proteins to process per batch
        
        Returns:
            Combined DataFrame of all interactions
        """
        all_interactions = []
        
        # Process in batches to avoid overwhelming the API
        for i in range(0, len(protein_list), batch_size):
            batch = protein_list[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(protein_list) + batch_size - 1)//batch_size}")
            
            batch_interactions = self.get_interactions_batch(batch)
            if not batch_interactions.empty:
                all_interactions.append(batch_interactions)
            
            # Rate limiting - be respectful to the API
            if i + batch_size < len(protein_list):
                time.sleep(2)
        
        if not all_interactions:
            return pd.DataFrame()
        
        # Combine all batches
        combined_df = pd.concat(all_interactions, ignore_index=True)
        
        # Remove duplicates (same interaction might appear in multiple batches)
        combined_df = combined_df.drop_duplicates(subset=['stringId_A', 'stringId_B'])
        
        print(f"Total unique interactions: {len(combined_df)}")
        return combined_df
    
    def convert_to_kg_edges(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert STRING interactions to knowledge graph edge format
        
        Args:
            interactions_df: DataFrame from STRING API
        
        Returns:
            DataFrame in KG edge format
        """
        if interactions_df.empty:
            return pd.DataFrame()
        
        # Convert to knowledge graph format
        kg_edges = []
        
        for _, row in interactions_df.iterrows():
            # Create bidirectional edges (protein A interacts with protein B)
            edge_forward = {
                'head': row['preferredName_A'],
                'relation': 'interacts_with',
                'tail': row['preferredName_B'],
                'source': 'STRING',
                'confidence': row['score'] / 1000.0,  # Normalize to 0-1
                'interaction_score': row['score'],
                'combined_score': row['tscore'] / 1000.0 if row['tscore'] > 0 else row['score'] / 1000.0
            }
            
            edge_backward = {
                'head': row['preferredName_B'],
                'relation': 'interacts_with',
                'tail': row['preferredName_A'],
                'source': 'STRING',
                'confidence': row['score'] / 1000.0,
                'interaction_score': row['score'],
                'combined_score': row['tscore'] / 1000.0 if row['tscore'] > 0 else row['score'] / 1000.0
            }
            
            kg_edges.extend([edge_forward, edge_backward])
        
        return pd.DataFrame(kg_edges)

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Fetch STRING protein interactions')
    parser.add_argument('--proteins', required=True, help='File containing protein identifiers (one per line)')
    parser.add_argument('--output', default='data/string_interactions.tsv', help='Output file for interactions')
    parser.add_argument('--species', type=int, default=511145, help='NCBI species ID (default: E. coli K12)')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size for API calls')
    parser.add_argument('--min-score', type=int, default=400, help='Minimum interaction score (0-1000)')
    
    args = parser.parse_args()
    
    # Read protein list
    if not os.path.exists(args.proteins):
        print(f"Error: Protein file {args.proteins} not found")
        return
    
    with open(args.proteins, 'r') as f:
        protein_list = [line.strip() for line in f if line.strip()]
    
    if not protein_list:
        print("Error: No proteins found in file")
        return
    
    print(f"Processing {len(protein_list)} proteins...")
    
    # Initialize STRING API
    string_api = STRINGAPI(species_id=args.species)
    
    # Fetch interactions
    interactions_df = string_api.get_interactions_for_proteins(
        protein_list, 
        batch_size=args.batch_size
    )
    
    if interactions_df.empty:
        print("No interactions found")
        return
    
    # Convert to knowledge graph format
    kg_edges = string_api.convert_to_kg_edges(interactions_df)
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Save interactions
    kg_edges.to_csv(args.output, sep='\t', index=False)
    print(f"Saved {len(kg_edges)} knowledge graph edges to {args.output}")
    
    # Print summary statistics
    print(f"\nSummary:")
    print(f"  Input proteins: {len(protein_list)}")
    print(f"  Raw interactions: {len(interactions_df)}")
    print(f"  KG edges: {len(kg_edges)}")
    print(f"  Average confidence: {kg_edges['confidence'].mean():.3f}")

if __name__ == "__main__":
    main()
