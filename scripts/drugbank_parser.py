#!/usr/bin/env python3
"""
DrugBank Full Database Parser
Processes the complete 1.5GB DrugBank XML to extract β-lactam antibiotics and relationships
"""

from lxml import etree
import csv
import sys
import gzip
from typing import Dict, Set, List, Tuple
import argparse

def iter_drugs(xml_path: str):
    """Stream parse DrugBank XML without loading entire file into memory"""
    context = etree.iterparse(xml_path, events=('end',), tag='{http://www.drugbank.ca}drug')
    
    for _, elem in context:
        yield elem
        elem.clear()
        # Clean up previous siblings to free memory
        while elem.getprevious() is not None:
            del elem.getparent()[0]

def extract_beta_lactam_info(drug_elem, ns: Dict[str, str]) -> Tuple[bool, Dict]:
    """Extract β-lactam specific information from drug element"""
    
    # Check if it's a β-lactam antibiotic
    drug_name = drug_elem.findtext('db:name', namespaces=ns, default='').lower()
    
    # Comprehensive β-lactam detection
    beta_lactam_keywords = [
        'penicillin', 'amoxicillin', 'ampicillin', 'cephalexin', 'cefazolin',
        'ceftriaxone', 'ceftazidime', 'meropenem', 'piperacillin', 'cefepime',
        'cefotaxime', 'cefoxitin', 'cefuroxime', 'cefaclor', 'cefadroxil',
        'carbapenem', 'monobactam', 'cephamycin', 'oxacillin', 'nafcillin',
        'dicloxacillin', 'cloxacillin', 'flucloxacillin', 'methicillin'
    ]
    
    is_beta_lactam = any(keyword in drug_name for keyword in beta_lactam_keywords)
    
    if not is_beta_lactam:
        return False, {}
    
    # Extract comprehensive drug information
    drug_info = {
        'drugbank_id': drug_elem.findtext('db:drugbank-id[@primary="true"]', namespaces=ns, default=''),
        'name': drug_elem.findtext('db:name', namespaces=ns, default=''),
        'atc_codes': [],
        'targets': [],
        'drug_interactions': [],
        'enzymes': [],
        'transporters': [],
        'carriers': []
    }
    
    # Extract ATC codes
    for atc_elem in drug_elem.findall('.//db:atc-code', namespaces=ns):
        atc_code = atc_elem.get('code')
        if atc_code:
            drug_info['atc_codes'].append(atc_code)
    
            # Extract targets
        for target_elem in drug_elem.findall('.//db:target', namespaces=ns):
            # Find polypeptide element first, then get its id attribute
            polypeptide = target_elem.find('.//db:polypeptide', namespaces=ns)
            uniprot_id = polypeptide.get('id') if polypeptide is not None else ''
            
            target_info = {
                'uniprot_id': uniprot_id,
                'gene_name': target_elem.findtext('.//db:gene-name', namespaces=ns, default=''),
                'organism': target_elem.findtext('db:organism', namespaces=ns, default=''),
                'known_action': target_elem.findtext('db:known-action', namespaces=ns, default='')
            }
            if target_info['uniprot_id']:
                drug_info['targets'].append(target_info)
    
    # Extract drug-drug interactions
    for ddi_elem in drug_elem.findall('.//db:drug-interactions/db:drug-interaction', namespaces=ns):
        ddi_id = ddi_elem.findtext('db:drugbank-id', namespaces=ns, default='')
        ddi_name = ddi_elem.findtext('db:name', namespaces=ns, default='')
        if ddi_id and ddi_name:
            drug_info['drug_interactions'].append({
                'drugbank_id': ddi_id,
                'name': ddi_name
            })
    
    # Extract enzymes (important for β-lactam resistance)
    for enzyme_elem in drug_elem.findall('.//db:enzymes/db:enzyme', namespaces=ns):
        # Find polypeptide element first, then get its id attribute
        polypeptide = enzyme_elem.find('.//db:polypeptide', namespaces=ns)
        uniprot_id = polypeptide.get('id') if polypeptide is not None else ''
        
        enzyme_info = {
            'uniprot_id': uniprot_id,
            'gene_name': enzyme_elem.findtext('.//db:gene-name', namespaces=ns, default=''),
            'organism': enzyme_elem.findtext('db:organism', namespaces=ns, default='')
        }
        if enzyme_info['uniprot_id']:
            drug_info['enzymes'].append(enzyme_info)
    
    return True, drug_info

def main(xml_path: str, out_drugs: str, out_proteins: str, out_edges: str):
    """Main parsing function - processes DrugBank XML and outputs structured data"""
    
    print(f"Processing DrugBank XML: {xml_path}")
    print(f"Output files: {out_drugs}, {out_proteins}, {out_edges}")
    
    # Open output files
    with open(out_drugs, 'w', newline='') as drugs_file, \
         open(out_proteins, 'w', newline='') as proteins_file, \
         open(out_edges, 'w', newline='') as edges_file:
        
        w_drugs = csv.writer(drugs_file, delimiter='\t')
        w_prots = csv.writer(proteins_file, delimiter='\t')
        w_edges = csv.writer(edges_file, delimiter='\t')
        
        # Write headers
        w_drugs.writerow(['drugbank_id', 'name', 'atc_codes', 'is_beta_lactam'])
        w_prots.writerow(['uniprot_id', 'gene_name', 'organism', 'protein_type'])
        w_edges.writerow(['head', 'relation', 'tail', 'source', 'confidence'])
        
        # Track seen entities to avoid duplicates
        seen_drugs: Set[str] = set()
        seen_proteins: Set[str] = set()
        edge_count = 0
        
        # DrugBank namespace
        ns = {'db': 'http://www.drugbank.ca'}
        
        # Process drugs
        for drug_elem in iter_drugs(xml_path):
            is_beta_lactam, drug_info = extract_beta_lactam_info(drug_elem, ns)
            
            # Skip if no drug info or no drugbank ID
            if not drug_info or not drug_info.get('drugbank_id'):
                continue
            
            # Write drug information
            if drug_info['drugbank_id'] not in seen_drugs:
                atc_codes_str = ';'.join(drug_info['atc_codes'])
                w_drugs.writerow([
                    drug_info['drugbank_id'],
                    drug_info['name'],
                    atc_codes_str,
                    '1' if is_beta_lactam else '0'
                ])
                seen_drugs.add(drug_info['drugbank_id'])
            
            # Write protein information and create edges
            for target in drug_info['targets']:
                if target['uniprot_id'] and target['uniprot_id'] not in seen_proteins:
                    w_prots.writerow([
                        target['uniprot_id'],
                        target['gene_name'],
                        target['organism'],
                        'target'
                    ])
                    seen_proteins.add(target['uniprot_id'])
                
                # Create drug-target edge
                w_edges.writerow([
                    drug_info['drugbank_id'],
                    'targets',
                    target['uniprot_id'],
                    'DrugBank',
                    '1.0'  # High confidence for direct DrugBank annotations
                ])
                edge_count += 1
            
            # Create drug-drug interaction edges
            for ddi in drug_info['drug_interactions']:
                w_edges.writerow([
                    drug_info['drugbank_id'],
                    'interacts_with',
                    ddi['drugbank_id'],
                    'DrugBank',
                    '0.8'  # Medium confidence for DDI
                ])
                edge_count += 1
            
            # Create drug-enzyme edges (important for resistance)
            for enzyme in drug_info['enzymes']:
                if enzyme['uniprot_id'] and enzyme['uniprot_id'] not in seen_proteins:
                    w_prots.writerow([
                        enzyme['uniprot_id'],
                        enzyme['gene_name'],
                        enzyme['organism'],
                        'enzyme'
                    ])
                    seen_proteins.add(enzyme['uniprot_id'])
                
                # Create drug-enzyme edge
                w_edges.writerow([
                    drug_info['drugbank_id'],
                    'metabolized_by',
                    enzyme['uniprot_id'],
                    'DrugBank',
                    '0.9'  # High confidence for enzyme relationships
                ])
                edge_count += 1
    
    print(f"Processing complete!")
    print(f"Drugs extracted: {len(seen_drugs)}")
    print(f"Proteins extracted: {len(seen_proteins)}")
    print(f"Edges created: {edge_count}")
    print(f"Output files: {out_drugs}, {out_proteins}, {out_edges}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse DrugBank XML and extract β-lactam antibiotics')
    parser.add_argument('--xml', required=True, help='Path to DrugBank full-database.xml')
    parser.add_argument('--out-drugs', default='data/drugbank_drugs.tsv', help='Output file for drugs')
    parser.add_argument('--out-proteins', default='data/drugbank_proteins.tsv', help='Output file for proteins')
    parser.add_argument('--out-edges', default='data/drugbank_edges.tsv', help='Output file for edges')
    
    args = parser.parse_args()
    
    # Create output directories if they don't exist
    import os
    os.makedirs(os.path.dirname(args.out_drugs), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_proteins), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_edges), exist_ok=True)
    
    main(args.xml, args.out_drugs, args.out_proteins, args.out_edges)