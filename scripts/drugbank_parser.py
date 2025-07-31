import xml.etree.ElementTree as ET

def parse_drugbank_xml(xml_path):
    """Parse DrugBank XML and extract β-lactam antibiotics with their targets"""
    print("🔍 Parsing DrugBank XML...")
    
    # Parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # DrugBank namespace
    ns = {'db': 'http://www.drugbank.ca'}
    
    beta_lactams = []
    
    for drug in root.findall('db:drug', ns):
        name = drug.find('db:name', ns)
        if name is not None:
            drug_name = name.text.lower()
            
            # Check if it's a β-lactam antibiotic
            if any(keyword in drug_name for keyword in ['penicillin', 'amoxicillin', 'ampicillin', 'cephalexin', 'cefazolin']):
                targets = []
                
                # Extract targets
                for target in drug.findall('.//db:target', ns):
                    target_name = target.find('db:name', ns)
                    if target_name is not None:
                        targets.append(target_name.text)
                
                beta_lactams.append({
                    'drug': name.text,
                    'targets': targets
                })
    
    return beta_lactams

def drugbank_to_triples(beta_lactams, output_path):
    """Convert β-lactam drug-target data to knowledge graph triples"""
    triples = []
    
    for drug_data in beta_lactams:
        drug = drug_data['drug']
        for target in drug_data['targets']:
            triples.append(f"{drug}\ttargets\t{target}")
    
    with open(output_path, 'w') as f:
        for triple in triples:
            f.write(triple + '\n')
    
    print(f"✅ Saved {len(triples)} DrugBank triples to {output_path}")