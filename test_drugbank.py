from scripts.drugbank_parser import parse_drugbank_xml, drugbank_to_triples

beta_lactams = parse_drugbank_xml("data/drugbank/full-database.xml")
print(f"Found {len(beta_lactams)} β-lactam antibiotics")
drugbank_to_triples(beta_lactams, "data/drugbank_triples.tsv")
