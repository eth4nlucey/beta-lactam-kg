from scripts.drugbank_parser import parse_drugbank_xml, drugbank_to_triples
from scripts.combine_kg_data import create_comprehensive_kg
from scripts.simple_link_predictor import predict_adjuvant_combinations
from scripts.literature_validation import validate_top_predictions
import pandas as pd

print("🧬 BETA-LACTAM ADJUVANT DISCOVERY PIPELINE")
print("=" * 50)

# Step 1: Knowledge Graph Statistics
print("\n📊 KNOWLEDGE GRAPH STATISTICS:")
kg_df = pd.read_csv("data/comprehensive_kg.tsv", sep='\t', names=['head', 'relation', 'tail'])
print(f"   Total triples: {len(kg_df)}")
print(f"   Unique drugs: {len(kg_df['head'].unique())}")
print(f"   Unique targets: {len(kg_df['tail'].unique())}")

# Step 2: Generate predictions
print("\n🔮 GENERATING ADJUVANT PREDICTIONS:")
predictions = predict_adjuvant_combinations("data/comprehensive_kg.tsv", beta_lactam="Ampicillin", top_k=5)

# Step 3: Literature validation
print("\n📚 LITERATURE VALIDATION:")
validated = validate_top_predictions(predictions, max_combinations=3)

# Step 4: Final ranked results
print("\n🏆 FINAL RANKED ADJUVANT CANDIDATES:")
for i, result in enumerate(validated, 1):
    print(f"\n{i}. {result['combination']}")
    print(f"   Computational score: {result['prediction_score']:.2f}")
    print(f"   Shared protein targets: {result['shared_targets']}")
    print(f"   Literature support: {result['literature_evidence']} articles")
    
    # Calculate combined confidence score
    confidence = (result['prediction_score'] * 0.7) + (result['literature_evidence'] * 0.3)
    print(f"   Combined confidence: {confidence:.2f}")

print("\n✅ Analysis complete!")