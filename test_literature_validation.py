from scripts.simple_link_predictor import predict_adjuvant_combinations
from scripts.literature_validation import validate_top_predictions

# Get predictions
predictions = predict_adjuvant_combinations("data/comprehensive_kg.tsv", beta_lactam="Ampicillin")

# Validate against literature
validated = validate_top_predictions(predictions, max_combinations=3)

# Display results
print("\n📊 VALIDATION RESULTS:")
for result in validated:
    print(f"\n💊 {result['combination']}")
    print(f"   Prediction score: {result['prediction_score']:.2f}")
    print(f"   Shared targets: {result['shared_targets']}")
    print(f"   Literature evidence: {result['literature_evidence']} articles")
    
    if result['articles']:
        print("   📄 Recent articles:")
        for article in result['articles'][:2]:
            print(f"     - {article['title'][:80]}...")