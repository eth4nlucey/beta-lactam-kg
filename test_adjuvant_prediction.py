from scripts.simple_link_predictor import predict_adjuvant_combinations

# Test adjuvant prediction
adjuvants = predict_adjuvant_combinations("data/comprehensive_kg.tsv", beta_lactam="Ampicillin")

print("💊 Top adjuvant predictions for Ampicillin:")
for drug, score, shared_targets in adjuvants:
    print(f" - {drug}: score = {score:.2f}")
    print(f"   Shared targets: {', '.join(shared_targets[:3])}...")
    print()python test_adjuvant_prediction.py