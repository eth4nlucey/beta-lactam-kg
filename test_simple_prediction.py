from scripts.simple_link_predictor import simple_link_prediction

# Test simple link prediction
predictions = simple_link_prediction("data/comprehensive_kg.tsv", target_drug="Ampicillin")

print("🔮 Top drug similarity predictions for Ampicillin:")
for drug, score, num_targets in predictions:
    print(f" - {drug}: similarity = {score:.3f} ({num_targets} targets)")