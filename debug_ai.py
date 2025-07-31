from scripts.ai_adjuvant_predictor import train_kg_embeddings, predict_drug_combinations

print("🧠 AI-POWERED ADJUVANT PREDICTION")

# Train model
model_result = train_kg_embeddings("data/comprehensive_kg.tsv", epochs=50)

# Debug: Check what entities we have
tf = model_result.training
print(f"\n🔍 All entities in KG:")
for entity in list(tf.entity_to_id.keys())[:15]:
    print(f"  - {entity}")

# Check if Ampicillin exists
if "Ampicillin" in tf.entity_to_id:
    print(f"\n✅ Ampicillin found in KG")
    predictions = predict_drug_combinations(model_result, "Ampicillin", top_k=5)
    
    print(f"\n🏆 AI PREDICTIONS:")
    for drug, score in predictions:
        print(f"  - {drug}: {score:.4f}")
else:
    print(f"\n❌ Ampicillin not found. Available drugs:")
    drugs = [e for e in tf.entity_to_id.keys() if not any(p in e.lower() for p in ['protein', 'binding'])]
    for drug in drugs[:10]:
        print(f"  - {drug}")