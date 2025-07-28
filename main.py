from scripts.string_api import get_string_interactions
from scripts.tsv_to_triples import string_tsv_to_triples
from scripts.train_kg_model import train_pykeen_model
from scripts.predict_links import predict_tail_entities

# --- Step 1: Define known targets of amoxicillin (E. coli proteins)
amoxicillin_targets = ["PBP2B", "ftsI", "murA"]

# --- Step 2: Fetch interactions from STRING (species ID 511145 = E. coli K12)
print(f"\n🔍 Fetching STRING interactions for amoxicillin targets:")
result = get_string_interactions(amoxicillin_targets, species=511145)

if result:
    with open("data/string_amoxicillin.tsv", "w") as f:
        f.write(result)
    print("✅ STRING interactions saved to data/string_amoxicillin.tsv")
else:
    print("❌ Failed to fetch STRING data")

# --- Step 3: Convert STRING data to KG triples
string_tsv_to_triples(
    input_path="data/string_amoxicillin.tsv",
    output_path="data/kg_triples.tsv"
)

# --- Step 4: Train a link prediction model using PyKEEN
train_pykeen_model(
    triple_path="data/kg_triples.tsv",
    output_path="results/amoxicillin_kg"
)

# --- Step 5: Predict interactions for ftsI
print("\n🔮 Predicting new interactions from model...")
candidates = ["murA", "PBP2B"]

try:
    top_predictions = predict_tail_entities(
        model_path="results/amoxicillin_kg/trained_model.pkl",
        triples_path="data/kg_triples.tsv",
        head="ftsI",
        relation="interacts_with",
        candidates=candidates
    )

    print("🧬 Ranked predictions:")
    for tail, score in top_predictions:
        print(f" - ftsI interacts_with {tail}: score = {score:.4f}")

except KeyError as e:
    print(f"⚠️ Skipped unknown entity: {e}")
