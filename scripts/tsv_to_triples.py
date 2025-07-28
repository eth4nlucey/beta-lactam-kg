import pandas as pd

def string_tsv_to_triples(input_path, output_path):
    df = pd.read_csv(input_path, sep="\t")

    # Show column names for verification
    print("🔍 Detected columns:", df.columns.tolist())

    # Use correct columns from STRING
    if "preferredName_A" in df.columns and "preferredName_B" in df.columns:
        triples = df[["preferredName_A", "preferredName_B"]].copy()
        triples.columns = ["head", "tail"]
        triples["relation"] = "interacts_with"
        triples = triples[["head", "relation", "tail"]]

        triples.to_csv(output_path, sep="\t", index=False, header=False)
        print(f"✅ Saved triples to {output_path}")
    else:
        print("❌ Expected STRING columns not found")
