from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

def train_pykeen_model(triple_path, output_path="results/"):
    # Load triples from file
    tf = TriplesFactory.from_path(triple_path)

    # Run the pipeline using manual triples factory (not a named dataset)
    result = pipeline(
        training=tf,
        testing=tf,
        model="TransE",
        model_kwargs={"embedding_dim": 32},
        training_kwargs={"num_epochs": 100},
        random_seed=42,
        dataset_kwargs={"create_inverse_triples": False},  # Explicitly bypass default dataset loading
    )

    result.save_to_directory(output_path)
    print(f"✅ Model trained and saved to {output_path}")
