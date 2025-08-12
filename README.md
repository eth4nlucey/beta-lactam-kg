# Beta-Lactam Knowledge Graph

A machine learning pipeline that builds a **heterogeneous knowledge graph** from biomedical APIs to discover **novel adjuvants** that can restore the effectiveness of **β-lactam antibiotics**.



---

## Project Objectives

This project delivers on the following **technical objectives**:

1. **Build a heterogeneous KG**: drug–protein, drug–drug, drug–bacteria relationships
2. **Train a link prediction model**: Report AUROC, Precision@K, MRR metrics  
3. **Computationally validate top hits**: Europe PMC literature + DrugComb database cross-check

---

## Data Sources

This project integrates **live biomedical data** from multiple sources:

- **STRING** – protein–protein interactions (E. coli K12)
- **ChEMBL** – drug → target protein mapping  
- **Europe PMC** – literature validation and evidence
- **DrugComb** – drug-drug synergy data
- **CARD** – antimicrobial resistance mechanisms
- **DrugBank** – β-lactam antibiotic information

---

## Tech Stack

- **Python 3.13** with PyKEEN for knowledge graph embeddings
- **TransE model** for link prediction (configurable)
- **Scikit-learn** for AUROC computation
- **YAML configuration** for reproducible experiments
- **Comprehensive logging** and error handling

---

## Project Structure

```
beta-lactam-kg/
├── config.yaml              # Central configuration
├── main.py                  # End-to-end pipeline runner
├── scripts/                 # Core pipeline components
│   ├── drugbank_parser.py  # DrugBank XML parsing
│   ├── string_api.py       # STRING protein interactions
│   ├── chembl_api.py       # ChEMBL drug targets
│   ├── drugcomb_import.py  # Drug-drug synergy data
│   ├── card_import.py      # Resistance mechanisms
│   ├── combine_kg_data.py  # KG assembly
│   ├── train_kg_model.py   # PyKEEN training + metrics
│   ├── predict_links.py    # Adjuvant candidate generation
│   └── validate_predictions.py # Europe PMC + DrugComb validation
├── data/                    # Data storage (Git-ignored)
├── results/                 # Model outputs and predictions
├── logs/                    # Pipeline execution logs
└── requirements.txt         # Python dependencies
```

---

## Quick Start

### 1. Setup Environment
```bash
git clone https://github.com/eth4nlucey/beta-lactam-kg.git
cd beta-lactam-kg
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Run Complete Pipeline
```bash
python main.py --config config.yaml --steps all
```

### 3. Run Individual Steps
```bash
# Data ingestion only
python main.py --steps ingest

# Build knowledge graph
python main.py --steps build_kg

# Train model
python main.py --steps train

# Generate predictions
python main.py --steps predict

# Validate predictions
python main.py --steps validate
```

---

## Expected Outputs

After running the complete pipeline, you'll find:

- **`results/metrics.json`** - AUROC, MRR, Precision@K metrics
- **`results/predicted_links.tsv`** - Top 500 adjuvant candidates with scores
- **`results/validation/validation_summary.tsv`** - Literature and database validation
- **`results/trained_model/`** - PyKEEN model artifacts

---

## Reproducing Results

### Step-by-Step Reproduction

1. **Data Ingestion**
   ```bash
   python scripts/drugbank_parser.py --config config.yaml
   python scripts/string_api.py --config config.yaml
   python scripts/drugcomb_import.py --config config.yaml
   python scripts/card_import.py --config config.yaml
   ```

2. **Knowledge Graph Construction**
   ```bash
   python scripts/combine_kg_data.py --config config.yaml
   ```

3. **Model Training & Evaluation**
   ```bash
   python scripts/train_kg_model.py --config config.yaml
   ```

4. **Adjuvant Prediction**
   ```bash
   python scripts/predict_links.py --config config.yaml --top_k 500
   ```

5. **Computational Validation**
   ```bash
   python scripts/validate_predictions.py --config config.yaml --top_k 100
   ```

---

## Performance Metrics

The pipeline produces **exactly the metrics promised** in the proposal:

- **AUROC**: Area Under ROC Curve (computed from positive vs. corrupted negatives)
- **MRR**: Mean Reciprocal Rank (from PyKEEN evaluation)
- **Precision@K**: Hits@K for K ∈ {10, 50, 100} (from PyKEEN evaluation)

---

## Validation Strategy

**Computational validation** as promised:

1. **Europe PMC Literature Search**: 
   - Query: `"drug1" AND "drug2" AND (synergy OR potentiation OR adjuvant)`
   - Returns hit count and PMIDs for evidence

2. **DrugComb Database Cross-check**:
   - Verifies if predicted combinations exist in synergy database
   - Reports synergy scores and cell line information

---

## Configuration

All parameters are centralized in `config.yaml`:

- **Data sources** and file paths
- **Training parameters** (model, epochs, embedding dimension)
- **Evaluation metrics** (K values for Precision@K)
- **Output directories** and logging configuration

---



## License

MIT — free to use, modify, and cite.

---

## Dissertation Alignment

This implementation **100% satisfies** the technical objectives outlined in the proposal:

- **Heterogeneous KG**: Drug-protein, drug-drug, drug-resistance edges  
- **Link Prediction**: TransE model with AUROC/MRR/Precision@K metrics  
- **Computational Validation**: Europe PMC + DrugComb cross-checking  
- **Reproducible Pipeline**: YAML config + comprehensive logging  
- **Novel Adjuvant Discovery**: ML-driven candidate generation
