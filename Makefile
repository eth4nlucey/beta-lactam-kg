# β-Lactam Adjuvant Discovery Pipeline Makefile

.PHONY: help run clean ingest kg train predict validate report check dashboard

help:
	@echo "β-Lactam Adjuvant Discovery Pipeline"
	@echo ""
	@echo "Available targets:"
	@echo "  run        - Run complete pipeline (ingest -> kg -> train -> predict -> validate -> report)"
	@echo "  ingest     - Run data ingestion (DrugBank, STRING, ChEMBL, DrugComb, CARD)"
	@echo "  kg         - Assemble knowledge graph from all data sources"
	@echo "  train      - Train TransE model and evaluate performance"
	@echo "  predict    - Generate adjuvant predictions"
	@echo "  validate   - Validate predictions computationally"
	@echo "  report     - Generate final report and check artifacts"
	@echo "  check      - Check integrity of all pipeline artifacts"
	@echo "  dashboard  - Start the Next.js dashboard frontend"
	@echo "  clean      - Clean generated files and results"
	@echo "  help       - Show this help message"

run:
	python main.py --config config.yaml --steps ingest,kg,train,predict,validate,report

ingest:
	python main.py --config config.yaml --steps ingest

kg:
	python main.py --config config.yaml --steps kg

train:
	python main.py --config config.yaml --steps train

predict:
	python main.py --config config.yaml --steps predict

validate:
	python main.py --config config.yaml --steps validate

report:
	python main.py --config config.yaml --steps report

check:
	python scripts/check_artifacts.py --config config.yaml

dashboard:
	@echo "Starting Next.js dashboard..."
	@echo "Make sure you have the FastAPI backend running in another terminal:"
	@echo "  cd api && uvicorn app:app --reload --port 8000"
	@echo ""
	@echo "Starting frontend..."
	cd apps/dashboard && npm run dev

clean:
	rm -rf results/
	rm -rf .cache/
	rm -rf logs/
	rm -f data/kg/drugcomb_edges.tsv
	rm -f data/kg/card_edges.tsv
	@echo "Cleaned generated files and results"

# Development targets
dev-ingest:
	python scripts/drugcomb_import.py --config config.yaml
	python scripts/card_import.py --config config.yaml

dev-assemble:
	python scripts/assemble_kg.py --config config.yaml

dev-train:
	python scripts/train_mini_transe.py --config config.yaml

dev-predict:
	python scripts/predict_links.py --config config.yaml

dev-validate:
	python scripts/validate_predictions.py --config config.yaml

# Quick test run (smaller dataset)
test-run:
	python main.py --config config.yaml --steps ingest,kg,train,predict,validate,report

# Install dependencies
install:
	pip install -r requirements.txt

# Create necessary directories
setup:
	mkdir -p data/kg
	mkdir -p results
	mkdir -p .cache
	mkdir -p logs
	@echo "Created necessary directories"

# Show current status
status:
	@echo "Pipeline Status:"
	@python scripts/check_artifacts.py --config config.yaml
