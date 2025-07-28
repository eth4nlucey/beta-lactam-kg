# 🧬 Beta-Lactam Knowledge Graph

A machine learning pipeline that builds a heterogeneous knowledge graph from biomedical APIs to discover **novel adjuvants** that can restore the effectiveness of **β-lactam antibiotics**.

---

## 🧠 Project Aim

Develop a computational framework to identify promising drug–adjuvant combinations for resistant bacterial infections — focusing on **β-lactam antibiotic synergy**.

---

## 🧪 Data Sources

This project pulls **live biomedical data** using APIs:

- 🔗 **STRING** – protein–protein interactions  
- 💊 **ChEMBL** – drug → target protein mapping  
- 📚 **Europe PMC** – literature validation (planned)  
- 🔄 Optional: **DrugComb**, **DGIdb**

---

## 🧰 Tech Stack

- Python 3.13  
- Requests + Pandas  
- PyKEEN (link prediction)  
- VS Code for development  
- GitHub for version control

---

## 📂 Project Structure

\`\`\`
beta-lactam-kg/
├── scripts/           # API wrappers (STRING, ChEMBL, etc.)
├── data/              # Temporary raw data (ignored by Git)
├── models/            # Link prediction training code
├── results/           # Ranked predictions & evaluation
├── main.py            # Entry point for testing
├── requirements.txt   # Python dependencies
└── README.md
\`\`\`

---

## 🚀 Getting Started

\`\`\`bash
git clone https://github.com/eth4nlucey/beta-lactam-kg.git
cd beta-lactam-kg
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python main.py
\`\`\`

---

## 📍 Status

✅ ChEMBL → Target API working  
✅ STRING → Protein interactions working  
🔜 Literature validation via Europe PMC  
🔜 KG assembly & link prediction training (PyKEEN)

---

## 🧑‍🔬 Author

**Ethan Lucey**  
Digital producer turned machine learning researcher.  
📍 Newcastle University, MSc Computer Science

---

## 📜 License

MIT — free to use, modify, and cite.
