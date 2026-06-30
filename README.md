# 🔮 ORI-GIN Redrob: Automated Universal Recruitment Analyzer

An AI-driven candidate evaluation engine designed to evaluate candidates based on deep contextual semantic understanding and structural profile alignment.

## 🚀 Core Architecture & Approach
Rather than relying on primitive string filtering, this system processes candidate profiles utilizing a Hybrid Scoring Logic Matrix:
1. **Semantic Layer (70% Weight):** Transforms the job description and candidate profiles into dense vector space embeddings using `all-MiniLM-L6-v2` to map true technical intent and contextual proximity.
2. **Seniority Timeline Layer (30% Weight):** Applies a mathematical deviation scaling factor against target experience parameters (default: 5 years).
3. **AURA Insights Engine:** Programmatically extracts high-signal contextual validation terms to generate a concise, human-readable justification for every candidate's ranking placement.

## ✨ Upgraded Features
- **Detailed Excel Export:** Generates formatted candidate shortlists in XLSX format containing full score breakdowns (`AI Semantic Match`, `Experience Match`, `Total AURA Score`, and `AURA Insight`).
- **Unified Insight Engine:** Modularized insight generator (`src/explainer.py`) used by both the Web and CLI versions.
- **Cross-Platform Compatibility:** Cleaned up dependencies for smooth installation on Windows, macOS, and Linux (CPU/device agnostic).

## 🛠️ Installation & Setup
Ensure you have Python 3.10+ installed locally, then initialize the environment by running these commands in your terminal:

```bash
# Clone the repository
git clone https://github.com/nh-44/ORI-GIN-Redrob.git
cd ORI-GIN-Redrob

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # On macOS/Linux use: source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

## 🎯 Usage

### 1. Run via CLI (Generates Excel Shortlist)
To run the ranking pipeline on the full dataset of 50 candidates and export the ranked output spreadsheet:
```bash
python main.py
```
This prints the leaderboard to the terminal and saves the output to `data/aura_ranked_candidates.xlsx`.

### 2. Run via Streamlit (Interactive Recruiter Dashboard)
To run the interactive recruiter canvas dashboard:
```bash
streamlit run app.py
```
From the web UI, you can dynamically tune weights and target experience parameters, view matches with color-coded status badges, and export the shortlist to CSV or Excel (`.xlsx`) with one click.