# 🔮 ORIGIN AURA: Automated Universal Recruitment Analyzer

An AI-driven candidate evaluation engine designed to move past rigid keyword matching and evaluate candidates based on deep contextual semantic understanding and structural profile alignment.

## 🚀 Core Architecture & Approach
Rather than relying on primitive string filtering (which misses exceptional cross-functional talent), AURA processes candidate profiles utilizing a Hybrid Scoring Logic Matrix:
1. Semantic Layer (70-100% Weight): Transforms the job description and full candidate profile arrays into dense vector space embeddings to map true technical intent and contextual proximity.
2. Seniority Timeline Layer (0-30% Weight): Applies a localized deviation scaling factor against target experience parameters.
3. AURA Insights Engine: Programmatically extracts high-signal contextual validation terms to generate a concise, human-readable justification for every candidate's ranking placement.

## 🛠️ Installation & Setup
Ensure you have Python 3.10+ installed locally, then initialize the node by running these commands in your terminal:

git clone your-github-repo-url
cd ai-recruitment-system

python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

pip install -r requirements.txt

streamlit run app.py         