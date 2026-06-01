import streamlit as st
import pandas as pd
import os
import numpy as np
from src.parser import load_and_preprocess_data
from src.embedder import RecruitmentEmbedder

# Configure page layout
st.set_page_config(page_title="ORIGIN AURA", page_icon="🔮", layout="wide")

# 🎨 Custom UI Architecture Engine
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght=300;400;600;700&display=swap');
    
    /* Global Canvas Foundation */
    .stApp {
        background-color: #0A0D0B !important;
        color: #E2E8F0 !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* 📋 STEP 2: TOTAL BLACKOUT FILE UPLOADER CONTRAST FIX */
    div[data-testid="stFileUploader"] {
        border: 2px solid #00FF66 !important;
        background-color: #0A0D0B !important;
        border-radius: 8px !important;
        padding: 20px !important;
    }
    
    /* Force Dropzone Box to be Solid Black with an Accent Border */
    div[data-testid="stFileUploaderDropzone"], 
    .stUploadDropzone,
    [data-testid="stFileUploader"] section {
        background-color: #0A0D0B !important;
        border: 2px dashed #FF8C00 !important;
        border-radius: 6px !important;
        padding: 15px !important;
    }
    
    /* Force every single label/text layer inside the uploader to be CRISP WHITE */
    div[data-testid="stFileUploaderDropzone"] span, 
    div[data-testid="stFileUploaderDropzone"] p,
    div[data-testid="stFileUploaderDropzone"] small,
    div[data-testid="stFileUploaderDropzone"] div,
    div[data-testid="stFileUploaderDropzone"] label,
    div[data-testid="stFileUploader"] section,
    div[data-testid="stFileUploader"] section * {
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
    }
    
    /* 💥 UNIFIED FEATURE FIX: MATCHING BROWSE FILES TO THE MAIN SYNTHESIS EXECUTION BUTTON */
    div[data-testid="stFileUploaderDropzone"] button,
    .stUploadDropzone button,
    [data-testid="stFileUploader"] button {
        background: linear-gradient(135deg, #FF8C00 0%, #FF5500 100%) !important;
        color: #0A0D0B !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 12px 35px !important;
        box-shadow: 0 0 15px rgba(255, 140, 0, 0.4) !important;
        text-shadow: none !important;
        transition: all 0.2s ease-in-out !important;
    }
    
    div[data-testid="stFileUploaderDropzone"] button:hover,
    .stUploadDropzone button:hover,
    [data-testid="stFileUploader"] button:hover {
        background: #00FF66 !important;
        color: #0A0D0B !important;
        border: none !important;
        box-shadow: 0 0 20px rgba(0, 255, 102, 0.6) !important;
    }

    /* 🧠 Streamlit Popups, Menus & Deploy Modal Overrides */
    header, [data-testid="stHeader"] { background-color: #0A0D0B !important; }
    div[data-testid="stModalBody"], div[role="dialog"] {
        background-color: #0D120F !important; color: #E2E8F0 !important; border: 1px solid #00FF66 !important;
    }
    div[data-testid="stModalBody"] div { background-color: #0D120F !important; color: #E2E8F0 !important; }
    div[data-testid="stModalBody"] p, div[data-testid="stModalBody"] h2 { color: #E2E8F0 !important; }
    div[data-baseweb="popover"], [role="listbox"] { background-color: #0D120F !important; border: 1px solid #00FF66 !important; }
    div[data-baseweb="popover"] li { background-color: #0D120F !important; color: #E2E8F0 !important; }
    div[data-baseweb="popover"] li:hover { background-color: #141A17 !important; color: #00FF66 !important; }

    /* Structural Theme Text Blocks */
    .main-title { color: #FF8C00 !important; font-weight: 700 !important; margin-bottom: 5px; }
    .sub-title { color: #00FF66 !important; font-weight: 400 !important; margin-bottom: 25px; }
    .step-header { color: #FF8C00 !important; font-size: 1.35rem !important; font-weight: 600 !important; margin-top: 25px !important; margin-bottom: 10px !important; }
    
    /* Sidebar Layout Configuration */
    [data-testid="stSidebar"] { background-color: #050706 !important; border-right: 2px solid #FF8C00 !important; }
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] p { color: #FF8C00 !important; font-weight: 600 !important; }
    p, label, span, .stMarkdown, [data-testid="stWidgetLabel"] p { color: #E2E8F0 !important; }
    
    /* Step Input Card Formatting */
    div.element-container:has(div.stTextArea), .stAlert {
        border: 1px solid #00FF66 !important; background-color: #0D120F !important; border-radius: 8px !important; padding: 20px !important; margin-bottom: 15px !important;
    }
    .stAlert p { color: #00FF66 !important; font-weight: 500 !important; }
    textarea, input { background-color: #141A17 !important; color: #FFFFFF !important; border: 1px solid #FF8C00 !important; border-radius: 4px !important; }
    
    /* 📝 CUSTOM SCHEMA INFOBAR NOTE */
    .schema-note {
        background-color: #0D120F !important;
        border-left: 4px solid #FF8C00 !important;
        padding: 12px 16px !important;
        border-radius: 4px !important;
        margin-bottom: 12px !important;
    }
    .schema-note code {
        background-color: #141A17 !important;
        color: #00FF66 !important;
        padding: 2px 6px !important;
        border-radius: 3px !important;
        font-family: monospace !important;
    }

    /* Central Execution Launch Engine Button */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #FF8C00 0%, #FF5500 100%) !important; color: #0A0D0B !important; font-weight: 700 !important; border: none !important; border-radius: 6px !important; padding: 12px 35px !important; box-shadow: 0 0 15px rgba(255, 140, 0, 0.4);
    }
    div.stButton > button:first-child:hover { background: #00FF66 !important; color: #0A0D0B !important; box-shadow: 0 0 20px rgba(0, 255, 102, 0.6); }

    /* 📊 LEADERBOARD DATA GRID COMPONENT DESIGN TWEAKS */
    .aura-grid-container {
        background-color: #0D120F;
        border: 1px solid #00FF66;
        border-radius: 8px;
        padding: 15px;
        margin-top: 15px;
        overflow-x: auto;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    }
    .aura-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.95rem;
        letter-spacing: 0.03em;
    }
    .aura-table th {
        color: #FF8C00 !important;
        background-color: #050706;
        padding: 14px 16px !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        font-size: 0.85rem;
        border-bottom: 2px solid #FF8C00 !important;
    }
    
    /* Layout clean columns alignment */
    .aura-table th:nth-child(1), .aura-table td:nth-child(1) { text-align: center !important; width: 60px; font-weight: bold; }
    .aura-table th:nth-child(2), .aura-table td:nth-child(2) { text-align: left !important; font-weight: 600; }
    .aura-table th:nth-child(3), .aura-table td:nth-child(3) { text-align: center !important; width: 130px; }
    .aura-table th:nth-child(4), .aura-table td:nth-child(4) { text-align: center !important; width: 150px; }
    .aura-table th:nth-child(5), .aura-table td:nth-child(5) { text-align: center !important; width: 150px; }
    .aura-table th:nth-child(6), .aura-table td:nth-child(6) { text-align: center !important; width: 140px; font-weight: bold; color: #00FF66; }
    
    /* Target last column (AURA Insight) for custom text layout packaging */
    .aura-table th:nth-child(7), .aura-table td:nth-child(7) { 
        text-align: left !important; 
        color: #94A3B8 !important;
        font-size: 0.9rem !important;
        max-width: 420px;
        line-height: 1.4;
    }

    .aura-table td {
        padding: 14px 16px !important;
        border-bottom: 1px solid #1C2420 !important;
        color: #E2E8F0 !important;
        vertical-align: middle !important;
    }
    .aura-table tr:hover {
        background-color: #141A17 !important;
    }

    /* Core Metrics Graphics Styling Blocks */
    .score-block {
        display: inline-block;
        width: 100px;
        padding: 5px 8px;
        text-align: center;
        border-radius: 4px;
        font-weight: bold;
        color: #0A0D0B;
        font-size: 0.9rem;
        box-shadow: 0 0 8px rgba(0,0,0,0.3);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Render Global Headers
st.markdown("<h1 class='main-title'>🔮 ORIGIN AURA: Automated Universal Recruitment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<h5 class='sub-title'><i>Intelligence Over Keywords. Precision Over Guesswork.</i></h5>", unsafe_allow_html=True)
st.markdown("<hr style='border-top: 1px solid #FF8C00; margin-bottom: 30px;'>", unsafe_allow_html=True)

# Sidebar Control Tuner
st.sidebar.markdown("<h2 style='color: #FF8C00; font-weight:700; font-size:1.6rem; margin-bottom:15px;'>🎛️ Core Tuning</h2>", unsafe_allow_html=True)
ranking_method = st.sidebar.selectbox("Select Ranking Method", ["Balanced Hybrid", "Strictly Semantic (AI Focus)", "Experience First"])
target_exp = st.sidebar.slider("Target Experience (Years)", 0, 15, 5)

if ranking_method == "Balanced Hybrid":
    semantic_weight, exp_weight = 0.70, 0.30
    st.sidebar.info("💡 Strategy: Balances semantic understanding with seniority markers.")
elif ranking_method == "Strictly Semantic (AI Focus)":
    semantic_weight, exp_weight = 1.00, 0.00
    st.sidebar.info("💡 Strategy: Prioritizes contextual skills mapping. Longevity is completely bypassed.")
else:
    semantic_weight, exp_weight = 0.40, 0.60
    st.sidebar.info("💡 Strategy: Weighs candidate timeline parameters heavier than semantic vectors.")

# Matrix Processing Pipelines
st.markdown("<div class='step-header'>📋 Step 1: Define Role Specifications</div>", unsafe_allow_html=True)
default_jd = "Looking for a Robotics & Edge AI Engineer experienced in ROS2, computer vision, C++, and deploying optimized deep learning models on NVIDIA Jetson hardware."
jd_text = st.text_area("Target Job Profile Parameters", value=default_jd, height=110, label_visibility="collapsed")

# 📁 Step 2 Layout with Frontend Guidelines Note Block
st.markdown("<div class='step-header'>📁 Step 2: Stream Candidate Profiles</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class='schema-note'>
        ⚠️ <b>Data Integrity Protocol:</b> The uploaded <code>.csv</code> file matrix must contain these exact headers: 
        <code>name</code>, <code>years_of_experience</code>, <code>skills</code>, and <code>experience_summary</code>.
    </div>
    """, 
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("Upload candidate dataset (.CSV) directly into the AURA gateway node", type=["csv"], label_visibility="collapsed")

# Custom micro-renderer for solid, high-contrast matrix color pills
def format_score_block(percentage):
    pct = int(round(percentage))
    if pct >= 85:
        bg_color = "#00FF66" # Premium Green
        text_color = "#0A0D0B"
    elif pct >= 60:
        bg_color = "#7FFF00" # Soft Light Green
        text_color = "#0A0D0B"
    elif pct >= 45:
        bg_color = "#FFA500" # Alert Orange
        text_color = "#0A0D0B"
    else:
        bg_color = "#FF3333" # Critical Red
        text_color = "#FFFFFF"
        
    return f'<div class="score-block" style="background-color: {bg_color}; color: {text_color};">{pct}%</div>'

# 🧠 INTERACTION ENGINE: Generates a 1-sentence analytical insight summary dynamically
def generate_aura_insight(row, jd_string, ai_score, exp_score, target_y):
    matched_keywords = []
    jd_words = [w.strip(",.()").lower() for w in jd_string.split() if len(w) > 2]
    cand_text = (str(row['skills']) + " " + str(row['experience_summary'])).lower()
    
    for word in jd_words:
        if word in cand_text and word not in matched_keywords and word not in ['for', 'and', 'with', 'the', 'looking', 'experienced']:
            matched_keywords.append(word)
            if len(matched_keywords) >= 3:
                break
                
    keywords_str = ", ".join(matched_keywords[:2]).upper()
    
    if ai_score >= 80 and abs(row['years_of_experience'] - target_y) <= 2:
        return f"Elite alignment: Exceptional structural grasp of {keywords_str} with optimal seniority timeline placement."
    elif ai_score >= 70:
        return f"Strong conceptual match with deep technical exposure highlighting {keywords_str} proficiencies."
    elif row['years_of_experience'] >= target_y and ai_score >= 50:
        return f"Ranked primarily on solid seniority foundation; technical exposure to {keywords_str if keywords_str else 'core role metrics'} acts as a secondary buffer."
    elif ai_score < 45:
        return "Critical contextual divergence; missing core technology proficiencies required for deployment."
    else:
        return f"Moderate match; displays baseline exposure to {keywords_str if keywords_str else 'role stack'} but requires alignment fine-tuning."

st.markdown("<div class='step-header'>👥 Step 3: Compute Leaderboard Matrices</div>", unsafe_allow_html=True)
if st.button("🚀 Execute AURA Synthesis Matrix", type="primary"):
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df['years_of_experience'] = pd.to_numeric(df['years_of_experience']).fillna(0)
        df['semantic_profile'] = df['skills'].fillna('') + " " + df['experience_summary'].fillna('')
        data_source_valid = True
    else:
        csv_path = os.path.join("data", "raw", "expanded_candidates.csv")
        if os.path.exists(csv_path):
            df = load_and_preprocess_data(csv_path)
            df['years_of_experience'] = pd.to_numeric(df['years_of_experience']).fillna(0)
            df['semantic_profile'] = df['skills'].fillna('') + " " + df['experience_summary'].fillna('')
            data_source_valid = True
            st.warning("📊 Gateway unpopulated. Streaming default fallback matrix.")
        else:
            data_source_valid = False
            st.error("❌ Data Stream Fault: Missing default dataset paths.")

    if data_source_valid:
        with st.spinner("Compiling contextual layers..."):
            embedder = RecruitmentEmbedder()
            
            cand_embeds = embedder.generate_embeddings(df['semantic_profile'].tolist())
            jd_embed = embedder.generate_embeddings(jd_text)
            
            semantic_scores = embedder.calculate_match_scores(cand_embeds, jd_embed)
            
            if isinstance(semantic_scores, np.ndarray):
                semantic_scores = semantic_scores.flatten()
                
            exp_diff = abs(df['years_of_experience'] - target_exp)
            experience_scores = np.clip(1.0 - (exp_diff / 10.0), 0.1, 1.0).to_numpy()
            
            final_scores = (semantic_scores * semantic_weight) + (experience_scores * exp_weight)
            
            # Formulating raw percentages to pass down to insight processor safely
            ai_pcts = (semantic_scores * 100).round(0)
            exp_pcts = (experience_scores * 100).round(0)
            
            # Compute analytical evaluation summaries column
            insights = []
            for idx, row in df.iterrows():
                ins = generate_aura_insight(row, jd_text, ai_pcts[idx], exp_pcts[idx], target_exp)
                insights.append(ins)
            df['AURA Insight'] = insights
            
            # 🛠️ HARD-LOCKED CONVERSION FIXED HERE: Explicitly pass safe Pandas Series mapping wrappers
            df['AI Match'] = pd.Series(ai_pcts.astype(int)).apply(format_score_block).values
            df['Exp Match'] = pd.Series(exp_pcts.astype(int)).apply(format_score_block).values
            df['OVERALL SCORE'] = (final_scores * 100).round(1)
            
            results = df.sort_values(by='OVERALL SCORE', ascending=False)
            results['Rank'] = results['OVERALL SCORE'].rank(method='min', ascending=False).astype(int)
            results = results.sort_values(by='Rank', ascending=True)
            
            st.success(f"⚡ Synthesis Operational. Mapped successfully.")
            st.balloons()
            
            # Restructuring display dataframe setup to match the fine-tuned CSS width configurations
            display_df = results[['Rank', 'name', 'years_of_experience', 'AI Match', 'Exp Match', 'OVERALL SCORE', 'AURA Insight']].copy()
            display_df.columns = ['Rank', 'Candidate Name', 'Experience (Yrs)', 'AI Semantic Match', 'Experience Match', 'Total AURA Score', 'AURA Insight']
            
            # Map clean matrix HTML container grid
            raw_html_table = display_df.to_html(escape=False, index=False, classes="aura-table")
            st.markdown(f'<div class="aura-grid-container">{raw_html_table}</div>', unsafe_allow_html=True)
            # 📥 Final Submission Output Gateway
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Prepare a clean, submission-ready CSV payload string without HTML styling blocks
            csv_download_df = results[['Rank', 'name', 'years_of_experience', 'OVERALL SCORE', 'AURA Insight']].copy()
            csv_download_df.columns = ['Rank', 'Candidate Name', 'Years of Experience', 'Total AURA Score', 'AURA Insight']
            csv_buffer = csv_download_df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 Export Ranked Candidate Shortlist (CSV)",
                data=csv_buffer,
                file_name="aura_ranked_candidates.csv",
                mime="text/csv",
                use_container_width=True
            )