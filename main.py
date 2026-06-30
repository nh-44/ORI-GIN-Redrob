import os
import numpy as np
import pandas as pd
from src.parser import load_and_preprocess_data, load_job_description
from src.embedder import RecruitmentEmbedder
from src.explainer import generate_aura_insight

def main():
    print("=== Starting Upgraded Hybrid AI Recruitment Engine ===")
    
    expanded_csv_path = os.path.join("data", "expanded_candidates.csv")
    mock_csv_path = os.path.join("data", "raw", "mock_candidates.csv")
    
    if os.path.exists(expanded_csv_path):
        csv_path = expanded_csv_path
        job_desc_text = "Looking for a Robotics & Edge AI Engineer experienced in ROS2, computer vision, C++, and deploying optimized deep learning models on NVIDIA Jetson hardware."
        print(f"Loading expanded candidates from: {csv_path}")
        print(f"Job Description: '{job_desc_text}'")
        TARGET_EXPERIENCE = 5
    else:
        csv_path = mock_csv_path
        jd_path = os.path.join("data", "raw", "mock_job_description.txt")
        if not os.path.exists(csv_path) or not os.path.exists(jd_path):
            print("[ERROR] Data files not found!")
            return
        print(f"Loading mock candidates from: {csv_path}")
        job_desc_text = load_job_description(jd_path)
        TARGET_EXPERIENCE = 5

    # 1. Load Data
    candidates_df = load_and_preprocess_data(csv_path)
    
    # 2. Generate AI Semantic Embeddings
    embedder = RecruitmentEmbedder()
    candidate_embeddings = embedder.generate_embeddings(candidates_df['semantic_profile'].tolist())
    jd_embedding = embedder.generate_embeddings(job_desc_text)

    # 3. Calculate Raw Semantic Match Scores (0 to 1)
    semantic_scores = embedder.calculate_match_scores(candidate_embeddings, jd_embedding)
    
    # 4. Calculate Structural Experience Match Scores
    exp_diff = abs(candidates_df['years_of_experience'] - TARGET_EXPERIENCE)
    experience_scores = 1.0 - (exp_diff / 10.0)
    experience_scores = np.clip(experience_scores, 0.1, 1.0) # Keep scores between 0.1 and 1.0

    # 5. Apply Hybrid Weights: 70% Semantic Model + 30% Hard Experience Match
    final_hybrid_scores = (semantic_scores * 0.7) + (experience_scores * 0.3)
    
    # 6. Inject back into dataframe
    candidates_df['semantic_score'] = semantic_scores
    candidates_df['experience_score'] = experience_scores
    candidates_df['match_score'] = final_hybrid_scores
    candidates_df['OVERALL SCORE'] = (final_hybrid_scores * 100).round(1)
    
    # 7. Generate AURA Insights
    ai_pcts = (semantic_scores * 100).round(0)
    exp_pcts = (experience_scores * 100).round(0)
    
    insights = []
    for idx, row in candidates_df.iterrows():
        ins = generate_aura_insight(row, job_desc_text, ai_pcts[idx], exp_pcts[idx], TARGET_EXPERIENCE)
        insights.append(ins)
    candidates_df['AURA Insight'] = insights

    # 8. Sort and Rank
    ranked_candidates = candidates_df.copy()
    ranked_candidates['Rank'] = ranked_candidates['OVERALL SCORE'].rank(method='min', ascending=False).astype(int)
    ranked_candidates = ranked_candidates.sort_values(by='Rank', ascending=True)

    # 9. Print out the Smart Leaderboard
    print("\n==============================================")
    print("[RESULTS] HYBRID SEMANTIC & STRUCTURAL RANKING RESULTS:")
    print("==============================================")
    for idx, row in ranked_candidates.iterrows():
        print(f"Rank {row['Rank']}: [FINAL SCORE: {row['OVERALL SCORE']}%]")
        print(f"Name: {row['name']} | Actual Experience: {row['years_of_experience']} Years (Target: {TARGET_EXPERIENCE} Years)")
        print(f"  -> breakdown -> AI Semantic Match: {row['semantic_score']*100:.1f}% | Experience Match: {row['experience_score']*100:.1f}%")
        print(f"  -> Insight: {row['AURA Insight']}")
        print("-" * 46)
    print("==============================================")
    
    # 10. Save to Excel
    xlsx_path = os.path.join("data", "aura_ranked_candidates.xlsx")
    excel_download_df = ranked_candidates[['Rank', 'name', 'years_of_experience', 'semantic_score', 'experience_score', 'OVERALL SCORE', 'AURA Insight']].copy()
    excel_download_df['semantic_score'] = (excel_download_df['semantic_score'] * 100).round(1)
    excel_download_df['experience_score'] = (excel_download_df['experience_score'] * 100).round(1)
    excel_download_df.columns = ['Rank', 'Candidate Name', 'Years of Experience', 'AI Semantic Match (%)', 'Experience Match (%)', 'Total AURA Score (%)', 'AURA Insight']
    
    try:
        excel_download_df.to_excel(xlsx_path, index=False, sheet_name='Ranked Candidates')
        print(f"[SUCCESS] Excel sheet saved to {xlsx_path}!")
    except Exception as e:
        print(f"[ERROR] Error saving to Excel: {e}")

if __name__ == "__main__":
    main()