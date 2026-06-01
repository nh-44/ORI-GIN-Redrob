import os
import numpy as np
from src.parser import load_and_preprocess_data, load_job_description
from src.embedder import RecruitmentEmbedder

def main():
    print("=== Starting Upgraded Hybrid AI Recruitment Engine ===")
    
    csv_path = os.path.join("data", "raw", "mock_candidates.csv")
    jd_path = os.path.join("data", "raw", "mock_job_description.txt")
    
    if not os.path.exists(csv_path) or not os.path.exists(jd_path):
        print("❌ Error: Data files not found!")
        return

    # 1. Load Data
    candidates_df = load_and_preprocess_data(csv_path)
    job_desc_text = load_job_description(jd_path)
    
    # 2. Define ideal criteria for the role
    TARGET_EXPERIENCE = 5  # Say the job description targets someone with 5 years of experience
    
    # 3. Generate AI Semantic Embeddings
    embedder = RecruitmentEmbedder()
    candidate_embeddings = embedder.generate_embeddings(candidates_df['semantic_profile'].tolist())
    jd_embedding = embedder.generate_embeddings(job_desc_text)

    # 4. Calculate Raw Semantic Match Scores (0 to 1)
    semantic_scores = embedder.calculate_match_scores(candidate_embeddings, jd_embedding)
    
    # 5. Calculate Structural Experience Match Scores
    # Scales down the score by 10% for every year a candidate deviates from the target experience
    exp_diff = abs(candidates_df['years_of_experience'] - TARGET_EXPERIENCE)
    experience_scores = 1.0 - (exp_diff / 10.0)
    experience_scores = np.clip(experience_scores, 0.1, 1.0) # Keep scores between 0.1 and 1.0

    # 6. Apply Hybrid Weights: 70% Semantic Model + 30% Hard Experience Match
    final_hybrid_scores = (semantic_scores * 0.7) + (experience_scores * 0.3)
    
    # 7. Inject back into dataframe and rank
    candidates_df['semantic_score'] = semantic_scores
    candidates_df['experience_score'] = experience_scores
    candidates_df['match_score'] = final_hybrid_scores
    
    ranked_candidates = candidates_df.sort_values(by='match_score', ascending=False)

    # 8. Print out the Smart Leaderboard
    print("\n==============================================")
    print("🎯 HYBRID SEMANTIC & STRUCTURAL RANKING RESULTS:")
    print("==============================================")
    for idx, row in ranked_candidates.iterrows():
        print(f"Rank: [FINAL SCORE: {row['match_score']*100:.1f}%]")
        print(f"Name: {row['name']} | Actual Experience: {row['years_of_experience']} Years (Target: {TARGET_EXPERIENCE} Years)")
        print(f"  ↳ breakdown -> AI Semantic Match: {row['semantic_score']*100:.1f}% | Experience Match: {row['experience_score']*100:.1f}%")
        print("-" * 46)
    print("==============================================")

if __name__ == "__main__":
    main()