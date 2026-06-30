def generate_aura_insight(row, jd_string, ai_score, exp_score, target_y):
    """
    Generates a 1-sentence analytical insight summary dynamically.
    """
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
