import pandas as pd

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    # Fill missing experience values with 0 and convert to numbers
    df['years_of_experience'] = pd.to_numeric(df['years_of_experience']).fillna(0)
    
    # Combine text fields into a unified profile string for the AI model
    df['semantic_profile'] = df['skills'].fillna('') + " " + df['experience_summary'].fillna('')
    return df

def load_job_description(file_path):
    with open(file_path, 'r') as f:
        return f.read().strip()