import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class RecruitmentEmbedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initializes the SentenceTransformer model.
        """
        print(f"Loading transformer model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("Model loaded successfully!")

    def generate_embeddings(self, texts):
        """
        Converts a list of strings into numerical AI embeddings.
        """
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, show_progress_bar=True)

    def calculate_match_scores(self, candidate_embeddings, job_description_embedding):
        """
        Calculates semantic similarity using Cosine Similarity.
        """
        if len(job_description_embedding.shape) == 1:
            job_description_embedding = job_description_embedding.reshape(1, -1)
            
        scores = cosine_similarity(candidate_embeddings, job_description_embedding)
        return scores.flatten()