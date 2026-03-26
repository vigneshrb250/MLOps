import os
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

def predict_recommendations(movie_title):
    similarity = joblib.load(os.path.join(MODEL_DIR, "similarity_matrix.pkl"))
    movies_df = joblib.load(os.path.join(MODEL_DIR, "movie_list.pkl"))

    idx = movies_df[movies_df['title'].str.lower() == movie_title.lower()].index[0]
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]

    movie_indices = [i[0] for i in sim_scores]
    return movies_df['title'].iloc[movie_indices].tolist()