import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from data import load_data, split_data

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

def train_recommender(X_train):
    print("Vectorizing text data...")
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(X_train)

    print("Calculating similarity matrix...")
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    print("Saving model artifacts...")
    # Ensure the model directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    joblib.dump(cosine_sim, os.path.join(MODEL_DIR, "similarity_matrix.pkl"))
    print("Training complete!")

if __name__ == "__main__":
    X, y = load_data()
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(y.to_frame(), os.path.join(MODEL_DIR, "movie_list.pkl"))
    
    train_recommender(X)