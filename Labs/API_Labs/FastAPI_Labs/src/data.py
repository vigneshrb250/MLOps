import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Automatically find the project root (one folder up from where this script lives)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, 'data', 'tmdb_5000_movies.csv')

def load_data(filepath=DEFAULT_DATA_PATH):
    """
    Load the Movie dataset and select the overview for content-based recommendation.
    """
    df = pd.read_csv(filepath)
    
    # Fill missing plot summaries with empty strings
    df['overview'] = df['overview'].fillna('')
    
    X = df['overview']
    y = df['title']
    
    return X, y

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test