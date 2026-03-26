import streamlit as st
import requests
import joblib

# Load the movie list just for the dropdown menu options
try:
    movies_df = joblib.load("model/movie_list.pkl")
    movie_titles = movies_df['title'].values
except FileNotFoundError:
    movie_titles = ["Please run train.py first!"]

st.title("🎬 Movie Matcher Lab")
st.write("Select a movie to get 5 content-based recommendations.")

selected_movie = st.selectbox("Pick a movie:", movie_titles)

if st.button("Find Similar Movies"):
    # Send a POST request to your FastAPI server
    api_url = "http://127.0.0.1:8000/predict"
    payload = {"movie_title": selected_movie}
    
    try:
        response = requests.post(api_url, json=payload)
        
        if response.status_code == 200:
            recs = response.json()["recommendations"]
            st.subheader("You might also like:")
            for idx, movie in enumerate(recs, 1):
                st.write(f"**{idx}.** {movie}")
        else:
            st.error(f"Error: {response.json()['detail']}")
            
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the API. Is FastAPI running on port 8000?")