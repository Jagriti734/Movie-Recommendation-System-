import pickle
import os
from dotenv import load_dotenv
from flask import Flask, render_template, request
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches
import requests
from concurrent.futures import ThreadPoolExecutor
import warnings

warnings.filterwarnings("ignore")

# 🔹 Load environment variables
load_dotenv()
OMDB_API_KEY = os.getenv("OMDB_API_KEY")

app = Flask(__name__)

# 🔹 Load ML files
tfidf_matrix = pickle.load(open('tfidf_matrix.pkl', 'rb'))
indices = pickle.load(open('indices.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# 🔹 Recommendation function (FINAL FIXED)
def recommend(movie, n=3):
    movie = movie.lower()

    matches = get_close_matches(movie, indices.keys(), n=1, cutoff=0.6)

    if not matches:
        return [], []

    idx = indices[matches[0]]

    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    # remove same movie
    sim_scores[idx] = -1

    sorted_indices = sim_scores.argsort()[::-1]

    movies = []
    scores = []

    for i in sorted_indices:
        # 🔥 SAFE popularity handling (fix TypeError)
        pop = df.iloc[i].get('popularity', 0)

        try:
            pop = float(pop)
        except:
            pop = 0

        if pop > 5:
            movies.append(df.iloc[i]['title'])
            scores.append(sim_scores[i])

        if len(movies) == n:
            break

    return movies, scores

# 🔹 Fetch movie details (FAST)
def fetch_movie(movie_name):
    try:
        url = f"https://www.omdbapi.com/?t={movie_name}&apikey={OMDB_API_KEY}"
        res = requests.get(url, timeout=2)
        data = res.json()

        if data.get("Response") == "False":
            return None

        poster = data.get("Poster")
        if poster == "N/A":
            poster = "https://via.placeholder.com/300x450?text=No+Image"

        return {
            "title": data.get("Title"),
            "poster": poster,
            "plot": data.get("Plot")
        }

    except:
        return None

# 🔹 Combine ML + API (REAL PARALLEL)
def get_movies(movie):
    recs, scores = recommend(movie)

    if not recs:
        return []

    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(fetch_movie, recs))

    movies = []
    for data, score in zip(results, scores):
        if data:
            data["score"] = round(score * 100, 2)
            movies.append(data)

    return movies

# 🔹 Flask route
@app.route("/", methods=["GET", "POST"])
def home():
    movies = []

    if request.method == "POST":
        movie = request.form.get("movie")

        if movie:
            print("Searching:", movie)
            movies = get_movies(movie)

    return render_template("index.html", movies=movies)

# 🔹 Run app
if __name__ == "__main__":
    app.run(debug=True)