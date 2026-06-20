import streamlit as st
import pickle
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# ---------------- OMDB API ----------------
API_KEY = "91149d89"
def fetch_poster(movie_name):
    url = f"https://www.omdbapi.com/?t={movie_name}&apikey={API_KEY}"

    try:
        data = requests.get(url).json()

        if data.get("Poster") and data["Poster"] != "N/A":
            return data["Poster"]

    except:
        pass

    return None
# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.stApp {
    background-color: #141414;
}

.main-title {
    text-align: center;
    color: #E50914;
    font-size: 55px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: white;
    font-size: 20px;
    margin-bottom: 30px;
}

.movie-card {
    background-color: #222222;
    padding: 15px;
    border-radius: 15px;
    text-align: center;
    color: white;
    font-weight: bold;
    min-height: 80px;
    margin-top: 10px;
}

.stButton > button {
    background-color: #E50914;
    color: white;
    border-radius: 10px;
    font-size: 18px;
    font-weight: bold;
    width: 100%;
    height: 50px;
}

.stButton > button:hover {
    background-color: #B20710;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
movies = pickle.load(open('movies.pkl', 'rb'))

# Generate similarity matrix at runtime
cv = CountVectorizer(
    max_features=5000,
    stop_words='english'
)

vectors = cv.fit_transform(movies['tags']).toarray()
similarity = cosine_similarity(vectors)

# ---------------- RECOMMEND FUNCTION ----------------
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]

    distances = similarity[movie_index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []
    posters = []

    for i in movies_list:
        movie_name = movies.iloc[i[0]].title

        recommended_movies.append(movie_name)
        posters.append(fetch_poster(movie_name))

    return recommended_movies, posters

# ---------------- HEADER ----------------
st.markdown(
    '<h1 class="main-title">🎬 Movie Recommendation System</h1>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Discover movies similar to your favorites</p>',
    unsafe_allow_html=True
)

# ---------------- MOVIE SELECT ----------------
selected_movie = st.selectbox(
    "Choose a Movie",
    movies['title'].values
)

# ---------------- BUTTON ----------------
if st.button("🍿 Recommend Movies"):

    recommendations, posters = recommend(selected_movie)

    st.markdown("## 🎯 Recommended For You")

    cols = st.columns(5)

    for idx in range(len(recommendations)):

        with cols[idx]:

            if posters[idx]:
                st.image(posters[idx], use_container_width=True)

            st.markdown(
                f"""
                <div class="movie-card">
                    {recommendations[idx]}
                </div>
                """,
                unsafe_allow_html=True
            )