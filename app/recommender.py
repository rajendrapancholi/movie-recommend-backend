# app/recommender.py
import os
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import process

BASE_DIR = os.path.dirname(__file__)
# location where the movielens dataset lives (set env var if different)
MOVIELENS_PATH = os.environ.get("MOVIELENS_DATA_PATH", os.path.join(BASE_DIR, "..", "data"))

MOVIES_CSV = os.path.join(MOVIELENS_PATH, "movies.csv")
RATINGS_CSV = os.path.join(MOVIELENS_PATH, "ratings.csv")

# model persistence dir
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

TFIDF_VECT_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
TFIDF_MATRIX_PATH = os.path.join(MODEL_DIR, "tfidf_matrix.joblib")
MOVIES_PKL = os.path.join(MODEL_DIR, "movies.joblib")
ITEM_USER_MATRIX_PATH = os.path.join(MODEL_DIR, "item_user_matrix.joblib")
ITEM_KNN_PATH = os.path.join(MODEL_DIR, "item_knn.joblib")

# helper to ensure input files exist
def _check_files():
    if not os.path.exists(MOVIES_CSV):
        raise FileNotFoundError(f"movies.csv not found at {MOVIES_CSV}. Set MOVIELENS_DATA_PATH accordingly.")
    if not os.path.exists(RATINGS_CSV):
        raise FileNotFoundError(f"ratings.csv not found at {RATINGS_CSV}. Set MOVIELENS_DATA_PATH accordingly.")

# Train or load persisted model components
def load_or_train():
    _check_files()

    if all(os.path.exists(p) for p in [TFIDF_VECT_PATH, TFIDF_MATRIX_PATH, MOVIES_PKL, ITEM_USER_MATRIX_PATH, ITEM_KNN_PATH]):
        # load
        tfidf = joblib.load(TFIDF_VECT_PATH)
        tfidf_matrix = joblib.load(TFIDF_MATRIX_PATH)
        movies = joblib.load(MOVIES_PKL)
        item_user_matrix = joblib.load(ITEM_USER_MATRIX_PATH)
        item_knn = joblib.load(ITEM_KNN_PATH)
        print("[recommender] Loaded persisted models.")
    else:
        # train
        print("[recommender] Training recommendation models (this may take a minute)...")
        movies = pd.read_csv(MOVIES_CSV)
        ratings = pd.read_csv(RATINGS_CSV)

        # preprocess genres column
        movies['genres'] = movies['genres'].fillna('').astype(str)

        # TF-IDF on genres (content)
        tfidf = TfidfVectorizer(stop_words='english', analyzer='word', ngram_range=(1, 1))
        tfidf_matrix = tfidf.fit_transform(movies['genres'])

        # Build item-user matrix (rows = movies.movieId in same order as movies dataframe)
        ratings_pivot = ratings.pivot_table(index='movieId', columns='userId', values='rating').fillna(0)
        # reindex pivot to match movies order (movieId)
        item_user_matrix_df = ratings_pivot.reindex(movies['movieId']).fillna(0)
        item_user_matrix = item_user_matrix_df.values  # shape: (n_movies, n_users)

        # Fit nearest neighbors on item-user matrix (collaborative)
        item_knn = NearestNeighbors(metric='cosine', algorithm='brute')
        item_knn.fit(item_user_matrix)

        # persist
        joblib.dump(tfidf, TFIDF_VECT_PATH)
        joblib.dump(tfidf_matrix, TFIDF_MATRIX_PATH)
        joblib.dump(movies, MOVIES_PKL)
        joblib.dump(item_user_matrix, ITEM_USER_MATRIX_PATH)
        joblib.dump(item_knn, ITEM_KNN_PATH)
        print("[recommender] Model trained and saved.")

    return tfidf, tfidf_matrix, movies, item_user_matrix, item_knn

# load on import
TFIDF_VECT, TFIDF_MATRIX, MOVIES_DF, ITEM_USER_MATRIX, ITEM_KNN = load_or_train()
MOVIE_TITLES = MOVIES_DF['title'].tolist()
N_MOVIES = len(MOVIES_DF)

# helper: fuzzy-match a title -> index in MOVIES_DF
def _match_title_get_index(query_title):
    match = process.extractOne(query_title, MOVIE_TITLES)
    if not match:
        return None
    matched_title = match[0]
    idx = MOVIES_DF[MOVIES_DF['title'] == matched_title].index
    if len(idx) == 0:
        return None
    return int(idx[0])

# main API: hybrid recommender
def get_recommendations(query_title: str, top_x: int = 10, user_history_titles: list | None = None,
                        alpha: float = 0.45, beta: float = 0.45, gamma: float = 0.10):
    """
    Return top_x recommended movies for query_title.
      - alpha: weight for content (genres)
      - beta: weight for collaborative (ratings)
      - gamma: weight for personalization from user's app-history (if provided)
    """
    idx = _match_title_get_index(query_title)
    if idx is None:
        return []

    # ----- content-based scores (TF-IDF genres) -----
    # compute similarity between query movie and all movies
    query_vec = TFIDF_VECT.transform([MOVIES_DF.loc[idx, 'genres']])
    content_scores = linear_kernel(query_vec, TFIDF_MATRIX).flatten()  # shape (n_movies,)

    # ----- collaborative scores (item-user matrix) -----
    # compute cosine similarity of item vectors (movie idx to all movies)
    from sklearn.metrics.pairwise import cosine_similarity as _cos_sim
    item_vec = ITEM_USER_MATRIX[idx].reshape(1, -1)
    # if item_vec is all zeros (no ratings for this movie) this will return zeros
    collab_scores = _cos_sim(item_vec, ITEM_USER_MATRIX).flatten()

    # ----- personalization from user's app history (if any) -----
    user_pref_scores = np.zeros(N_MOVIES, dtype=float)
    if user_history_titles:
        # treat each user-history title as a liked item (weight 1)
        # for each history movie find index and add its similarity vector to user_pref_scores
        for ht in user_history_titles:
            h_idx = _match_title_get_index(ht)
            if h_idx is None:
                continue
            h_vec = ITEM_USER_MATRIX[h_idx].reshape(1, -1)
            sim = _cos_sim(h_vec, ITEM_USER_MATRIX).flatten()  # similarity of this liked movie to all movies
            user_pref_scores += sim
        # normalize
        if user_pref_scores.sum() > 0:
            user_pref_scores = user_pref_scores / np.max(user_pref_scores)

    # If user_pref_scores are empty (no history), zero-gamma
    if user_pref_scores.sum() == 0:
        gamma = 0.0
        # renormalize alpha+beta to sum 1
        total = alpha + beta
        if total > 0:
            alpha = alpha / total
            beta = beta / total

    # combine scores
    combined = alpha * (content_scores / (content_scores.max() if content_scores.max() else 1)) + \
               beta * (collab_scores / (collab_scores.max() if collab_scores.max() else 1)) + \
               gamma * (user_pref_scores / (user_pref_scores.max() if user_pref_scores.max() else 1))

    # exclude the queried movie itself
    combined[idx] = -1.0

    # get top indices
    top_indices = np.argsort(combined)[::-1][:top_x]
    recommended = []
    for i in top_indices:
        recommended.append({
            "movieId": int(MOVIES_DF.loc[i, "movieId"]),
            "title": MOVIES_DF.loc[i, "title"],
            "genres": MOVIES_DF.loc[i, "genres"],
            "score": float(combined[i])
        })

    return recommended
