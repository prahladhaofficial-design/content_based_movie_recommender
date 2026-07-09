import argparse
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_movies():
    frames = []
    for csv_path in DATA_DIR.glob("*.csv"):
        df = pd.read_csv(csv_path)
        df["source_file"] = csv_path.name
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")
    return pd.concat(frames, ignore_index=True, sort=False)


def choose_title_column(df):
    for candidate in ["Series_Title", "title", "movie_name", "Movie", "Title", "original_title"]:
        if candidate in df.columns:
            return candidate
    for column in df.columns:
        if "title" in column.lower() or "name" in column.lower():
            return column
    raise ValueError("Could not infer a movie title column from the datasets.")


def build_text_features(df, title_col):
    useful_columns = [
        col for col in df.columns
        if any(key in col.lower() for key in ["genre", "overview", "plot", "director", "star", "actor", "cast", "certificate"])
    ]
    if not useful_columns:
        useful_columns = [title_col]
    return df[useful_columns].fillna("").astype(str).agg(" ".join, axis=1)


def recommend(title, top_k=10):
    movies = load_movies()
    title_col = choose_title_column(movies)
    movies = movies.dropna(subset=[title_col]).drop_duplicates(subset=[title_col]).reset_index(drop=True)

    matches = movies[movies[title_col].astype(str).str.contains(title, case=False, na=False)]
    if matches.empty:
        raise ValueError(f"No movie title matched: {title}")

    query_index = matches.index[0]
    features = build_text_features(movies, title_col)
    matrix = TfidfVectorizer(stop_words="english", max_features=10000).fit_transform(features)
    scores = cosine_similarity(matrix[query_index], matrix).ravel()
    ranking = scores.argsort()[::-1]

    results = []
    for idx in ranking:
        if idx == query_index:
            continue
        row = movies.iloc[idx]
        results.append({
            "title": row[title_col],
            "score": round(float(scores[idx]), 4),
            "source": row.get("source_file", ""),
        })
        if len(results) == top_k:
            break
    return movies.iloc[query_index][title_col], results


def main():
    parser = argparse.ArgumentParser(description="Content-based movie recommender.")
    parser.add_argument("--title", help="Movie title or partial title to search for.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of recommendations to return.")
    parser.add_argument("--list", action="store_true", help="List sample movie titles and exit.")
    args = parser.parse_args()

    movies = load_movies()
    title_col = choose_title_column(movies)
    if args.list:
        print(movies[title_col].dropna().drop_duplicates().head(30).to_string(index=False))
        return
    if not args.title:
        parser.error("--title is required unless --list is used")

    matched_title, results = recommend(args.title, args.top_k)
    print(f"Recommendations similar to: {matched_title}")
    for rank, item in enumerate(results, 1):
        print(f"{rank:2d}. {item['title']} | score={item['score']} | source={item['source']}")


if __name__ == "__main__":
    main()
