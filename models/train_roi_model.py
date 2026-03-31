
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from utils.db import run_query


MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "roi_random_forest.joblib"


def clip_target_roi(value: float) -> float:
    if pd.isna(value):
        return 0.0
    return float(np.clip(float(value), -100.0, 250.0))


def compress_signal(value: float, cap: float = 250.0, scale: float = 25.0) -> float:
    if pd.isna(value):
        return 0.0
    value = float(np.clip(float(value), -cap, cap))
    return float(np.sign(value) * np.log1p(abs(value)) * scale)


def weighted_top_k(values, weights=None, k=3):
    values = [float(v) for v in values if pd.notna(v)]
    if not values:
        return 0.0
    values = sorted(values, reverse=True)[:k]
    if weights is None:
        weights = [0.55, 0.30, 0.15]
    weights = np.array(weights[: len(values)], dtype=float)
    weights = weights / weights.sum()
    return float(np.dot(values, weights))


def load_data():
    base = run_query(
        """
        SELECT movie_id, title, budget, revenue, roi_pct
        FROM core.v_roi_movie_base
        """
    )

    person_movies = run_query(
        """
        SELECT role_group, person_id, name, movie_id, title, revenue, budget, roi_pct
        FROM core.v_roi_person_movies
        """
    )

    movie_genres = run_query(
        """
        SELECT mg.movie_id, g.genre_id, g.genre_name
        FROM core.movie_genres mg
        JOIN core.genres g
          ON mg.genre_id = g.genre_id
        """
    )

    movie_cast = run_query(
        """
        SELECT movie_id, person_id
        FROM core.movie_cast
        """
    )

    movie_directors = run_query(
        """
        SELECT movie_id, person_id
        FROM core.movie_crew
        WHERE job = 'Director'
        """
    )

    return base, person_movies, movie_genres, movie_cast, movie_directors


def build_leave_one_out_maps(base, person_movies, movie_genres, movie_cast, movie_directors):
    roi_map = {int(r.movie_id): float(r.roi_pct) for r in base.itertuples(index=False)}

    # Person totals by role
    person_totals = defaultdict(lambda: {"sum_roi": 0.0, "sum_success": 0.0, "count": 0})
    for r in person_movies.itertuples(index=False):
        key = (str(r.role_group), int(r.person_id))
        person_totals[key]["sum_roi"] += float(r.roi_pct)
        person_totals[key]["sum_success"] += 1.0 if float(r.roi_pct) > 0 else 0.0
        person_totals[key]["count"] += 1

    # Pair totals (cast-cast)
    cast_by_movie = defaultdict(list)
    for r in movie_cast.itertuples(index=False):
        cast_by_movie[int(r.movie_id)].append(int(r.person_id))

    pair_totals = defaultdict(lambda: {"sum_roi": 0.0, "count": 0})
    for movie_id, cast_ids in cast_by_movie.items():
        roi = roi_map.get(movie_id)
        if roi is None:
            continue
        unique_cast = sorted(set(cast_ids))
        for a, b in combinations(unique_cast, 2):
            key = (a, b)
            pair_totals[key]["sum_roi"] += roi
            pair_totals[key]["count"] += 1

    # Pair totals (director-cast)
    directors_by_movie = defaultdict(list)
    for r in movie_directors.itertuples(index=False):
        directors_by_movie[int(r.movie_id)].append(int(r.person_id))

    director_cast_totals = defaultdict(lambda: {"sum_roi": 0.0, "count": 0})
    for movie_id, director_ids in directors_by_movie.items():
        roi = roi_map.get(movie_id)
        if roi is None:
            continue
        cast_ids = sorted(set(cast_by_movie.get(movie_id, [])))
        for director_id in set(director_ids):
            for cast_id in cast_ids:
                key = (director_id, cast_id)
                director_cast_totals[key]["sum_roi"] += roi
                director_cast_totals[key]["count"] += 1

    # Genre totals
    genres_by_movie = defaultdict(list)
    for r in movie_genres.itertuples(index=False):
        genres_by_movie[int(r.movie_id)].append((int(r.genre_id), str(r.genre_name)))

    genre_totals = defaultdict(lambda: {"sum_roi": 0.0, "sum_success": 0.0, "count": 0})
    for movie_id, genre_rows in genres_by_movie.items():
        roi = roi_map.get(movie_id)
        if roi is None:
            continue
        for genre_id, _ in set(genre_rows):
            genre_totals[genre_id]["sum_roi"] += roi
            genre_totals[genre_id]["sum_success"] += 1.0 if roi > 0 else 0.0
            genre_totals[genre_id]["count"] += 1

    # Person-genre totals
    person_genre_totals = defaultdict(lambda: {"sum_roi": 0.0, "sum_success": 0.0, "count": 0})
    person_movies_map = defaultdict(list)
    for r in person_movies.itertuples(index=False):
        person_movies_map[int(r.movie_id)].append((str(r.role_group), int(r.person_id), float(r.roi_pct)))

    for movie_id, persons in person_movies_map.items():
        genre_ids = [gid for gid, _ in set(genres_by_movie.get(movie_id, []))]
        roi = roi_map.get(movie_id)
        if roi is None:
            continue
        for role_group, person_id, _ in persons:
            for genre_id in genre_ids:
                key = (role_group, person_id, genre_id)
                person_genre_totals[key]["sum_roi"] += roi
                person_genre_totals[key]["sum_success"] += 1.0 if roi > 0 else 0.0
                person_genre_totals[key]["count"] += 1

    return (
        roi_map,
        person_totals,
        pair_totals,
        director_cast_totals,
        genre_totals,
        person_genre_totals,
        cast_by_movie,
        directors_by_movie,
        genres_by_movie,
    )


def loo_avg(sum_value, current_value, count):
    denom = count - 1
    if denom <= 0:
        return 0.0
    return float((sum_value - current_value) / denom)


def loo_success(sum_success, current_success, count):
    denom = count - 1
    if denom <= 0:
        return 0.0
    return float(((sum_success - current_success) / denom) * 100.0)


def build_training_frame():
    base, person_movies, movie_genres, movie_cast, movie_directors = load_data()
    (
        roi_map,
        person_totals,
        pair_totals,
        director_cast_totals,
        genre_totals,
        person_genre_totals,
        cast_by_movie,
        directors_by_movie,
        genres_by_movie,
    ) = build_leave_one_out_maps(base, person_movies, movie_genres, movie_cast, movie_directors)

    persons_by_movie = defaultdict(list)
    for r in person_movies.itertuples(index=False):
        persons_by_movie[int(r.movie_id)].append(
            {
                "role_group": str(r.role_group),
                "person_id": int(r.person_id),
                "roi_pct": float(r.roi_pct),
            }
        )

    rows = []
    for b in base.itertuples(index=False):
        movie_id = int(b.movie_id)
        current_roi = float(b.roi_pct)
        current_success = 1.0 if current_roi > 0 else 0.0

        persons = persons_by_movie.get(movie_id, [])
        genres = [gid for gid, _ in set(genres_by_movie.get(movie_id, []))]

        # Director features
        director_values = []
        director_success_values = []
        director_count_values = []
        director_genre_fit_values = []

        # Cast features
        cast_ranked = []

        for p in persons:
            role_group = p["role_group"]
            person_id = p["person_id"]
            key = (role_group, person_id)
            totals = person_totals[key]

            loo_roi = loo_avg(totals["sum_roi"], current_roi, totals["count"])
            loo_success_pct = loo_success(totals["sum_success"], current_success, totals["count"])
            loo_count = max(totals["count"] - 1, 0)

            person_genre_scores = []
            for genre_id in genres:
                g_key = (role_group, person_id, genre_id)
                if g_key in person_genre_totals:
                    g_tot = person_genre_totals[g_key]
                    g_loo_roi = loo_avg(g_tot["sum_roi"], current_roi, g_tot["count"])
                    g_loo_success = loo_success(g_tot["sum_success"], current_success, g_tot["count"])
                    raw_score = (
                        0.60 * clip_target_roi(g_loo_roi)
                        + 0.25 * ((g_loo_success - 50.0) * 2.0)
                        + 0.15 * min((g_tot["count"] - 1) * 2.0, 20.0)
                    )
                    person_genre_scores.append(compress_signal(raw_score, cap=220.0, scale=18.0))
            genre_fit = weighted_top_k(person_genre_scores, k=3) if person_genre_scores else 0.0

            if role_group == "Director":
                director_values.append(compress_signal(loo_roi, cap=220.0, scale=18.0))
                director_success_values.append(loo_success_pct)
                director_count_values.append(loo_count)
                director_genre_fit_values.append(genre_fit)
            else:
                rank_score = (
                    0.60 * clip_target_roi(loo_roi)
                    + 0.25 * loo_success_pct
                    + 0.15 * min(loo_count, 40.0)
                )
                cast_ranked.append(
                    {
                        "person_id": person_id,
                        "avg_roi_feature": compress_signal(loo_roi, cap=220.0, scale=18.0),
                        "success_feature": loo_success_pct,
                        "count_feature": loo_count,
                        "genre_fit": genre_fit,
                        "rank_score": rank_score,
                    }
                )

        cast_ranked = sorted(cast_ranked, key=lambda x: x["rank_score"], reverse=True)[:3]
        selected_cast_ids = [x["person_id"] for x in cast_ranked]

        # Pair features leave-one-out
        cast_pair_rois = []
        cast_pair_counts = []
        for a, b2 in combinations(sorted(selected_cast_ids), 2):
            key = (a, b2)
            if key in pair_totals:
                totals = pair_totals[key]
                loo_pair_roi = loo_avg(totals["sum_roi"], current_roi, totals["count"])
                loo_pair_count = max(totals["count"] - 1, 0)
                cast_pair_rois.append(compress_signal(loo_pair_roi, cap=180.0, scale=16.0))
                cast_pair_counts.append(loo_pair_count)

        director_cast_rois = []
        director_cast_counts = []
        director_ids = sorted(set(directors_by_movie.get(movie_id, [])))
        for director_id in director_ids:
            for cast_id in selected_cast_ids:
                key = (director_id, cast_id)
                if key in director_cast_totals:
                    totals = director_cast_totals[key]
                    loo_dc_roi = loo_avg(totals["sum_roi"], current_roi, totals["count"])
                    loo_dc_count = max(totals["count"] - 1, 0)
                    director_cast_rois.append(compress_signal(loo_dc_roi, cap=180.0, scale=16.0))
                    director_cast_counts.append(loo_dc_count)

        genre_roi_features = []
        genre_success_features = []
        for genre_id in genres:
            if genre_id in genre_totals:
                totals = genre_totals[genre_id]
                loo_g_roi = loo_avg(totals["sum_roi"], current_roi, totals["count"])
                loo_g_success = loo_success(totals["sum_success"], current_success, totals["count"])
                genre_roi_features.append(compress_signal(loo_g_roi, cap=200.0, scale=15.0))
                genre_success_features.append(loo_g_success)

        rows.append(
            {
                "movie_id": movie_id,
                "title": b.title,
                "target_roi_pct": clip_target_roi(current_roi),
                "director_avg_roi": weighted_top_k(director_values, k=1),
                "director_success_rate": weighted_top_k(director_success_values, k=1),
                "director_film_count": weighted_top_k(director_count_values, k=1),
                "director_count": len(director_values),
                "cast_avg_roi": weighted_top_k([x["avg_roi_feature"] for x in cast_ranked], k=3),
                "cast_success_rate": weighted_top_k([x["success_feature"] for x in cast_ranked], k=3),
                "cast_film_count": weighted_top_k([x["count_feature"] for x in cast_ranked], k=3),
                "cast_count": len(cast_ranked),
                "cast_pair_avg_roi": weighted_top_k(cast_pair_rois, k=3),
                "cast_pair_avg_count": weighted_top_k(cast_pair_counts, k=3),
                "cast_pair_match_count": len(cast_pair_rois),
                "director_cast_avg_roi": weighted_top_k(director_cast_rois, k=3),
                "director_cast_avg_count": weighted_top_k(director_cast_counts, k=3),
                "director_cast_match_count": len(director_cast_rois),
                "genre_count": len(genres),
                "genre_avg_roi": weighted_top_k(genre_roi_features, k=3),
                "genre_success_rate": weighted_top_k(genre_success_features, k=3),
                "director_genre_fit": weighted_top_k(director_genre_fit_values, k=1),
                "cast_genre_fit": weighted_top_k([x["genre_fit"] for x in cast_ranked], k=3),
            }
        )

    df = pd.DataFrame(rows)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def train_and_save():
    df = build_training_frame()

    feature_cols = [
        "director_avg_roi",
        "director_success_rate",
        "director_film_count",
        "director_count",
        "cast_avg_roi",
        "cast_success_rate",
        "cast_film_count",
        "cast_count",
        "cast_pair_avg_roi",
        "cast_pair_avg_count",
        "cast_pair_match_count",
        "director_cast_avg_roi",
        "director_cast_avg_count",
        "director_cast_match_count",
        "genre_count",
        "genre_avg_roi",
        "genre_success_rate",
        "director_genre_fit",
        "cast_genre_fit",
    ]

    X = df[feature_cols].copy()
    y = df["target_roi_pct"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=350,
        max_depth=10,
        min_samples_leaf=8,
        min_samples_split=16,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    preds = np.clip(model.predict(X_test), -100.0, 250.0)

    mae = float(mean_absolute_error(y_test, preds))
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = float(r2_score(y_test, preds))

    feature_importance = (
        pd.DataFrame(
            {"feature": feature_cols, "importance": model.feature_importances_}
        )
        .sort_values("importance", ascending=False)
        .to_dict("records")
    )

    artifact = {
        "model": model,
        "feature_cols": feature_cols,
        "metrics": {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "full_rows": int(len(df)),
        },
        "feature_importance": feature_importance,
        "notes": {
            "target_clip": "[-100, 250]",
            "stabilization": "leave-one-out features, top-3 cast weighting, compressed synergy signals, no budget input",
        },
    }

    joblib.dump(artifact, MODEL_PATH)

    print(f"Saved model to {MODEL_PATH}")
    print(f"Rows: {len(df)}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.3f}")
    print("Top features:")
    print(pd.DataFrame(feature_importance).head(10).to_string(index=False))


if __name__ == "__main__":
    train_and_save()
