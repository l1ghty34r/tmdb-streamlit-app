import os
from itertools import combinations
from math import log1p
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


st.set_page_config(
    page_title="TMDB Project App",
    page_icon="🎬",
    layout="wide",
)


# =========================================================
# Helper functions
# =========================================================
def format_money(value) -> str:
    if pd.isna(value):
        return "n/a"
    try:
        value = float(value)
    except Exception:
        return "n/a"

    abs_value = abs(value)
    if abs_value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    if abs_value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    return f"${value:,.0f}"


def format_number(value, decimals: int = 2) -> str:
    if pd.isna(value):
        return "n/a"
    try:
        return f"{float(value):,.{decimals}f}"
    except Exception:
        return "n/a"


def format_int(value) -> str:
    if pd.isna(value):
        return "n/a"
    try:
        return f"{int(float(value)):,}"
    except Exception:
        return "n/a"


def format_date(value) -> str:
    if pd.isna(value):
        return "n/a"
    try:
        return pd.to_datetime(value).strftime("%Y-%m-%d")
    except Exception:
        return str(value)


def format_bool(value) -> str:
    if pd.isna(value):
        return "n/a"
    return "Yes" if bool(value) else "No"


def poster_url(path_value: Optional[str]) -> Optional[str]:
    if not path_value or pd.isna(path_value):
        return None
    path_value = str(path_value).strip()
    if not path_value:
        return None
    return f"https://image.tmdb.org/t/p/w500{path_value}"


def safe_mean(values: List[float], fallback: float = 0.0) -> float:
    clean = [float(v) for v in values if pd.notna(v)]
    if not clean:
        return float(fallback)
    return float(np.mean(clean))


def safe_median(values: List[float], fallback: float = 0.0) -> float:
    clean = [float(v) for v in values if pd.notna(v)]
    if not clean:
        return float(fallback)
    return float(np.median(clean))


def safe_max(values: List[float], fallback: float = 0.0) -> float:
    clean = [float(v) for v in values if pd.notna(v)]
    if not clean:
        return float(fallback)
    return float(np.max(clean))


def weighted_roi_score(avg_roi: float, median_roi: float, success_rate: float, film_count: float) -> float:
    avg_roi = 0.0 if pd.isna(avg_roi) else float(avg_roi)
    median_roi = 0.0 if pd.isna(median_roi) else float(median_roi)
    success_rate = 0.0 if pd.isna(success_rate) else float(success_rate)
    film_count = 0.0 if pd.isna(film_count) else float(film_count)

    # konservativerer Score als nur Avg ROI
    base = (0.35 * avg_roi) + (0.35 * median_roi) + (0.30 * success_rate)
    experience_factor = log1p(max(film_count, 0.0))
    return float(base * experience_factor)


# =========================================================
# Database connection
# =========================================================
def get_database_url() -> str:
    try:
        database_url = st.secrets["DATABASE_URL"]
    except Exception:
        database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError("DATABASE_URL not found in Streamlit secrets or environment variables.")

    return database_url


@st.cache_data(ttl=300)
def run_query(query: str, params: Optional[tuple] = None) -> pd.DataFrame:
    database_url = get_database_url()
    with psycopg2.connect(database_url) as conn:
        return pd.read_sql_query(query, conn, params=params)


# =========================================================
# Recommendation Engine
# =========================================================
@st.cache_data(ttl=600)
def load_recommender_data() -> pd.DataFrame:
    query = """
    WITH unique_movies AS (
        SELECT DISTINCT ON (m.title)
            m.movie_id,
            m.title,
            m.overview,
            m.poster_path,
            m.popularity,
            m.runtime,
            m.vote_average
        FROM core.movies m
        ORDER BY m.title, m.popularity DESC
    ),
    cast_info AS (
        SELECT
            mc.movie_id,
            STRING_AGG(p.name, ' ' ORDER BY mc.cast_order ASC) AS actors
        FROM core.movie_cast mc
        JOIN core.people p ON mc.person_id = p.person_id
        WHERE mc.cast_order <= 3
        GROUP BY mc.movie_id
    ),
    crew_info AS (
        SELECT
            x.movie_id,
            x.director
        FROM (
            SELECT
                mcr.movie_id,
                p.name AS director,
                ROW_NUMBER() OVER (
                    PARTITION BY mcr.movie_id
                    ORDER BY p.name
                ) AS rn
            FROM core.movie_crew mcr
            JOIN core.people p ON mcr.person_id = p.person_id
            WHERE mcr.job = 'Director'
        ) x
        WHERE x.rn = 1
    ),
    genre_info AS (
        SELECT
            mg.movie_id,
            STRING_AGG(g.genre_name, ' ' ORDER BY g.genre_name) AS genre_list
        FROM core.movie_genres mg
        JOIN core.genres g ON mg.genre_id = g.genre_id
        GROUP BY mg.movie_id
    )
    SELECT
        um.*,
        COALESCE(gi.genre_list, '') AS genres,
        COALESCE(ci.actors, '') AS top_cast,
        COALESCE(cr.director, '') AS director
    FROM unique_movies um
    LEFT JOIN genre_info gi ON um.movie_id = gi.movie_id
    LEFT JOIN cast_info ci ON um.movie_id = ci.movie_id
    LEFT JOIN crew_info cr ON um.movie_id = cr.movie_id
    ORDER BY um.title;
    """
    return run_query(query)


@st.cache_resource
def build_recommender_engine(df: pd.DataFrame):
    work_df = df.copy()

    def clean_data(x):
        return str(x).lower().replace(" ", "")

    temp_cast = work_df["top_cast"].fillna("").apply(clean_data)
    temp_director = work_df["director"].fillna("").apply(clean_data)
    temp_genres = work_df["genres"].fillna("").apply(clean_data)
    temp_overview = work_df["overview"].fillna("").str.lower()

    work_df["soup"] = (
        temp_cast
        + " "
        + temp_director
        + " "
        + temp_director
        + " "
        + temp_genres
        + " "
        + temp_genres
        + " "
        + temp_overview
    )

    count = CountVectorizer(stop_words="english")
    count_matrix = count.fit_transform(work_df["soup"])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    indices = pd.Series(work_df.index, index=work_df["title"]).drop_duplicates()

    return work_df, indices, cosine_sim


def show_recommendation_engine_page() -> None:
    st.title("🎬 Recommendation Engine")
    st.write(
        "This module recommends similar movies based on content, genres, cast, director, and overview."
    )

    with st.spinner("Loading recommendation engine..."):
        df = load_recommender_data()
        df, indices, cosine_sim = build_recommender_engine(df)

    if df.empty:
        st.warning("No data could be loaded for the recommendation engine.")
        return

    st.sidebar.markdown("### Filters for Recommendation Engine")
    min_rating = st.sidebar.slider(
        "Minimum Rating",
        min_value=0.0,
        max_value=10.0,
        value=6.0,
        step=0.5,
        key="rec_min_rating",
    )

    valid_runtime = df["runtime"].dropna()
    max_runtime = int(valid_runtime.max()) if not valid_runtime.empty else 300
    selected_runtime = st.sidebar.slider(
        "Maximum Runtime",
        min_value=60,
        max_value=max_runtime,
        value=min(120, max_runtime),
        key="rec_runtime",
    )

    all_genres = sorted(
        list(
            set(
                [
                    g
                    for sublist in df["genres"].fillna("").str.split()
                    for g in sublist
                    if str(g).strip()
                ]
            )
        )
    )
    selected_genres = st.sidebar.multiselect(
        "Limit by Genre",
        all_genres,
        key="rec_genres",
    )

    num_recommendations = st.sidebar.slider(
        "Number of Movies",
        5,
        20,
        10,
        key="rec_num",
    )

    selected_movie = st.selectbox(
        "Pick a movie you love:",
        df["title"].dropna().sort_values().unique(),
        key="rec_selected_movie",
    )

    if st.button("Find Similar", key="rec_find_similar"):
        if selected_movie not in indices:
            st.error("Movie not found in the recommendation index.")
            return

        idx = indices[selected_movie]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        recommended_list = []
        seen_titles = {selected_movie.lower().strip()}

        for i, score in sim_scores[1:]:
            movie_row = df.iloc[i]
            current_title = str(movie_row["title"]).lower().strip()

            if current_title in seen_titles:
                continue

            vote_avg = movie_row.get("vote_average")
            runtime_val = movie_row.get("runtime")

            if pd.notna(vote_avg) and float(vote_avg) < min_rating:
                continue

            if pd.notna(runtime_val) and float(runtime_val) > selected_runtime:
                continue

            if selected_genres:
                movie_genres = [g.lower().strip() for g in str(movie_row["genres"]).split()]
                selected_lower = [s.lower().strip() for s in selected_genres]
                if not any(genre in movie_genres for genre in selected_lower):
                    continue

            recommended_list.append(movie_row)
            seen_titles.add(current_title)

            if len(recommended_list) >= num_recommendations:
                break

        if not recommended_list:
            st.warning("No recommendations found. Try adjusting the filters.")
            return

        st.markdown("---")
        st.subheader(f"Recommendations for {selected_movie}")

        cols = st.columns(5)
        for i, row in enumerate(recommended_list):
            with cols[i % 5]:
                img_url = poster_url(row.get("poster_path"))
                if img_url:
                    st.image(img_url, use_container_width=True)
                else:
                    st.markdown("**No Poster**")

                st.write(f"**{row['title']}**")
                st.caption(f"Director: {row['director'] if pd.notna(row['director']) else 'n/a'}")
                st.caption(f"Genres: {row['genres'] if pd.notna(row['genres']) else 'n/a'}")
                st.caption(f"Rating: {format_number(row['vote_average'], 1)}")


# =========================================================
# ROI Builder - data loading
# =========================================================
@st.cache_data(ttl=600)
def load_person_role_stats() -> pd.DataFrame:
    query = """
    WITH movie_roi AS (
        SELECT
            m.movie_id,
            m.title,
            m.budget,
            m.revenue,
            ((m.revenue - m.budget) / NULLIF(m.budget, 0)) * 100.0 AS roi_pct
        FROM core.movies m
        WHERE COALESCE(m.budget, 0) >= 100000
          AND COALESCE(m.revenue, 0) > 0
    ),
    person_movies AS (
        SELECT DISTINCT
            'Cast' AS role_group,
            p.person_id,
            p.name,
            mr.movie_id,
            mr.roi_pct
        FROM movie_roi mr
        JOIN core.movie_cast mc
            ON mr.movie_id = mc.movie_id
        JOIN core.people p
            ON mc.person_id = p.person_id

        UNION ALL

        SELECT DISTINCT
            'Director' AS role_group,
            p.person_id,
            p.name,
            mr.movie_id,
            mr.roi_pct
        FROM movie_roi mr
        JOIN core.movie_crew mcr
            ON mr.movie_id = mcr.movie_id
        JOIN core.people p
            ON mcr.person_id = p.person_id
        WHERE mcr.job = 'Director'
    )
    SELECT
        role_group,
        person_id,
        name,
        COUNT(DISTINCT movie_id) AS film_count,
        AVG(roi_pct) AS avg_roi_pct,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY roi_pct) AS median_roi_pct,
        AVG(CASE WHEN roi_pct > 0 THEN 1 ELSE 0 END) * 100.0 AS success_rate_pct
    FROM person_movies
    GROUP BY role_group, person_id, name
    ORDER BY role_group, film_count DESC, name ASC;
    """
    df = run_query(query)
    if not df.empty:
        df["avg_roi_pct"] = df["avg_roi_pct"].clip(-100, 300)
        df["median_roi_pct"] = df["median_roi_pct"].clip(-100, 300)
        df["weighted_roi_score"] = df.apply(
            lambda row: weighted_roi_score(
                row["avg_roi_pct"],
                row["median_roi_pct"],
                row["success_rate_pct"],
                row["film_count"],
            ),
            axis=1,
        )
    return df


@st.cache_data(ttl=600)
def load_cast_pair_synergy_stats() -> pd.DataFrame:
    query = """
    WITH movie_roi AS (
        SELECT
            m.movie_id,
            ((m.revenue - m.budget) / NULLIF(m.budget, 0)) * 100.0 AS roi_pct
        FROM core.movies m
        WHERE COALESCE(m.budget, 0) >= 100000
          AND COALESCE(m.revenue, 0) > 0
    ),
    cast_pairs AS (
        SELECT
            mc1.movie_id,
            LEAST(mc1.person_id, mc2.person_id) AS person_id_1,
            GREATEST(mc1.person_id, mc2.person_id) AS person_id_2
        FROM core.movie_cast mc1
        JOIN core.movie_cast mc2
            ON mc1.movie_id = mc2.movie_id
           AND mc1.person_id < mc2.person_id
    )
    SELECT
        cp.person_id_1,
        cp.person_id_2,
        COUNT(DISTINCT cp.movie_id) AS pair_film_count,
        AVG(mr.roi_pct) AS pair_avg_roi_pct
    FROM cast_pairs cp
    JOIN movie_roi mr
        ON cp.movie_id = mr.movie_id
    GROUP BY cp.person_id_1, cp.person_id_2;
    """
    df = run_query(query)
    if not df.empty:
        df["pair_avg_roi_pct"] = df["pair_avg_roi_pct"].clip(-100, 300)
    return df


@st.cache_data(ttl=600)
def load_director_cast_synergy_stats() -> pd.DataFrame:
    query = """
    WITH movie_roi AS (
        SELECT
            m.movie_id,
            ((m.revenue - m.budget) / NULLIF(m.budget, 0)) * 100.0 AS roi_pct
        FROM core.movies m
        WHERE COALESCE(m.budget, 0) >= 100000
          AND COALESCE(m.revenue, 0) > 0
    ),
    director_cast_pairs AS (
        SELECT DISTINCT
            mcr.movie_id,
            mcr.person_id AS director_id,
            mc.person_id AS cast_id
        FROM core.movie_crew mcr
        JOIN core.movie_cast mc
            ON mcr.movie_id = mc.movie_id
        WHERE mcr.job = 'Director'
    )
    SELECT
        dcp.director_id,
        dcp.cast_id,
        COUNT(DISTINCT dcp.movie_id) AS pair_film_count,
        AVG(mr.roi_pct) AS pair_avg_roi_pct
    FROM director_cast_pairs dcp
    JOIN movie_roi mr
        ON dcp.movie_id = mr.movie_id
    GROUP BY dcp.director_id, dcp.cast_id;
    """
    df = run_query(query)
    if not df.empty:
        df["pair_avg_roi_pct"] = df["pair_avg_roi_pct"].clip(-100, 300)
    return df


@st.cache_data(ttl=600)
def load_roi_model_source_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    movies_query = """
    SELECT
        m.movie_id,
        m.title,
        m.budget,
        m.revenue
    FROM core.movies m
    WHERE COALESCE(m.budget, 0) >= 100000
      AND COALESCE(m.revenue, 0) > 0;
    """

    cast_query = """
    SELECT DISTINCT
        mc.movie_id,
        mc.person_id
    FROM core.movie_cast mc
    WHERE mc.person_id IS NOT NULL;
    """

    director_query = """
    SELECT DISTINCT
        mcr.movie_id,
        mcr.person_id
    FROM core.movie_crew mcr
    WHERE mcr.job = 'Director'
      AND mcr.person_id IS NOT NULL;
    """

    return (
        run_query(movies_query),
        run_query(cast_query),
        run_query(director_query),
    )


# =========================================================
# ROI Builder - model training and feature engineering
# =========================================================
def build_person_lookup_maps(person_stats: pd.DataFrame):
    director_stats_map = (
        person_stats[person_stats["role_group"] == "Director"]
        .set_index("person_id")
        .to_dict("index")
    )
    cast_stats_map = (
        person_stats[person_stats["role_group"] == "Cast"]
        .set_index("person_id")
        .to_dict("index")
    )
    return director_stats_map, cast_stats_map


def build_synergy_lookup_maps(cast_pair_df: pd.DataFrame, director_cast_df: pd.DataFrame):
    cast_pair_map = {}
    for _, row in cast_pair_df.iterrows():
        key = (int(row["person_id_1"]), int(row["person_id_2"]))
        cast_pair_map[key] = {
            "pair_film_count": int(row["pair_film_count"]),
            "pair_avg_roi_pct": float(row["pair_avg_roi_pct"]),
        }

    director_cast_map = {}
    for _, row in director_cast_df.iterrows():
        key = (int(row["director_id"]), int(row["cast_id"]))
        director_cast_map[key] = {
            "pair_film_count": int(row["pair_film_count"]),
            "pair_avg_roi_pct": float(row["pair_avg_roi_pct"]),
        }

    return cast_pair_map, director_cast_map


def compute_team_features(
    director_id: Optional[int],
    cast_ids: List[int],
    director_stats_map: Dict[int, dict],
    cast_stats_map: Dict[int, dict],
    cast_pair_map: Dict[Tuple[int, int], dict],
    director_cast_map: Dict[Tuple[int, int], dict],
) -> Dict[str, float]:
    director_avg_roi = 0.0
    director_median_roi = 0.0
    director_success_rate = 0.0
    director_film_count = 0.0
    director_weighted_score = 0.0

    if director_id is not None and director_id in director_stats_map:
        d = director_stats_map[director_id]
        director_avg_roi = float(d.get("avg_roi_pct", 0.0))
        director_median_roi = float(d.get("median_roi_pct", 0.0))
        director_success_rate = float(d.get("success_rate_pct", 0.0))
        director_film_count = float(d.get("film_count", 0.0))
        director_weighted_score = float(d.get("weighted_roi_score", 0.0))

    cast_avg_rois = []
    cast_median_rois = []
    cast_success_rates = []
    cast_film_counts = []
    cast_weighted_scores = []

    for cast_id in cast_ids:
        if cast_id in cast_stats_map:
            c = cast_stats_map[cast_id]
            cast_avg_rois.append(float(c.get("avg_roi_pct", 0.0)))
            cast_median_rois.append(float(c.get("median_roi_pct", 0.0)))
            cast_success_rates.append(float(c.get("success_rate_pct", 0.0)))
            cast_film_counts.append(float(c.get("film_count", 0.0)))
            cast_weighted_scores.append(float(c.get("weighted_roi_score", 0.0)))

    cast_avg_roi = safe_mean(cast_avg_rois, 0.0)
    cast_median_roi = safe_median(cast_avg_rois, 0.0)
    cast_best_roi = safe_max(cast_avg_rois, 0.0)
    cast_top2_avg_roi = safe_mean(sorted(cast_avg_rois, reverse=True)[:2], 0.0)
    cast_success_rate = safe_mean(cast_success_rates, 0.0)
    cast_film_count_avg = safe_mean(cast_film_counts, 0.0)
    cast_weighted_score_avg = safe_mean(cast_weighted_scores, 0.0)
    cast_count = float(len(cast_ids))

    cast_pair_rois = []
    cast_pair_counts = []
    for a, b in combinations(sorted(cast_ids), 2):
        key = (min(a, b), max(a, b))
        if key in cast_pair_map:
            cast_pair_rois.append(float(cast_pair_map[key]["pair_avg_roi_pct"]))
            cast_pair_counts.append(float(cast_pair_map[key]["pair_film_count"]))

    cast_pair_synergy_avg_roi = safe_mean(cast_pair_rois, 0.0)
    cast_pair_synergy_avg_count = safe_mean(cast_pair_counts, 0.0)
    known_cast_pair_count = float(len(cast_pair_rois))

    director_cast_rois = []
    director_cast_counts = []
    if director_id is not None:
        for cast_id in cast_ids:
            key = (director_id, cast_id)
            if key in director_cast_map:
                director_cast_rois.append(float(director_cast_map[key]["pair_avg_roi_pct"]))
                director_cast_counts.append(float(director_cast_map[key]["pair_film_count"]))

    director_cast_synergy_avg_roi = safe_mean(director_cast_rois, 0.0)
    director_cast_synergy_avg_count = safe_mean(director_cast_counts, 0.0)
    known_director_cast_count = float(len(director_cast_rois))

    has_director = 1.0 if director_id is not None else 0.0

    return {
        "director_avg_roi_pct": director_avg_roi,
        "director_median_roi_pct": director_median_roi,
        "director_success_rate_pct": director_success_rate,
        "director_film_count": director_film_count,
        "director_weighted_score": director_weighted_score,
        "cast_avg_roi_pct": cast_avg_roi,
        "cast_median_roi_pct": cast_median_roi,
        "cast_best_roi_pct": cast_best_roi,
        "cast_top2_avg_roi_pct": cast_top2_avg_roi,
        "cast_success_rate_pct": cast_success_rate,
        "cast_film_count_avg": cast_film_count_avg,
        "cast_weighted_score_avg": cast_weighted_score_avg,
        "cast_count": cast_count,
        "cast_pair_synergy_avg_roi_pct": cast_pair_synergy_avg_roi,
        "cast_pair_synergy_avg_film_count": cast_pair_synergy_avg_count,
        "known_cast_pair_count": known_cast_pair_count,
        "director_cast_synergy_avg_roi_pct": director_cast_synergy_avg_roi,
        "director_cast_synergy_avg_film_count": director_cast_synergy_avg_count,
        "known_director_cast_count": known_director_cast_count,
        "has_director": has_director,
    }


@st.cache_resource
def train_roi_builder_model():
    person_stats = load_person_role_stats()
    cast_pair_df = load_cast_pair_synergy_stats()
    director_cast_df = load_director_cast_synergy_stats()
    movies_df, cast_assignments_df, director_assignments_df = load_roi_model_source_data()

    director_stats_map, cast_stats_map = build_person_lookup_maps(person_stats)
    cast_pair_map, director_cast_map = build_synergy_lookup_maps(cast_pair_df, director_cast_df)

    cast_by_movie = (
        cast_assignments_df.groupby("movie_id")["person_id"]
        .apply(lambda x: sorted(list(set(int(v) for v in x.tolist()))))
        .to_dict()
    )

    director_by_movie = (
        director_assignments_df.groupby("movie_id")["person_id"]
        .apply(lambda x: int(x.iloc[0]) if not x.empty else None)
        .to_dict()
    )

    rows = []
    for _, movie in movies_df.iterrows():
        movie_id = int(movie["movie_id"])
        budget = float(movie["budget"])
        revenue = float(movie["revenue"])

        if budget < 100000:
            continue

        roi_pct = ((revenue - budget) / budget) * 100.0
        roi_pct_clipped = float(np.clip(roi_pct, -100.0, 300.0))

        director_id = director_by_movie.get(movie_id)
        cast_ids = cast_by_movie.get(movie_id, [])

        team_features = compute_team_features(
            director_id=director_id,
            cast_ids=cast_ids,
            director_stats_map=director_stats_map,
            cast_stats_map=cast_stats_map,
            cast_pair_map=cast_pair_map,
            director_cast_map=director_cast_map,
        )

        rows.append(
            {
                "movie_id": movie_id,
                "title": movie["title"],
                "target_roi_pct": roi_pct_clipped,
                **team_features,
            }
        )

    feature_df = pd.DataFrame(rows)
    if feature_df.empty:
        raise ValueError("No training data could be built for ROI Builder.")

    feature_columns = [
        "director_avg_roi_pct",
        "director_median_roi_pct",
        "director_success_rate_pct",
        "director_film_count",
        "director_weighted_score",
        "cast_avg_roi_pct",
        "cast_median_roi_pct",
        "cast_best_roi_pct",
        "cast_top2_avg_roi_pct",
        "cast_success_rate_pct",
        "cast_film_count_avg",
        "cast_weighted_score_avg",
        "cast_count",
        "cast_pair_synergy_avg_roi_pct",
        "cast_pair_synergy_avg_film_count",
        "known_cast_pair_count",
        "director_cast_synergy_avg_roi_pct",
        "director_cast_synergy_avg_film_count",
        "known_director_cast_count",
        "has_director",
    ]

    fill_values = {col: float(feature_df[col].median()) for col in feature_columns}
    X = feature_df[feature_columns].fillna(fill_values).copy()
    y = feature_df["target_roi_pct"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = RandomForestRegressor(
        n_estimators=350,
        max_depth=9,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    return {
        "model": model,
        "feature_columns": feature_columns,
        "fill_values": fill_values,
        "person_stats": person_stats,
        "director_stats_map": director_stats_map,
        "cast_stats_map": cast_stats_map,
        "cast_pair_map": cast_pair_map,
        "director_cast_map": director_cast_map,
    }


# =========================================================
# ROI Builder - UI helpers
# =========================================================
def ensure_roi_builder_session_state() -> None:
    if "roi_builder_selected_director_id" not in st.session_state:
        st.session_state["roi_builder_selected_director_id"] = None
    if "roi_builder_selected_cast_ids" not in st.session_state:
        st.session_state["roi_builder_selected_cast_ids"] = []


def get_filtered_role_options(person_stats: pd.DataFrame, role_group: str, min_films: int) -> dict:
    role_df = person_stats[
        (person_stats["role_group"] == role_group)
        & (person_stats["film_count"] >= min_films)
    ].copy()

    if role_df.empty:
        return {}

    role_df = role_df.sort_values(
        ["weighted_roi_score", "film_count", "name"],
        ascending=[False, False, True]
    )

    label_to_id = {}
    for _, row in role_df.iterrows():
        label = (
            f"{row['name']} "
            f"(Films: {int(row['film_count'])} | "
            f"Avg ROI: {format_number(row['avg_roi_pct'], 1)}%)"
        )
        label_to_id[label] = int(row["person_id"])

    return label_to_id


def sync_roi_builder_state_with_filters(director_options: dict, cast_options: dict) -> None:
    available_director_ids = set(director_options.values())
    available_cast_ids = set(cast_options.values())

    if st.session_state["roi_builder_selected_director_id"] not in available_director_ids:
        st.session_state["roi_builder_selected_director_id"] = None

    st.session_state["roi_builder_selected_cast_ids"] = [
        person_id
        for person_id in st.session_state["roi_builder_selected_cast_ids"]
        if person_id in available_cast_ids
    ]


def add_cast_member(person_id: int) -> None:
    if person_id not in st.session_state["roi_builder_selected_cast_ids"]:
        st.session_state["roi_builder_selected_cast_ids"].append(person_id)


def remove_cast_member(person_id: int) -> None:
    st.session_state["roi_builder_selected_cast_ids"] = [
        pid for pid in st.session_state["roi_builder_selected_cast_ids"] if pid != person_id
    ]


def get_person_row(person_stats: pd.DataFrame, role_group: str, person_id: int) -> Optional[pd.Series]:
    df = person_stats[
        (person_stats["role_group"] == role_group)
        & (person_stats["person_id"] == person_id)
    ]
    if df.empty:
        return None
    return df.iloc[0]


def build_current_team_prediction(artifacts: dict) -> Tuple[float, Dict[str, float]]:
    director_id = st.session_state["roi_builder_selected_director_id"]
    cast_ids = st.session_state["roi_builder_selected_cast_ids"]

    if director_id is None and not cast_ids:
        return 0.0, {
            "director_avg_roi_pct": 0.0,
            "cast_avg_roi_pct": 0.0,
            "cast_pair_synergy_avg_roi_pct": 0.0,
            "director_cast_synergy_avg_roi_pct": 0.0,
            "known_cast_pair_count": 0.0,
            "known_director_cast_count": 0.0,
            "cast_count": 0.0,
            "cast_top2_avg_roi_pct": 0.0,
        }

    team_features = compute_team_features(
        director_id=director_id,
        cast_ids=cast_ids,
        director_stats_map=artifacts["director_stats_map"],
        cast_stats_map=artifacts["cast_stats_map"],
        cast_pair_map=artifacts["cast_pair_map"],
        director_cast_map=artifacts["director_cast_map"],
    )

    row = pd.DataFrame(
        [[team_features.get(col, 0.0) for col in artifacts["feature_columns"]]],
        columns=artifacts["feature_columns"],
    ).fillna(artifacts["fill_values"])

    predicted_roi = float(artifacts["model"].predict(row)[0])
    predicted_roi = float(np.clip(predicted_roi, -100.0, 300.0))

    return predicted_roi, team_features


def build_selected_people_table(person_stats: pd.DataFrame) -> pd.DataFrame:
    rows = []

    director_id = st.session_state["roi_builder_selected_director_id"]
    if director_id is not None:
        row = get_person_row(person_stats, "Director", director_id)
        if row is not None:
            rows.append(
                {
                    "Role": "Director",
                    "Name": row["name"],
                    "Films": int(row["film_count"]),
                    "Avg ROI %": format_number(row["avg_roi_pct"], 1),
                    "Median ROI %": format_number(row["median_roi_pct"], 1),
                    "Success Rate %": format_number(row["success_rate_pct"], 1),
                    "Weighted Score": format_number(row["weighted_roi_score"], 1),
                }
            )

    for cast_id in st.session_state["roi_builder_selected_cast_ids"]:
        row = get_person_row(person_stats, "Cast", cast_id)
        if row is not None:
            rows.append(
                {
                    "Role": "Cast",
                    "Name": row["name"],
                    "Films": int(row["film_count"]),
                    "Avg ROI %": format_number(row["avg_roi_pct"], 1),
                    "Median ROI %": format_number(row["median_roi_pct"], 1),
                    "Success Rate %": format_number(row["success_rate_pct"], 1),
                    "Weighted Score": format_number(row["weighted_roi_score"], 1),
                }
            )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def top_people_table(person_stats: pd.DataFrame, role_group: str, min_films: int, limit: int = 15) -> pd.DataFrame:
    df = person_stats[
        (person_stats["role_group"] == role_group)
        & (person_stats["film_count"] >= min_films)
    ].copy()

    if df.empty:
        return pd.DataFrame()

    df = df.sort_values(["weighted_roi_score", "film_count"], ascending=[False, False]).head(limit)
    display_df = pd.DataFrame(
        {
            "Name": df["name"],
            "Films": df["film_count"].astype(int),
            "Avg ROI %": df["avg_roi_pct"].apply(lambda x: format_number(x, 1)),
            "Median ROI %": df["median_roi_pct"].apply(lambda x: format_number(x, 1)),
            "Success Rate %": df["success_rate_pct"].apply(lambda x: format_number(x, 1)),
            "Weighted Score": df["weighted_roi_score"].apply(lambda x: format_number(x, 1)),
        }
    )
    return display_df


# =========================================================
# ROI Builder page
# =========================================================
def show_roi_builder_page() -> None:
    st.title("📈 ROI Builder")
    st.write(
        "Build a new movie team by selecting a director and cast members. "
        "The expected ROI is estimated with a model that combines track record, team strength, and collaboration synergy."
    )

    with st.spinner("Loading ROI Builder..."):
        artifacts = train_roi_builder_model()

    person_stats = artifacts["person_stats"]
    ensure_roi_builder_session_state()

    st.sidebar.markdown("### Filters for ROI Builder")
    min_films = st.sidebar.selectbox(
        "Minimum Number of Films",
        options=[1, 3, 5, 10],
        index=1,
        key="roi_min_films",
    )

    director_options = get_filtered_role_options(person_stats, "Director", min_films)
    cast_options = get_filtered_role_options(person_stats, "Cast", min_films)
    sync_roi_builder_state_with_filters(director_options, cast_options)

    if not director_options and not cast_options:
        st.warning("No matching people found for the selected minimum number of films.")
        return

    director_id_to_label = {v: k for k, v in director_options.items()}

    current_director_id = st.session_state["roi_builder_selected_director_id"]
    current_director_label = (
        director_id_to_label[current_director_id]
        if current_director_id is not None and current_director_id in director_id_to_label
        else "No Selection"
    )

    director_labels = ["No Selection"] + list(director_options.keys())

    selected_director_label = st.selectbox(
        "Select Director:",
        options=director_labels,
        index=director_labels.index(current_director_label),
        key="roi_director_selectbox",
    )

    if selected_director_label == "No Selection":
        st.session_state["roi_builder_selected_director_id"] = None
    else:
        st.session_state["roi_builder_selected_director_id"] = director_options[selected_director_label]

    st.markdown("### Build Cast")

    available_cast_labels = [
        label
        for label, person_id in cast_options.items()
        if person_id not in st.session_state["roi_builder_selected_cast_ids"]
    ]

    add_col1, add_col2 = st.columns([4, 1])

    with add_col1:
        selected_cast_to_add = st.selectbox(
            "Search and select cast member:",
            options=["Please choose"] + available_cast_labels,
            key="roi_cast_add_selectbox",
        )

    with add_col2:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        if st.button("Add", key="roi_add_cast_button", use_container_width=True):
            if selected_cast_to_add != "Please choose":
                add_cast_member(cast_options[selected_cast_to_add])
                st.rerun()

    predicted_roi, team_features = build_current_team_prediction(artifacts)

    st.markdown("---")
    st.subheader("Expected Team Performance")

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Expected ROI", f"{format_number(predicted_roi, 1)}%")
    metric_col2.metric("Director Avg ROI", f"{format_number(team_features['director_avg_roi_pct'], 1)}%")
    metric_col3.metric("Cast Avg ROI", f"{format_number(team_features['cast_avg_roi_pct'], 1)}%")
    metric_col4.metric("Top-2 Cast ROI", f"{format_number(team_features['cast_top2_avg_roi_pct'], 1)}%")

    metric_col5, metric_col6, metric_col7, metric_col8 = st.columns(4)
    metric_col5.metric("Cast Pair Synergy", f"{format_number(team_features['cast_pair_synergy_avg_roi_pct'], 1)}%")
    metric_col6.metric("Director-Cast Synergy", f"{format_number(team_features['director_cast_synergy_avg_roi_pct'], 1)}%")
    metric_col7.metric("Known Cast Pairs", format_int(team_features["known_cast_pair_count"]))
    metric_col8.metric("Known Director-Cast Links", format_int(team_features["known_director_cast_count"]))

    if predicted_roi == 0 and st.session_state["roi_builder_selected_director_id"] is None and not st.session_state["roi_builder_selected_cast_ids"]:
        st.info("Current expected ROI is 0 because no director or cast has been selected yet.")
    elif predicted_roi < 0:
        st.error("This team currently suggests a negative ROI scenario.")
    elif predicted_roi < 50:
        st.warning("This team currently suggests a moderate ROI scenario.")
    else:
        st.success("This team currently suggests a strong ROI scenario.")

    with st.expander("How is the ROI calculated?"):
        st.markdown(
            """
### Goal
This tool estimates the **expected ROI** of a newly assembled movie team.

### Step 1: Historical performance is collected
For directors and actors, the app looks at past films and calculates:
- average ROI
- median ROI
- success rate
- number of films

### Step 2: Team strength is derived
For your selected team, the model builds features such as:
- director average ROI
- cast average ROI
- top-2 cast ROI
- average experience
- weighted performance score

### Step 3: Collaboration synergy is included
The app also checks:
- whether selected actors have worked together before
- whether the selected director has worked with the chosen actors before

If those collaborations existed and performed well historically, the synergy features improve the estimate.

### Step 4: Machine learning prediction
A **Random Forest Regressor** is trained on historical films from the database.
Target variable:
- ROI = (Revenue - Budget) / Budget * 100

### Stabilization
To avoid unrealistic outliers:
- only films with budget >= 100,000 are used
- ROI values are clipped to the range **-100% to 300%**

### Interpretation
- below 0% = likely loss
- 0% to 50% = moderate
- 50% to 150% = strong
- above 150% = very strong

This is a **data-based estimate**, not a guarantee.
            """
        )

    st.markdown("---")
    st.subheader("Selected People")

    selected_people_df = build_selected_people_table(person_stats)
    if selected_people_df.empty:
        st.info("No people selected yet.")
    else:
        st.dataframe(selected_people_df, use_container_width=True, hide_index=True)

    if st.session_state["roi_builder_selected_cast_ids"]:
        st.markdown("---")
        st.subheader("Remove Cast Members")
        for cast_id in st.session_state["roi_builder_selected_cast_ids"]:
            cast_row = get_person_row(person_stats, "Cast", cast_id)
            if cast_row is None:
                continue

            remove_col1, remove_col2 = st.columns([5, 1])
            with remove_col1:
                st.write(
                    f"**{cast_row['name']}** — Films: {int(cast_row['film_count'])} | "
                    f"Avg ROI: {format_number(cast_row['avg_roi_pct'], 1)}% | "
                    f"Weighted Score: {format_number(cast_row['weighted_roi_score'], 1)}"
                )
            with remove_col2:
                if st.button("Remove", key=f"roi_remove_cast_{cast_id}", use_container_width=True):
                    remove_cast_member(cast_id)
                    st.rerun()

    st.markdown("---")
    st.subheader("Top Directors")
    top_directors_df = top_people_table(person_stats, "Director", min_films=min_films, limit=15)
    if top_directors_df.empty:
        st.write("No data available.")
    else:
        st.dataframe(top_directors_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Top Actors")
    top_actors_df = top_people_table(person_stats, "Cast", min_films=min_films, limit=20)
    if top_actors_df.empty:
        st.write("No data available.")
    else:
        st.dataframe(top_actors_df, use_container_width=True, hide_index=True)


# =========================================================
# Movie Database page
# =========================================================
def show_movie_database_page() -> None:
    st.title("🎬 Movie Database")
    st.write(
        "Browse the movie database with filters and search terms. "
        "Below the result list, you can select one movie for the detailed view."
    )

    st.sidebar.markdown("### Filters for Movie Database")
    search_term = st.sidebar.text_input("Movie Title", placeholder="leave empty = browse with filters only")
    min_vote = st.sidebar.slider("Minimum Rating", 0.0, 10.0, 0.0, 0.1)
    year_from = st.sidebar.number_input("Release Year From", min_value=1900, max_value=2100, value=1900, step=1)
    year_to = st.sidebar.number_input("Release Year To", min_value=1900, max_value=2100, value=2100, step=1)
    min_votes = st.sidebar.number_input("Minimum Votes", min_value=0, max_value=1000000, value=0, step=100)
    only_with_reviews = st.sidebar.checkbox("Only Movies with Reviews", value=False)
    only_with_poster = st.sidebar.checkbox("Only Movies with Poster", value=False)
    results_limit = st.sidebar.selectbox("Maximum Results", [25, 50, 100, 200], index=2)
    sort_by = st.sidebar.selectbox(
        "Sort by",
        ["Popularity", "Vote Average", "Vote Count", "Revenue", "Budget", "Runtime", "Year", "Title"],
    )
    sort_order = st.sidebar.selectbox("Order", ["Descending", "Ascending"])

    genre_filter_query = """
        SELECT genre_name
        FROM core.genres
        ORDER BY genre_name;
    """
    all_genres_df = run_query(genre_filter_query)
    genre_options = all_genres_df["genre_name"].dropna().tolist() if not all_genres_df.empty else []
    selected_genre = st.sidebar.selectbox("Genre", ["All"] + genre_options)

    search_value = f"%{search_term.strip()}%" if search_term and search_term.strip() else "%"

    sort_column_map = {
        "Popularity": "m.popularity",
        "Vote Average": "m.vote_average",
        "Vote Count": "m.vote_count",
        "Revenue": "m.revenue",
        "Budget": "m.budget",
        "Runtime": "m.runtime",
        "Year": "release_year",
        "Title": "m.title",
    }
    sort_direction = "DESC" if sort_order == "Descending" else "ASC"
    sort_column = sort_column_map[sort_by]

    movie_search_query = f"""
        SELECT DISTINCT
            m.movie_id,
            m.title,
            NULLIF(m.release_date, '')::date AS release_date_cast,
            m.runtime,
            m.budget,
            m.revenue,
            m.vote_average,
            m.vote_count,
            m.popularity,
            m.poster_path,
            EXTRACT(YEAR FROM NULLIF(m.release_date, '')::date) AS release_year,
            EXISTS (
                SELECT 1
                FROM core.reviews r
                WHERE r.movie_id = m.movie_id
            ) AS has_reviews
        FROM core.movies m
        LEFT JOIN core.movie_genres mg
            ON m.movie_id = mg.movie_id
        LEFT JOIN core.genres g
            ON mg.genre_id = g.genre_id
        WHERE LOWER(m.title) LIKE LOWER(%s)
          AND COALESCE(m.vote_average, 0) >= %s
          AND COALESCE(m.vote_count, 0) >= %s
          AND (
              m.release_date IS NULL
              OR m.release_date = ''
              OR EXTRACT(YEAR FROM NULLIF(m.release_date, '')::date) BETWEEN %s AND %s
          )
          AND (
              %s = 'All'
              OR g.genre_name = %s
          )
          AND (
              %s = FALSE
              OR EXISTS (
                    SELECT 1
                    FROM core.reviews r
                    WHERE r.movie_id = m.movie_id
              )
          )
          AND (
              %s = FALSE
              OR (m.poster_path IS NOT NULL AND m.poster_path <> '')
          )
        ORDER BY {sort_column} {sort_direction} NULLS LAST, m.title ASC
        LIMIT {results_limit};
    """

    movie_results = run_query(
        movie_search_query,
        (
            search_value,
            min_vote,
            min_votes,
            year_from,
            year_to,
            selected_genre,
            selected_genre,
            only_with_reviews,
            only_with_poster,
        ),
    )

    st.caption(f"Movies found: {len(movie_results)}")

    if movie_results.empty:
        st.warning("No results found. Adjust the filters or broaden the search.")
        return

    st.markdown("### Quick Selection")
    st.caption("Compact preview of the current results. Select a movie below for the detail view.")

    if "selected_movie_id" not in st.session_state:
        st.session_state["selected_movie_id"] = None

    filter_signature = (
        search_value,
        min_vote,
        min_votes,
        year_from,
        year_to,
        selected_genre,
        only_with_reviews,
        only_with_poster,
        results_limit,
        sort_by,
        sort_order,
        len(movie_results),
    )

    if "quick_pick_page" not in st.session_state:
        st.session_state["quick_pick_page"] = 1
    if st.session_state.get("quick_pick_filter_signature") != filter_signature:
        st.session_state["quick_pick_page"] = 1
        st.session_state["quick_pick_filter_signature"] = filter_signature

    cards_per_page = 6
    total_cards = len(movie_results)
    total_pages = max(1, (total_cards + cards_per_page - 1) // cards_per_page)

    if st.session_state["quick_pick_page"] > total_pages:
        st.session_state["quick_pick_page"] = total_pages

    start_idx = (st.session_state["quick_pick_page"] - 1) * cards_per_page
    end_idx = start_idx + cards_per_page
    quick_pick_df = movie_results.iloc[start_idx:end_idx]
    quick_pick_cols = st.columns(6, gap="small")

    for idx, (_, row) in enumerate(quick_pick_df.iterrows()):
        with quick_pick_cols[idx % 6]:
            img_url = poster_url(row.get("poster_path"))
            if img_url:
                st.image(img_url, width=220)
            else:
                st.markdown("**No Poster**")

            st.markdown(f"**{row['title']}**")
            st.caption(
                f"{int(row['release_year']) if pd.notna(row['release_year']) else 'n/a'} | "
                f"Rating: {format_number(row['vote_average'], 1)}"
            )
            if st.button("Details", key=f"detail_btn_{row['movie_id']}", use_container_width=True):
                st.session_state["selected_movie_id"] = int(row["movie_id"])
                st.rerun()

    nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
    with nav_col1:
        if st.button("← Previous", disabled=st.session_state["quick_pick_page"] <= 1, use_container_width=True):
            st.session_state["quick_pick_page"] -= 1
            st.rerun()
    with nav_col2:
        st.markdown(
            f"<div style='text-align:center; padding-top:8px;'><b>Page {st.session_state['quick_pick_page']} of {total_pages}</b></div>",
            unsafe_allow_html=True,
        )
    with nav_col3:
        if st.button("Next →", disabled=st.session_state["quick_pick_page"] >= total_pages, use_container_width=True):
            st.session_state["quick_pick_page"] += 1
            st.rerun()

    st.markdown("---")
    st.markdown("### Detail View")

    movie_options = {
        (
            f"{row['title']} "
            f"({int(row['release_year']) if pd.notna(row['release_year']) else 'n/a'})"
            f" | Rating: {format_number(row['vote_average'], 1)}"
            f" | ID: {row['movie_id']}"
        ): row["movie_id"]
        for _, row in movie_results.iterrows()
    }

    label_by_id = {value: key for key, value in movie_options.items()}

    default_movie_id = st.session_state.get("selected_movie_id")
    if default_movie_id not in label_by_id:
        default_movie_id = list(movie_options.values())[0]
        st.session_state["selected_movie_id"] = default_movie_id

    movie_labels = list(movie_options.keys())
    default_index = movie_labels.index(label_by_id[default_movie_id])

    selected_label = st.selectbox(
        "Or select a movie directly from the result list",
        movie_labels,
        index=default_index,
    )
    selected_movie_id = movie_options[selected_label]
    st.session_state["selected_movie_id"] = selected_movie_id

    movie_detail_query = """
        SELECT
            m.movie_id,
            m.title,
            NULLIF(m.release_date, '')::date AS release_date,
            m.runtime,
            m.budget,
            m.revenue,
            m.vote_average,
            m.vote_count,
            m.popularity,
            m.original_title,
            m.overview,
            m.poster_path,
            m.backdrop_path,
            m.status,
            m.tagline,
            m.homepage,
            m.original_language,
            m.adult,
            m.video,
            m.created_at,
            m.updated_at
        FROM core.movies m
        WHERE m.movie_id = %s;
    """

    genre_query = """
        SELECT g.genre_name
        FROM core.movie_genres mg
        JOIN core.genres g
            ON mg.genre_id = g.genre_id
        WHERE mg.movie_id = %s
        ORDER BY g.genre_name;
    """

    cast_query = """
        SELECT
            p.name,
            mc.character,
            mc.cast_order
        FROM core.movie_cast mc
        JOIN core.people p
            ON mc.person_id = p.person_id
        WHERE mc.movie_id = %s
        ORDER BY mc.cast_order ASC NULLS LAST, p.name ASC
        LIMIT 30;
    """

    crew_query = """
        SELECT
            p.name,
            mc.job,
            mc.department
        FROM core.movie_crew mc
        JOIN core.people p
            ON mc.person_id = p.person_id
        WHERE mc.movie_id = %s
        ORDER BY mc.department ASC NULLS LAST, mc.job ASC NULLS LAST, p.name ASC
        LIMIT 50;
    """

    review_query = """
        SELECT
            author,
            author_username,
            author_rating,
            movie_rating,
            content,
            content_length,
            created_at,
            updated_at,
            url
        FROM core.reviews
        WHERE movie_id = %s
        ORDER BY author_rating DESC NULLS LAST, content_length DESC NULLS LAST
        LIMIT 10;
    """

    movie_df = run_query(movie_detail_query, (selected_movie_id,))
    genres_df = run_query(genre_query, (selected_movie_id,))
    cast_df = run_query(cast_query, (selected_movie_id,))
    crew_df = run_query(crew_query, (selected_movie_id,))
    reviews_df = run_query(review_query, (selected_movie_id,))

    if movie_df.empty:
        st.error("Movie details could not be loaded.")
        return

    movie = movie_df.iloc[0]

    poster_col, header_col = st.columns([1, 3])
    with poster_col:
        img_url = poster_url(movie.get("poster_path"))
        if img_url:
            st.image(img_url, use_container_width=True)
        else:
            st.info("No poster available")

    with header_col:
        st.subheader(movie["title"])
        subtitle_parts = []
        if pd.notna(movie.get("original_title")) and str(movie.get("original_title")) != str(movie.get("title")):
            subtitle_parts.append(f"Original Title: {movie['original_title']}")
        if pd.notna(movie.get("tagline")) and str(movie.get("tagline")).strip():
            subtitle_parts.append(f"Tagline: {movie['tagline']}")
        if subtitle_parts:
            for part in subtitle_parts:
                st.write(part)

        genre_tags = genres_df["genre_name"].dropna().tolist() if not genres_df.empty else []
        if genre_tags:
            st.markdown("**Genres:** " + " | ".join(genre_tags))

        if pd.notna(movie.get("homepage")) and str(movie.get("homepage")).strip():
            st.markdown(f"[Open Homepage]({movie['homepage']})")

    st.markdown("---")

    metric_cols = st.columns(4)
    metric_cols[0].metric("Release Date", format_date(movie.get("release_date")))
    metric_cols[0].metric("Runtime", f"{format_int(movie.get('runtime'))} min" if pd.notna(movie.get("runtime")) else "n/a")
    metric_cols[1].metric("Vote Average", format_number(movie.get("vote_average"), 2))
    metric_cols[1].metric("Vote Count", format_int(movie.get("vote_count")))
    metric_cols[2].metric("Popularity", format_number(movie.get("popularity"), 2))
    metric_cols[2].metric("Budget", format_money(movie.get("budget")))
    metric_cols[3].metric("Revenue", format_money(movie.get("revenue")))
    metric_cols[3].metric("Reviews", str(len(reviews_df)))

    st.markdown("---")

    info_col, summary_col = st.columns([1, 2])
    with info_col:
        st.markdown("### Metadata")
        info_df = pd.DataFrame(
            {
                "Field": [
                    "Movie ID",
                    "Original Language",
                    "Status",
                    "Adult",
                    "Video",
                    "Created At",
                    "Updated At",
                ],
                "Value": [
                    movie.get("movie_id"),
                    movie.get("original_language") if pd.notna(movie.get("original_language")) else "n/a",
                    movie.get("status") if pd.notna(movie.get("status")) else "n/a",
                    format_bool(movie.get("adult")),
                    format_bool(movie.get("video")),
                    movie.get("created_at") if pd.notna(movie.get("created_at")) else "n/a",
                    movie.get("updated_at") if pd.notna(movie.get("updated_at")) else "n/a",
                ],
            }
        )
        st.dataframe(info_df, use_container_width=True, hide_index=True)

    with summary_col:
        st.markdown("### Overview")
        overview = movie.get("overview")
        if pd.notna(overview) and str(overview).strip() and str(overview).lower() != "nan":
            st.write(overview)
        else:
            st.write("No description available.")

    st.markdown("---")

    cast_col, crew_col = st.columns(2)
    with cast_col:
        st.markdown("### Cast")
        if cast_df.empty:
            st.write("No cast data available.")
        else:
            display_cast = cast_df.copy()
            display_cast["cast_order"] = display_cast["cast_order"].astype("Int64")
            display_cast = display_cast.rename(
                columns={
                    "name": "Name",
                    "character": "Character",
                    "cast_order": "Cast Order",
                }
            )
            st.dataframe(display_cast, use_container_width=True, hide_index=True)

    with crew_col:
        st.markdown("### Crew")
        if crew_df.empty:
            st.write("No crew data available.")
        else:
            display_crew = crew_df.rename(
                columns={
                    "name": "Name",
                    "job": "Job",
                    "department": "Department",
                }
            )
            st.dataframe(display_crew, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Reviews")
    st.caption("The 10 most relevant or longest reviews are displayed.")

    if reviews_df.empty:
        st.write("No reviews available.")
    else:
        for _, row in reviews_df.iterrows():
            reviewer = row["author"] if pd.notna(row["author"]) else "Unknown"
            username = row["author_username"] if pd.notna(row["author_username"]) else "n/a"
            rating = row["author_rating"] if pd.notna(row["author_rating"]) else row["movie_rating"]
            content = row["content"] if pd.notna(row["content"]) else ""
            preview = content[:800] + ("..." if len(content) > 800 else "")
            review_title = (
                f"{reviewer} | Username: {username} | "
                f"Rating: {format_number(rating, 1) if pd.notna(rating) else 'n/a'}"
            )

            with st.expander(review_title):
                st.write(preview)
                meta_left, meta_right = st.columns(2)
                meta_left.caption(f"Created: {row['created_at'] if pd.notna(row['created_at']) else 'n/a'}")
                meta_right.caption(f"Updated: {row['updated_at'] if pd.notna(row['updated_at']) else 'n/a'}")
                if pd.notna(row["url"]) and str(row["url"]).strip():
                    st.markdown(f"[Open Review Link]({row['url']})")

    st.markdown("---")
    st.markdown("### Result List")

    display_results = movie_results.copy()
    if "release_year" in display_results.columns:
        display_results["release_year"] = display_results["release_year"].astype("Int64")

    display_results["Poster"] = display_results["poster_path"].apply(
        lambda x: "Yes" if pd.notna(x) and str(x).strip() else "No"
    )
    display_results["Reviews"] = display_results["has_reviews"].apply(lambda x: "Yes" if bool(x) else "No")
    display_results["Budget"] = display_results["budget"].apply(format_money)
    display_results["Revenue"] = display_results["revenue"].apply(format_money)
    display_results["Popularity"] = display_results["popularity"].apply(lambda x: format_number(x, 2))
    display_results["Vote Average"] = display_results["vote_average"].apply(lambda x: format_number(x, 2))
    display_results["Vote Count"] = display_results["vote_count"].apply(format_int)
    display_results["Runtime"] = display_results["runtime"].apply(format_int)
    display_results["Year"] = display_results["release_year"]
    display_results["Movie ID"] = display_results["movie_id"]
    display_results["Title"] = display_results["title"]

    st.dataframe(
        display_results[
            [
                "Movie ID",
                "Poster",
                "Title",
                "Year",
                "Vote Average",
                "Vote Count",
                "Popularity",
                "Runtime",
                "Budget",
                "Revenue",
                "Reviews",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    csv_export = display_results[
        [
            "Movie ID",
            "Title",
            "Year",
            "Vote Average",
            "Vote Count",
            "Popularity",
            "Runtime",
            "Budget",
            "Revenue",
            "Reviews",
        ]
    ].to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Result List as CSV",
        data=csv_export,
        file_name="filtered_movies.csv",
        mime="text/csv",
    )


# =========================================================
# Navigation
# =========================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    [
        "Movie Database",
        "Recommendation Engine",
        "ROI Builder",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("TMDB Final Project")


# =========================================================
# Router
# =========================================================
if page == "Movie Database":
    show_movie_database_page()
elif page == "Recommendation Engine":
    show_recommendation_engine_page()
else:
    show_roi_builder_page()