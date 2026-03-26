import os
from typing import Optional, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
    return "Ja" if bool(value) else "Nein"


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


def safe_max(values: List[float], fallback: float = 0.0) -> float:
    clean = [float(v) for v in values if pd.notna(v)]
    if not clean:
        return float(fallback)
    return float(np.max(clean))


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
# Recommender Engine (ML Projekt 1)
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


def show_ml_project_1_page() -> None:
    st.title("🤖 ML Projekt 1 – Recommendation Engine")
    st.write(
        "Dieses Modul empfiehlt ähnliche Filme auf Basis von Inhalt, Genres, Cast, Director und Overview."
    )

    with st.spinner("Lade Recommendation Engine..."):
        df = load_recommender_data()
        df, indices, cosine_sim = build_recommender_engine(df)

    if df.empty:
        st.warning("Für die Recommendation Engine konnten keine Daten geladen werden.")
        return

    st.sidebar.markdown("### Filter für ML Projekt 1")
    min_rating = st.sidebar.slider(
        "Minimum Rating (ML 1)",
        min_value=0.0,
        max_value=10.0,
        value=6.0,
        step=0.5,
        key="ml1_min_rating",
    )

    valid_runtime = df["runtime"].dropna()
    max_runtime = int(valid_runtime.max()) if not valid_runtime.empty else 300
    selected_runtime = st.sidebar.slider(
        "Maximum Runtime (ML 1)",
        min_value=60,
        max_value=max_runtime,
        value=min(120, max_runtime),
        key="ml1_runtime",
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
        "Limit by Genre (ML 1)",
        all_genres,
        key="ml1_genres",
    )

    num_recommendations = st.sidebar.slider(
        "Number of movies (ML 1)",
        5,
        20,
        10,
        key="ml1_num_recs",
    )

    selected_movie = st.selectbox(
        "Pick a movie you love:",
        df["title"].dropna().sort_values().unique(),
        key="ml1_selected_movie",
    )

    if st.button("Find Similar", key="ml1_find_similar"):
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
            st.warning("Keine Empfehlungen gefunden. Passe die Filter an.")
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
                    st.markdown("**Kein Poster**")

                st.write(f"**{row['title']}**")
                st.caption(f"Director: {row['director'] if pd.notna(row['director']) else 'n/a'}")
                st.caption(f"Genres: {row['genres'] if pd.notna(row['genres']) else 'n/a'}")
                st.caption(f"Rating: {format_number(row['vote_average'], 1)}")


# =========================================================
# ML Projekt 2 – Movie Team ROI Builder
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
            m.runtime,
            ((m.revenue - m.budget) / NULLIF(m.budget, 0)) * 100.0 AS roi_pct
        FROM core.movies m
        WHERE COALESCE(m.budget, 0) > 0
          AND COALESCE(m.revenue, 0) > 0
          AND COALESCE(m.runtime, 0) > 0
    ),
    person_movies AS (
        SELECT DISTINCT
            'Cast' AS role_group,
            p.person_id,
            p.name,
            mr.movie_id,
            mr.budget,
            mr.revenue,
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
            mr.budget,
            mr.revenue,
            mr.roi_pct
        FROM movie_roi mr
        JOIN core.movie_crew mcr
            ON mr.movie_id = mcr.movie_id
        JOIN core.people p
            ON mcr.person_id = p.person_id
        WHERE mcr.job = 'Director'

        UNION ALL

        SELECT DISTINCT
            'Producer' AS role_group,
            p.person_id,
            p.name,
            mr.movie_id,
            mr.budget,
            mr.revenue,
            mr.roi_pct
        FROM movie_roi mr
        JOIN core.movie_crew mcr
            ON mr.movie_id = mcr.movie_id
        JOIN core.people p
            ON mcr.person_id = p.person_id
        WHERE mcr.job IN ('Producer', 'Executive Producer')

        UNION ALL

        SELECT DISTINCT
            'Writer' AS role_group,
            p.person_id,
            p.name,
            mr.movie_id,
            mr.budget,
            mr.revenue,
            mr.roi_pct
        FROM movie_roi mr
        JOIN core.movie_crew mcr
            ON mr.movie_id = mcr.movie_id
        JOIN core.people p
            ON mcr.person_id = p.person_id
        WHERE mcr.job IN ('Writer', 'Screenplay', 'Story')
    )
    SELECT
        role_group,
        person_id,
        name,
        COUNT(DISTINCT movie_id) AS film_count,
        AVG(budget) AS avg_budget,
        AVG(revenue) AS avg_revenue,
        AVG(roi_pct) AS avg_roi_pct,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY roi_pct) AS median_roi_pct,
        AVG(CASE WHEN roi_pct > 0 THEN 1 ELSE 0 END) * 100.0 AS success_rate_pct
    FROM person_movies
    GROUP BY role_group, person_id, name
    ORDER BY role_group, film_count DESC, name ASC;
    """
    return run_query(query)


@st.cache_data(ttl=600)
def load_team_model_source_data():
    movie_query = """
    SELECT
        m.movie_id,
        m.title,
        m.budget,
        m.revenue,
        m.runtime
    FROM core.movies m
    WHERE COALESCE(m.budget, 0) > 0
      AND COALESCE(m.revenue, 0) > 0
      AND COALESCE(m.runtime, 0) > 0
    ORDER BY m.movie_id;
    """

    cast_query = """
    SELECT DISTINCT
        mc.movie_id,
        mc.person_id
    FROM core.movie_cast mc
    WHERE mc.person_id IS NOT NULL;
    """

    crew_query = """
    SELECT DISTINCT
        mcr.movie_id,
        mcr.person_id,
        CASE
            WHEN mcr.job = 'Director' THEN 'Director'
            WHEN mcr.job IN ('Producer', 'Executive Producer') THEN 'Producer'
            WHEN mcr.job IN ('Writer', 'Screenplay', 'Story') THEN 'Writer'
            ELSE NULL
        END AS role_group
    FROM core.movie_crew mcr
    WHERE mcr.person_id IS NOT NULL
      AND (
          mcr.job = 'Director'
          OR mcr.job IN ('Producer', 'Executive Producer')
          OR mcr.job IN ('Writer', 'Screenplay', 'Story')
      );
    """

    return (
        run_query(movie_query),
        run_query(cast_query),
        run_query(crew_query),
    )


def aggregate_role_features(
    assignments_df: pd.DataFrame,
    role_group: str,
    stats_df: pd.DataFrame,
    prefix: str,
    include_best_roi: bool = False,
) -> pd.DataFrame:
    role_assignments = assignments_df.copy()
    if "role_group" in role_assignments.columns:
        role_assignments = role_assignments[role_assignments["role_group"] == role_group].copy()

    role_stats = stats_df[stats_df["role_group"] == role_group][
        ["person_id", "avg_roi_pct", "success_rate_pct", "film_count"]
    ].copy()

    merged = role_assignments.merge(role_stats, on="person_id", how="left")
    if merged.empty:
        return pd.DataFrame(columns=["movie_id"])

    agg_dict = {
        "avg_roi_pct": "mean",
        "success_rate_pct": "mean",
        "film_count": "mean",
    }
    if include_best_roi:
        agg_dict["avg_roi_pct"] = ["mean", "max"]

    grouped = merged.groupby("movie_id").agg(agg_dict)

    if include_best_roi:
        grouped.columns = [
            f"{prefix}_avg_roi_pct",
            f"{prefix}_best_roi_pct",
            f"{prefix}_success_rate_pct",
            f"{prefix}_film_count",
        ]
    else:
        grouped.columns = [
            f"{prefix}_avg_roi_pct",
            f"{prefix}_success_rate_pct",
            f"{prefix}_film_count",
        ]

    grouped = grouped.reset_index()
    return grouped


@st.cache_resource
def train_movie_team_roi_model():
    person_stats = load_person_role_stats()
    movies_df, cast_df, crew_df = load_team_model_source_data()

    work_df = movies_df.copy()
    work_df["roi_pct"] = ((work_df["revenue"] - work_df["budget"]) / work_df["budget"]) * 100.0
    work_df["roi_pct_clipped"] = work_df["roi_pct"].clip(-100, 500)
    work_df["log_budget"] = np.log1p(work_df["budget"])

    director_features = aggregate_role_features(
        assignments_df=crew_df,
        role_group="Director",
        stats_df=person_stats,
        prefix="director",
        include_best_roi=False,
    )
    producer_features = aggregate_role_features(
        assignments_df=crew_df,
        role_group="Producer",
        stats_df=person_stats,
        prefix="producer",
        include_best_roi=False,
    )
    writer_features = aggregate_role_features(
        assignments_df=crew_df,
        role_group="Writer",
        stats_df=person_stats,
        prefix="writer",
        include_best_roi=False,
    )
    cast_features = aggregate_role_features(
        assignments_df=cast_df,
        role_group="Cast",
        stats_df=person_stats,
        prefix="cast",
        include_best_roi=True,
    )

    feature_df = work_df.merge(director_features, on="movie_id", how="left")
    feature_df = feature_df.merge(producer_features, on="movie_id", how="left")
    feature_df = feature_df.merge(writer_features, on="movie_id", how="left")
    feature_df = feature_df.merge(cast_features, on="movie_id", how="left")

    role_defaults = {}
    for role_group, prefix in [
        ("Director", "director"),
        ("Producer", "producer"),
        ("Writer", "writer"),
        ("Cast", "cast"),
    ]:
        role_slice = person_stats[person_stats["role_group"] == role_group]
        role_defaults[f"{prefix}_avg_roi_pct"] = float(role_slice["avg_roi_pct"].mean()) if not role_slice.empty else 0.0
        role_defaults[f"{prefix}_success_rate_pct"] = float(role_slice["success_rate_pct"].mean()) if not role_slice.empty else 0.0
        role_defaults[f"{prefix}_film_count"] = float(role_slice["film_count"].mean()) if not role_slice.empty else 0.0
        if prefix == "cast":
            role_defaults["cast_best_roi_pct"] = float(role_slice["avg_roi_pct"].max()) if not role_slice.empty else 0.0

    feature_columns = [
        "budget",
        "log_budget",
        "runtime",
        "director_avg_roi_pct",
        "director_success_rate_pct",
        "director_film_count",
        "producer_avg_roi_pct",
        "producer_success_rate_pct",
        "producer_film_count",
        "writer_avg_roi_pct",
        "writer_success_rate_pct",
        "writer_film_count",
        "cast_avg_roi_pct",
        "cast_best_roi_pct",
        "cast_success_rate_pct",
        "cast_film_count",
    ]

    for col in feature_columns:
        if col not in feature_df.columns:
            feature_df[col] = np.nan

    fill_values = {
        "budget": float(feature_df["budget"].median()),
        "log_budget": float(feature_df["log_budget"].median()),
        "runtime": float(feature_df["runtime"].median()),
        **role_defaults,
    }

    X = feature_df[feature_columns].copy()
    y = feature_df["roi_pct_clipped"].copy()

    X = X.fillna(fill_values)
    valid_mask = X.notna().all(axis=1) & y.notna()
    X = X.loc[valid_mask].copy()
    y = y.loc[valid_mask].copy()
    eval_df = feature_df.loc[valid_mask, ["movie_id", "title", "roi_pct_clipped"]].copy()

    X_train, X_test, y_train, y_test, eval_train, eval_test = train_test_split(
        X,
        y,
        eval_df,
        test_size=0.2,
        random_state=42,
    )

    model = RandomForestRegressor(
        n_estimators=350,
        max_depth=10,
        min_samples_split=6,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    pred_test = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred_test))
    r2 = r2_score(y_test, pred_test)

    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    evaluation_results = eval_test.copy()
    evaluation_results["actual_roi_pct"] = y_test.values
    evaluation_results["predicted_roi_pct"] = pred_test

    return {
        "model": model,
        "feature_columns": feature_columns,
        "fill_values": fill_values,
        "person_stats": person_stats,
        "metrics": {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        },
        "importance_df": importance_df,
        "evaluation_df": evaluation_results,
        "runtime_min": int(max(60, feature_df["runtime"].min())),
        "runtime_max": int(min(300, max(180, feature_df["runtime"].max()))),
        "budget_median": float(feature_df["budget"].median()),
        "budget_max": float(max(250_000_000, feature_df["budget"].quantile(0.95))),
        "film_count": int(len(feature_df)),
    }


def get_filtered_role_options(
    person_stats: pd.DataFrame,
    role_group: str,
    min_films: int,
) -> Dict[str, int]:
    role_df = person_stats[
        (person_stats["role_group"] == role_group)
        & (person_stats["film_count"] >= min_films)
    ].copy()

    if role_df.empty:
        return {}

    role_df = role_df.sort_values(["film_count", "name"], ascending=[False, True])

    label_to_id = {}
    for _, row in role_df.iterrows():
        label = (
            f"{row['name']} "
            f"(Filme: {int(row['film_count'])} | "
            f"Ø ROI: {format_number(row['avg_roi_pct'], 1)}%)"
        )
        label_to_id[label] = int(row["person_id"])

    return label_to_id


def get_selected_person_stats(
    person_stats: pd.DataFrame,
    role_group: str,
    person_ids: List[int],
) -> pd.DataFrame:
    if not person_ids:
        return pd.DataFrame()

    return person_stats[
        (person_stats["role_group"] == role_group)
        & (person_stats["person_id"].isin(person_ids))
    ].copy()


def build_role_feature_values(
    person_stats: pd.DataFrame,
    role_group: str,
    person_ids: List[int],
    prefix: str,
    fill_values: Dict[str, float],
) -> Dict[str, float]:
    role_df = get_selected_person_stats(person_stats, role_group, person_ids)

    if role_df.empty:
        values = {
            f"{prefix}_avg_roi_pct": float(fill_values.get(f"{prefix}_avg_roi_pct", 0.0)),
            f"{prefix}_success_rate_pct": float(fill_values.get(f"{prefix}_success_rate_pct", 0.0)),
            f"{prefix}_film_count": float(fill_values.get(f"{prefix}_film_count", 0.0)),
        }
        if prefix == "cast":
            values["cast_best_roi_pct"] = float(fill_values.get("cast_best_roi_pct", 0.0))
        return values

    values = {
        f"{prefix}_avg_roi_pct": float(role_df["avg_roi_pct"].mean()),
        f"{prefix}_success_rate_pct": float(role_df["success_rate_pct"].mean()),
        f"{prefix}_film_count": float(role_df["film_count"].mean()),
    }
    if prefix == "cast":
        values["cast_best_roi_pct"] = float(role_df["avg_roi_pct"].max())

    return values


def build_team_feature_row(
    budget: float,
    runtime: int,
    person_stats: pd.DataFrame,
    fill_values: Dict[str, float],
    director_ids: List[int],
    producer_ids: List[int],
    writer_ids: List[int],
    cast_ids: List[int],
    feature_columns: List[str],
) -> pd.DataFrame:
    row = {col: 0.0 for col in feature_columns}
    row["budget"] = float(budget)
    row["log_budget"] = float(np.log1p(budget))
    row["runtime"] = float(runtime)

    row.update(
        build_role_feature_values(
            person_stats=person_stats,
            role_group="Director",
            person_ids=director_ids,
            prefix="director",
            fill_values=fill_values,
        )
    )
    row.update(
        build_role_feature_values(
            person_stats=person_stats,
            role_group="Producer",
            person_ids=producer_ids,
            prefix="producer",
            fill_values=fill_values,
        )
    )
    row.update(
        build_role_feature_values(
            person_stats=person_stats,
            role_group="Writer",
            person_ids=writer_ids,
            prefix="writer",
            fill_values=fill_values,
        )
    )
    row.update(
        build_role_feature_values(
            person_stats=person_stats,
            role_group="Cast",
            person_ids=cast_ids,
            prefix="cast",
            fill_values=fill_values,
        )
    )

    df = pd.DataFrame([row], columns=feature_columns)
    return df.fillna(fill_values)


def build_selected_people_table(
    person_stats: pd.DataFrame,
    director_ids: List[int],
    producer_ids: List[int],
    writer_ids: List[int],
    cast_ids: List[int],
) -> pd.DataFrame:
    pieces = []

    for role_group, ids in [
        ("Director", director_ids),
        ("Producer", producer_ids),
        ("Writer", writer_ids),
        ("Cast", cast_ids),
    ]:
        role_df = get_selected_person_stats(person_stats, role_group, ids)
        if role_df.empty:
            continue
        role_df = role_df.copy()
        role_df["role_group"] = role_group
        pieces.append(role_df)

    if not pieces:
        return pd.DataFrame()

    display_df = pd.concat(pieces, ignore_index=True)
    display_df = display_df.rename(
        columns={
            "role_group": "Rolle",
            "name": "Name",
            "film_count": "Filme",
            "avg_roi_pct": "Ø ROI %",
            "median_roi_pct": "Median ROI %",
            "success_rate_pct": "Erfolgsquote %",
            "avg_budget": "Ø Budget",
            "avg_revenue": "Ø Revenue",
        }
    )
    display_df["Filme"] = display_df["Filme"].astype(int)
    display_df["Ø ROI %"] = display_df["Ø ROI %"].apply(lambda x: format_number(x, 1))
    display_df["Median ROI %"] = display_df["Median ROI %"].apply(lambda x: format_number(x, 1))
    display_df["Erfolgsquote %"] = display_df["Erfolgsquote %"].apply(lambda x: format_number(x, 1))
    display_df["Ø Budget"] = display_df["Ø Budget"].apply(format_money)
    display_df["Ø Revenue"] = display_df["Ø Revenue"].apply(format_money)

    return display_df[
        ["Rolle", "Name", "Filme", "Ø ROI %", "Median ROI %", "Erfolgsquote %", "Ø Budget", "Ø Revenue"]
    ].sort_values(["Rolle", "Name"])


def show_ml_project_2_page() -> None:
    st.title("🤖 ML Projekt 2 – Movie Team ROI Builder")
    st.write(
        "Dieses Modul simuliert die wirtschaftliche Performance eines neuen Filmprojekts. "
        "Du stellst Budget, Director, Producer, Writer und Cast zusammen und erhältst eine geschätzte ROI-Erwartung."
    )

    with st.spinner("Lade Personenscores und trainiere ROI-Modell..."):
        artifacts = train_movie_team_roi_model()

    person_stats = artifacts["person_stats"]
    metrics = artifacts["metrics"]
    importance_df = artifacts["importance_df"]
    evaluation_df = artifacts["evaluation_df"]

    st.markdown("### Modellidee")
    st.write(
        """
        - Zielgröße: **ROI %** statt reiner Revenue  
        - Quelle: historische Performance aus **core.movies**, **movie_cast**, **movie_crew** und **people**  
        - Rollen in V1: **Director, Producer, Writer, Cast**  
        - Das Modell nutzt **Budget, Runtime** und aggregierte historische ROI-Scores der gewählten Personen
        """
    )

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("MAE", f"{format_number(metrics['mae'], 1)} %-Punkte")
    metric_col2.metric("RMSE", f"{format_number(metrics['rmse'], 1)} %-Punkte")
    metric_col3.metric("R²", format_number(metrics["r2"], 3))
    metric_col4.metric("Trainingsfilme", format_int(artifacts["film_count"]))

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("#### Actual vs Predicted ROI")
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ax1.scatter(
            evaluation_df["actual_roi_pct"],
            evaluation_df["predicted_roi_pct"],
            alpha=0.55,
        )
        min_val = min(evaluation_df["actual_roi_pct"].min(), evaluation_df["predicted_roi_pct"].min())
        max_val = max(evaluation_df["actual_roi_pct"].max(), evaluation_df["predicted_roi_pct"].max())
        ax1.plot([min_val, max_val], [min_val, max_val])
        ax1.set_xlabel("Actual ROI %")
        ax1.set_ylabel("Predicted ROI %")
        ax1.set_title("Actual vs Predicted")
        st.pyplot(fig1)

    with chart_col2:
        st.markdown("#### Wichtigste Features")
        top_features = importance_df.head(12).sort_values("importance", ascending=True)
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.barh(top_features["feature"], top_features["importance"])
        ax2.set_xlabel("Importance")
        ax2.set_ylabel("Feature")
        ax2.set_title("Top Feature Importances")
        st.pyplot(fig2)

    st.markdown("---")
    st.markdown("### Team konfigurieren")

    setup_col1, setup_col2, setup_col3 = st.columns(3)

    with setup_col1:
        budget_input = st.number_input(
            "Budget (USD)",
            min_value=1_000_000.0,
            max_value=max(500_000_000.0, artifacts["budget_max"]),
            value=max(20_000_000.0, artifacts["budget_median"]),
            step=5_000_000.0,
        )

    with setup_col2:
        runtime_input = st.slider(
            "Runtime (Minuten)",
            min_value=artifacts["runtime_min"],
            max_value=artifacts["runtime_max"],
            value=min(max(120, artifacts["runtime_min"]), artifacts["runtime_max"]),
        )

    with setup_col3:
        min_films_input = st.selectbox(
            "Mindestanzahl Filme pro Person",
            options=[1, 3, 5, 10],
            index=1,
        )

    director_options = get_filtered_role_options(person_stats, "Director", min_films_input)
    producer_options = get_filtered_role_options(person_stats, "Producer", min_films_input)
    writer_options = get_filtered_role_options(person_stats, "Writer", min_films_input)
    cast_options = get_filtered_role_options(person_stats, "Cast", min_films_input)

    if not director_options or not cast_options:
        st.warning("Für die gewählte Mindestanzahl Filme sind nicht genug Personen verfügbar.")
        return

    crew_col1, crew_col2 = st.columns(2)

    with crew_col1:
        st.markdown("#### Crew")
        director_label = st.selectbox(
            "Director",
            options=["Keine Auswahl"] + list(director_options.keys()),
        )
        producer_labels = st.multiselect(
            "Producer (0–3 empfohlen)",
            options=list(producer_options.keys()),
            default=[],
        )
        writer_labels = st.multiselect(
            "Writer (0–2 empfohlen)",
            options=list(writer_options.keys()),
            default=[],
        )

    with crew_col2:
        st.markdown("#### Cast")
        cast_labels = st.multiselect(
            "Cast (1–5 empfohlen)",
            options=list(cast_options.keys()),
            default=[],
        )

    director_ids = [director_options[director_label]] if director_label != "Keine Auswahl" else []
    producer_ids = [producer_options[label] for label in producer_labels]
    writer_ids = [writer_options[label] for label in writer_labels]
    cast_ids = [cast_options[label] for label in cast_labels]

    current_row = build_team_feature_row(
        budget=budget_input,
        runtime=runtime_input,
        person_stats=person_stats,
        fill_values=artifacts["fill_values"],
        director_ids=director_ids,
        producer_ids=producer_ids,
        writer_ids=writer_ids,
        cast_ids=cast_ids,
        feature_columns=artifacts["feature_columns"],
    )

    baseline_row = build_team_feature_row(
        budget=budget_input,
        runtime=runtime_input,
        person_stats=person_stats,
        fill_values=artifacts["fill_values"],
        director_ids=[],
        producer_ids=[],
        writer_ids=[],
        cast_ids=[],
        feature_columns=artifacts["feature_columns"],
    )

    model = artifacts["model"]
    predicted_roi = float(model.predict(current_row)[0])
    baseline_roi = float(model.predict(baseline_row)[0])
    roi_delta = predicted_roi - baseline_roi

    predicted_revenue = max(0.0, budget_input * (1 + predicted_roi / 100.0))
    break_even_revenue = budget_input

    st.markdown("---")
    st.markdown("### Erwartete Performance")

    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
    perf_col1.metric("Erwarteter ROI", f"{format_number(predicted_roi, 1)}%")
    perf_col2.metric("Veränderung vs. Baseline", f"{format_number(roi_delta, 1)}%")
    perf_col3.metric("Erwarteter Revenue", format_money(predicted_revenue))
    perf_col4.metric("Break-even Revenue", format_money(break_even_revenue))

    if predicted_roi < 0:
        st.error("Einordnung: negatives ROI-Szenario.")
    elif predicted_roi < 50:
        st.warning("Einordnung: moderates ROI-Szenario.")
    else:
        st.success("Einordnung: starkes ROI-Szenario.")

    st.caption(
        "Baseline = Modellprognose mit gleichem Budget und gleicher Runtime, aber ohne explizit ausgewählte Personen. "
        "So startet der Unterschiedseffekt bei 0 und verändert sich mit jeder Team-Auswahl."
    )

    st.markdown("---")
    st.markdown("### Ausgewählte Personen und historische Kennzahlen")

    selected_people_df = build_selected_people_table(
        person_stats=person_stats,
        director_ids=director_ids,
        producer_ids=producer_ids,
        writer_ids=writer_ids,
        cast_ids=cast_ids,
    )

    if selected_people_df.empty:
        st.info("Noch keine Personen ausgewählt.")
    else:
        st.dataframe(
            selected_people_df,
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")
    st.markdown("### Top ROI-Talente nach Rolle")
    role_tabs = st.tabs(["Director", "Producer", "Writer", "Cast"])

    for tab, role_group in zip(role_tabs, ["Director", "Producer", "Writer", "Cast"]):
        with tab:
            role_df = person_stats[
                (person_stats["role_group"] == role_group)
                & (person_stats["film_count"] >= min_films_input)
            ].copy()

            if role_df.empty:
                st.write("Keine Daten verfügbar.")
            else:
                role_df = role_df.sort_values(
                    ["avg_roi_pct", "film_count"],
                    ascending=[False, False],
                ).head(15)

                display_df = role_df.rename(
                    columns={
                        "name": "Name",
                        "film_count": "Filme",
                        "avg_roi_pct": "Ø ROI %",
                        "median_roi_pct": "Median ROI %",
                        "success_rate_pct": "Erfolgsquote %",
                        "avg_budget": "Ø Budget",
                        "avg_revenue": "Ø Revenue",
                    }
                ).copy()

                display_df["Filme"] = display_df["Filme"].astype(int)
                display_df["Ø ROI %"] = display_df["Ø ROI %"].apply(lambda x: format_number(x, 1))
                display_df["Median ROI %"] = display_df["Median ROI %"].apply(lambda x: format_number(x, 1))
                display_df["Erfolgsquote %"] = display_df["Erfolgsquote %"].apply(lambda x: format_number(x, 1))
                display_df["Ø Budget"] = display_df["Ø Budget"].apply(format_money)
                display_df["Ø Revenue"] = display_df["Ø Revenue"].apply(format_money)

                st.dataframe(
                    display_df[
                        ["Name", "Filme", "Ø ROI %", "Median ROI %", "Erfolgsquote %", "Ø Budget", "Ø Revenue"]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )


# =========================================================
# Page 1 – Film Database
# =========================================================
def show_film_database_page() -> None:
    st.title("🎬 The Movie Database")
    st.write(
        "Durchsuche die Filmdatenbank über Filter oder Suchbegriffe. Unterhalb der Trefferliste kannst du optional einen einzelnen Film für die Detailansicht auswählen."
    )

    st.sidebar.markdown("### Filter für Film datenbank")
    search_term = st.sidebar.text_input("Filmtitel", placeholder="leer lassen = nur über Filter browsen")
    min_vote = st.sidebar.slider("Mindestbewertung", 0.0, 10.0, 0.0, 0.1)
    year_from = st.sidebar.number_input("Release Year von", min_value=1900, max_value=2100, value=1900, step=1)
    year_to = st.sidebar.number_input("Release Year bis", min_value=1900, max_value=2100, value=2100, step=1)
    min_votes = st.sidebar.number_input("Mindestens Votes", min_value=0, max_value=1000000, value=0, step=100)
    only_with_reviews = st.sidebar.checkbox("Nur Filme mit Reviews", value=False)
    only_with_poster = st.sidebar.checkbox("Nur Filme mit Poster", value=False)
    results_limit = st.sidebar.selectbox("Maximale Trefferzahl", [25, 50, 100, 200], index=2)
    sort_by = st.sidebar.selectbox(
        "Sortieren nach",
        ["Popularity", "Vote Average", "Vote Count", "Revenue", "Budget", "Runtime", "Year", "Title"],
    )
    sort_order = st.sidebar.selectbox("Reihenfolge", ["Absteigend", "Aufsteigend"])

    genre_filter_query = """
        SELECT genre_name
        FROM core.genres
        ORDER BY genre_name;
    """
    all_genres_df = run_query(genre_filter_query)
    genre_options = all_genres_df["genre_name"].dropna().tolist() if not all_genres_df.empty else []
    selected_genre = st.sidebar.selectbox("Genre", ["Alle"] + genre_options)

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
    sort_direction = "DESC" if sort_order == "Absteigend" else "ASC"
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
              %s = 'Alle'
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

    st.caption(f"Gefundene Filme: {len(movie_results)}")

    if movie_results.empty:
        st.warning("Keine Treffer gefunden. Passe die Filter an oder erweitere die Suche.")
        return

    st.markdown("### Schnellauswahl")
    st.caption("Kompakte Vorschau der aktuellen Treffer. Die Detailansicht wählst du darunter im Dropdown aus.")

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
                st.markdown("**Kein Poster**")

            st.markdown(f"**{row['title']}**")
            st.caption(
                f"{int(row['release_year']) if pd.notna(row['release_year']) else 'n/a'} | "
                f"Rating: {format_number(row['vote_average'], 1)}"
            )
            if st.button("Detailansicht", key=f"detail_btn_{row['movie_id']}", use_container_width=True):
                st.session_state["selected_movie_id"] = int(row["movie_id"])
                st.rerun()

    nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
    with nav_col1:
        if st.button("← Vorherige", disabled=st.session_state["quick_pick_page"] <= 1, use_container_width=True):
            st.session_state["quick_pick_page"] -= 1
            st.rerun()
    with nav_col2:
        st.markdown(
            f"<div style='text-align:center; padding-top:8px;'><b>Seite {st.session_state['quick_pick_page']} von {total_pages}</b></div>",
            unsafe_allow_html=True,
        )
    with nav_col3:
        if st.button("Nächste →", disabled=st.session_state["quick_pick_page"] >= total_pages, use_container_width=True):
            st.session_state["quick_pick_page"] += 1
            st.rerun()

    st.markdown("---")
    st.markdown("### Detailansicht")

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
        "Oder Film direkt aus der Trefferliste auswählen",
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
        st.error("Die Filmdetails konnten nicht geladen werden.")
        return

    movie = movie_df.iloc[0]

    poster_col, header_col = st.columns([1, 3])
    with poster_col:
        img_url = poster_url(movie.get("poster_path"))
        if img_url:
            st.image(img_url, use_container_width=True)
        else:
            st.info("Kein Poster vorhanden")

    with header_col:
        st.subheader(movie["title"])
        subtitle_parts = []
        if pd.notna(movie.get("original_title")) and str(movie.get("original_title")) != str(movie.get("title")):
            subtitle_parts.append(f"Originaltitel: {movie['original_title']}")
        if pd.notna(movie.get("tagline")) and str(movie.get("tagline")).strip():
            subtitle_parts.append(f"Tagline: {movie['tagline']}")
        if subtitle_parts:
            for part in subtitle_parts:
                st.write(part)

        genre_tags = genres_df["genre_name"].dropna().tolist() if not genres_df.empty else []
        if genre_tags:
            st.markdown("**Genres:** " + " | ".join(genre_tags))

        if pd.notna(movie.get("homepage")) and str(movie.get("homepage")).strip():
            st.markdown(f"[Zur Homepage]({movie['homepage']})")

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
        st.markdown("### Stammdaten")
        info_df = pd.DataFrame(
            {
                "Feld": [
                    "Movie ID",
                    "Original Language",
                    "Status",
                    "Adult",
                    "Video",
                    "Created At",
                    "Updated At",
                ],
                "Wert": [
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
            st.write("Keine Beschreibung vorhanden.")

    st.markdown("---")

    cast_col, crew_col = st.columns(2)
    with cast_col:
        st.markdown("### Cast")
        if cast_df.empty:
            st.write("Keine Cast-Daten vorhanden.")
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
            st.write("Keine Crew-Daten vorhanden.")
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
    st.caption("Es werden die 10 relevantesten bzw. längsten verfügbaren Reviews angezeigt.")

    if reviews_df.empty:
        st.write("Keine Reviews vorhanden.")
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
                    st.markdown(f"[Review-Link öffnen]({row['url']})")

    st.markdown("---")
    st.markdown("### Trefferliste")
    st.caption("Die Trefferliste reagiert vollständig auf die gesetzten Filter. Du kannst die gesamte Filmdatenbank eingrenzen, exportieren und anschließend einzelne Filme im Detail öffnen.")

    display_results = movie_results.copy()
    if "release_year" in display_results.columns:
        display_results["release_year"] = display_results["release_year"].astype("Int64")

    display_results["Poster"] = display_results["poster_path"].apply(
        lambda x: "Ja" if pd.notna(x) and str(x).strip() else "Nein"
    )
    display_results["Reviews"] = display_results["has_reviews"].apply(lambda x: "Ja" if bool(x) else "Nein")
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
        label="Trefferliste als CSV herunterladen",
        data=csv_export,
        file_name="filtered_movies.csv",
        mime="text/csv",
    )


# =========================================================
# Placeholder page
# =========================================================
def show_placeholder_page(title: str, text: str) -> None:
    st.title(title)
    st.info(text)


# =========================================================
# Navigation
# =========================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Seite auswählen",
    [
        "1. Film datenbank",
        "2. Analysen mit Power BI",
        "3. ML Projekt 1",
        "4. ML Projekt 2",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("TMDB Abschlussprojekt")


# =========================================================
# Router
# =========================================================
if page == "1. Film datenbank":
    show_film_database_page()
elif page == "2. Analysen mit Power BI":
    show_placeholder_page(
        "📊 Analysen mit Power BI",
        "Diese Seite wird später mit den Power-BI-Ergebnissen oder Screenshots ergänzt.",
    )
elif page == "3. ML Projekt 1":
    show_ml_project_1_page()
else:
    show_ml_project_2_page()