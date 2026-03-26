from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from utils.db import run_query
from utils.helpers import format_money, format_number, safe_mean, safe_max


def clip_roi(value: float) -> float:
    if pd.isna(value):
        return 0.0
    return float(np.clip(float(value), -100.0, 400.0))


def roi_category_label(value: float, has_selection: bool) -> str:
    if not has_selection:
        return "No setup selected"
    if value < 0:
        return "High Risk"
    if value < 50:
        return "Moderate"
    if value < 150:
        return "Strong"
    return "Very Strong"


def entity_component(avg_roi: float, success_rate: float) -> float:
    avg_roi = clip_roi(avg_roi)
    success_signal = (0.0 if pd.isna(success_rate) else float(success_rate) - 50.0) * 2.0
    return float((0.75 * avg_roi) + (0.25 * success_signal))


def budget_fit_component(selected_budget: float, typical_budget: float) -> float:
    if selected_budget <= 0 or typical_budget <= 0:
        return 0.0
    ratio = selected_budget / typical_budget
    log_distance = abs(np.log(ratio))
    score = 35.0 - (45.0 * log_distance)
    return float(np.clip(score, -80.0, 35.0))


@st.cache_data(ttl=600)
def load_person_role_stats() -> pd.DataFrame:
    query = """
    WITH movie_roi AS (
        SELECT
            m.movie_id,
            m.title,
            m.revenue,
            m.budget,
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
            mr.title,
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
            mr.title,
            mr.revenue,
            mr.roi_pct
        FROM movie_roi mr
        JOIN core.movie_crew mcr
            ON mr.movie_id = mcr.movie_id
        JOIN core.people p
            ON mcr.person_id = p.person_id
        WHERE mcr.job = 'Director'
    ),
    top_genre_per_person AS (
        SELECT
            x.role_group,
            x.person_id,
            x.genre_name
        FROM (
            SELECT
                pm.role_group,
                pm.person_id,
                g.genre_name,
                COUNT(*) AS genre_count,
                ROW_NUMBER() OVER (
                    PARTITION BY pm.role_group, pm.person_id
                    ORDER BY COUNT(*) DESC, g.genre_name ASC
                ) AS rn
            FROM person_movies pm
            JOIN core.movie_genres mg
                ON pm.movie_id = mg.movie_id
            JOIN core.genres g
                ON mg.genre_id = g.genre_id
            GROUP BY pm.role_group, pm.person_id, g.genre_name
        ) x
        WHERE x.rn = 1
    ),
    best_film_per_person AS (
        SELECT
            x.role_group,
            x.person_id,
            x.title AS best_film_title
        FROM (
            SELECT
                pm.role_group,
                pm.person_id,
                pm.title,
                pm.roi_pct,
                ROW_NUMBER() OVER (
                    PARTITION BY pm.role_group, pm.person_id
                    ORDER BY pm.roi_pct DESC NULLS LAST, pm.title ASC
                ) AS rn
            FROM person_movies pm
        ) x
        WHERE x.rn = 1
    )
    SELECT
        pm.role_group,
        pm.person_id,
        pm.name,
        COUNT(DISTINCT pm.movie_id) AS film_count,
        AVG(pm.roi_pct) AS avg_roi_pct,
        AVG(CASE WHEN pm.roi_pct > 0 THEN 1 ELSE 0 END) * 100.0 AS success_rate_pct,
        SUM(pm.revenue) AS total_revenue,
        tg.genre_name AS top_genre,
        bf.best_film_title
    FROM person_movies pm
    LEFT JOIN top_genre_per_person tg
        ON pm.role_group = tg.role_group
       AND pm.person_id = tg.person_id
    LEFT JOIN best_film_per_person bf
        ON pm.role_group = bf.role_group
       AND pm.person_id = bf.person_id
    GROUP BY
        pm.role_group,
        pm.person_id,
        pm.name,
        tg.genre_name,
        bf.best_film_title
    ORDER BY pm.role_group, film_count DESC, pm.name ASC;
    """
    df = run_query(query)
    if not df.empty:
        df["component_score"] = df.apply(
            lambda row: entity_component(row["avg_roi_pct"], row["success_rate_pct"]),
            axis=1,
        )
    return df


@st.cache_data(ttl=600)
def load_genre_stats() -> pd.DataFrame:
    query = """
    WITH movie_roi AS (
        SELECT
            m.movie_id,
            m.title,
            m.revenue,
            m.budget,
            ((m.revenue - m.budget) / NULLIF(m.budget, 0)) * 100.0 AS roi_pct
        FROM core.movies m
        WHERE COALESCE(m.budget, 0) >= 100000
          AND COALESCE(m.revenue, 0) > 0
    ),
    genre_movies AS (
        SELECT
            g.genre_id,
            g.genre_name,
            mr.movie_id,
            mr.title,
            mr.revenue,
            mr.roi_pct,
            m.budget
        FROM movie_roi mr
        JOIN core.movies m
            ON mr.movie_id = m.movie_id
        JOIN core.movie_genres mg
            ON mr.movie_id = mg.movie_id
        JOIN core.genres g
            ON mg.genre_id = g.genre_id
    ),
    best_film_per_genre AS (
        SELECT
            x.genre_id,
            x.title AS best_film_title
        FROM (
            SELECT
                gm.genre_id,
                gm.title,
                gm.roi_pct,
                ROW_NUMBER() OVER (
                    PARTITION BY gm.genre_id
                    ORDER BY gm.roi_pct DESC NULLS LAST, gm.title ASC
                ) AS rn
            FROM genre_movies gm
        ) x
        WHERE x.rn = 1
    )
    SELECT
        gm.genre_id,
        gm.genre_name,
        COUNT(DISTINCT gm.movie_id) AS film_count,
        AVG(gm.roi_pct) AS avg_roi_pct,
        AVG(CASE WHEN gm.roi_pct > 0 THEN 1 ELSE 0 END) * 100.0 AS success_rate_pct,
        SUM(gm.revenue) AS total_revenue,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY gm.budget) AS median_budget,
        bf.best_film_title
    FROM genre_movies gm
    LEFT JOIN best_film_per_genre bf
        ON gm.genre_id = bf.genre_id
    GROUP BY
        gm.genre_id,
        gm.genre_name,
        bf.best_film_title
    ORDER BY film_count DESC, gm.genre_name ASC;
    """
    df = run_query(query)
    if not df.empty:
        df["component_score"] = df.apply(
            lambda row: entity_component(row["avg_roi_pct"], row["success_rate_pct"]),
            axis=1,
        )
    return df


@st.cache_data(ttl=600)
def load_global_budget_stats() -> pd.DataFrame:
    query = """
    SELECT
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY m.budget) AS global_median_budget
    FROM core.movies m
    WHERE COALESCE(m.budget, 0) >= 100000
      AND COALESCE(m.revenue, 0) > 0;
    """
    return run_query(query)


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
    return run_query(query)


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
    return run_query(query)


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


def build_genre_lookup_map(genre_stats: pd.DataFrame):
    return genre_stats.set_index("genre_id").to_dict("index")


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


def ensure_roi_builder_session_state() -> None:
    if "roi_builder_selected_director_id" not in st.session_state:
        st.session_state["roi_builder_selected_director_id"] = None
    if "roi_builder_selected_cast_ids" not in st.session_state:
        st.session_state["roi_builder_selected_cast_ids"] = []
    if "roi_builder_selected_genre_id" not in st.session_state:
        st.session_state["roi_builder_selected_genre_id"] = None
    if "roi_builder_budget_input" not in st.session_state:
        st.session_state["roi_builder_budget_input"] = 50000000


def get_filtered_role_options(person_stats: pd.DataFrame, role_group: str, min_films: int) -> dict:
    role_df = person_stats[
        (person_stats["role_group"] == role_group)
        & (person_stats["film_count"] >= min_films)
    ].copy()

    if role_df.empty:
        return {}

    role_df = role_df.sort_values(
        ["film_count", "name"],
        ascending=[False, True],
    )

    label_to_id = {}
    for _, row in role_df.iterrows():
        label = (
            f"{row['name']} "
            f"(Films: {int(row['film_count'])} | "
            f"Success: {format_number(row['success_rate_pct'], 1)}%)"
        )
        label_to_id[label] = int(row["person_id"])

    return label_to_id


def get_genre_options(genre_stats: pd.DataFrame, min_films: int) -> dict:
    genre_df = genre_stats[genre_stats["film_count"] >= min_films].copy()

    if genre_df.empty:
        return {}

    genre_df = genre_df.sort_values(
        ["film_count", "genre_name"],
        ascending=[False, True],
    )

    label_to_id = {}
    for _, row in genre_df.iterrows():
        label = (
            f"{row['genre_name']} "
            f"(Films: {int(row['film_count'])} | "
            f"Success: {format_number(row['success_rate_pct'], 1)}%)"
        )
        label_to_id[label] = int(row["genre_id"])

    return label_to_id


def sync_roi_builder_state_with_filters(director_options: dict, cast_options: dict, genre_options: dict) -> None:
    available_director_ids = set(director_options.values())
    available_cast_ids = set(cast_options.values())
    available_genre_ids = set(genre_options.values())

    if st.session_state["roi_builder_selected_director_id"] not in available_director_ids:
        st.session_state["roi_builder_selected_director_id"] = None

    if st.session_state["roi_builder_selected_genre_id"] not in available_genre_ids:
        st.session_state["roi_builder_selected_genre_id"] = None

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


def get_genre_row(genre_stats: pd.DataFrame, genre_id: int) -> Optional[pd.Series]:
    df = genre_stats[genre_stats["genre_id"] == genre_id]
    if df.empty:
        return None
    return df.iloc[0]


def build_current_team_prediction(
    person_stats: pd.DataFrame,
    genre_stats: pd.DataFrame,
    cast_pair_df: pd.DataFrame,
    director_cast_df: pd.DataFrame,
    global_budget_df: pd.DataFrame,
) -> Tuple[float, bool]:
    director_id = st.session_state["roi_builder_selected_director_id"]
    cast_ids = st.session_state["roi_builder_selected_cast_ids"]
    genre_id = st.session_state["roi_builder_selected_genre_id"]
    selected_budget = float(st.session_state["roi_builder_budget_input"])

    has_selection = bool(director_id is not None or genre_id is not None or len(cast_ids) > 0)
    if not has_selection:
        return 0.0, False

    director_stats_map, cast_stats_map = build_person_lookup_maps(person_stats)
    genre_stats_map = build_genre_lookup_map(genre_stats)
    cast_pair_map, director_cast_map = build_synergy_lookup_maps(cast_pair_df, director_cast_df)

    active_scores = []
    active_weights = []

    if director_id is not None and director_id in director_stats_map:
        d = director_stats_map[director_id]
        active_scores.append(float(d["component_score"]))
        active_weights.append(0.30)

    cast_components = []
    for cast_id in cast_ids:
        if cast_id in cast_stats_map:
            cast_components.append(float(cast_stats_map[cast_id]["component_score"]))

    if cast_components:
        active_scores.append(float(safe_mean(cast_components, 0.0)))
        active_weights.append(0.25)

    typical_budget = None
    if genre_id is not None and genre_id in genre_stats_map:
        g = genre_stats_map[genre_id]
        active_scores.append(float(g["component_score"]))
        active_weights.append(0.20)
        typical_budget = float(g["median_budget"]) if pd.notna(g["median_budget"]) else None

    cast_pair_rois = []
    for a, b in combinations(sorted(cast_ids), 2):
        key = (min(a, b), max(a, b))
        if key in cast_pair_map:
            cast_pair_rois.append(clip_roi(cast_pair_map[key]["pair_avg_roi_pct"]))

    if cast_pair_rois:
        active_scores.append(float(safe_mean(cast_pair_rois, 0.0)))
        active_weights.append(0.10)

    director_cast_rois = []
    if director_id is not None:
        for cast_id in cast_ids:
            key = (director_id, cast_id)
            if key in director_cast_map:
                director_cast_rois.append(clip_roi(director_cast_map[key]["pair_avg_roi_pct"]))

    if director_cast_rois:
        active_scores.append(float(safe_mean(director_cast_rois, 0.0)))
        active_weights.append(0.10)

    if typical_budget is None:
        typical_budget = float(global_budget_df.iloc[0]["global_median_budget"]) if not global_budget_df.empty else 50000000.0

    budget_score = budget_fit_component(selected_budget, typical_budget)
    active_scores.append(float(budget_score))
    active_weights.append(0.15)

    predicted_roi = float(np.average(active_scores, weights=active_weights))
    predicted_roi = float(np.clip(predicted_roi, -100.0, 400.0))

    return predicted_roi, True


def build_selected_team_table(person_stats: pd.DataFrame, genre_stats: pd.DataFrame) -> pd.DataFrame:
    rows = []

    director_id = st.session_state["roi_builder_selected_director_id"]
    if director_id is not None:
        row = get_person_row(person_stats, "Director", director_id)
        if row is not None:
            rows.append(
                {
                    "Type": "Director",
                    "Name": row["name"],
                    "Films": int(row["film_count"]),
                    "Success Rate %": format_number(row["success_rate_pct"], 1),
                    "Avg ROI %": format_number(row["avg_roi_pct"], 1),
                    "Top Genre": row["top_genre"] if pd.notna(row["top_genre"]) else "n/a",
                    "Most Successful Film": row["best_film_title"] if pd.notna(row["best_film_title"]) else "n/a",
                }
            )

    for cast_id in st.session_state["roi_builder_selected_cast_ids"]:
        row = get_person_row(person_stats, "Cast", cast_id)
        if row is not None:
            rows.append(
                {
                    "Type": "Cast",
                    "Name": row["name"],
                    "Films": int(row["film_count"]),
                    "Success Rate %": format_number(row["success_rate_pct"], 1),
                    "Avg ROI %": format_number(row["avg_roi_pct"], 1),
                    "Top Genre": row["top_genre"] if pd.notna(row["top_genre"]) else "n/a",
                    "Most Successful Film": row["best_film_title"] if pd.notna(row["best_film_title"]) else "n/a",
                }
            )

    genre_id = st.session_state["roi_builder_selected_genre_id"]
    if genre_id is not None:
        row = get_genre_row(genre_stats, genre_id)
        if row is not None:
            rows.append(
                {
                    "Type": "Genre",
                    "Name": row["genre_name"],
                    "Films": int(row["film_count"]),
                    "Success Rate %": format_number(row["success_rate_pct"], 1),
                    "Avg ROI %": format_number(row["avg_roi_pct"], 1),
                    "Top Genre": row["genre_name"],
                    "Most Successful Film": row["best_film_title"] if pd.notna(row["best_film_title"]) else "n/a",
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

    df = df.sort_values(["total_revenue", "film_count", "name"], ascending=[False, False, True]).head(limit)

    return pd.DataFrame(
        {
            "Name": df["name"],
            "Films": df["film_count"].astype(int),
            "Success Rate %": df["success_rate_pct"].apply(lambda x: format_number(x, 1)),
            "Avg ROI %": df["avg_roi_pct"].apply(lambda x: format_number(x, 1)),
            "Total Revenue": df["total_revenue"].apply(format_money),
            "Top Genre": df["top_genre"].fillna("n/a"),
            "Most Successful Film": df["best_film_title"].fillna("n/a"),
        }
    )


def top_genres_table(genre_stats: pd.DataFrame, min_films: int, limit: int = 15) -> pd.DataFrame:
    df = genre_stats[genre_stats["film_count"] >= min_films].copy()

    if df.empty:
        return pd.DataFrame()

    df = df.sort_values(["total_revenue", "film_count", "genre_name"], ascending=[False, False, True]).head(limit)

    return pd.DataFrame(
        {
            "Genre": df["genre_name"],
            "Films": df["film_count"].astype(int),
            "Success Rate %": df["success_rate_pct"].apply(lambda x: format_number(x, 1)),
            "Avg ROI %": df["avg_roi_pct"].apply(lambda x: format_number(x, 1)),
            "Total Revenue": df["total_revenue"].apply(format_money),
            "Most Successful Film": df["best_film_title"].fillna("n/a"),
        }
    )


def show_roi_builder_page() -> None:
    st.title("📈 ROI Builder")
    st.write(
        "Build a movie setup with director, cast, genre, and budget. The tool estimates the expected ROI based on historical performance, genre quality, collaboration patterns, and budget fit."
    )

    with st.spinner("Loading ROI Builder..."):
        person_stats = load_person_role_stats()
        genre_stats = load_genre_stats()
        cast_pair_df = load_cast_pair_synergy_stats()
        director_cast_df = load_director_cast_synergy_stats()
        global_budget_df = load_global_budget_stats()

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
    genre_options = get_genre_options(genre_stats, min_films)
    sync_roi_builder_state_with_filters(director_options, cast_options, genre_options)

    st.subheader("Build Your Movie Setup")

    input_col1, input_col2 = st.columns(2)

    with input_col1:
        director_id_to_label = {v: k for k, v in director_options.items()}
        current_director_id = st.session_state["roi_builder_selected_director_id"]
        current_director_label = (
            director_id_to_label[current_director_id]
            if current_director_id is not None and current_director_id in director_id_to_label
            else "No Selection"
        )
        director_labels = ["No Selection"] + list(director_options.keys())
        selected_director_label = st.selectbox(
            "Director",
            options=director_labels,
            index=director_labels.index(current_director_label),
            key="roi_director_selectbox",
        )
        if selected_director_label == "No Selection":
            st.session_state["roi_builder_selected_director_id"] = None
        else:
            st.session_state["roi_builder_selected_director_id"] = director_options[selected_director_label]

    with input_col2:
        genre_id_to_label = {v: k for k, v in genre_options.items()}
        current_genre_id = st.session_state["roi_builder_selected_genre_id"]
        current_genre_label = (
            genre_id_to_label[current_genre_id]
            if current_genre_id is not None and current_genre_id in genre_id_to_label
            else "No Selection"
        )
        genre_labels = ["No Selection"] + list(genre_options.keys())
        selected_genre_label = st.selectbox(
            "Genre",
            options=genre_labels,
            index=genre_labels.index(current_genre_label),
            key="roi_genre_selectbox",
        )
        if selected_genre_label == "No Selection":
            st.session_state["roi_builder_selected_genre_id"] = None
        else:
            st.session_state["roi_builder_selected_genre_id"] = genre_options[selected_genre_label]

    st.number_input(
        "Estimated Budget (USD)",
        min_value=100000,
        max_value=1000000000,
        step=1000000,
        key="roi_builder_budget_input",
    )

    st.markdown("### Add Cast")

    available_cast_labels = [
        label
        for label, person_id in cast_options.items()
        if person_id not in st.session_state["roi_builder_selected_cast_ids"]
    ]

    add_col1, add_col2 = st.columns([5, 1])

    with add_col1:
        selected_cast_to_add = st.selectbox(
            "Cast Member",
            options=["Please choose"] + available_cast_labels,
            key="roi_cast_add_selectbox",
        )

    with add_col2:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        if st.button("Add", key="roi_add_cast_button", use_container_width=True):
            if selected_cast_to_add != "Please choose":
                add_cast_member(cast_options[selected_cast_to_add])
                st.rerun()

    predicted_roi, has_selection = build_current_team_prediction(
        person_stats=person_stats,
        genre_stats=genre_stats,
        cast_pair_df=cast_pair_df,
        director_cast_df=director_cast_df,
        global_budget_df=global_budget_df,
    )
    category = roi_category_label(predicted_roi, has_selection)

    st.markdown("---")
    st.subheader("Expected Result")

    metric_col1, metric_col2 = st.columns(2)
    metric_col1.metric("Expected ROI", f"{format_number(predicted_roi, 1)}%")
    metric_col2.metric("Category", category)

    if not has_selection:
        st.info("Select at least a director, a genre, or one cast member to generate an expected ROI.")
    elif predicted_roi < 0:
        st.error("The selected setup suggests a negative return scenario.")
    elif predicted_roi < 50:
        st.warning("The selected setup suggests a moderate ROI outlook.")
    else:
        st.success("The selected setup suggests a strong ROI outlook.")

    with st.expander("How is this calculated and how should the result be interpreted?"):
        st.markdown(
            """
This version uses a **transparent weighted scoring model** instead of the previous black-box ML approach.

The expected ROI is derived from these building blocks:
- director historical performance
- cast historical performance
- genre historical performance
- cast collaboration history
- director-cast collaboration history
- budget fit relative to the selected genre

That means:
- **genre now directly affects the ROI**
- **budget now directly affects the ROI**
- results are easier to interpret

The budget is compared with the **typical historical budget** of the selected genre. If the entered budget is far above or below that historical level, the model applies a penalty.

The category means:
- **No setup selected**: no forecast yet
- **High Risk**: expected ROI below 0%
- **Moderate**: expected ROI from 0% to below 50%
- **Strong**: expected ROI from 50% to below 150%
- **Very Strong**: expected ROI of 150% or more

This is still an estimate, not a guarantee. Real results can differ because of marketing, release timing, competition, franchise effects, and audience trends.
            """
        )

    st.markdown("---")
    st.subheader("Selected Setup")

    selected_team_df = build_selected_team_table(person_stats, genre_stats)
    if selected_team_df.empty:
        st.info("No setup selected yet.")
    else:
        st.dataframe(selected_team_df, use_container_width=True, hide_index=True)

    if st.session_state["roi_builder_selected_cast_ids"]:
        st.markdown("### Remove Cast Members")
        for cast_id in st.session_state["roi_builder_selected_cast_ids"]:
            cast_row = get_person_row(person_stats, "Cast", cast_id)
            if cast_row is None:
                continue

            remove_col1, remove_col2 = st.columns([5, 1])
            with remove_col1:
                st.write(
                    f"**{cast_row['name']}** — Films: {int(cast_row['film_count'])} | Success Rate: {format_number(cast_row['success_rate_pct'], 1)}% | Top Genre: {cast_row['top_genre'] if pd.notna(cast_row['top_genre']) else 'n/a'}"
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

    st.markdown("---")
    st.subheader("Top Genres")
    top_genres_df = top_genres_table(genre_stats, min_films=min_films, limit=15)
    if top_genres_df.empty:
        st.write("No data available.")
    else:
        st.dataframe(top_genres_df, use_container_width=True, hide_index=True)