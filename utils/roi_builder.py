
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from utils.db import run_query
from utils.helpers import format_money, format_number, safe_mean


MODEL_PATH = Path("models/roi_random_forest.joblib")


# ----------------------------
# Stabilization helpers
# ----------------------------
def clip_target_roi(value: float) -> float:
    if pd.isna(value):
        return 0.0
    return float(np.clip(float(value), -100.0, 250.0))


def clip_display_roi(value: float) -> float:
    if pd.isna(value):
        return 0.0
    return float(np.clip(float(value), -100.0, 250.0))


def compress_signal(value: float, cap: float = 250.0, scale: float = 25.0) -> float:
    """
    Compresses extreme ROI-like values so that outliers do not dominate the model.
    """
    if pd.isna(value):
        return 0.0
    value = float(np.clip(float(value), -cap, cap))
    return float(np.sign(value) * np.log1p(abs(value)) * scale)


def roi_category_label(value: float, has_setup: bool) -> str:
    if not has_setup:
        return "Kein Setup"
    if value < 0:
        return "Hohes Risiko"
    if value < 40:
        return "Moderat"
    if value < 120:
        return "Stark"
    return "Sehr stark"


def weighted_top_k(values: List[float], weights: Optional[List[float]] = None, k: int = 3) -> float:
    clean = [float(v) for v in values if pd.notna(v)]
    if not clean:
        return 0.0
    clean = sorted(clean, reverse=True)[:k]
    if weights is None:
        weights = [0.55, 0.30, 0.15]
    weights = weights[: len(clean)]
    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()
    return float(np.dot(clean, weights))


@st.cache_data(ttl=3600)
def load_person_role_stats() -> pd.DataFrame:
    query = """
    SELECT
        role_group,
        person_id,
        name,
        film_count,
        avg_roi_pct,
        success_rate_pct,
        total_revenue,
        top_genre,
        best_film_title
    FROM core.v_roi_person_stats
    ORDER BY role_group, film_count DESC, name ASC;
    """
    return run_query(query)


@st.cache_data(ttl=3600)
def load_genre_stats() -> pd.DataFrame:
    query = """
    SELECT
        genre_id,
        genre_name,
        film_count,
        avg_roi_pct,
        success_rate_pct,
        total_revenue,
        median_budget,
        best_film_title
    FROM core.v_roi_genre_stats
    ORDER BY film_count DESC, genre_name ASC;
    """
    return run_query(query)


@st.cache_data(ttl=3600)
def load_cast_pair_synergy_stats() -> pd.DataFrame:
    query = """
    SELECT
        person_id_1,
        person_id_2,
        pair_film_count,
        pair_avg_roi_pct
    FROM core.v_roi_cast_pair_stats;
    """
    return run_query(query)


@st.cache_data(ttl=3600)
def load_director_cast_synergy_stats() -> pd.DataFrame:
    query = """
    SELECT
        director_id,
        cast_id,
        pair_film_count,
        pair_avg_roi_pct
    FROM core.v_roi_director_cast_pair_stats;
    """
    return run_query(query)


@st.cache_data(ttl=3600)
def load_genre_affinity_stats() -> pd.DataFrame:
    query = """
    SELECT
        pm.role_group,
        pm.person_id,
        g.genre_id,
        g.genre_name,
        COUNT(DISTINCT pm.movie_id) AS genre_film_count,
        AVG(pm.roi_pct) AS genre_avg_roi_pct,
        AVG(CASE WHEN pm.roi_pct > 0 THEN 1 ELSE 0 END) * 100.0 AS genre_success_rate_pct
    FROM core.v_roi_person_movies pm
    JOIN core.movie_genres mg
      ON pm.movie_id = mg.movie_id
    JOIN core.genres g
      ON mg.genre_id = g.genre_id
    GROUP BY pm.role_group, pm.person_id, g.genre_id, g.genre_name;
    """
    return run_query(query)


@st.cache_resource
def load_model_artifact():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_resource
def build_roi_lookup_artifacts():
    person_stats = load_person_role_stats()
    genre_stats = load_genre_stats()
    cast_pair_df = load_cast_pair_synergy_stats()
    director_cast_df = load_director_cast_synergy_stats()
    genre_affinity_df = load_genre_affinity_stats()

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
    genre_stats_map = genre_stats.set_index("genre_id").to_dict("index")

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

    genre_affinity_map = {}
    for _, row in genre_affinity_df.iterrows():
        key = (str(row["role_group"]), int(row["person_id"]), int(row["genre_id"]))
        genre_affinity_map[key] = {
            "genre_name": row["genre_name"],
            "genre_film_count": int(row["genre_film_count"]),
            "genre_avg_roi_pct": float(row["genre_avg_roi_pct"]),
            "genre_success_rate_pct": float(row["genre_success_rate_pct"]),
        }

    global_median_budget = 50_000_000.0
    if not genre_stats.empty and genre_stats["median_budget"].notna().any():
        global_median_budget = float(genre_stats["median_budget"].dropna().median())

    return {
        "person_stats": person_stats,
        "genre_stats": genre_stats,
        "director_stats_map": director_stats_map,
        "cast_stats_map": cast_stats_map,
        "genre_stats_map": genre_stats_map,
        "cast_pair_map": cast_pair_map,
        "director_cast_map": director_cast_map,
        "genre_affinity_map": genre_affinity_map,
        "global_median_budget": global_median_budget,
    }


def ensure_session_state():
    if "roi_builder_selected_director_id" not in st.session_state:
        st.session_state["roi_builder_selected_director_id"] = None
    if "roi_builder_selected_cast_ids" not in st.session_state:
        st.session_state["roi_builder_selected_cast_ids"] = []
    if "roi_builder_selected_genre_id" not in st.session_state:
        st.session_state["roi_builder_selected_genre_id"] = None


def get_filtered_role_options(person_stats: pd.DataFrame, role_group: str, min_films: int) -> Dict[str, int]:
    role_df = person_stats[
        (person_stats["role_group"] == role_group)
        & (person_stats["film_count"] >= min_films)
    ].copy()

    if role_df.empty:
        return {}

    role_df = role_df.sort_values(
        ["film_count", "avg_roi_pct", "name"],
        ascending=[False, False, True],
    )

    label_to_id = {}
    for _, row in role_df.iterrows():
        label = (
            f"{row['name']} "
            f"(Filme: {int(row['film_count'])} | "
            f"Ø ROI: {format_number(row['avg_roi_pct'], 1)}%)"
        )
        label_to_id[label] = int(row["person_id"])

    return label_to_id


def get_genre_options(genre_stats: pd.DataFrame, min_films: int) -> Dict[str, int]:
    genre_df = genre_stats[genre_stats["film_count"] >= min_films].copy()

    if genre_df.empty:
        return {}

    genre_df = genre_df.sort_values(
        ["film_count", "avg_roi_pct", "genre_name"],
        ascending=[False, False, True],
    )

    label_to_id = {}
    for _, row in genre_df.iterrows():
        label = (
            f"{row['genre_name']} "
            f"(Filme: {int(row['film_count'])} | "
            f"Ø ROI: {format_number(row['avg_roi_pct'], 1)}%)"
        )
        label_to_id[label] = int(row["genre_id"])

    return label_to_id


def sync_state_with_filters(
    director_options: Dict[str, int],
    cast_options: Dict[str, int],
    genre_options: Dict[str, int],
):
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


def add_cast_member(person_id: int):
    if person_id not in st.session_state["roi_builder_selected_cast_ids"]:
        st.session_state["roi_builder_selected_cast_ids"].append(person_id)


def remove_cast_member(person_id: int):
    st.session_state["roi_builder_selected_cast_ids"] = [
        pid for pid in st.session_state["roi_builder_selected_cast_ids"] if pid != person_id
    ]


def compute_genre_fit(role_group: str, person_id: int, genre_id: Optional[int], genre_affinity_map: dict) -> float:
    if genre_id is None:
        return 0.0
    key = (role_group, int(person_id), int(genre_id))
    if key not in genre_affinity_map:
        return 0.0
    row = genre_affinity_map[key]
    raw_score = (
        0.60 * clip_display_roi(row["genre_avg_roi_pct"])
        + 0.25 * ((row["genre_success_rate_pct"] - 50.0) * 2.0)
        + 0.15 * min(row["genre_film_count"] * 2.0, 20.0)
    )
    return compress_signal(raw_score, cap=220.0, scale=18.0)


def build_feature_row(artifacts) -> Tuple[pd.DataFrame, bool, dict]:
    director_id = st.session_state["roi_builder_selected_director_id"]
    cast_ids = st.session_state["roi_builder_selected_cast_ids"]
    genre_id = st.session_state["roi_builder_selected_genre_id"]

    has_setup = bool(director_id is not None or len(cast_ids) > 0)
    if not has_setup:
        return pd.DataFrame(), False, {}

    director_stats_map = artifacts["director_stats_map"]
    cast_stats_map = artifacts["cast_stats_map"]
    genre_stats_map = artifacts["genre_stats_map"]
    cast_pair_map = artifacts["cast_pair_map"]
    director_cast_map = artifacts["director_cast_map"]
    genre_affinity_map = artifacts["genre_affinity_map"]

    director_avg_roi = 0.0
    director_success_rate = 0.0
    director_film_count = 0.0
    director_count = 0
    director_genre_fit = 0.0

    if director_id is not None and director_id in director_stats_map:
        d = director_stats_map[director_id]
        director_avg_roi = compress_signal(d["avg_roi_pct"], cap=220.0, scale=18.0)
        director_success_rate = float(d["success_rate_pct"])
        director_film_count = float(d["film_count"])
        director_count = 1
        director_genre_fit = compute_genre_fit("Director", director_id, genre_id, genre_affinity_map)

    cast_ranked = []
    for cast_id in cast_ids:
        if cast_id in cast_stats_map:
            c = cast_stats_map[cast_id]
            score_rank = (
                0.60 * clip_display_roi(c["avg_roi_pct"])
                + 0.25 * float(c["success_rate_pct"])
                + 0.15 * min(float(c["film_count"]), 40.0)
            )
            cast_ranked.append(
                {
                    "person_id": cast_id,
                    "avg_roi_pct": float(c["avg_roi_pct"]),
                    "success_rate_pct": float(c["success_rate_pct"]),
                    "film_count": float(c["film_count"]),
                    "rank_score": score_rank,
                }
            )

    cast_ranked = sorted(cast_ranked, key=lambda x: x["rank_score"], reverse=True)[:3]
    selected_cast_ids_for_features = [x["person_id"] for x in cast_ranked]

    cast_avg_roi = weighted_top_k(
        [compress_signal(x["avg_roi_pct"], cap=220.0, scale=18.0) for x in cast_ranked]
    )
    cast_success_rate = weighted_top_k([x["success_rate_pct"] for x in cast_ranked], k=3)
    cast_film_count = weighted_top_k([x["film_count"] for x in cast_ranked], k=3)
    cast_count = len(cast_ranked)
    cast_genre_fit = weighted_top_k(
        [
            compute_genre_fit("Cast", x["person_id"], genre_id, genre_affinity_map)
            for x in cast_ranked
        ],
        k=3,
    )

    cast_pair_rois = []
    cast_pair_counts = []
    for a, b in combinations(sorted(selected_cast_ids_for_features), 2):
        key = (min(a, b), max(a, b))
        if key in cast_pair_map:
            cast_pair_rois.append(compress_signal(cast_pair_map[key]["pair_avg_roi_pct"], cap=180.0, scale=16.0))
            cast_pair_counts.append(float(cast_pair_map[key]["pair_film_count"]))

    director_cast_rois = []
    director_cast_counts = []
    if director_id is not None:
        for cast_id in selected_cast_ids_for_features:
            key = (director_id, cast_id)
            if key in director_cast_map:
                director_cast_rois.append(
                    compress_signal(director_cast_map[key]["pair_avg_roi_pct"], cap=180.0, scale=16.0)
                )
                director_cast_counts.append(float(director_cast_map[key]["pair_film_count"]))

    genre_avg_roi = 0.0
    genre_success_rate = 0.0
    genre_count = 0
    if genre_id is not None and genre_id in genre_stats_map:
        g = genre_stats_map[genre_id]
        genre_avg_roi = compress_signal(g["avg_roi_pct"], cap=200.0, scale=15.0)
        genre_success_rate = float(g["success_rate_pct"])
        genre_count = 1

    feature_row = pd.DataFrame(
        [
            {
                "director_avg_roi": director_avg_roi,
                "director_success_rate": director_success_rate,
                "director_film_count": director_film_count,
                "director_count": director_count,
                "cast_avg_roi": cast_avg_roi,
                "cast_success_rate": cast_success_rate,
                "cast_film_count": cast_film_count,
                "cast_count": cast_count,
                "cast_pair_avg_roi": safe_mean(cast_pair_rois, 0.0),
                "cast_pair_avg_count": safe_mean(cast_pair_counts, 0.0),
                "cast_pair_match_count": len(cast_pair_rois),
                "director_cast_avg_roi": safe_mean(director_cast_rois, 0.0),
                "director_cast_avg_count": safe_mean(director_cast_counts, 0.0),
                "director_cast_match_count": len(director_cast_rois),
                "genre_count": genre_count,
                "genre_avg_roi": genre_avg_roi,
                "genre_success_rate": genre_success_rate,
                "director_genre_fit": director_genre_fit,
                "cast_genre_fit": cast_genre_fit,
            }
        ]
    )

    diagnostics = {
        "director_genre_fit_raw": float(np.clip(director_genre_fit, -120.0, 120.0)),
        "cast_genre_fit_raw": float(np.clip(cast_genre_fit, -120.0, 120.0)),
        "cast_synergy_raw": float(np.clip(safe_mean(cast_pair_rois, 0.0), -120.0, 120.0)),
        "director_cast_synergy_raw": float(np.clip(safe_mean(director_cast_rois, 0.0), -120.0, 120.0)),
        "used_cast_count_for_features": cast_count,
        "selected_cast_count_total": len(cast_ids),
    }

    return feature_row, True, diagnostics


def predict_roi(model_artifact, feature_row: pd.DataFrame) -> Tuple[float, float]:
    model = model_artifact["model"]
    feature_cols = model_artifact["feature_cols"]

    X = feature_row.reindex(columns=feature_cols, fill_value=0.0)
    pred = float(model.predict(X)[0])
    pred = clip_target_roi(pred)

    if hasattr(model, "estimators_"):
        tree_preds = np.array([tree.predict(X)[0] for tree in model.estimators_], dtype=float)
        tree_preds = np.clip(tree_preds, -100.0, 250.0)
        uncertainty = float(np.std(tree_preds))
    else:
        uncertainty = 18.0

    return pred, uncertainty


def success_probability(predicted_roi: float, uncertainty: float) -> float:
    if uncertainty <= 1e-6:
        return 100.0 if predicted_roi > 0 else 0.0
    z = predicted_roi / max(uncertainty, 1.0)
    prob = 1.0 / (1.0 + np.exp(-z))
    return float(np.clip(prob * 100.0, 0.0, 100.0))


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


def build_selected_team_table(person_stats: pd.DataFrame, genre_stats: pd.DataFrame) -> pd.DataFrame:
    rows = []

    director_id = st.session_state["roi_builder_selected_director_id"]
    if director_id is not None:
        row = get_person_row(person_stats, "Director", director_id)
        if row is not None:
            rows.append(
                {
                    "Typ": "Director",
                    "Name": row["name"],
                    "Filme": int(row["film_count"]),
                    "Success Rate %": format_number(row["success_rate_pct"], 1),
                    "Ø ROI %": format_number(row["avg_roi_pct"], 1),
                    "Top-Genre": row["top_genre"] if pd.notna(row["top_genre"]) else "n/a",
                    "Erfolgreichster Film": row["best_film_title"] if pd.notna(row["best_film_title"]) else "n/a",
                }
            )

    for cast_id in st.session_state["roi_builder_selected_cast_ids"]:
        row = get_person_row(person_stats, "Cast", cast_id)
        if row is not None:
            rows.append(
                {
                    "Typ": "Cast",
                    "Name": row["name"],
                    "Filme": int(row["film_count"]),
                    "Success Rate %": format_number(row["success_rate_pct"], 1),
                    "Ø ROI %": format_number(row["avg_roi_pct"], 1),
                    "Top-Genre": row["top_genre"] if pd.notna(row["top_genre"]) else "n/a",
                    "Erfolgreichster Film": row["best_film_title"] if pd.notna(row["best_film_title"]) else "n/a",
                }
            )

    genre_id = st.session_state["roi_builder_selected_genre_id"]
    if genre_id is not None:
        row = get_genre_row(genre_stats, genre_id)
        if row is not None:
            rows.append(
                {
                    "Typ": "Genre",
                    "Name": row["genre_name"],
                    "Filme": int(row["film_count"]),
                    "Success Rate %": format_number(row["success_rate_pct"], 1),
                    "Ø ROI %": format_number(row["avg_roi_pct"], 1),
                    "Top-Genre": row["genre_name"],
                    "Erfolgreichster Film": row["best_film_title"] if pd.notna(row["best_film_title"]) else "n/a",
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
            "Filme": df["film_count"].astype(int),
            "Success Rate %": df["success_rate_pct"].apply(lambda x: format_number(x, 1)),
            "Ø ROI %": df["avg_roi_pct"].apply(lambda x: format_number(x, 1)),
            "Gesamtumsatz": df["total_revenue"].apply(format_money),
            "Top-Genre": df["top_genre"].fillna("n/a"),
            "Erfolgreichster Film": df["best_film_title"].fillna("n/a"),
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
            "Filme": df["film_count"].astype(int),
            "Success Rate %": df["success_rate_pct"].apply(lambda x: format_number(x, 1)),
            "Ø ROI %": df["avg_roi_pct"].apply(lambda x: format_number(x, 1)),
            "Gesamtumsatz": df["total_revenue"].apply(format_money),
            "Erfolgreichster Film": df["best_film_title"].fillna("n/a"),
        }
    )


def show_feature_importance(model_artifact):
    feature_importance = model_artifact.get("feature_importance", [])
    if not feature_importance:
        return

    importance_df = pd.DataFrame(feature_importance).head(10)
    if importance_df.empty:
        return

    st.markdown("### Wichtigste Modell-Features")
    st.dataframe(
        importance_df.assign(importance=lambda x: x["importance"].map(lambda v: format_number(v, 4))),
        use_container_width=True,
        hide_index=True,
    )


def show_roi_builder_page() -> None:
    st.title("📈 ROI Builder")
    st.write(
        "Entwicklung eines datengetriebenen Tools, das den erwarteten ROI eines Film-Setups (Director, Cast, Genre) prognostiziert."
    )

    with st.spinner("ROI Builder wird geladen..."):
        artifacts = build_roi_lookup_artifacts()
        model_artifact = load_model_artifact()

    if model_artifact is None:
        st.error("Kein trainiertes Modell gefunden. Lege zuerst die Datei models/roi_random_forest.joblib an.")
        st.code("python -m models.train_roi_model")
        return

    person_stats = artifacts["person_stats"]
    genre_stats = artifacts["genre_stats"]

    ensure_session_state()

    st.sidebar.markdown("### Filter für den ROI Builder")
    min_films = st.sidebar.selectbox(
        "Minimale Anzahl an Filmen",
        options=[1, 3, 5, 10],
        index=1,
        key="roi_min_films",
    )

    director_options = get_filtered_role_options(person_stats, "Director", min_films)
    cast_options = get_filtered_role_options(person_stats, "Cast", min_films)
    genre_options = get_genre_options(genre_stats, min_films)
    sync_state_with_filters(director_options, cast_options, genre_options)

    st.subheader("Film-Setup erstellen")

    input_col1, input_col2 = st.columns(2)

    with input_col1:
        director_id_to_label = {v: k for k, v in director_options.items()}
        current_director_id = st.session_state["roi_builder_selected_director_id"]
        current_director_label = (
            director_id_to_label[current_director_id]
            if current_director_id is not None and current_director_id in director_id_to_label
            else "Keine Auswahl"
        )
        director_labels = ["Keine Auswahl"] + list(director_options.keys())
        selected_director_label = st.selectbox(
            "Director",
            options=director_labels,
            index=director_labels.index(current_director_label),
            key="roi_director_selectbox",
        )
        if selected_director_label == "Keine Auswahl":
            st.session_state["roi_builder_selected_director_id"] = None
        else:
            st.session_state["roi_builder_selected_director_id"] = director_options[selected_director_label]

    with input_col2:
        genre_id_to_label = {v: k for k, v in genre_options.items()}
        current_genre_id = st.session_state["roi_builder_selected_genre_id"]
        current_genre_label = (
            genre_id_to_label[current_genre_id]
            if current_genre_id is not None and current_genre_id in genre_id_to_label
            else "Keine Auswahl"
        )
        genre_labels = ["Keine Auswahl"] + list(genre_options.keys())
        selected_genre_label = st.selectbox(
            "Genre",
            options=genre_labels,
            index=genre_labels.index(current_genre_label),
            key="roi_genre_selectbox",
        )
        if selected_genre_label == "Keine Auswahl":
            st.session_state["roi_builder_selected_genre_id"] = None
        else:
            st.session_state["roi_builder_selected_genre_id"] = genre_options[selected_genre_label]

    st.markdown("### Cast hinzufügen")

    available_cast_labels = [
        label
        for label, person_id in cast_options.items()
        if person_id not in st.session_state["roi_builder_selected_cast_ids"]
    ]

    add_col1, add_col2 = st.columns([5, 1])

    with add_col1:
        selected_cast_to_add = st.selectbox(
            "Cast-Mitglied",
            options=["Bitte auswählen"] + available_cast_labels,
            key="roi_cast_add_selectbox",
        )

    with add_col2:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        if st.button("Hinzufügen", key="roi_add_cast_button", use_container_width=True):
            if selected_cast_to_add != "Bitte auswählen":
                add_cast_member(cast_options[selected_cast_to_add])
                st.rerun()

    feature_row, has_setup, diagnostics = build_feature_row(artifacts)

    st.markdown("---")
    st.subheader("Erwartetes Ergebnis")

    if not has_setup:
        st.info("Wähle mindestens einen Director oder ein Cast-Mitglied aus, um eine Prognose zu berechnen.")
    else:
        predicted_roi, uncertainty = predict_roi(model_artifact, feature_row)
        category = roi_category_label(predicted_roi, True)
        prob_success = success_probability(predicted_roi, uncertainty)
        roi_low = max(-100.0, predicted_roi - uncertainty)
        roi_high = min(250.0, predicted_roi + uncertainty)

        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        metric_col1.metric("Expected ROI", f"{format_number(predicted_roi, 1)}%")
        metric_col2.metric("ROI Range", f"{format_number(roi_low, 1)}% bis {format_number(roi_high, 1)}%")
        metric_col3.metric("Success Probability", f"{format_number(prob_success, 1)}%")
        metric_col4.metric("Kategorie", category)

        if predicted_roi < 0:
            st.error("Das gewählte Setup deutet auf ein negatives Rendite-Szenario hin.")
        elif predicted_roi < 40:
            st.warning("Das gewählte Setup deutet auf ein moderates ROI-Szenario hin.")
        else:
            st.success("Das gewählte Setup deutet auf ein starkes ROI-Szenario hin.")

        diag1, diag2, diag3, diag4 = st.columns(4)
        diag1.metric("Director Genre Fit", format_number(diagnostics["director_genre_fit_raw"], 1))
        diag2.metric("Cast Genre Fit", format_number(diagnostics["cast_genre_fit_raw"], 1))
        diag3.metric("Cast Synergy", format_number(diagnostics["cast_synergy_raw"], 1))
        diag4.metric("Director-Cast Synergy", format_number(diagnostics["director_cast_synergy_raw"], 1))

        with st.expander("Erklärung der einzelnen Kennzahlen"):
            st.markdown(
                """
**Expected ROI**  
Geschätzter ROI des ausgewählten Setups. Die finale Prognose kommt aus dem trainierten Random-Forest-Modell.

**ROI Range**  
Grobe Spannbreite der Prognose. Sie basiert auf der Streuung der einzelnen Bäume im Random Forest.

**Success Probability**  
Vereinfachte Wahrscheinlichkeit, dass das Setup profitabel ist. Ein positiver ROI führt zu einer höheren Wahrscheinlichkeit.

**Kategorie**  
Zusammenfassung des erwarteten Ergebnisses:
- Hohes Risiko
- Moderat
- Stark
- Sehr stark

**Director Genre Fit**  
Wie gut der gewählte Director historisch zum ausgewählten Genre passt.

**Cast Genre Fit**  
Wie gut die wichtigsten ausgewählten Schauspieler historisch zum Genre passen.

**Cast Synergy**  
Historische gemeinsame Performance der wichtigsten Schauspieler untereinander.

**Director-Cast Synergy**  
Historische gemeinsame Performance zwischen Director und den wichtigsten ausgewählten Schauspielern.

**Wichtig für die Logik**  
Für die Modell-Features werden maximal die 3 stärksten ausgewählten Schauspieler genutzt. Dadurch wird verhindert, dass ein zusätzliches Cast-Mitglied das Ergebnis nur wegen einfacher Durchschnittsbildung stark verzerrt.
                """
            )

        with st.expander("Wie wird die Prognose berechnet?"):
            st.markdown(
                """
Die finale Version des ROI Builders nutzt ein stabilisiertes Random-Forest-Regressionsmodell.

Berücksichtigt werden unter anderem:
- historische Performance des Directors
- historische Performance des Casts
- Director-Cast-Synergien
- Cast-Cast-Synergien
- Genre-Fit

Um das Modell robuster zu machen, wurden:
- extreme ROI-Werte begrenzt
- Signale komprimiert
- Cast-Features auf die stärksten 3 Schauspieler fokussiert
- Budget aus dem Interface entfernt
                """
            )

        with st.expander("Modellgüte"):
            metrics = model_artifact.get("metrics", {})
            st.write(f"MAE: {format_number(metrics.get('mae', 0.0), 2)}")
            st.write(f"RMSE: {format_number(metrics.get('rmse', 0.0), 2)}")
            st.write(f"R²: {format_number(metrics.get('r2', 0.0), 3)}")
            st.write(f"Train Rows: {metrics.get('train_rows', 0)}")
            st.write(f"Test Rows: {metrics.get('test_rows', 0)}")
            st.write(f"Gesamtzeilen: {metrics.get('full_rows', 0)}")
            
        with st.expander("Data Leakage – Problem und Lösung"):
            st.markdown("### Warum war das ein Problem?")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(
                    """
        <div style="
            background-color:#3a1f1f;
            padding:18px;
            border-radius:12px;
            border:1px solid #7a3a3a;
        ">
            <h4 style="margin-top:0;">🟥 Vorher: Problem</h4>
            <p><strong>Film A vorhersagen</strong></p>
            <p><strong>Feature:</strong><br>
            Actor ROI = 180%<br>
            <em>(inkl. Film A)</em></p>
            <p>👉 Das Modell kennt indirekt schon einen Teil der Antwort</p>
            <p>👉 Ergebnis: <strong>R² ≈ 0.99</strong></p>
            <p>👉 Für dieses Problem unrealistisch</p>
        </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown(
                    """
        <div style="
            background-color:#1f3a28;
            padding:18px;
            border-radius:12px;
            border:1px solid #3d7a52;
        ">
            <h4 style="margin-top:0;">🟩 Nachher: Lösung</h4>
            <p><strong>Film A vorhersagen</strong></p>
            <p><strong>Feature:</strong><br>
            Actor ROI = 50%<br>
            <em>(ohne Film A)</em></p>
            <p>👉 Der aktuelle Film wird aus der Berechnung entfernt</p>
            <p>👉 Ergebnis: <strong>R² ≈ 0.15</strong></p>
            <p>👉 Realistischer und besser übertragbar</p>
        </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("### Was bedeutet Leave-One-Out?")
            st.markdown(
                """
        Beim Training wird so getan, als wäre der aktuelle Film noch unbekannt.

        Das bedeutet:
        - der Film, den das Modell gerade vorhersagen soll, wird aus allen Durchschnittswerten entfernt
        - dadurch kann das Modell nicht mehr auf die Lösung „spicken“
        - die Vorhersage wird ehrlicher und realistischer

        **Einfach gesagt:**  
        Das Modell berechnet seine Eingaben immer so, als würde es den Film noch nicht kennen.
                """
            )

            st.markdown("### Fazit")
            st.info(
                "Ein sehr hoher R²-Wert ist nur dann gut, wenn das Modell fair trainiert wurde. "
                "Hier war der erste Wert zu gut, um realistisch zu sein. "
                "Durch Leave-One-Out wurde das Modell robuster und die Ergebnisse wurden glaubwürdiger."
            )

    st.markdown("---")
    st.subheader("Ausgewähltes Setup")

    selected_team_df = build_selected_team_table(person_stats, genre_stats)
    if selected_team_df.empty:
        st.info("Noch kein Setup ausgewählt.")
    else:
        st.dataframe(selected_team_df, use_container_width=True, hide_index=True)

    if st.session_state["roi_builder_selected_cast_ids"]:
        st.caption(
            f"Für die Modell-Features werden aktuell die {diagnostics.get('used_cast_count_for_features', 0)} "
            f"stärksten Schauspieler aus insgesamt {diagnostics.get('selected_cast_count_total', 0)} ausgewählten verwendet."
        )
        st.markdown("### Cast entfernen")
        for cast_id in st.session_state["roi_builder_selected_cast_ids"]:
            cast_row = get_person_row(person_stats, "Cast", cast_id)
            if cast_row is None:
                continue

            remove_col1, remove_col2 = st.columns([5, 1])
            with remove_col1:
                st.write(
                    f"**{cast_row['name']}** — Filme: {int(cast_row['film_count'])} | "
                    f"Ø ROI: {format_number(cast_row['avg_roi_pct'], 1)}% | "
                    f"Top-Genre: {cast_row['top_genre'] if pd.notna(cast_row['top_genre']) else 'n/a'}"
                )
            with remove_col2:
                if st.button("Entfernen", key=f"roi_remove_cast_{cast_id}", use_container_width=True):
                    remove_cast_member(cast_id)
                    st.rerun()

    st.markdown("---")
    show_feature_importance(model_artifact)

    st.markdown("---")
    st.subheader("Top Directors")
    top_directors_df = top_people_table(person_stats, "Director", min_films=min_films, limit=15)
    if top_directors_df.empty:
        st.write("Keine Daten verfügbar.")
    else:
        st.dataframe(top_directors_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Top Schauspieler")
    top_actors_df = top_people_table(person_stats, "Cast", min_films=min_films, limit=20)
    if top_actors_df.empty:
        st.write("Keine Daten verfügbar.")
    else:
        st.dataframe(top_actors_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Top Genres")
    top_genres_df = top_genres_table(genre_stats, min_films=min_films, limit=15)
    if top_genres_df.empty:
        st.write("Keine Daten verfügbar.")
    else:
        st.dataframe(top_genres_df, use_container_width=True, hide_index=True)
