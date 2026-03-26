from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from utils.db import run_query
from utils.helpers import format_money, format_number, safe_mean


def clip_roi(value: float) -> float:
    if pd.isna(value):
        return 0.0
    return float(np.clip(float(value), -100.0, 400.0))


def roi_category_label(value: float, has_setup: bool) -> str:
    if not has_setup:
        return "Kein Setup ausgewählt"
    if value < 0:
        return "Hohes Risiko"
    if value < 50:
        return "Moderat"
    if value < 150:
        return "Stark"
    return "Sehr stark"


def entity_component(avg_roi: float, success_rate: float) -> float:
    avg_roi = clip_roi(avg_roi)
    success_signal = (0.0 if pd.isna(success_rate) else float(success_rate) - 50.0) * 2.0
    return float((0.75 * avg_roi) + (0.25 * success_signal))


def genre_affinity_component(
    affinity_map: Dict[Tuple[str, int, int], dict],
    role_group: str,
    entity_id: int,
    genre_id: Optional[int],
    fallback_top_genre: Optional[str],
    selected_genre_name: Optional[str],
    base_avg_roi: float,
) -> float:
    if genre_id is None:
        return 0.0

    key = (role_group, entity_id, genre_id)
    if key in affinity_map:
        row = affinity_map[key]
        genre_avg_roi = clip_roi(row.get("genre_avg_roi_pct", 0.0))
        genre_success = float(row.get("genre_success_rate_pct", 0.0))
        genre_count = float(row.get("genre_film_count", 0.0))
        score = (0.70 * genre_avg_roi) + (0.20 * ((genre_success - 50.0) * 2.0)) + (0.10 * min(genre_count * 3.0, 30.0))
        return float(np.clip(score * 0.35, -60.0, 60.0))

    if fallback_top_genre and selected_genre_name:
        if str(fallback_top_genre).strip().lower() == str(selected_genre_name).strip().lower():
            return float(np.clip(base_avg_roi * 0.10, 5.0, 25.0))
        return float(np.clip(-0.25 * abs(base_avg_roi), -45.0, -10.0))

    return -15.0


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
    df = run_query(query)
    if not df.empty:
        df["component_score"] = df.apply(
            lambda row: entity_component(row["avg_roi_pct"], row["success_rate_pct"]),
            axis=1,
        )
    return df


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
def load_person_genre_affinity_stats() -> pd.DataFrame:
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
    GROUP BY
        pm.role_group,
        pm.person_id,
        g.genre_id,
        g.genre_name;
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


def build_person_genre_affinity_map(person_genre_df: pd.DataFrame):
    affinity_map = {}
    for _, row in person_genre_df.iterrows():
        key = (str(row["role_group"]), int(row["person_id"]), int(row["genre_id"]))
        affinity_map[key] = {
            "genre_name": row["genre_name"],
            "genre_film_count": int(row["genre_film_count"]),
            "genre_avg_roi_pct": float(row["genre_avg_roi_pct"]),
            "genre_success_rate_pct": float(row["genre_success_rate_pct"]),
        }
    return affinity_map


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


@st.cache_resource
def build_roi_lookup_artifacts():
    person_stats = load_person_role_stats()
    genre_stats = load_genre_stats()
    person_genre_df = load_person_genre_affinity_stats()
    cast_pair_df = load_cast_pair_synergy_stats()
    director_cast_df = load_director_cast_synergy_stats()

    director_stats_map, cast_stats_map = build_person_lookup_maps(person_stats)
    genre_stats_map = build_genre_lookup_map(genre_stats)
    person_genre_affinity_map = build_person_genre_affinity_map(person_genre_df)
    cast_pair_map, director_cast_map = build_synergy_lookup_maps(cast_pair_df, director_cast_df)

    return {
        "person_stats": person_stats,
        "genre_stats": genre_stats,
        "person_genre_df": person_genre_df,
        "cast_pair_df": cast_pair_df,
        "director_cast_df": director_cast_df,
        "director_stats_map": director_stats_map,
        "cast_stats_map": cast_stats_map,
        "genre_stats_map": genre_stats_map,
        "person_genre_affinity_map": person_genre_affinity_map,
        "cast_pair_map": cast_pair_map,
        "director_cast_map": director_cast_map,
    }


def ensure_roi_builder_session_state() -> None:
    if "roi_builder_selected_director_id" not in st.session_state:
        st.session_state["roi_builder_selected_director_id"] = None
    if "roi_builder_selected_cast_ids" not in st.session_state:
        st.session_state["roi_builder_selected_cast_ids"] = []
    if "roi_builder_selected_genre_id" not in st.session_state:
        st.session_state["roi_builder_selected_genre_id"] = None


def get_filtered_role_options(person_stats: pd.DataFrame, role_group: str, min_films: int) -> dict:
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


def get_genre_options(genre_stats: pd.DataFrame, min_films: int) -> dict:
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


def build_current_team_prediction(artifacts) -> Tuple[float, bool]:
    director_id = st.session_state["roi_builder_selected_director_id"]
    cast_ids = st.session_state["roi_builder_selected_cast_ids"]
    genre_id = st.session_state["roi_builder_selected_genre_id"]

    has_core_setup = bool(director_id is not None or len(cast_ids) > 0)
    if not has_core_setup:
        return 0.0, False

    director_stats_map = artifacts["director_stats_map"]
    cast_stats_map = artifacts["cast_stats_map"]
    genre_stats_map = artifacts["genre_stats_map"]
    person_genre_affinity_map = artifacts["person_genre_affinity_map"]
    cast_pair_map = artifacts["cast_pair_map"]
    director_cast_map = artifacts["director_cast_map"]

    active_scores = []
    active_weights = []

    selected_genre_name = None
    if genre_id is not None and genre_id in genre_stats_map:
        selected_genre_name = genre_stats_map[genre_id]["genre_name"]

    if director_id is not None and director_id in director_stats_map:
        d = director_stats_map[director_id]
        director_base = float(d["component_score"])
        director_genre_fit = genre_affinity_component(
            affinity_map=person_genre_affinity_map,
            role_group="Director",
            entity_id=director_id,
            genre_id=genre_id,
            fallback_top_genre=d.get("top_genre"),
            selected_genre_name=selected_genre_name,
            base_avg_roi=d.get("avg_roi_pct", 0.0),
        )
        active_scores.append(float(director_base + director_genre_fit))
        active_weights.append(0.42)

    cast_adjusted_scores = []
    for cast_id in cast_ids:
        if cast_id in cast_stats_map:
            c = cast_stats_map[cast_id]
            cast_base = float(c["component_score"])
            cast_genre_fit = genre_affinity_component(
                affinity_map=person_genre_affinity_map,
                role_group="Cast",
                entity_id=cast_id,
                genre_id=genre_id,
                fallback_top_genre=c.get("top_genre"),
                selected_genre_name=selected_genre_name,
                base_avg_roi=c.get("avg_roi_pct", 0.0),
            )
            cast_adjusted_scores.append(float(cast_base + cast_genre_fit))

    if cast_adjusted_scores:
        active_scores.append(float(safe_mean(cast_adjusted_scores, 0.0)))
        active_weights.append(0.33)

    cast_pair_rois = []
    for a, b in combinations(sorted(cast_ids), 2):
        key = (min(a, b), max(a, b))
        if key in cast_pair_map:
            cast_pair_rois.append(clip_roi(cast_pair_map[key]["pair_avg_roi_pct"]))

    if cast_pair_rois:
        active_scores.append(float(safe_mean(cast_pair_rois, 0.0)))
        active_weights.append(0.15)

    director_cast_rois = []
    if director_id is not None:
        for cast_id in cast_ids:
            key = (director_id, cast_id)
            if key in director_cast_map:
                director_cast_rois.append(clip_roi(director_cast_map[key]["pair_avg_roi_pct"]))

    if director_cast_rois:
        active_scores.append(float(safe_mean(director_cast_rois, 0.0)))
        active_weights.append(0.10)

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


def show_roi_builder_page() -> None:
    st.title("📈 ROI Builder")
    st.write(
        "Stelle ein Film-Setup aus Director, Cast und Genre zusammen. Der erwartete ROI basiert auf historischen Performance-Daten. Marketingkosten sind in den zugrunde liegenden Budgets nicht enthalten, daher fallen die hier gezeigten ROI-Werte in der Regel höher aus als in der Realität."
    )

    with st.spinner("ROI Builder wird geladen..."):
        artifacts = build_roi_lookup_artifacts()

    person_stats = artifacts["person_stats"]
    genre_stats = artifacts["genre_stats"]

    ensure_roi_builder_session_state()

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
    sync_roi_builder_state_with_filters(director_options, cast_options, genre_options)

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

    predicted_roi, has_setup = build_current_team_prediction(artifacts)
    category = roi_category_label(predicted_roi, has_setup)

    st.markdown("---")
    st.subheader("Erwartetes Ergebnis")

    metric_col1, metric_col2 = st.columns(2)
    metric_col1.metric("Expected ROI", f"{format_number(predicted_roi, 1)}%")
    metric_col2.metric("Kategorie", category)

    if not has_setup:
        st.info("Wähle mindestens einen Director oder ein Cast-Mitglied aus, um einen erwarteten ROI zu berechnen. Ein Genre allein erzeugt noch keine Prognose.")
    elif predicted_roi < 0:
        st.error("Das gewählte Setup deutet auf ein negatives Rendite-Szenario hin.")
    elif predicted_roi < 50:
        st.warning("Das gewählte Setup deutet auf ein moderates ROI-Szenario hin.")
    else:
        st.success("Das gewählte Setup deutet auf ein starkes ROI-Szenario hin.")

    with st.expander("Wie wird der ROI berechnet und wie ist das Ergebnis zu verstehen?"):
        st.markdown(
            """
Der erwartete ROI basiert auf einem transparenten Scoring-Modell mit historischen Filmdaten.

Ein Genre allein erzeugt keinen ROI. Das Genre wirkt hier nur indirekt als Passungsfaktor für Director und Cast.

Berücksichtigt werden:
- die historische Performance des Directors
- die historische Performance des Casts
- gemeinsame Performance von Cast-Kombinationen
- gemeinsame Performance von Director und Cast
- die historische Genre-Passung von Director und Cast

Wenn sich Director und Cast historisch eher in anderen Genres bewegt haben, wirkt sich das negativ auf den erwarteten ROI aus. Dadurch fällt das Ergebnis bei unpassenden Genre-Kombinationen deutlich stärker ab.

Die Kategorie bedeutet:
- Kein Setup ausgewählt
- Hohes Risiko
- Moderat
- Stark
- Sehr stark

Wichtig: Marketingkosten sind in den zugrunde liegenden Budgets nicht enthalten. Deshalb liegen die hier berechneten ROI-Werte meist über dem, was man in der Realität erwarten würde.
            """
        )

    st.markdown("---")
    st.subheader("Ausgewähltes Setup")

    selected_team_df = build_selected_team_table(person_stats, genre_stats)
    if selected_team_df.empty:
        st.info("Noch kein Setup ausgewählt.")
    else:
        st.dataframe(selected_team_df, use_container_width=True, hide_index=True)

    if st.session_state["roi_builder_selected_cast_ids"]:
        st.markdown("### Cast entfernen")
        for cast_id in st.session_state["roi_builder_selected_cast_ids"]:
            cast_row = get_person_row(person_stats, "Cast", cast_id)
            if cast_row is None:
                continue

            remove_col1, remove_col2 = st.columns([5, 1])
            with remove_col1:
                st.write(
                    f"**{cast_row['name']}** — Filme: {int(cast_row['film_count'])} | Ø ROI: {format_number(cast_row['avg_roi_pct'], 1)}% | Top-Genre: {cast_row['top_genre'] if pd.notna(cast_row['top_genre']) else 'n/a'}"
                )
            with remove_col2:
                if st.button("Entfernen", key=f"roi_remove_cast_{cast_id}", use_container_width=True):
                    remove_cast_member(cast_id)
                    st.rerun()

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