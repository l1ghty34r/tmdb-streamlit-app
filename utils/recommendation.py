import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from utils.db import run_query
from utils.helpers import format_number, poster_url


# =========================================================
# DATA LOADING
# =========================================================
@st.cache_data(ttl=600)
def load_recommender_data() -> pd.DataFrame:
    query = """
    WITH unique_movies AS (
        SELECT DISTINCT ON (m.title, EXTRACT(YEAR FROM m.release_date::date))
            m.movie_id,
            m.title,
            m.overview,
            m.poster_path,
            m.popularity,
            m.runtime,
            m.vote_average,
            m.release_date
        FROM core.movies m
        ORDER BY m.title, EXTRACT(YEAR FROM m.release_date::date), m.popularity DESC
    ),
    cast_info AS (
        SELECT
            mc.movie_id,
            STRING_AGG(
                REPLACE(LOWER(p.name), ' ', '_'),
                '|' ORDER BY mc.cast_order ASC
            ) AS actors
        FROM core.movie_cast mc
        JOIN core.people p
            ON mc.person_id = p.person_id
        WHERE mc.cast_order <= 3
        GROUP BY mc.movie_id
    ),
    crew_info AS (
        SELECT
            x.movie_id,
            REPLACE(LOWER(x.director), ' ', '_') AS director
        FROM (
            SELECT
                mcr.movie_id,
                p.name AS director,
                ROW_NUMBER() OVER (
                    PARTITION BY mcr.movie_id
                    ORDER BY p.name
                ) AS rn
            FROM core.movie_crew mcr
            JOIN core.people p
                ON mcr.person_id = p.person_id
            WHERE mcr.job = 'Director'
        ) x
        WHERE x.rn = 1
    ),
    genre_info AS (
        SELECT
            mg.movie_id,
            STRING_AGG(
                REPLACE(LOWER(g.genre_name), ' ', '_'),
                '|' ORDER BY g.genre_name
            ) AS genre_list
        FROM core.movie_genres mg
        JOIN core.genres g
            ON mg.genre_id = g.genre_id
        GROUP BY mg.movie_id
    )
    SELECT
        um.movie_id,
        um.title,
        um.overview,
        um.poster_path,
        um.popularity,
        um.runtime,
        um.vote_average,
        um.release_date,
        COALESCE(gi.genre_list, '') AS genres,
        COALESCE(ci.actors, '') AS top_cast,
        COALESCE(cr.director, '') AS director
    FROM unique_movies um
    LEFT JOIN genre_info gi
        ON um.movie_id = gi.movie_id
    LEFT JOIN cast_info ci
        ON um.movie_id = ci.movie_id
    LEFT JOIN crew_info cr
        ON um.movie_id = cr.movie_id
    ORDER BY um.title;
    """
    return run_query(query)


# =========================================================
# HELPERS
# =========================================================
def _safe_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def _pipe_to_weighted_tokens(text: str, repeat: int = 1) -> str:
    """
    Wandelt 'tom_hanks|meg_ryan' in 'tom_hanks meg_ryan' um
    und wiederholt Tokens optional zur Gewichtung.
    """
    if not text:
        return ""
    tokens = [token.strip() for token in str(text).split("|") if token.strip()]
    if not tokens:
        return ""
    return " ".join(tokens * repeat)


def _format_genres_for_display(genres: str) -> str:
    if pd.isna(genres) or not str(genres).strip():
        return "n/a"
    return str(genres).replace("|", ", ").replace("_", " ")


def _format_director_for_display(director: str) -> str:
    if pd.isna(director) or not str(director).strip():
        return "n/a"
    return str(director).replace("_", " ")


def _build_movie_label(row: pd.Series) -> str:
    year = pd.to_datetime(row.get("release_date"), errors="coerce")
    if pd.notna(year):
        return f"{row['title']} ({year.year})"
    return str(row["title"])


# =========================================================
# MODEL BUILDING
# =========================================================
@st.cache_resource
def build_recommender_engine(df: pd.DataFrame):
    work_df = df.copy()

    work_df["overview_clean"] = work_df["overview"].fillna("").apply(_safe_text)
    work_df["director_clean"] = work_df["director"].fillna("").apply(_safe_text)
    work_df["cast_clean"] = work_df["top_cast"].fillna("").apply(_safe_text)
    work_df["genres_clean"] = work_df["genres"].fillna("").apply(_safe_text)

    # Gewichtung:
    # Director = 3x
    # Cast = 2x
    # Genres = 2x
    # Overview = 1x
    work_df["soup"] = (
        work_df["director_clean"].apply(lambda x: _pipe_to_weighted_tokens(x, repeat=3))
        + " "
        + work_df["cast_clean"].apply(lambda x: _pipe_to_weighted_tokens(x, repeat=2))
        + " "
        + work_df["genres_clean"].apply(lambda x: _pipe_to_weighted_tokens(x, repeat=2))
        + " "
        + work_df["overview_clean"]
    ).str.strip()

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8,
        sublinear_tf=True,
    )

    tfidf_matrix = vectorizer.fit_transform(work_df["soup"])

    # Eindeutiger Index über movie_id
    indices = pd.Series(work_df.index, index=work_df["movie_id"])

    return work_df, indices, tfidf_matrix


# =========================================================
# RECOMMENDATION LOGIC
# =========================================================
def get_recommendations(
    df: pd.DataFrame,
    indices: pd.Series,
    tfidf_matrix,
    selected_movie_id: int,
    min_rating: float,
    max_runtime: int,
    selected_genres: list[str],
    num_recommendations: int,
) -> pd.DataFrame:
    if selected_movie_id not in indices:
        return pd.DataFrame()

    idx = indices[selected_movie_id]

    # Similarity nur für ausgewählten Film berechnen
    sim_scores = linear_kernel(tfidf_matrix[idx:idx + 1], tfidf_matrix).flatten()

    candidates = df.copy()
    candidates["similarity"] = sim_scores

    # gewählten Film selbst entfernen
    candidates = candidates[candidates["movie_id"] != selected_movie_id].copy()

    # Basisfilter
    candidates = candidates[
        (candidates["vote_average"].fillna(0) >= min_rating) &
        (candidates["runtime"].fillna(9999) <= max_runtime)
    ].copy()

    # Genre-Filter
    if selected_genres:
        selected_genres_set = {g.strip().lower() for g in selected_genres}

        def has_matching_genre(genre_string: str) -> bool:
            movie_genres = {
                g.strip().lower()
                for g in str(genre_string).split("|")
                if g.strip()
            }
            return len(selected_genres_set.intersection(movie_genres)) > 0

        candidates = candidates[candidates["genres"].apply(has_matching_genre)].copy()

    if candidates.empty:
        return pd.DataFrame()

    # optionale Bonus-Scores
    candidates["vote_norm"] = candidates["vote_average"].fillna(0) / 10

    max_popularity = candidates["popularity"].fillna(0).max()
    if max_popularity and max_popularity > 0:
        candidates["pop_norm"] = candidates["popularity"].fillna(0) / max_popularity
    else:
        candidates["pop_norm"] = 0.0

    # leichter Bonus für exakt gleichen Director
    selected_director = str(
        df.loc[df["movie_id"] == selected_movie_id, "director"].iloc[0]
    ).strip()

    if selected_director:
        candidates["director_bonus"] = np.where(
            candidates["director"].fillna("").str.strip() == selected_director,
            1.0,
            0.0
        )
    else:
        candidates["director_bonus"] = 0.0

    # Bonus für Überschneidung im Top-Cast
    selected_cast = set(
        token.strip()
        for token in str(
            df.loc[df["movie_id"] == selected_movie_id, "top_cast"].iloc[0]
        ).split("|")
        if token.strip()
    )

    def cast_overlap_score(cast_string: str) -> float:
        current_cast = {
            token.strip()
            for token in str(cast_string).split("|")
            if token.strip()
        }
        if not selected_cast or not current_cast:
            return 0.0
        overlap = len(selected_cast.intersection(current_cast))
        return overlap / max(len(selected_cast), 1)

    candidates["cast_bonus"] = candidates["top_cast"].apply(cast_overlap_score)

    # finaler Score
    candidates["final_score"] = (
        candidates["similarity"] * 0.72
        + candidates["vote_norm"] * 0.13
        + candidates["pop_norm"] * 0.05
        + candidates["director_bonus"] * 0.05
        + candidates["cast_bonus"] * 0.05
    )

    candidates = candidates.sort_values("final_score", ascending=False)

    # zur Sicherheit doppelte Titel entfernen
    candidates["title_clean"] = candidates["title"].astype(str).str.strip().str.lower()
    candidates = candidates.drop_duplicates(subset=["title_clean"])

    return candidates.head(num_recommendations).copy()


# =========================================================
# UI
# =========================================================
def show_recommendation_engine_page() -> None:
    st.title("🎬 Recommendation Engine")
    st.write(
        "This module recommends similar movies based on genres, cast, director, and plot overview."
    )

    with st.expander("Wie funktioniert dieser Recommender?"):
        st.markdown(
            """
    ### Modelllogik einfach erklärt

    - Jeder Film wird in ein Textprofil („Feature Soup“) umgewandelt  

    **Dieses Profil enthält:**
    - Regisseur  
    - Hauptdarsteller  
    - Genres  
    - Inhaltsbeschreibung (Overview)  
    - Diese Features werden mit **TF-IDF (Term Frequency – Inverse Document Frequency)** vektorisiert  
    - Anschließend berechnet die App die **Cosine Similarity** zwischen Filmen  

    **Das finale Ranking berücksichtigt zusätzlich:**
    - Bewertung (Rating)  
    - Popularität  
    - Bonus für gleichen Regisseur  
    - Bonus für gemeinsame Schauspieler  

    ---

    ### Warum ist das besser als einfaches Keyword-Zählen?

    - TF-IDF reduziert den Einfluss sehr häufiger Wörter  
    - Regisseur, Cast und Genres werden stärker gewichtet als allgemeiner Plot-Text  
    - Exakte Übereinstimmungen in wichtigen Metadaten werden belohnt  
            """
        )

    with st.spinner("Loading recommendation engine..."):
        df = load_recommender_data()

    if df.empty:
        st.warning("No data could be loaded for the recommendation engine.")
        return

    df["release_year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year

    with st.spinner("Building recommendation model..."):
        df, indices, tfidf_matrix = build_recommender_engine(df)

    # =========================
    # SIDEBAR FILTERS
    # =========================
    st.sidebar.markdown("### Recommendation Filters")

    min_rating = st.sidebar.slider(
        "Minimum Rating",
        min_value=0.0,
        max_value=10.0,
        value=6.0,
        step=0.5,
        key="rec_min_rating",
    )

    valid_runtime = df["runtime"].dropna()
    max_runtime_available = int(valid_runtime.max()) if not valid_runtime.empty else 300

    selected_runtime = st.sidebar.slider(
        "Maximum Runtime",
        min_value=60,
        max_value=max_runtime_available,
        value=min(140, max_runtime_available),
        key="rec_runtime",
    )

    all_genres = sorted(
        {
            genre.strip()
            for genre_string in df["genres"].fillna("")
            for genre in str(genre_string).split("|")
            if genre.strip()
        }
    )

    selected_genres = st.sidebar.multiselect(
        "Limit by Genre",
        all_genres,
        key="rec_genres",
    )

    num_recommendations = st.sidebar.slider(
        "Number of Movies",
        min_value=5,
        max_value=20,
        value=10,
        key="rec_num",
    )

    # =========================
    # MOVIE SELECTION
    # =========================
    df["movie_label"] = df.apply(_build_movie_label, axis=1)

    movie_options = (
        df[["movie_id", "movie_label"]]
        .drop_duplicates()
        .sort_values("movie_label")
    )

    label_to_id = dict(zip(movie_options["movie_label"], movie_options["movie_id"]))

    selected_label = st.selectbox(
        "Pick a movie you love:",
        movie_options["movie_label"].tolist(),
        key="rec_selected_movie",
    )

    # =========================
    # RUN RECOMMENDER
    # =========================
    if st.button("Find Similar", key="rec_find_similar"):
        selected_movie_id = label_to_id[selected_label]

        rec_df = get_recommendations(
            df=df,
            indices=indices,
            tfidf_matrix=tfidf_matrix,
            selected_movie_id=selected_movie_id,
            min_rating=min_rating,
            max_runtime=selected_runtime,
            selected_genres=selected_genres,
            num_recommendations=num_recommendations,
        )

        if rec_df.empty:
            st.warning("No recommendations found. Try adjusting the filters.")
            return

        st.markdown("---")
        st.subheader(f"Recommendations for {selected_label}")

        cols_per_row = 5
        rows = [
            rec_df.iloc[i:i + cols_per_row]
            for i in range(0, len(rec_df), cols_per_row)
        ]

        for row_chunk in rows:
            cols = st.columns(cols_per_row)
            for col, (_, movie) in zip(cols, row_chunk.iterrows()):
                with col:
                    img_url = poster_url(movie.get("poster_path"))
                    if img_url:
                        st.image(img_url, use_container_width=True)
                    else:
                        st.markdown("**No Poster**")

                    release_year = movie.get("release_year")
                    year_suffix = f" ({int(release_year)})" if pd.notna(release_year) else ""

                    st.markdown(f"**{movie['title']}{year_suffix}**")
                    st.caption(f"Director: {_format_director_for_display(movie.get('director'))}")
                    st.caption(f"Genres: {_format_genres_for_display(movie.get('genres'))}")
                    st.caption(f"Rating: {format_number(movie.get('vote_average'), 1)}")
                    st.caption(f"Similarity: {movie.get('similarity', 0):.3f}")