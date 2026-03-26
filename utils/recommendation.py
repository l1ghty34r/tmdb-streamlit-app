import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.db import run_query
from utils.helpers import format_number, poster_url


@st.cache_data(ttl=3600)
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
            m.vote_average,
            m.vote_count
        FROM core.movies m
        ORDER BY m.title, m.popularity DESC
    ),
    cast_info AS (
        SELECT
            mc.movie_id,
            STRING_AGG(p.name, ' ' ORDER BY mc.cast_order ASC) AS actors
        FROM core.movie_cast mc
        JOIN core.people p ON mc.person_id = p.person_id
        WHERE mc.cast_order <= 5
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

    def clean_token_text(x):
        if pd.isna(x):
            return ""
        return str(x).lower().replace(" ", "")

    work_df["genres_clean"] = work_df["genres"].fillna("").apply(clean_token_text)
    work_df["cast_clean"] = work_df["top_cast"].fillna("").apply(clean_token_text)
    work_df["director_clean"] = work_df["director"].fillna("").apply(clean_token_text)
    work_df["overview_clean"] = work_df["overview"].fillna("").astype(str).str.lower()

    work_df["soup"] = (
        work_df["genres_clean"] + " "
        + work_df["genres_clean"] + " "
        + work_df["cast_clean"] + " "
        + work_df["director_clean"] + " "
        + work_df["director_clean"] + " "
        + work_df["overview_clean"]
    )

    tfidf = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8,
    )

    tfidf_matrix = tfidf.fit_transform(work_df["soup"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(work_df.index, index=work_df["title"]).drop_duplicates()

    return work_df, indices, cosine_sim


def show_recommendation_engine_page() -> None:
    st.title("🎬 Recommendation Engine")
    st.write(
        "Diese Recommendation Engine nutzt einen Content-Based Filtering Ansatz mit TF-IDF und Cosine Similarity. "
        "Filme werden auf Basis von Genre, Cast, Director und Overview miteinander verglichen."
    )

    with st.expander("Wie funktioniert die Recommendation Engine?"):
        st.markdown(
            """
Die Recommendation Engine basiert auf **Content-Based Filtering**.

Verwendet werden:
- **Genres**
- **Cast**
- **Director**
- **Overview**

Diese Informationen werden mit **TF-IDF** in numerische Vektoren umgewandelt.
Anschließend wird mit **Cosine Similarity** berechnet, welche Filme sich inhaltlich am stärksten ähneln.

Das ist kein klassisch trainiertes Machine-Learning-Modell mit Target Variable, sondern ein
**similarity-basiertes Recommender-System**.
            """
        )

    with st.spinner("Recommendation Engine wird geladen..."):
        df = load_recommender_data()
        df, indices, cosine_sim = build_recommender_engine(df)

    if df.empty:
        st.warning("Es konnten keine Daten für die Recommendation Engine geladen werden.")
        return

    st.sidebar.markdown("### Filter für die Recommendation Engine")

    min_rating = st.sidebar.slider(
        "Mindestbewertung",
        min_value=0.0,
        max_value=10.0,
        value=6.0,
        step=0.5,
        key="rec_min_rating",
    )

    min_votes = st.sidebar.slider(
        "Minimale Anzahl Votes",
        min_value=0,
        max_value=5000,
        value=100,
        step=50,
        key="rec_min_votes",
    )

    valid_runtime = df["runtime"].dropna()
    max_runtime = int(valid_runtime.max()) if not valid_runtime.empty else 300
    selected_runtime = st.sidebar.slider(
        "Maximale Laufzeit",
        min_value=60,
        max_value=max_runtime,
        value=min(140, max_runtime),
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
        "Genre eingrenzen",
        all_genres,
        key="rec_genres",
    )

    num_recommendations = st.sidebar.slider(
        "Anzahl Empfehlungen",
        5,
        20,
        10,
        key="rec_num",
    )

    selected_movie = st.selectbox(
        "Wähle einen Film:",
        df["title"].dropna().sort_values().unique(),
        key="rec_selected_movie",
    )

    if st.button("Ähnliche Filme finden", key="rec_find_similar"):
        if selected_movie not in indices:
            st.error("Film nicht im Recommendation Index gefunden.")
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
            vote_count = movie_row.get("vote_count")
            runtime_val = movie_row.get("runtime")

            if pd.notna(vote_avg) and float(vote_avg) < min_rating:
                continue

            if pd.notna(vote_count) and int(vote_count) < min_votes:
                continue

            if pd.notna(runtime_val) and float(runtime_val) > selected_runtime:
                continue

            if selected_genres:
                movie_genres = [g.lower().strip() for g in str(movie_row["genres"]).split()]
                selected_lower = [s.lower().strip() for s in selected_genres]
                if not any(genre in movie_genres for genre in selected_lower):
                    continue

            recommended_list.append((movie_row, float(score)))
            seen_titles.add(current_title)

            if len(recommended_list) >= num_recommendations:
                break

        if not recommended_list:
            st.warning("Keine passenden Empfehlungen gefunden. Passe die Filter an.")
            return

        st.markdown("---")
        st.subheader(f"Empfehlungen für: {selected_movie}")

        cols = st.columns(5)
        for i, (row, score) in enumerate(recommended_list):
            with cols[i % 5]:
                img_url = poster_url(row.get("poster_path"))
                if img_url:
                    st.image(img_url, use_container_width=True)
                else:
                    st.markdown("**Kein Poster**")

                st.write(f"**{row['title']}**")
                st.caption(f"Similarity Score: {format_number(score, 3)}")
                st.caption(f"Director: {row['director'] if pd.notna(row['director']) else 'n/a'}")
                st.caption(f"Genres: {row['genres'] if pd.notna(row['genres']) else 'n/a'}")
                st.caption(f"Rating: {format_number(row['vote_average'], 1)}")
                st.caption(f"Votes: {int(row['vote_count']) if pd.notna(row['vote_count']) else 'n/a'}")