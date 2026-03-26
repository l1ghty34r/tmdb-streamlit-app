import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.db import run_query
from utils.helpers import format_number, poster_url


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