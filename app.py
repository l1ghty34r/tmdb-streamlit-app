import os
from typing import Optional

import pandas as pd
import psycopg2
import streamlit as st


st.set_page_config(
    page_title="TMDB Project App",
    page_icon="🎬",
    layout="wide",
)


# -----------------------------
# Helper functions
# -----------------------------
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


def poster_url(path_value: Optional[str]) -> Optional[str]:
    if not path_value or pd.isna(path_value):
        return None
    path_value = str(path_value).strip()
    if not path_value:
        return None
    return f"https://image.tmdb.org/t/p/w500{path_value}"


# -----------------------------
# Database connection
# -----------------------------
@st.cache_resource
def get_connection():
    database_url = None

    try:
        database_url = st.secrets["DATABASE_URL"]
    except Exception:
        database_url = os.getenv("DATABASE_URL")

    if database_url:
        return psycopg2.connect(database_url)

    return psycopg2.connect(
        host=os.getenv("PGHOST", "localhost"),
        port=os.getenv("PGPORT", "5432"),
        dbname=os.getenv("PGDATABASE", "postgres"),
        user=os.getenv("PGUSER", "postgres"),
        password=os.getenv("PGPASSWORD", "postgres"),
        sslmode=os.getenv("PGSSLMODE", "prefer"),
    )


@st.cache_data(ttl=300)
def run_query(query: str, params: Optional[tuple] = None) -> pd.DataFrame:
    conn = get_connection()
    return pd.read_sql_query(query, conn, params=params)


# -----------------------------
# UI: Navigation
# -----------------------------
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


# -----------------------------
# Page 1
# -----------------------------
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
            rm.poster_path,
            EXTRACT(YEAR FROM NULLIF(m.release_date, '')::date) AS release_year,
            EXISTS (
                SELECT 1
                FROM core.reviews r
                WHERE r.movie_id = m.movie_id
            ) AS has_reviews
        FROM core.movies m
        LEFT JOIN raw.movies rm
            ON m.movie_id = rm.id
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
              OR (rm.poster_path IS NOT NULL AND rm.poster_path <> '')
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

    # -----------------------------
    # Quick pick / cards
    # -----------------------------
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

    # -----------------------------
    # Detail view selector
    # -----------------------------
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

    # -----------------------------
    # Detail queries
    # -----------------------------
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
            rm.original_title,
            rm.overview,
            rm.poster_path,
            rm.backdrop_path,
            rm.status,
            rm.tagline,
            rm.homepage,
            rm.original_language,
            rm.adult,
            rm.video,
            rm.created_at,
            rm.updated_at
        FROM core.movies m
        LEFT JOIN raw.movies rm
            ON m.movie_id = rm.id
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
                    movie.get("adult") if pd.notna(movie.get("adult")) else "n/a",
                    movie.get("video") if pd.notna(movie.get("video")) else "n/a",
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

    display_results["Poster"] = display_results["poster_path"].apply(lambda x: "Ja" if pd.notna(x) and str(x).strip() else "Nein")
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


# -----------------------------
# Placeholder pages
# -----------------------------
def show_placeholder_page(title: str, text: str) -> None:
    st.title(title)
    st.info(text)


# -----------------------------
# Router
# -----------------------------
if page == "1. Film datenbank":
    show_film_database_page()
elif page == "2. Analysen mit Power BI":
    show_placeholder_page(
        "📊 Analysen mit Power BI",
        "Diese Seite wird später mit den Power-BI-Ergebnissen oder Screenshots ergänzt.",
    )
elif page == "3. ML Projekt 1":
    show_placeholder_page(
        "🤖 ML Projekt 1",
        "Diese Seite dient später zur Präsentation des ersten Machine-Learning-Projekts.",
    )
else:
    show_placeholder_page(
        "🤖 ML Projekt 2",
        "Diese Seite dient später zur Präsentation des zweiten Machine-Learning-Projekts.",
    )
