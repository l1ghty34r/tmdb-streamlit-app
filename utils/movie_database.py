import pandas as pd
import streamlit as st

from utils.db import run_query
from utils.helpers import (
    format_bool,
    format_date,
    format_int,
    format_money,
    format_number,
    poster_url,
)


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