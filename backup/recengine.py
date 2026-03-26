import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page Config
st.set_page_config(page_title="Recommendation Engine", layout="wide")

# --- 1. Database Connection ---
@st.cache_data
def load_movie_data():
    db_url = st.secrets["DATABASE_URL"]
    engine = create_engine(db_url)
    
    # We use DISTINCT ON (title) to ensure each title appears only once.
    # We order by title and popularity so we keep the "main" version of a movie.
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
        SELECT mc.movie_id, STRING_AGG(p.name, ' ') as actors
        FROM core.movie_cast mc
        JOIN core.people p ON mc.person_id = p.person_id
        WHERE mc.cast_order <= 3
        GROUP BY mc.movie_id
    ),
    crew_info AS (
        SELECT mcr.movie_id, p.name as director
        FROM core.movie_crew mcr
        JOIN core.people p ON mcr.person_id = p.person_id
        WHERE mcr.job = 'Director'
        -- Ensure only one director per movie if data is messy
        GROUP BY mcr.movie_id, p.name 
    ),
    genre_info AS (
        SELECT mg.movie_id, STRING_AGG(g.genre_name, ' ') as genre_list
        FROM core.movie_genres mg
        JOIN core.genres g ON mg.genre_id = g.genre_id
        GROUP BY mg.movie_id
    )
    SELECT 
        um.*,
        COALESCE(gi.genre_list, '') as genres,
        COALESCE(ci.actors, '') as top_cast,
        COALESCE(cr.director, '') as director
    FROM unique_movies um
    LEFT JOIN genre_info gi ON um.movie_id = gi.movie_id
    LEFT JOIN cast_info ci ON um.movie_id = ci.movie_id
    LEFT JOIN crew_info cr ON um.movie_id = cr.movie_id;
    """
    df = pd.read_sql_query(query, engine)
    engine.dispose()
    return df

# --- 2. Engine Logic ---
@st.cache_resource
def build_engine(df):
    def clean_data(x):
        return str(x).lower().replace(" ", "")

    temp_cast = df['top_cast'].apply(clean_data)
    temp_director = df['director'].apply(clean_data)
    temp_genres = df['genres'].apply(clean_data)
    temp_overview = df['overview'].fillna('').str.lower()

    # Metadata Soup
    df['soup'] = temp_cast + ' ' + temp_director + ' ' + temp_director + ' ' + temp_genres + ' ' + temp_genres + ' ' + temp_overview

    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(df['soup'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    
    return indices, cosine_sim

# --- 3. Streamlit Interface ---
st.title("🎬 Recommendation Engine")

with st.spinner("Initializing Database..."):
    df = load_movie_data()
    indices, cosine_sim = build_engine(df)

# --- SIDEBAR FILTERS ---
st.sidebar.header("Filter Results")

# Rating Slider (0.0 to 10.0)
min_rating = st.sidebar.slider(
    "Minimum Rating (0-10):", 
    min_value=0.0, 
    max_value=10.0, 
    value=6.0, # Default to 6.0 for decent quality
    step=0.5
)

# Runtime Slider
max_runtime = int(df['runtime'].max()) if not df.empty else 300
selected_runtime = st.sidebar.slider(
    "Maximum Runtime (minutes):", 
    min_value=60, 
    max_value=max_runtime, 
    value=120  # Default to 2 hours
)
# Get a unique list of all genres available in the DB
all_genres = sorted(list(set([g for sublist in df['genres'].str.split() for g in sublist])))
selected_genres = st.sidebar.multiselect("Limit by Genre:", all_genres)

num_recommendations = st.sidebar.slider("Number of movies:", 5, 20, 10)

# --- MAIN PAGE ---
if not df.empty:
    selected_movie = st.selectbox("Pick a movie you love:", df['title'].values)

    if st.button('Find Similar'):
        with st.spinner(f"Analyzing '{selected_movie}'..."):
            if selected_movie in indices:
                # 1. Get the raw similarity scores
                idx = indices[selected_movie]
                sim_scores = list(enumerate(cosine_sim[idx]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                
                recommended_list = []
                seen_titles = set()
                seen_titles.add(selected_movie.lower().strip())
                # 2. Loop through the sorted scores (skipping the first one as it's the movie itself)
                for i, score in sim_scores[1:]:
                    movie_row = df.iloc[i]
                    current_title = str(movie_row['title']).lower().strip()
                    
                    # --- DUPLICATE TITLE FILTER ---
                    if current_title in seen_titles:
                        continue # Skip if we've already added this movie name

                    # --- 2. RATING FILTER (NEW) ---
                    if movie_row['vote_average'] < min_rating:
                        continue
                    # --- RUNTIME FILTER ---
                    # We check this first because it's a simple number comparison (very fast)
                    if movie_row['runtime'] > selected_runtime:
                        continue
                    # If the user has selected genres, we check them.
                    # If they HAVEN'T, we let everything through.
                    if selected_genres:
                        # Ensure we are comparing apples to apples (lowercase, no extra spaces)
                        movie_genres = [g.lower().strip() for g in str(movie_row['genres']).split()]
                        selected_lower = [s.lower().strip() for s in selected_genres]
                        
                        # Check: Does this movie have ANY of the genres the user picked?
                        if not any(genre in movie_genres for genre in selected_lower):
                            continue # Skip to the next movie in the similarity list
                    
                    # 3. If we got here, the movie passed (or there was no filter)
                    recommended_list.append(movie_row)
                    seen_titles.add(current_title)
                    # 4. Stop when we reach the user's limit
                    if len(recommended_list) >= num_recommendations:
                        break
                
                # --- DISPLAY RESULTS ---
                if len(recommended_list) > 0:
                    st.write("---")
                    st.subheader(f"Recommendations for {selected_movie}")
                    
                    cols = st.columns(5)
                    base_url = "https://image.tmdb.org/t/p/w500"

                    for i, row in enumerate(recommended_list):
                        with cols[i % 5]:
                            path = row['poster_path']
                            img_url = base_url + path if path else "https://via.placeholder.com/500x750?text=No+Poster"
                            st.image(img_url, use_container_width=True)
                            st.write(f"**{row['title']}**")
                            st.caption(f"Director: {row['director']}")
                else:
                    st.warning("No movies found. Try adjusting your genre filters or search for a different movie.")
            else:
                st.error("Movie not found in the index.")