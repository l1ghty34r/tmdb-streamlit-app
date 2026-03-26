import streamlit as st

from utils.movie_database import show_movie_database_page
from utils.recommendation import show_recommendation_engine_page
from utils.roi_builder import show_roi_builder_page

st.set_page_config(
    page_title="TMDB Project",
    page_icon="🎬",
    layout="wide",
)

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    [
        "Movie Database",
        "Recommendation Engine",
        "ROI Builder",
    ],
    index=0,
)

if page == "Movie Database":
    show_movie_database_page()
elif page == "Recommendation Engine":
    show_recommendation_engine_page()
else:
    show_roi_builder_page()