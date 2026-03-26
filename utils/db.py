import os
from typing import Optional

import pandas as pd
import psycopg2
import streamlit as st


def get_database_url() -> str:
    try:
        database_url = st.secrets["DATABASE_URL"]
    except Exception:
        database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError("DATABASE_URL not found in Streamlit secrets or environment variables.")

    return database_url


@st.cache_data(ttl=300)
def run_query(query: str, params: Optional[tuple] = None) -> pd.DataFrame:
    database_url = get_database_url()
    with psycopg2.connect(database_url) as conn:
        return pd.read_sql_query(query, conn, params=params)