from math import log1p
from typing import List, Optional

import numpy as np
import pandas as pd


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


def format_bool(value) -> str:
    if pd.isna(value):
        return "n/a"
    return "Yes" if bool(value) else "No"


def poster_url(path_value: Optional[str]) -> Optional[str]:
    if not path_value or pd.isna(path_value):
        return None
    path_value = str(path_value).strip()
    if not path_value:
        return None
    return f"https://image.tmdb.org/t/p/w500{path_value}"


def safe_mean(values: List[float], fallback: float = 0.0) -> float:
    clean = [float(v) for v in values if pd.notna(v)]
    if not clean:
        return float(fallback)
    return float(np.mean(clean))


def safe_median(values: List[float], fallback: float = 0.0) -> float:
    clean = [float(v) for v in values if pd.notna(v)]
    if not clean:
        return float(fallback)
    return float(np.median(clean))


def safe_max(values: List[float], fallback: float = 0.0) -> float:
    clean = [float(v) for v in values if pd.notna(v)]
    if not clean:
        return float(fallback)
    return float(np.max(clean))


def weighted_roi_score(avg_roi: float, median_roi: float, success_rate: float, film_count: float) -> float:
    avg_roi = 0.0 if pd.isna(avg_roi) else float(avg_roi)
    median_roi = 0.0 if pd.isna(median_roi) else float(median_roi)
    success_rate = 0.0 if pd.isna(success_rate) else float(success_rate)
    film_count = 0.0 if pd.isna(film_count) else float(film_count)

    base = (0.35 * avg_roi) + (0.35 * median_roi) + (0.30 * success_rate)
    experience_factor = log1p(max(film_count, 0.0))
    return float(base * experience_factor)