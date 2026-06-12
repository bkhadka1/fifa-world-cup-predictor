"""
utils.py — shared helpers for the FIFA World Cup Prediction Model
"""

import os
import logging
import pandas as pd
import numpy as np


# ── Logging ──────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """Return a consistently-formatted logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(name)


# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def data_path(*parts: str) -> str:
    return os.path.join(ROOT, "data", *parts)

def model_path(*parts: str) -> str:
    return os.path.join(ROOT, "models", *parts)

def output_path(*parts: str) -> str:
    return os.path.join(ROOT, "outputs", *parts)


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_results(path: str | None = None) -> pd.DataFrame:
    """
    Load the main match results CSV.
    Expected columns: date, home_team, away_team, home_score, away_score,
                      tournament, city, country, neutral
    """
    path = path or data_path("raw", "results.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_shootouts(path: str | None = None) -> pd.DataFrame:
    """Load penalty shootout outcomes (used for knockout resolution)."""
    path = path or data_path("raw", "shootouts.csv")
    return pd.read_csv(path, parse_dates=["date"])


def load_rankings(path: str | None = None) -> pd.DataFrame:
    """
    Load FIFA ranking snapshots.
    Expected columns: rank_date, country_full, rank, total_points, ...
    """
    path = path or data_path("raw", "fifa_ranking.csv")
    df = pd.read_csv(path, parse_dates=["rank_date"])
    df = df.sort_values("rank_date").reset_index(drop=True)
    return df


# ── Match outcome helpers ─────────────────────────────────────────────────────

def match_outcome(home_score: int, away_score: int) -> str:
    """Return 'home', 'away', or 'draw'."""
    if home_score > away_score:
        return "home"
    elif away_score > home_score:
        return "away"
    return "draw"


def add_outcome_column(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a string outcome column and a numeric result (1=home, 0=draw, -1=away)."""
    df = df.copy()
    df["outcome"] = df.apply(
        lambda r: match_outcome(r["home_score"], r["away_score"]), axis=1
    )
    df["result"] = df["outcome"].map({"home": 1, "draw": 0, "away": -1})
    return df


# ── Date utilities ────────────────────────────────────────────────────────────

def filter_date_range(
    df: pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
    date_col: str = "date",
) -> pd.DataFrame:
    if start:
        df = df[df[date_col] >= pd.Timestamp(start)]
    if end:
        df = df[df[date_col] <= pd.Timestamp(end)]
    return df


# ── Normalisation ─────────────────────────────────────────────────────────────

# Canonical team names — add more as needed
TEAM_ALIASES: dict[str, str] = {
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "Türkiye": "Turkey",
    "Czech Republic": "Czechia",
    "USA": "United States",
    "Northern Ireland": "Northern Ireland",
}

def normalize_team(name: str) -> str:
    return TEAM_ALIASES.get(name.strip(), name.strip())


def normalize_teams(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["home_team", "away_team"]:
        if col in df.columns:
            df[col] = df[col].map(normalize_team)
    return df
