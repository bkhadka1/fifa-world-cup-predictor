"""
simulate.py — Monte Carlo World Cup tournament simulation

How it works:
  1. Load the trained ensemble model
  2. For each simulated tournament:
     a. Simulate group stage — each match played probabilistically
     b. Rank groups, determine knockout qualifiers
     c. Simulate knockout rounds (R16 → QF → SF → Final)
  3. Repeat N times (default 100,000)
  4. Aggregate win/finalist/semifinal probabilities per team
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

from src.utils import get_logger
from src.features import FEATURE_COLS

logger = get_logger(__name__)



# Official draw confirmed December 5, 2025 — Washington DC
# Tournament started June 11, 2026

GROUPS_2026: dict = {
    "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}


# ── Feature builder for a single match ───────────────────────────────────────

def match_features(
    home, away, elo_ratings, form_stats,
    is_neutral=True, tournament_weight=1.0
):
    elo_home = elo_ratings.get(home, 1500.0)
    elo_away = elo_ratings.get(away, 1500.0)

    def f(team, key, default):
        return form_stats.get(team, {}).get(key, default)

    row = {
        "elo_home":             elo_home,
        "elo_away":             elo_away,
        "elo_diff":             elo_home - elo_away,
        "form_ppg_home":        f(home, "form_ppg", 1.2),
        "goals_scored_home":    f(home, "goals_scored", 1.3),
        "goals_conceded_home":  f(home, "goals_conceded", 1.1),
        "win_rate_home":        f(home, "win_rate", 0.4),
        "days_since_last_home": f(home, "days_since_last", 14),
        "form_ppg_away":        f(away, "form_ppg", 1.2),
        "goals_scored_away":    f(away, "goals_scored", 1.3),
        "goals_conceded_away":  f(away, "goals_conceded", 1.1),
        "win_rate_away":        f(away, "win_rate", 0.4),
        "days_since_last_away": f(away, "days_since_last", 14),
        "form_diff":            f(home, "form_ppg", 1.2) - f(away, "form_ppg", 1.2),
        "goals_scored_diff":    f(home, "goals_scored", 1.3) - f(away, "goals_scored", 1.3),
        "goals_conceded_diff":  f(home, "goals_conceded", 1.1) - f(away, "goals_conceded", 1.1),
        "h2h_win_rate":         0.5,
        "h2h_goal_diff":        0.0,
        "is_neutral":           int(is_neutral),
        "tournament_weight":    tournament_weight,
    }
    return pd.DataFrame([row])[FEATURE_COLS]


# ── Single match simulation ───────────────────────────────────────────────────

def simulate_match(home, away, model, elo_ratings, form_stats,
                   is_neutral=True, allow_draw=True):
    X = match_features(home, away, elo_ratings, form_stats, is_neutral)
    lam_home, lam_away = model.poisson.predict_lambda(X)
    home_goals = int(np.random.poisson(lam_home[0]))
    away_goals = int(np.random.poisson(lam_away[0]))

    if home_goals > away_goals:
        return home, home_goals, away_goals
    elif away_goals > home_goals:
        return away, home_goals, away_goals
    else:
        if allow_draw:
            return "draw", home_goals, away_goals
        else:
            winner = home if np.random.random() < 0.5 else away
            return winner, home_goals, away_goals


# ── Group stage ───────────────────────────────────────────────────────────────

def simulate_group(teams, model, elo_ratings, form_stats):
    records = {t: {"pts": 0, "gd": 0, "gs": 0} for t in teams}
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            home, away = teams[i], teams[j]
            result, hg, ag = simulate_match(
                home, away, model, elo_ratings, form_stats,
                is_neutral=True, allow_draw=True
            )
            records[home]["gs"] += hg
            records[away]["gs"] += ag
            records[home]["gd"] += hg - ag
            records[away]["gd"] += ag - hg
            if result == home:
                records[home]["pts"] += 3
            elif result == away:
                records[away]["pts"] += 3
            else:
                records[home]["pts"] += 1
                records[away]["pts"] += 1

    return sorted(
        teams,
        key=lambda t: (records[t]["pts"], records[t]["gd"], records[t]["gs"]),
        reverse=True,
    )


# ── Full tournament simulation ────────────────────────────────────────────────

def simulate_tournament(groups, model, elo_ratings, form_stats):
    stages = {team: "group" for g in groups.values() for team in g}

    # Group stage
    group_standings = {}
    for gname, teams in groups.items():
        standings = simulate_group(teams, model, elo_ratings, form_stats)
        group_standings[gname] = standings
        for team in standings[2:]:
            stages[team] = "group_exit"

    # Build R16: top 2 from each group, pair adjacent groups
    group_keys = sorted(group_standings.keys())
    qualifiers = {g: group_standings[g][:2] for g in group_keys}

    r16_matches = []
    for i in range(0, len(group_keys), 2):
        if i + 1 < len(group_keys):
            g1, g2 = group_keys[i], group_keys[i + 1]
            r16_matches.append((qualifiers[g1][0], qualifiers[g2][1]))
            r16_matches.append((qualifiers[g2][0], qualifiers[g1][1]))

    # Knockout rounds
    round_names = ["r16", "quarterfinal", "semifinal", "final"]
    current_round = r16_matches

    for round_name in round_names:
        if not current_round:
            break
        winners = []
        for home, away in current_round:
            winner, _, _ = simulate_match(
                home, away, model, elo_ratings, form_stats,
                is_neutral=True, allow_draw=False
            )
            winners.append(winner)
            loser = away if winner == home else home
            stages[loser] = round_name + "_exit"

        if round_name == "final":
            for w in winners:
                stages[w] = "winner"
        else:
            current_round = [
                (winners[i], winners[i + 1])
                for i in range(0, len(winners) - 1, 2)
            ]

    return stages


# ── Monte Carlo runner ────────────────────────────────────────────────────────

def run_simulation(model, elo_ratings, form_stats,
                   groups=None, n_simulations=100_000, seed=42):
    np.random.seed(seed)
    groups = groups or GROUPS_2026
    all_teams = [team for g in groups.values() for team in g]
    counters = {team: defaultdict(int) for team in all_teams}

    for _ in tqdm(range(n_simulations), desc="Simulating tournaments"):
        result = simulate_tournament(groups, model, elo_ratings, form_stats)
        for team, stage in result.items():
            counters[team][stage] += 1

    rows = []
    for team in all_teams:
        c = counters[team]
        t = n_simulations
        win        = c["winner"]
        final      = win + c["final_exit"]
        semi       = final + c["semifinal_exit"]
        quarter    = semi + c["quarterfinal_exit"]
        r16        = quarter + c["r16_exit"]
        rows.append({
            "team":             team,
            "win_pct":          win / t * 100,
            "final_pct":        final / t * 100,
            "semifinal_pct":    semi / t * 100,
            "quarterfinal_pct": quarter / t * 100,
            "r16_pct":          r16 / t * 100,
        })

    return (
        pd.DataFrame(rows)
        .sort_values("win_pct", ascending=False)
        .reset_index(drop=True)
    )
