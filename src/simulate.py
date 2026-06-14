"""
simulate.py — Monte Carlo World Cup tournament simulation (optimized)

Key optimization: all form stats and ELO ratings are pre-computed once
before the simulation loop. Each simulation only does numpy random sampling
— no DataFrame lookups inside the loop.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

from src.utils import get_logger
from src.features import FEATURE_COLS

logger = get_logger(__name__)


# ── 2026 World Cup group draw ─────────────────────────────────────────────────
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


# ── Pre-compute team stats ────────────────────────────────────────────────────

def build_team_stats(
    teams: list[str],
    elo_ratings: dict[str, float],
    form_stats: dict[str, dict],
    use_real_elo: bool = True,
) -> dict[str, dict]:
    """
    Pre-compute a flat stats dict for every team.
    This runs ONCE before the simulation loop.
    Returns: {team_name: {elo, form_ppg, goals_scored, goals_conceded, win_rate, days_since_last}}
    """
    if use_real_elo:
        try:
            from src.real_elo import REAL_ELO_2026
            elo_ratings = {**elo_ratings, **REAL_ELO_2026}
        except ImportError:
            pass

    stats = {}
    for team in teams:
        f = form_stats.get(team, {})
        stats[team] = {
            "elo":              elo_ratings.get(team, 1500.0),
            "form_ppg":         f.get("form_ppg", 1.2),
            "goals_scored":     f.get("goals_scored", 1.3),
            "goals_conceded":   f.get("goals_conceded", 1.1),
            "win_rate":         f.get("win_rate", 0.4),
            "days_since_last":  f.get("days_since_last", 14),
        }
    return stats


# ── Fast feature vector (numpy, no DataFrame) ─────────────────────────────────

def fast_features(home: str, away: str, team_stats: dict[str, dict]) -> np.ndarray:
    """
    Build a feature vector as a numpy array — no DataFrame overhead.
    Order must match FEATURE_COLS exactly.
    """
    h = team_stats[home]
    a = team_stats[away]
    return np.array([[
        h["elo"],                                        # elo_home
        a["elo"],                                        # elo_away
        h["elo"] - a["elo"],                             # elo_diff
        h["form_ppg"],                                   # form_ppg_home
        h["goals_scored"],                               # goals_scored_home
        h["goals_conceded"],                             # goals_conceded_home
        h["win_rate"],                                   # win_rate_home
        h["days_since_last"],                            # days_since_last_home
        a["form_ppg"],                                   # form_ppg_away
        a["goals_scored"],                               # goals_scored_away
        a["goals_conceded"],                             # goals_conceded_away
        a["win_rate"],                                   # win_rate_away
        a["days_since_last"],                            # days_since_last_away
        h["form_ppg"] - a["form_ppg"],                  # form_diff
        h["goals_scored"] - a["goals_scored"],           # goals_scored_diff
        h["goals_conceded"] - a["goals_conceded"],       # goals_conceded_diff
        0.5,                                             # h2h_win_rate
        0.0,                                             # h2h_goal_diff
        1,                                               # is_neutral
        1.0,                                             # tournament_weight
    ]], dtype=np.float32)


# ── Pre-compute Poisson lambdas for every possible matchup ───────────────────

def precompute_lambdas(
    teams: list[str],
    team_stats: dict[str, dict],
    poisson_model,
) -> dict[tuple[str, str], tuple[float, float]]:
    """
    Pre-compute (lambda_home, lambda_away) for every team pair.
    Called ONCE before the simulation loop — eliminates model.predict() calls
    inside the loop entirely.

    Returns: {(home, away): (lam_home, lam_away)}
    """
    import pandas as pd
    logger.info(f"Pre-computing Poisson lambdas for {len(teams)} teams...")

    pairs, feature_rows = [], []
    for i, home in enumerate(teams):
        for j, away in enumerate(teams):
            if home == away:
                continue
            pairs.append((home, away))
            feature_rows.append(fast_features(home, away, team_stats)[0])

    X = pd.DataFrame(feature_rows, columns=FEATURE_COLS)
    lam_home_arr, lam_away_arr = poisson_model.predict_lambda(X)

    lambdas = {}
    for (home, away), lh, la in zip(pairs, lam_home_arr, lam_away_arr):
        lambdas[(home, away)] = (float(lh), float(la))

    logger.info(f"Pre-computed {len(lambdas):,} matchup lambdas ✓")
    return lambdas


# ── Fast single match simulation ──────────────────────────────────────────────

def fast_simulate_match(
    home: str,
    away: str,
    lambdas: dict[tuple[str, str], tuple[float, float]],
    allow_draw: bool = True,
) -> tuple[str, int, int]:
    """
    Simulate one match using pre-computed Poisson lambdas.
    Pure numpy — no model calls, no DataFrames.
    """
    lam_h, lam_a = lambdas[(home, away)]
    hg = np.random.poisson(lam_h)
    ag = np.random.poisson(lam_a)

    if hg > ag:
        return home, hg, ag
    elif ag > hg:
        return away, hg, ag
    else:
        if allow_draw:
            return "draw", hg, ag
        else:
            winner = home if np.random.random() < 0.5 else away
            return winner, hg, ag


# ── Group stage ───────────────────────────────────────────────────────────────

def fast_simulate_group(
    teams: list[str],
    lambdas: dict[tuple[str, str], tuple[float, float]],
) -> list[str]:
    records = {t: [0, 0, 0] for t in teams}  # [pts, gd, gs]

    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            home, away = teams[i], teams[j]
            result, hg, ag = fast_simulate_match(home, away, lambdas, allow_draw=True)
            records[home][1] += hg - ag
            records[away][1] += ag - hg
            records[home][2] += hg
            records[away][2] += ag
            if result == home:
                records[home][0] += 3
            elif result == away:
                records[away][0] += 3
            else:
                records[home][0] += 1
                records[away][0] += 1

    return sorted(teams, key=lambda t: tuple(records[t]), reverse=True)


# ── Full tournament (fast) ────────────────────────────────────────────────────

def fast_simulate_tournament(
    groups: dict[str, list[str]],
    lambdas: dict[tuple[str, str], tuple[float, float]],
) -> dict[str, str]:
    stages = {team: "group" for g in groups.values() for team in g}

    # Group stage
    group_standings = {}
    for gname, teams in groups.items():
        standings = fast_simulate_group(teams, lambdas)
        group_standings[gname] = standings
        for team in standings[2:]:
            stages[team] = "group_exit"

    # R16 bracket
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
            winner, _, _ = fast_simulate_match(home, away, lambdas, allow_draw=False)
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

def run_simulation(
    model,
    elo_ratings: dict[str, float],
    form_stats: dict[str, dict],
    groups: dict | None = None,
    n_simulations: int = 100_000,
    seed: int = 42,
    use_real_elo: bool = True,
) -> pd.DataFrame:
    """
    Run N Monte Carlo simulations.

    Optimizations vs naive version:
    - All form stats pre-computed before loop
    - All Poisson lambdas pre-computed before loop
    - Pure numpy inside the simulation loop
    - Expected time: ~30 seconds for 100,000 simulations
    """
    np.random.seed(seed)
    groups = groups or GROUPS_2026
    all_teams = [team for g in groups.values() for team in g]

    # ── Pre-compute everything once ──
    logger.info("Pre-computing team stats...")
    team_stats = build_team_stats(all_teams, elo_ratings, form_stats, use_real_elo)

    logger.info("Pre-computing Poisson lambdas for all matchups...")
    lambdas = precompute_lambdas(all_teams, team_stats, model.poisson)

    # ── Simulation loop (pure numpy, very fast) ──
    logger.info(f"Running {n_simulations:,} simulations...")
    counters = {team: defaultdict(int) for team in all_teams}

    for _ in tqdm(range(n_simulations), desc="Simulating"):
        result = fast_simulate_tournament(groups, lambdas)
        for team, stage in result.items():
            counters[team][stage] += 1

    # ── Aggregate results ──
    rows = []
    for team in all_teams:
        c = counters[team]
        t = n_simulations
        win     = c["winner"]
        final   = win     + c["final_exit"]
        semi    = final   + c["semifinal_exit"]
        quarter = semi    + c["quarterfinal_exit"]
        r16     = quarter + c["r16_exit"]
        rows.append({
            "team":             team,
            "win_pct":          win     / t * 100,
            "final_pct":        final   / t * 100,
            "semifinal_pct":    semi    / t * 100,
            "quarterfinal_pct": quarter / t * 100,
            "r16_pct":          r16     / t * 100,
        })

    return (
        pd.DataFrame(rows)
        .sort_values("win_pct", ascending=False)
        .reset_index(drop=True)
    )


# ── Convenience: simulate a single match (for dashboard/notebook use) ─────────

def simulate_match(
    home: str,
    away: str,
    model,
    elo_ratings: dict[str, float],
    form_stats: dict[str, dict],
    is_neutral: bool = True,
    allow_draw: bool = True,
) -> tuple[str, int, int]:
    """Single match simulation for ad-hoc use (e.g. dashboard match simulator)."""
    import pandas as pd
    from src.simulate import fast_features

    try:
        from src.real_elo import REAL_ELO_2026
        elo_ratings = {**elo_ratings, **REAL_ELO_2026}
    except ImportError:
        pass

    all_teams = list(set([home, away]))
    team_stats = build_team_stats(all_teams, elo_ratings, form_stats, use_real_elo=False)

    X = pd.DataFrame(fast_features(home, away, team_stats), columns=FEATURE_COLS)
    lam_h, lam_a = model.poisson.predict_lambda(X)
    hg = int(np.random.poisson(lam_h[0]))
    ag = int(np.random.poisson(lam_a[0]))

    if hg > ag:
        return home, hg, ag
    elif ag > hg:
        return away, hg, ag
    else:
        if allow_draw:
            return "draw", hg, ag
        winner = home if np.random.random() < 0.5 else away
        return winner, hg, ag
