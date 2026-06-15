"""
simulate.py — Monte Carlo World Cup 2026 simulation

Official FIFA 2026 format (from official bracket):
  - 48 teams, 12 groups of 4
  - Top 2 from each group + 8 best 3rd-place teams = 32 teams
  - Round of 32 (Last 32): 16 matches → 16 winners
  - Round of 16 (Last 16): 8 matches → 8 winners
  - Quarter-finals: 4 matches → 4 winners
  - Semi-finals: 2 matches → 2 winners
  - Final: 1 match → champion

Official bracket pairings (from CBS Sports official wall chart):
  Left side:
    R32: 1E vs best3, 1A vs 2C, 1F vs 2C, 2K vs 2L, 1H vs 1J, 1D vs 2E, 1G vs 2F, 1B vs 1C... 
  
  Simplified: we use seeded bracket where:
  - 12 group winners (1st place) are top seeds
  - 12 group runners-up (2nd place) are mid seeds  
  - 8 best 3rd-place teams are lowest seeds
  - Bracket pairs: 1st vs 3rd-place qualifiers, 2nd vs 2nd
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

from src.utils import get_logger
from src.features import FEATURE_COLS

logger = get_logger(__name__)


# ── 2026 World Cup groups ─────────────────────────────────────────────────────

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

# Official R32 bracket pairings from CBS Sports wall chart
# Format: (team1_source, team2_source)
# 1X = 1st place group X, 2X = 2nd place group X, 3rd = best 3rd place
# Left side of bracket feeds into SF1, right side into SF2
BRACKET_R32 = [
    # Left side (→ Semifinal 1)
    ("1E", "3ABC"),   # R32-77
    ("1A", "2B"),     # R32-74  → feeds into R16-89
    ("2A", "2C"),     # R32-73
    ("1B", "3DEF"),   # R32-77  → feeds into R16-89
    ("1F", "2C"),     # R32-75
    ("1C", "3ABCD"),  # R32-75  → feeds into R16-90
    ("2K", "2L"),     # R32-83
    ("1H", "1J"),     # R32-84  → feeds into R16-93/94

    # Right side (→ Semifinal 2)
    ("1C", "2F"),     # R32-78  → simplified
    ("1G", "3GHI"),   # R32-79
    ("1A", "2D"),     # R32-80
    ("1L", "3JKL"),   # R32-86
    ("1I", "2J"),     # R32-85  → simplified  
    ("1D", "2E"),     # R32-87
    ("1K", "2L"),     # R32-88
    ("1B", "2A"),     # R32-81  → simplified
]


# ── Pre-compute team stats ────────────────────────────────────────────────────

def build_team_stats(teams, elo_ratings, form_stats, use_real_elo=True):
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
            "elo":             elo_ratings.get(team, 1500.0),
            "form_ppg":        f.get("form_ppg", 1.2),
            "goals_scored":    f.get("goals_scored", 1.3),
            "goals_conceded":  f.get("goals_conceded", 1.1),
            "win_rate":        f.get("win_rate", 0.4),
            "days_since_last": f.get("days_since_last", 14),
        }
    return stats


# ── Fast feature vector ───────────────────────────────────────────────────────

def fast_features(home, away, team_stats):
    h = team_stats[home]
    a = team_stats[away]
    return np.array([[
        h["elo"], a["elo"], h["elo"] - a["elo"],
        h["form_ppg"], h["goals_scored"], h["goals_conceded"],
        h["win_rate"], h["days_since_last"],
        a["form_ppg"], a["goals_scored"], a["goals_conceded"],
        a["win_rate"], a["days_since_last"],
        h["form_ppg"] - a["form_ppg"],
        h["goals_scored"] - a["goals_scored"],
        h["goals_conceded"] - a["goals_conceded"],
        0.5, 0.0, 1, 1.0,
    ]], dtype=np.float32)


# ── Pre-compute Poisson lambdas ───────────────────────────────────────────────

def precompute_lambdas(teams, team_stats, poisson_model):
    logger.info(f"Pre-computing Poisson lambdas for {len(teams)} teams...")
    pairs, rows = [], []
    for home in teams:
        for away in teams:
            if home != away:
                pairs.append((home, away))
                rows.append(fast_features(home, away, team_stats)[0])
    X = pd.DataFrame(rows, columns=FEATURE_COLS)
    lam_h, lam_a = poisson_model.predict_lambda(X)
    lambdas = {pair: (float(h), float(a)) for pair, h, a in zip(pairs, lam_h, lam_a)}
    logger.info(f"Pre-computed {len(lambdas):,} matchup lambdas ✓")
    return lambdas


# ── Match simulation ──────────────────────────────────────────────────────────

def fast_simulate_match(home, away, lambdas, allow_draw=True):
    lam_h, lam_a = lambdas[(home, away)]
    hg = np.random.poisson(lam_h)
    ag = np.random.poisson(lam_a)
    if hg > ag:      return home, hg, ag
    elif ag > hg:    return away, hg, ag
    elif allow_draw: return "draw", hg, ag
    else:            return (home if np.random.random() < 0.5 else away), hg, ag


# ── Group stage ───────────────────────────────────────────────────────────────

def fast_simulate_group(teams, lambdas):
    """Round-robin. Returns [1st, 2nd, 3rd, 4th]."""
    rec = {t: [0, 0, 0] for t in teams}  # pts, gd, gs
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            h, a = teams[i], teams[j]
            result, hg, ag = fast_simulate_match(h, a, lambdas, allow_draw=True)
            rec[h][1] += hg - ag; rec[a][1] += ag - hg
            rec[h][2] += hg;      rec[a][2] += ag
            if result == h:       rec[h][0] += 3
            elif result == a:     rec[a][0] += 3
            else:                 rec[h][0] += 1; rec[a][0] += 1
    return sorted(teams, key=lambda t: tuple(rec[t]), reverse=True)


# ── Select best 8 third-place teams ──────────────────────────────────────────

def best_third_place(third_place_teams, team_stats):
    """Rank 3rd-place teams by ELO and return top 8."""
    ranked = sorted(third_place_teams, key=lambda t: team_stats[t]["elo"], reverse=True)
    return ranked[:8]


# ── Knockout round helper ─────────────────────────────────────────────────────

def play_round(matches, lambdas, stages, stage_name):
    """Play a list of (home, away) matches. Returns winners."""
    winners = []
    for home, away in matches:
        winner, _, _ = fast_simulate_match(home, away, lambdas, allow_draw=False)
        loser = away if winner == home else home
        stages[loser] = stage_name + "_exit"
        winners.append(winner)
    return winners


# ── Full tournament ───────────────────────────────────────────────────────────

def fast_simulate_tournament(groups, lambdas, team_stats):
    """
    Official FIFA 2026 format:
      - 12 groups → top 2 + best 8 third-place = 32 teams
      - R32 (Last 32): 16 matches → 16 winners
      - R16 (Last 16): 8 matches → 8 winners
      - QF → SF → Final
    
    Bracket: fixed seeding based on group finish.
    1st place teams are top seeds, 2nd place mid seeds, 3rd place lowest seeds.
    Bracket pairs: 1st place of one group vs 2nd/3rd of another group.
    Adjacent groups are paired: A&B, C&D, E&F, G&H, I&J, K&L
    """
    stages = {team: "group" for g in groups.values() for team in g}

    # ── Group stage ──
    group_standings = {}
    group_keys = sorted(groups.keys())

    for gname in group_keys:
        standings = fast_simulate_group(groups[gname], lambdas)
        group_standings[gname] = standings
        # 4th place always eliminated
        stages[standings[3]] = "group_exit"

    # Collect finishers
    first  = {g: group_standings[g][0] for g in group_keys}
    second = {g: group_standings[g][1] for g in group_keys}
    third  = {g: group_standings[g][2] for g in group_keys}

    # Best 8 third-place teams advance, rest eliminated
    all_third = [third[g] for g in group_keys]
    best8_third = best_third_place(all_third, team_stats)
    for t in all_third:
        if t not in best8_third:
            stages[t] = "group_exit"

    # ── Build R32 bracket: exactly 16 matches, 32 teams ──
    # 24 teams from top 2 per group + 8 best 3rd place = 32 teams
    # Bracket: pair adjacent groups crossover + 3rd place vs weakest 1st place
    #
    # 6 group pairs × 2 crossover matches = 12 matches (24 teams)
    # 8 best 3rd place teams vs 8 weakest 1st place = but 1st places already used
    # So: 3rd place teams vs 2nd place of non-adjacent groups
    #
    # SIMPLE CLEAN APPROACH matching CBS bracket:
    # Match 1-12:  1st(A) vs 2nd(B), 1st(B) vs 2nd(A), ... for all 6 pairs
    # Match 13-16: 4 best 3rd place vs 4 worst 2nd place (from remaining groups)
    # Match 17-20 would be remaining 3rd place — but we only need 16 total
    # So we pair all 8 best 3rd place teams into 4 play-in matches among themselves
    # Winners of those 4 play-in matches fill slots 13-16

    # ── Build R32 bracket: exactly 16 matches, 32 teams ──
    # No play-in matches — all 32 teams enter R32 directly.
    # 1. Top 8 Group Winners vs 8 best 3rd-place (strongest 1st vs weakest 3rd)
    # 2. Bottom 4 Group Winners vs bottom 4 Group Runners-up
    # 3. Top 8 Group Runners-up play each other (4 matches)
    # Total: 8 + 4 + 4 = 16 matches

    r32_matches = []

    ranked_firsts  = sorted([first[g]  for g in group_keys], key=lambda t: team_stats[t]["elo"], reverse=True)
    ranked_seconds = sorted([second[g] for g in group_keys], key=lambda t: team_stats[t]["elo"], reverse=True)

    # 1. Top 8 Group Winners vs 8 best 3rd-place (strongest vs weakest — bracket seeding)
    best8_third_reversed = best8_third[::-1]
    for i in range(8):
        r32_matches.append((ranked_firsts[i], best8_third_reversed[i]))

    # 2. Bottom 4 Group Winners vs bottom 4 Group Runners-up
    bottom4_firsts  = ranked_firsts[8:]
    bottom4_seconds = ranked_seconds[8:][::-1]
    for i in range(4):
        r32_matches.append((bottom4_firsts[i], bottom4_seconds[i]))

    # 3. Top 8 Group Runners-up play each other
    top8_seconds = ranked_seconds[:8]
    for i in range(0, 8, 2):
        r32_matches.append((top8_seconds[i], top8_seconds[i+1]))

    assert len(r32_matches) == 16, f"Expected 16 R32 matches, got {len(r32_matches)}"

    # ── Play R32: 16 matches → 16 winners ──
    r32_winners = play_round(r32_matches, lambdas, stages, "r32")

    # ── R16: pair consecutive winners → 8 matches → 8 winners ──
    r16_matches = [(r32_winners[i], r32_winners[i+1]) for i in range(0, 16, 2)]
    r16_winners = play_round(r16_matches, lambdas, stages, "r16")

    # ── QF: 8 → 4 ──
    qf_matches = [(r16_winners[i], r16_winners[i+1]) for i in range(0, 8, 2)]
    qf_winners = play_round(qf_matches, lambdas, stages, "quarterfinal")

    # ── SF: 4 → 2 ──
    sf_matches = [(qf_winners[0], qf_winners[1]), (qf_winners[2], qf_winners[3])]
    sf_winners = play_round(sf_matches, lambdas, stages, "semifinal")

    # ── Final: 2 → 1 ──
    final_winner = play_round([(sf_winners[0], sf_winners[1])], lambdas, stages, "final")
    stages[final_winner[0]] = "winner"

    return stages


# ── Monte Carlo runner ────────────────────────────────────────────────────────

def run_simulation(model, elo_ratings, form_stats,
                   groups=None, n_simulations=100_000,
                   seed=42, use_real_elo=True):
    np.random.seed(seed)
    groups = groups or GROUPS_2026
    all_teams = [t for g in groups.values() for t in g]

    logger.info("Pre-computing team stats...")
    team_stats = build_team_stats(all_teams, elo_ratings, form_stats, use_real_elo)

    logger.info("Pre-computing Poisson lambdas...")
    lambdas = precompute_lambdas(all_teams, team_stats, model.poisson)

    logger.info(f"Running {n_simulations:,} simulations...")
    counters = {t: defaultdict(int) for t in all_teams}

    for _ in tqdm(range(n_simulations), desc="Simulating"):
        result = fast_simulate_tournament(groups, lambdas, team_stats)
        for team, stage in result.items():
            counters[team][stage] += 1

    rows = []
    for team in all_teams:
        c = counters[team]
        t = n_simulations
        win     = c["winner"]
        final   = win     + c["final_exit"]
        semi    = final   + c["semifinal_exit"]
        quarter = semi    + c["quarterfinal_exit"]
        r16     = quarter + c["r16_exit"]
        r32     = r16     + c["r32_exit"]
        rows.append({
            "team":             team,
            "win_pct":          win     / t * 100,
            "final_pct":        final   / t * 100,
            "semifinal_pct":    semi    / t * 100,
            "quarterfinal_pct": quarter / t * 100,
            "r16_pct":          r16     / t * 100,
            "r32_pct":          r32     / t * 100,
        })

    return (
        pd.DataFrame(rows)
        .sort_values("win_pct", ascending=False)
        .reset_index(drop=True)
    )


# ── Single match helper ───────────────────────────────────────────────────────

def simulate_match(home, away, model, elo_ratings, form_stats,
                   is_neutral=True, allow_draw=True):
    try:
        from src.real_elo import REAL_ELO_2026
        elo_ratings = {**elo_ratings, **REAL_ELO_2026}
    except ImportError:
        pass
    team_stats = build_team_stats([home, away], elo_ratings, form_stats, use_real_elo=False)
    X = pd.DataFrame(fast_features(home, away, team_stats), columns=FEATURE_COLS)
    lam_h, lam_a = model.poisson.predict_lambda(X)
    hg = int(np.random.poisson(lam_h[0]))
    ag = int(np.random.poisson(lam_a[0]))
    if hg > ag:      return home, hg, ag
    elif ag > hg:    return away, hg, ag
    elif allow_draw: return "draw", hg, ag
    else:            return (home if np.random.random() < 0.5 else away), hg, ag
