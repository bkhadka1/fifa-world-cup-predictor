"""
dashboard/app.py — FIFA World Cup 2026 Prediction Dashboard

Run with:
    streamlit run dashboard/app.py

Two tabs:
  🏆 Predictions  — championship probabilities, group breakdown, match simulator
  📊 Model Stats  — ELO ratings, feature importance, model comparison
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib

from src.simulate import GROUPS_2026, simulate_match
from dashboard.train_on_startup import get_or_train_model
from src.elo import EloSystem
from src.features import compute_form

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="WC 2026 Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Bebas+Neue&display=swap');

  html, body, [class*="css"] {
      font-family: 'Inter', sans-serif;
  }

  /* Dark football-pitch background */
  .stApp {
      background-color: #0a0e1a;
      color: #e8eaf0;
  }

  /* Hero header */
  .hero {
      background: linear-gradient(135deg, #0d1b2a 0%, #1a2f4a 50%, #0d1b2a 100%);
      border-bottom: 2px solid #2ecc71;
      padding: 2rem 2.5rem 1.5rem;
      margin: -1rem -1rem 2rem -1rem;
  }
  .hero-title {
      font-family: 'Bebas Neue', sans-serif;
      font-size: 3.2rem;
      letter-spacing: 0.08em;
      color: #ffffff;
      line-height: 1;
      margin: 0;
  }
  .hero-subtitle {
      font-size: 0.85rem;
      color: #2ecc71;
      letter-spacing: 0.2em;
      text-transform: uppercase;
      margin-top: 0.4rem;
  }
  .hero-badge {
      display: inline-block;
      background: #2ecc71;
      color: #0a0e1a;
      font-size: 0.7rem;
      font-weight: 700;
      letter-spacing: 0.15em;
      text-transform: uppercase;
      padding: 0.25rem 0.6rem;
      border-radius: 2px;
      margin-top: 0.6rem;
  }

  /* Metric cards */
  .metric-card {
      background: #111827;
      border: 1px solid #1e2d40;
      border-radius: 8px;
      padding: 1.1rem 1.3rem;
      text-align: center;
  }
  .metric-label {
      font-size: 0.7rem;
      color: #6b7280;
      letter-spacing: 0.15em;
      text-transform: uppercase;
      margin-bottom: 0.3rem;
  }
  .metric-value {
      font-family: 'Bebas Neue', sans-serif;
      font-size: 2.2rem;
      color: #2ecc71;
      line-height: 1;
  }
  .metric-sub {
      font-size: 0.75rem;
      color: #9ca3af;
      margin-top: 0.2rem;
  }

  /* Section headers */
  .section-label {
      font-size: 0.7rem;
      color: #2ecc71;
      letter-spacing: 0.2em;
      text-transform: uppercase;
      font-weight: 600;
      margin-bottom: 0.8rem;
      border-bottom: 1px solid #1e2d40;
      padding-bottom: 0.5rem;
  }

  /* Team row in predictions */
  .team-row {
      display: flex;
      align-items: center;
      padding: 0.55rem 0;
      border-bottom: 1px solid #1a2235;
  }
  .team-rank {
      font-family: 'Bebas Neue', sans-serif;
      font-size: 1.1rem;
      color: #374151;
      width: 2rem;
  }
  .team-name {
      flex: 1;
      font-weight: 500;
      font-size: 0.9rem;
  }
  .team-pct {
      font-family: 'Bebas Neue', sans-serif;
      font-size: 1rem;
      color: #2ecc71;
      width: 3.5rem;
      text-align: right;
  }

  /* Tab styling */
  .stTabs [data-baseweb="tab-list"] {
      background: #111827;
      border-radius: 8px;
      padding: 4px;
      gap: 4px;
  }
  .stTabs [data-baseweb="tab"] {
      color: #6b7280;
      font-size: 0.85rem;
      font-weight: 500;
      border-radius: 6px;
      padding: 0.5rem 1.2rem;
  }
  .stTabs [aria-selected="true"] {
      background: #2ecc71 !important;
      color: #0a0e1a !important;
      font-weight: 700;
  }

  /* Match simulator */
  .match-result {
      background: #111827;
      border: 1px solid #1e2d40;
      border-radius: 8px;
      padding: 1.5rem;
      text-align: center;
  }
  .vs-text {
      font-family: 'Bebas Neue', sans-serif;
      font-size: 2.5rem;
      color: #2ecc71;
  }
  .prob-bar-label {
      font-size: 0.75rem;
      color: #9ca3af;
      margin-bottom: 0.2rem;
  }

  /* Hide streamlit branding */
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
  .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_predictions():
    path = os.path.join(os.path.dirname(__file__), "..", "outputs", "wc2026_predictions.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_data
def load_features():
    path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "features.csv")
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["date"])
    return None


@st.cache_resource
def load_model():
    try:
        return get_or_train_model()
    except Exception as e:
        st.warning(f"Model unavailable: {e}")
        return None


@st.cache_resource
def load_elo():
    matches_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "matches_modern.csv")
    if not os.path.exists(matches_path):
        return None, None
    matches = pd.read_csv(matches_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    matches = matches.dropna(subset=["home_score", "away_score"])
    elo = EloSystem(default_elo=1500, home_advantage=100)
    elo.fit(matches, verbose=False)

    today = pd.Timestamp("today")
    all_teams = [t for g in GROUPS_2026.values() for t in g]
    form_stats = {t: compute_form(matches, t, before_date=today, window=10) for t in all_teams}
    # Blend computed ELO with real-world ratings (real takes priority)
    elo.ratings = {**elo.ratings, **REAL_ELO_2026}
    return elo, form_stats


predictions  = load_predictions()
features_df  = load_features()
model        = load_model()
elo, form_stats = load_elo()

all_wc_teams = [t for g in GROUPS_2026.values() for t in g]

# ── Hero ──────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
  <div class="hero-subtitle">Machine Learning · Monte Carlo Simulation</div>
  <div class="hero-title">World Cup 2026<br>Predictor</div>
  <div class="hero-badge">⚽ 100,000 simulations · 48 teams · 4-model ensemble</div>
</div>
""", unsafe_allow_html=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2 = st.tabs(["🏆  Predictions", "📊  Model Stats"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:

    if predictions is None:
        st.warning("No predictions file found. Run `notebooks/04_simulation.ipynb` first to generate `outputs/wc2026_predictions.csv`.")
        st.stop()

    top3 = predictions.head(3)

    # ── Top 3 metric cards ──
    col1, col2, col3, col_spacer = st.columns([1, 1, 1, 1])
    medals = ["🥇", "🥈", "🥉"]
    for col, (_, row), medal in zip([col1, col2, col3], top3.iterrows(), medals):
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">{medal} Favourite</div>
              <div class="metric-value">{row['team']}</div>
              <div class="metric-sub">{row['win_pct']:.1f}% chance to win</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Two-column layout ──
    left, right = st.columns([1.1, 1], gap="large")

    with left:
        # Championship probability chart
        st.markdown('<div class="section-label">Championship Probability — Top 16</div>', unsafe_allow_html=True)
        top16 = predictions.head(16)

        colors = ["#FFD700" if i == 0 else "#C0C0C0" if i == 1 else "#CD7F32" if i == 2 else "#2ecc71"
                  for i in range(len(top16))]

        fig = go.Figure(go.Bar(
            x=top16["win_pct"],
            y=top16["team"],
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:.1f}%" for v in top16["win_pct"]],
            textposition="outside",
            textfont=dict(size=11, color="#e8eaf0"),
        ))
        fig.update_layout(
            plot_bgcolor="#111827",
            paper_bgcolor="#0a0e1a",
            font=dict(color="#e8eaf0", family="Inter"),
            margin=dict(l=10, r=60, t=10, b=10),
            xaxis=dict(
                showgrid=True, gridcolor="#1e2d40",
                zeroline=False, showticklabels=False,
            ),
            yaxis=dict(
                showgrid=False, autorange="reversed",
                tickfont=dict(size=12),
            ),
            height=430,
            bargap=0.35,
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        # Stage reach stacked chart
        st.markdown('<div class="section-label">Tournament Reach — Top 12</div>', unsafe_allow_html=True)
        top12 = predictions.head(12).copy()

        fig2 = go.Figure()
        stage_cols  = ["win_pct",   "final_pct",   "semifinal_pct",   "quarterfinal_pct", "r16_pct"]
        stage_labels = ["Champion", "Runner-up",   "Semifinal",       "Quarterfinal",     "Round of 16"]
        stage_colors = ["#FFD700",  "#C0C0C0",     "#2ecc71",         "#3b82f6",          "#374151"]

        # Convert to marginal probabilities
        marginals = pd.DataFrame()
        marginals["team"]      = top12["team"]
        marginals["champion"]  = top12["win_pct"]
        marginals["runner_up"] = top12["final_pct"]        - top12["win_pct"]
        marginals["semi"]      = top12["semifinal_pct"]    - top12["final_pct"]
        marginals["quarter"]   = top12["quarterfinal_pct"] - top12["semifinal_pct"]
        marginals["r16"]       = top12["r16_pct"]          - top12["quarterfinal_pct"]

        for col, label, color in zip(
            ["champion", "runner_up", "semi", "quarter", "r16"],
            stage_labels, stage_colors
        ):
            fig2.add_trace(go.Bar(
                name=label,
                y=marginals["team"],
                x=marginals[col],
                orientation="h",
                marker=dict(color=color),
                text=[f"{v:.0f}%" if v > 3 else "" for v in marginals[col]],
                textposition="inside",
                textfont=dict(size=9, color="#0a0e1a"),
            ))

        fig2.update_layout(
            barmode="stack",
            plot_bgcolor="#111827",
            paper_bgcolor="#0a0e1a",
            font=dict(color="#e8eaf0", family="Inter"),
            margin=dict(l=10, r=20, t=10, b=10),
            xaxis=dict(showgrid=True, gridcolor="#1e2d40", zeroline=False, title="Probability (%)"),
            yaxis=dict(showgrid=False, autorange="reversed", tickfont=dict(size=11)),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.01,
                xanchor="left", x=0,
                font=dict(size=10),
                bgcolor="rgba(0,0,0,0)",
            ),
            height=430,
            bargap=0.3,
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # ── Group breakdown ──
    st.markdown('<div class="section-label">Group-by-Group Breakdown</div>', unsafe_allow_html=True)

    group_cols = st.columns(4)
    group_items = list(GROUPS_2026.items())

    for i, (gname, teams) in enumerate(group_items):
        col = group_cols[i % 4]
        with col:
            group_data = predictions[predictions.team.isin(teams)].sort_values("win_pct", ascending=False)
            fig_g = go.Figure(go.Bar(
                x=group_data["team"],
                y=group_data["win_pct"],
                marker=dict(
                    color=["#FFD700", "#3b82f6", "#374151", "#374151"],
                    line=dict(width=0),
                ),
                text=[f"{v:.1f}%" for v in group_data["win_pct"]],
                textposition="outside",
                textfont=dict(size=10),
            ))
            fig_g.update_layout(
                title=dict(text=f"Group {gname}", font=dict(size=13, color="#e8eaf0"), x=0.5),
                plot_bgcolor="#111827",
                paper_bgcolor="#111827",
                font=dict(color="#e8eaf0", family="Inter"),
                margin=dict(l=5, r=5, t=30, b=5),
                xaxis=dict(showgrid=False, tickfont=dict(size=9)),
                yaxis=dict(showgrid=False, showticklabels=False),
                height=200,
                bargap=0.3,
            )
            st.plotly_chart(fig_g, use_container_width=True)

    st.markdown("---")

    # ── Match simulator ──
    st.markdown('<div class="section-label">Match Simulator</div>', unsafe_allow_html=True)

    sim_col1, sim_col2, sim_col3 = st.columns([1, 0.3, 1])

    with sim_col1:
        home_team = st.selectbox("Home team", sorted(all_wc_teams), index=sorted(all_wc_teams).index("Brazil") if "Brazil" in all_wc_teams else 0)
    with sim_col3:
        away_options = [t for t in sorted(all_wc_teams) if t != home_team]
        away_team = st.selectbox("Away team", away_options, index=away_options.index("Argentina") if "Argentina" in away_options else 0)
    with sim_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        simulate_btn = st.button("⚽  Simulate", use_container_width=True)

    if model and elo and form_stats:
        N_SIM = 5000
        h_wins = d_wins = a_wins = 0
        for _ in range(N_SIM):
            result, _, _ = simulate_match(
                home_team, away_team, model,
                elo.ratings, form_stats,
                is_neutral=True, allow_draw=True
            )
            if result == home_team:   h_wins += 1
            elif result == away_team: a_wins += 1
            else:                     d_wins += 1

        h_pct = h_wins / N_SIM * 100
        d_pct = d_wins / N_SIM * 100
        a_pct = a_wins / N_SIM * 100

        r1, r2, r3 = st.columns(3)
        for col, label, pct, color in [
            (r1, f"{home_team} win", h_pct, "#2ecc71"),
            (r2, "Draw",             d_pct, "#9ca3af"),
            (r3, f"{away_team} win", a_pct, "#3b82f6"),
        ]:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="metric-label">{label}</div>
                  <div class="metric-value" style="color:{color}">{pct:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

        # Prob bar
        fig_m = go.Figure(go.Bar(
            x=[h_pct, d_pct, a_pct],
            y=[home_team, "Draw", away_team],
            orientation="h",
            marker=dict(color=["#2ecc71", "#9ca3af", "#3b82f6"], line=dict(width=0)),
            text=[f"{v:.1f}%" for v in [h_pct, d_pct, a_pct]],
            textposition="outside",
            textfont=dict(size=12, color="#e8eaf0"),
        ))
        fig_m.update_layout(
            plot_bgcolor="#111827", paper_bgcolor="#0a0e1a",
            font=dict(color="#e8eaf0", family="Inter"),
            margin=dict(l=10, r=60, t=10, b=10),
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[0, 110]),
            yaxis=dict(showgrid=False, tickfont=dict(size=13)),
            height=160, bargap=0.4,
        )
        st.plotly_chart(fig_m, use_container_width=True)
    else:
        st.info("Run notebooks 01 and 02 to generate ELO ratings and enable the match simulator.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL STATS
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:

    left2, right2 = st.columns([1, 1], gap="large")

    with left2:
        # ELO ratings
        st.markdown('<div class="section-label">Current ELO Ratings — Top 25</div>', unsafe_allow_html=True)

        if elo:
            top25_elo = elo.top_n(25)
            colors_elo = ["#FFD700" if i == 0 else "#C0C0C0" if i == 1 else "#CD7F32" if i == 2 else "#2ecc71"
                          for i in range(len(top25_elo))]
            fig_elo = go.Figure(go.Bar(
                x=top25_elo.values,
                y=top25_elo.index,
                orientation="h",
                marker=dict(color=colors_elo, line=dict(width=0)),
                text=[f"{v:.0f}" for v in top25_elo.values],
                textposition="outside",
                textfont=dict(size=10, color="#e8eaf0"),
            ))
            fig_elo.add_vline(x=1500, line_dash="dot", line_color="#374151", annotation_text="avg", annotation_font_color="#6b7280")
            fig_elo.update_layout(
                plot_bgcolor="#111827", paper_bgcolor="#0a0e1a",
                font=dict(color="#e8eaf0", family="Inter"),
                margin=dict(l=10, r=60, t=10, b=10),
                xaxis=dict(showgrid=True, gridcolor="#1e2d40", zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, autorange="reversed", tickfont=dict(size=11)),
                height=560, bargap=0.3,
            )
            st.plotly_chart(fig_elo, use_container_width=True)
        else:
            st.info("Run notebooks 01 and 02 to generate ELO ratings.")

    with right2:
        # Feature importance
        st.markdown('<div class="section-label">Feature Importance (XGBoost)</div>', unsafe_allow_html=True)

        if model:
            try:
                from src.features import FEATURE_COLS
                importances = model.xgb.named_steps["clf"].feature_importances_
                feat_imp = pd.Series(importances, index=FEATURE_COLS).sort_values(ascending=False).head(15)
                colors_fi = ["#FFD700" if i < 3 else "#2ecc71" for i in range(len(feat_imp))]
                fig_fi = go.Figure(go.Bar(
                    x=feat_imp.values,
                    y=feat_imp.index,
                    orientation="h",
                    marker=dict(color=colors_fi, line=dict(width=0)),
                    text=[f"{v:.3f}" for v in feat_imp.values],
                    textposition="outside",
                    textfont=dict(size=10, color="#e8eaf0"),
                ))
                fig_fi.update_layout(
                    plot_bgcolor="#111827", paper_bgcolor="#0a0e1a",
                    font=dict(color="#e8eaf0", family="Inter"),
                    margin=dict(l=10, r=70, t=10, b=10),
                    xaxis=dict(showgrid=True, gridcolor="#1e2d40", zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, autorange="reversed", tickfont=dict(size=11)),
                    height=340, bargap=0.35,
                )
                st.plotly_chart(fig_fi, use_container_width=True)
            except Exception:
                st.info("Train the model first to see feature importance.")
        else:
            st.info("Run notebook 03 to train models.")

        # Model comparison table
        st.markdown('<div class="section-label">Model Performance on Test Set</div>', unsafe_allow_html=True)

        model_metrics_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "model_metrics.csv")
        if os.path.exists(model_metrics_path):
            metrics_df = pd.read_csv(model_metrics_path)
            st.dataframe(
                metrics_df.style
                    .format({"accuracy": "{:.3f}", "log_loss": "{:.3f}", "brier_score": "{:.3f}"})
                    .highlight_max(subset=["accuracy"], color="#1a3a2a")
                    .highlight_min(subset=["log_loss", "brier_score"], color="#1a3a2a"),
                use_container_width=True, hide_index=True,
            )
        else:
            # Show placeholder table
            placeholder = pd.DataFrame({
                "Model":       ["Poisson", "XGBoost", "Random Forest", "Logistic Reg.", "Ensemble"],
                "Accuracy":    ["—", "—", "—", "—", "—"],
                "Log Loss":    ["—", "—", "—", "—", "—"],
                "Brier Score": ["—", "—", "—", "—", "—"],
            })
            st.dataframe(placeholder, use_container_width=True, hide_index=True)
            st.caption("Run notebook 03 and save metrics to `outputs/model_metrics.csv` to populate this table.")

    # ELO history chart (full width)
    st.markdown("---")
    st.markdown('<div class="section-label">ELO Rating History — Top Nations</div>', unsafe_allow_html=True)

    if elo:
        elo_hist = elo.history_df()
        nations = ["Brazil", "France", "Argentina", "Germany", "Spain", "England"]
        nation_colors = ["#009C3B", "#002395", "#74ACDF", "#000000", "#AA151B", "#CF081F"]

        fig_hist = go.Figure()
        for team, color in zip(nations, nation_colors):
            home_d = elo_hist[elo_hist.home_team == team][["date", "elo_home_pre"]].rename(columns={"elo_home_pre": "elo"})
            away_d = elo_hist[elo_hist.away_team == team][["date", "elo_away_pre"]].rename(columns={"elo_away_pre": "elo"})
            combined = pd.concat([home_d, away_d]).sort_values("date")
            fig_hist.add_trace(go.Scatter(
                x=combined.date, y=combined.elo,
                name=team, line=dict(color=color, width=2),
                mode="lines", opacity=0.9,
            ))

        fig_hist.add_hline(y=1500, line_dash="dot", line_color="#374151")
        fig_hist.update_layout(
            plot_bgcolor="#111827", paper_bgcolor="#0a0e1a",
            font=dict(color="#e8eaf0", family="Inter"),
            margin=dict(l=10, r=20, t=10, b=10),
            xaxis=dict(showgrid=True, gridcolor="#1e2d40"),
            yaxis=dict(showgrid=True, gridcolor="#1e2d40", title="ELO Rating"),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
            height=320,
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("Run notebooks 01 and 02 to generate ELO history.")

    # Quick stats row
    st.markdown("---")
    st.markdown('<div class="section-label">Dataset Overview</div>', unsafe_allow_html=True)

    s1, s2, s3, s4, s5 = st.columns(5)
    if features_df is not None:
        for col, label, value in [
            (s1, "Total matches",   f"{len(features_df):,}"),
            (s2, "Teams",           f"{pd.concat([features_df.home_team, features_df.away_team]).nunique()}"),
            (s3, "Date range",      f"1990 – {features_df.date.dt.year.max()}"),
            (s4, "Features",        "20"),
            (s5, "Simulations",     "100,000"),
        ]:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="metric-label">{label}</div>
                  <div class="metric-value" style="font-size:1.4rem">{value}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Run notebooks 01–04 to populate dataset stats.")
