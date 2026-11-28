"""
Streamlit frontend for the Football ML Prediction System (v2)
Run locally:  streamlit run app.py
Deploy:       push to GitHub → Streamlit Community Cloud
"""

from __future__ import annotations

import sys
import traceback
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

# Make project root importable
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.betting import DEFAULT_SETTINGS, apply_daily_cap, find_value_bets
from src.downloader import download_league_data, find_leagues_with_matches_today
from src.predictor import predict_league
from src.trainer import train_league_model
from src.utils import (
    PRED_DIR,
    get_all_leagues,
    load_leagues_config,
    load_manual_odds,
    save_active_leagues,
)

# ---------------------------------------------------------------------------
# Page config – mobile friendly
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Football Value Bets",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better mobile readability + sticky table header
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
    .stMetric {background: #0e1117; border-radius: 8px; padding: 8px 12px;}
    div[data-testid="stDataFrame"] {font-size: 0.85rem;}
    .value-bet-card {
        border: 1px solid #2e7d32;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 10px;
        background: linear-gradient(135deg, #0d1f12 0%, #0e1117 100%);
    }
    .no-bet {color: #888; font-style: italic;}

    /* Sticky header for the predictions table */
    div[data-testid="stDataFrame"] thead tr th {
        position: sticky;
        top: 0;
        z-index: 2;
        background-color: #1a2332 !important;
        color: #e0e0e0 !important;
        border-bottom: 2px solid #69f0ae !important;
    }
    div[data-testid="stDataFrame"] thead {
        position: sticky;
        top: 0;
        z-index: 2;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _today_utc() -> str:
    return datetime.now(timezone.utc).date().strftime("%Y-%m-%d")


@st.cache_data(show_spinner=False)
def load_all_league_options():
    """Return (all_league_dicts, name_to_league, default_selected_names)."""
    all_leagues = get_all_leagues("leagues.yaml")
    name_to_league = {lg["name"]: lg for lg in all_leagues}
    # Default selection = those marked enabled: true in the YAML
    default_names = [lg["name"] for lg in all_leagues if lg.get("enabled", True)]
    return all_leagues, name_to_league, default_names


def run_pipeline(
    selected_leagues: list[dict],
    download_fresh: bool,
    only_today: bool,
    target_date: str | None,
    progress_callback=None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Core pipeline. Only runs the leagues the user selected.
    If only_today is True, further filters to leagues that actually have fixtures.
    """
    if target_date is None:
        target_date = _today_utc()

    master = load_leagues_config("leagues.yaml")
    settings = master.get("settings", {})
    bet_settings = {**DEFAULT_SETTINGS, **master.get("betting", {})}
    max_matches = settings.get("max_matches_per_league", 12)
    min_history = settings.get("min_history_matches", 120)
    seasons_back = settings.get("history_seasons", 4)

    leagues = list(selected_leagues)

    if only_today and leagues:
        # Keep only those that have fixtures on the target date
        with_fixtures = find_leagues_with_matches_today(leagues, target_date=target_date)
        # Preserve order / user selection as much as possible
        fixture_ids = {lg["id"] for lg in with_fixtures}
        leagues = [lg for lg in leagues if lg["id"] in fixture_ids]
        save_active_leagues(leagues, target_date=target_date)

    if not leagues:
        return pd.DataFrame(), pd.DataFrame(), {
            "message": "No selected leagues have fixtures on that date (or none selected).",
            "success": 0,
            "skipped": 0,
            "bet_settings": bet_settings,
            "n_leagues": 0,
            "target_date": target_date,
        }

    manual_odds = load_manual_odds()
    all_results: list[dict] = []
    all_bets: list[dict] = []
    success = skipped = 0
    n = len(leagues)

    for i, league in enumerate(leagues, 1):
        name, slug = league["name"], league["slug"]
        if progress_callback:
            progress_callback(i, n, name)

        if download_fresh:
            try:
                download_league_data(league, seasons_back=seasons_back)
            except Exception as e:
                st.warning(f"Download warning for {name}: {e}")

        try:
            bundle = train_league_model(name, slug, min_matches=min_history)
        except Exception as e:
            st.warning(f"Training error for {name}: {e}")
            bundle = None

        if bundle is None:
            skipped += 1
            continue

        try:
            preds = predict_league(
                league=league,
                bundle=bundle,
                max_matches=max_matches,
                target_date=target_date if only_today else None,
                manual_odds=manual_odds,
            )
        except Exception as e:
            st.warning(f"Prediction error for {name}: {e}")
            skipped += 1
            continue

        for p in preds:
            odds = p.pop("_odds", {})
            all_results.append(p)
            if odds:
                all_bets.extend(
                    find_value_bets(p, odds, bundle["gates"], bet_settings)
                )
        success += 1

    preds_df = pd.DataFrame(all_results) if all_results else pd.DataFrame()
    bets = apply_daily_cap(all_bets, bet_settings)
    bets_df = pd.DataFrame(bets) if bets else pd.DataFrame()

    # Persist latest CSVs
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    if not preds_df.empty:
        preds_df.to_csv(PRED_DIR / "latest_predictions.csv", index=False)
        preds_df.to_csv(PRED_DIR / f"predictions_{stamp}.csv", index=False)
    if not bets_df.empty:
        bets_df.to_csv(PRED_DIR / "latest_value_bets.csv", index=False)
        bets_df.to_csv(PRED_DIR / f"value_bets_{stamp}.csv", index=False)

    return preds_df, bets_df, {
        "success": success,
        "skipped": skipped,
        "bet_settings": bet_settings,
        "n_leagues": n,
        "target_date": target_date,
    }


def load_cached_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    preds_path = PRED_DIR / "latest_predictions.csv"
    bets_path = PRED_DIR / "latest_value_bets.csv"
    preds = pd.read_csv(preds_path) if preds_path.exists() else pd.DataFrame()
    bets = pd.read_csv(bets_path) if bets_path.exists() else pd.DataFrame()
    return preds, bets


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
all_leagues, name_to_league, default_names = load_all_league_options()
all_names = sorted(name_to_league.keys())

with st.sidebar:
    st.title("⚽ Controls")
    st.caption("Football ML · Dixon-Coles v2")

    target = st.date_input(
        "Target date",
        value=date.today(),
        help="UTC date used for fixtures",
    )
    target_str = target.strftime("%Y-%m-%d")

    only_today = st.toggle(
        "Only leagues with fixtures on this date",
        value=True,
        help="Further filters your selected leagues to those that actually have matches",
    )
    download_fresh = st.toggle(
        "Download fresh data",
        value=True,
        help="Turn off to reuse CSVs already in data/raw (faster)",
    )

    st.divider()
    st.subheader("Leagues to run")

    # Quick action buttons
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("All", use_container_width=True):
            st.session_state["selected_leagues"] = all_names
    with col_b:
        if st.button("YAML", use_container_width=True, help="Only those enabled in leagues.yaml"):
            st.session_state["selected_leagues"] = default_names
    with col_c:
        if st.button("None", use_container_width=True):
            st.session_state["selected_leagues"] = []

    # Initialise session state once
    if "selected_leagues" not in st.session_state:
        st.session_state["selected_leagues"] = default_names

    selected_names = st.multiselect(
        "Select leagues",
        options=all_names,
        default=st.session_state["selected_leagues"],
        help="Only the leagues you tick here will be processed",
        key="league_multiselect",
    )
    # Keep session state in sync
    st.session_state["selected_leagues"] = selected_names

    st.caption(f"{len(selected_names)} league(s) selected")

    st.divider()
    st.subheader("Odds (optional)")
    uploaded = st.file_uploader(
        "Upload odds.csv",
        type=["csv"],
        help="Columns: Date, HomeTeam, AwayTeam, Market, Odds",
    )
    if uploaded is not None:
        odds_path = ROOT / "data" / "odds.csv"
        odds_path.parent.mkdir(parents=True, exist_ok=True)
        odds_path.write_bytes(uploaded.getvalue())
        st.success("odds.csv saved")

    st.divider()
    run_btn = st.button("🚀 Run predictions", type="primary", use_container_width=True)

    st.divider()
    st.caption(
        "⚠️ No model guarantees profit. Only risk money you can afford to lose."
    )

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("Football Value Bet Finder")
st.markdown(
    f"Target date: **{target_str}** · "
    f"Selected: **{len(selected_names)}** leagues · "
    f"{'fixtures filter ON' if only_today else 'fixtures filter OFF'}"
)

# Resolve selected league dicts
selected_leagues = [name_to_league[n] for n in selected_names if n in name_to_league]

# Run or show cached
if run_btn:
    if not selected_leagues:
        st.warning("Please select at least one league in the sidebar.")
        st.stop()

    progress = st.progress(0, text="Starting…")
    status = st.empty()

    def _progress(i, n, name):
        progress.progress(i / n, text=f"[{i}/{n}] {name}")
        status.info(f"Processing **{name}** ({i}/{n})")

    try:
        with st.spinner("Running pipeline (download → train → predict)…"):
            preds_df, bets_df, summary = run_pipeline(
                selected_leagues=selected_leagues,
                download_fresh=download_fresh,
                only_today=only_today,
                target_date=target_str,
                progress_callback=_progress,
            )
        progress.progress(1.0, text="Done")
        status.success(
            f"Finished · {summary['success']} leagues OK · "
            f"{summary['skipped']} skipped"
        )
        st.session_state["preds"] = preds_df
        st.session_state["bets"] = bets_df
        st.session_state["summary"] = summary
    except Exception:
        st.error("Pipeline failed")
        st.code(traceback.format_exc())
        st.stop()
else:
    # Try to show last run
    if "preds" not in st.session_state:
        preds_df, bets_df = load_cached_results()
        st.session_state["preds"] = preds_df
        st.session_state["bets"] = bets_df
        st.session_state["summary"] = {}
    preds_df = st.session_state.get("preds", pd.DataFrame())
    bets_df = st.session_state.get("bets", pd.DataFrame())
    summary = st.session_state.get("summary", {})

# ---------------------------------------------------------------------------
# Value bets (most important)
# ---------------------------------------------------------------------------
st.header("🎯 Value bets")

if bets_df is not None and not bets_df.empty:
    bs = summary.get("bet_settings", DEFAULT_SETTINGS)
    st.caption(
        f"Bankroll {bs.get('bankroll', 100):.0f} · "
        f"max {bs.get('max_stake_pct', 0.015):.1%} per bet · "
        f"max {bs.get('max_daily_exposure_pct', 0.05):.0%} daily"
    )
    for _, b in bets_df.iterrows():
        edge = b.get("Edge", 0)
        st.markdown(
            f"""
            <div class="value-bet-card">
              <strong>{b.get('HomeTeam', '')} vs {b.get('AwayTeam', '')}</strong>
              &nbsp;·&nbsp; {b.get('League', '')}<br>
              <span style="font-size:1.15rem; color:#69f0ae;">
                {b.get('Market', '')} @ {b.get('Odds', '')}
              </span>
              &nbsp; edge <b>{edge:+.1%}</b>
              &nbsp; model {b.get('ModelProb', 0):.0%}
              &nbsp; stake <b>{b.get('Stake', 0):.2f}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )
    csv = bets_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download value bets CSV",
        data=csv,
        file_name="value_bets.csv",
        mime="text/csv",
    )
else:
    st.markdown(
        '<p class="no-bet">No value bets cleared the filters. '
        "That is usually the correct outcome.</p>",
        unsafe_allow_html=True,
    )
    st.info(
        "Tips: upload bookmaker odds via the sidebar, or use the **Fair_*** columns "
        "below and only bet when real odds are clearly higher than fair odds."
    )

# ---------------------------------------------------------------------------
# Predictions table
# ---------------------------------------------------------------------------
st.header("📊 All predictions")

if preds_df is not None and not preds_df.empty:
    # Friendly column order + the extra markets the user requested
    preferred_cols = [
        "Date",
        "League",
        "HomeTeam",
        "AwayTeam",
        "MostLikely",
        "TopProb",
        "HomeXG",
        "AwayXG",
        "H",
        "D",
        "A",
        "Over15",          # over_1.5
        "Over25",
        "Under35",         # under_3.5
        "HomeOver15",      # home_over_1.5
        "AwayOver15",      # away_over_1.5
        "BTTS_Yes",
        "Fair_H",
        "Fair_D",
        "Fair_A",
        "Fair_Over15",
        "Fair_Over25",
        "Fair_Under35",
        "Fair_HomeOver15",
        "Fair_AwayOver15",
        "ModelQuality",
        "ApprovedMarkets",
        "NewTeam",
    ]
    show_cols = [c for c in preferred_cols if c in preds_df.columns]

    display = preds_df[show_cols].copy()

    # Rename for clearer headers on the phone
    rename_map = {
        "Over15": "Over 1.5",
        "Over25": "Over 2.5",
        "Under35": "Under 3.5",
        "HomeOver15": "Home Over 1.5",
        "AwayOver15": "Away Over 1.5",
        "BTTS_Yes": "BTTS Yes",
        "Fair_Over15": "Fair Over 1.5",
        "Fair_Under35": "Fair Under 3.5",
        "Fair_HomeOver15": "Fair Home O1.5",
        "Fair_AwayOver15": "Fair Away O1.5",
    }
    display = display.rename(columns=rename_map)

    # Format probability columns as percentages
    pct_cols = [
        "TopProb", "H", "D", "A",
        "Over 1.5", "Over 2.5", "Under 3.5",
        "Home Over 1.5", "Away Over 1.5", "BTTS Yes",
    ]
    for c in pct_cols:
        if c in display.columns:
            display[c] = (pd.to_numeric(display[c], errors="coerce") * 100).round(1).astype(str) + "%"

    # Alternating row colours (4 soft dark tones that cycle)
    row_colors = ["#15202b", "#1a2733", "#1e2d3d", "#243447"]

    def _style_rows(row):
        color = row_colors[row.name % 4]
        return [f"background-color: {color}; color: #e8e8e8"] * len(row)

    styled = (
        display.style
        .apply(_style_rows, axis=1)
        .set_properties(**{"font-size": "0.82rem", "border-color": "#2a3a4a"})
    )

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        height=min(480, 48 + 36 * len(display)),
    )

    csv = preds_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download full predictions CSV",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv",
    )
else:
    st.write("No predictions yet. Select leagues in the sidebar and press **Run predictions**.")

# ---------------------------------------------------------------------------
# Quick stats
# ---------------------------------------------------------------------------
if summary:
    c1, c2, c3 = st.columns(3)
    c1.metric("Leagues OK", summary.get("success", "–"))
    c2.metric("Skipped", summary.get("skipped", "–"))
    c3.metric("Target", summary.get("target_date", target_str))
