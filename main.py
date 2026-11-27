"""
=============================================================
Football ML Prediction System  (v2 - Dixon-Coles engine)
=============================================================
FLOW
  Find leagues with fixtures today
    -> download history (several seasons)
    -> walk-forward validate + fit one model per league
    -> predict today's matches (all markets from one scoreline model)
    -> if bookmaker odds are available: flag VALUE bets only,
       with capped fractional-Kelly stakes
    -> save predictions + value bets

Usage
  python main.py                    # today, re-download data
  python main.py --no-download      # reuse CSVs already in data/raw
  python main.py --date 2026-11-30  # a specific day
  python main.py --all-enabled      # ignore "today", use enabled leagues
=============================================================
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.betting import DEFAULT_SETTINGS, apply_daily_cap, find_value_bets
from src.downloader import download_league_data, find_leagues_with_matches_today
from src.predictor import predict_league
from src.trainer import train_league_model
from src.utils import (
    PRED_DIR,
    get_all_leagues,
    get_enabled_leagues,
    load_leagues_config,
    load_manual_odds,
    save_active_leagues,
)


def discover_and_activate_todays_leagues(target_date: str = None):
    if target_date is None:
        target_date = datetime.now(timezone.utc).date().strftime("%Y-%m-%d")
    all_leagues = get_all_leagues("leagues.yaml")
    active = find_leagues_with_matches_today(all_leagues, target_date=target_date)
    save_active_leagues(active, target_date=target_date)
    if not active:
        print("  No leagues have fixtures on that date. Nothing to do.")
    return active


def run_full_pipeline(download_fresh: bool = True, only_today: bool = True,
                      target_date: str = None):
    print("=" * 70)
    print("  FOOTBALL ML PREDICTION SYSTEM  (v2)")
    print("  Dixon-Coles goal model • walk-forward validated • value-bet filter")
    print("=" * 70)
    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    if target_date is None:
        target_date = datetime.now(timezone.utc).date().strftime("%Y-%m-%d")

    if only_today:
        print(f"  Mode: ONLY {target_date}\n")
        leagues = discover_and_activate_todays_leagues(target_date)
        if not leagues:
            return
        config = load_leagues_config("active_leagues.yaml")
    else:
        print("  Mode: ALL ENABLED LEAGUES (from leagues.yaml)\n")
        leagues = get_enabled_leagues("leagues.yaml")
        config = load_leagues_config("leagues.yaml")

    settings = config.get("settings", {})
    master = load_leagues_config("leagues.yaml")
    bet_settings = {**DEFAULT_SETTINGS, **master.get("betting", {})}
    max_matches = settings.get("max_matches_per_league", 12)
    min_history = settings.get("min_history_matches", 120)
    seasons_back = settings.get("history_seasons", 4)
    manual_odds = load_manual_odds()

    print(f"  Active leagues this run: {len(leagues)}")
    for lg in leagues:
        extra = f" ({lg.get('today_fixtures')} fixtures)" if lg.get("today_fixtures") else ""
        print(f"    • {lg['name']}{extra}")
    print(f"  Min history required: {min_history} matches | seasons: {seasons_back}\n")

    all_results, all_bets = [], []
    success, skipped = 0, 0

    for i, league in enumerate(leagues, 1):
        name, slug = league["name"], league["slug"]
        print(f"\n{'#'*70}\n#  [{i}/{len(leagues)}]  {name}\n{'#'*70}")

        if download_fresh:
            try:
                download_league_data(league, seasons_back=seasons_back)
            except Exception as e:
                print(f"  Download warning: {e}")

        try:
            bundle = train_league_model(name, slug, min_matches=min_history)
        except Exception as e:
            print(f"  Training error: {e}")
            bundle = None
        if bundle is None:
            print(f"  → Skipping predictions for {name}")
            skipped += 1
            continue

        try:
            preds = predict_league(
                league=league, bundle=bundle, max_matches=max_matches,
                target_date=target_date if only_today else None,
                manual_odds=manual_odds,
            )
        except Exception as e:
            print(f"  Prediction error: {e}")
            skipped += 1
            continue

        for p in preds:
            odds = p.pop("_odds", {})
            all_results.append(p)
            if odds:
                all_bets.extend(find_value_bets(p, odds, bundle["gates"], bet_settings))
        success += 1

    print("\n" + "=" * 70)
    print(f"  Leagues processed: {success} | skipped: {skipped}")

    if not all_results:
        print("\nNo predictions were generated.")
        print("=" * 70)
        return

    df = pd.DataFrame(all_results)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    df.to_csv(PRED_DIR / f"predictions_{stamp}.csv", index=False)
    df.to_csv(PRED_DIR / "latest_predictions.csv", index=False)
    print(f"\n✅  {len(df)} predictions saved → data/predictions/latest_predictions.csv")

    bets = apply_daily_cap(all_bets, bet_settings)
    bets_df = pd.DataFrame(bets)
    bets_df.to_csv(PRED_DIR / "latest_value_bets.csv", index=False)
    if bets:
        bets_df.to_csv(PRED_DIR / f"value_bets_{stamp}.csv", index=False)

    print("\n" + "-" * 70)
    print(f"  VALUE BETS  (bankroll {bet_settings['bankroll']:.2f}, "
          f"max {bet_settings['max_stake_pct']:.1%} per bet, "
          f"max {bet_settings['max_daily_exposure_pct']:.0%} per day)")
    print("-" * 70)
    if bets:
        for b in bets:
            print(f"  {b['HomeTeam']} vs {b['AwayTeam']}  [{b['League']}]")
            print(f"     {b['Market']} @ {b['Odds']}   edge {b['Edge']:+.1%}   "
                  f"model {b['ModelProb']:.0%}   stake {b['Stake']:.2f}")
    else:
        has_odds = bool(manual_odds) or bool(all_bets)
        if has_odds:
            print("  No bet clears the value filter today. NOT betting is the right call.")
        else:
            print("  No bookmaker odds were available, so no bets can be judged.")
            print("  Use the Fair_* columns in latest_predictions.csv: only bet when the")
            print("  bookmaker's odds are clearly ABOVE the fair odds (see README),")
            print("  or put odds in data/odds.csv and re-run.")
    print("=" * 70)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-download", action="store_true", help="reuse CSVs in data/raw")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (default: today UTC)")
    ap.add_argument("--all-enabled", action="store_true",
                    help="use leagues enabled in leagues.yaml instead of 'today only'")
    args = ap.parse_args()
    run_full_pipeline(
        download_fresh=not args.no_download,
        only_today=not args.all_enabled,
        target_date=args.date,
    )
