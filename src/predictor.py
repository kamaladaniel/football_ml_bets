"""
Prediction Module
All markets come from ONE scoreline distribution, then are shrunk toward the
league base rate by the amount that validated best out-of-sample.
"""

import pandas as pd

from src.dixon_coles import market_probs
from src.downloader import download_upcoming_fixtures
from src.utils import find_team_match, normalize_team_name

PROB_KEYS = ["H", "D", "A", "Over15", "Over25", "Under25", "Under35",
             "HomeOver15", "AwayOver15", "BTTS_Yes", "BTTS_No"]


def _shrink(p: dict, base: dict, s: float) -> dict:
    if s <= 0:
        return p
    out = dict(p)
    for k in ("H", "D", "A", "Over15", "Over25", "Under35",
              "HomeOver15", "AwayOver15", "BTTS_Yes"):
        if k in base:
            out[k] = (1 - s) * p[k] + s * base[k]
    # keep complements consistent
    out["Under25"] = 1 - out["Over25"]
    out["BTTS_No"] = 1 - out["BTTS_Yes"]
    tot = out["H"] + out["D"] + out["A"]
    for k in ("H", "D", "A"):
        out[k] /= tot
    return out


def _lookup_odds(fix, home_key, away_key, manual_odds):
    """API odds first, then anything from data/odds.csv (manual wins on clash)."""
    odds = dict(fix.get("odds") or {})
    if manual_odds:
        for h, a in ((fix["HomeTeam"], fix["AwayTeam"]), (home_key, away_key)):
            key = (fix["Date"], normalize_team_name(h), normalize_team_name(a))
            if key in manual_odds:
                odds.update(manual_odds[key])
                break
    return odds


def predict_league(league: dict, bundle: dict, max_matches: int = 12,
                   target_date: str = None, manual_odds: dict = None):
    """Returns a list of prediction dicts (each may carry an '_odds' dict)."""
    league_name = league["name"]
    model = bundle["model"]

    print(f"\n  Fetching upcoming matches for {league_name}...")
    fixtures = download_upcoming_fixtures(
        league["id"], max_matches=max_matches, target_date=target_date
    )
    if not fixtures:
        print("  No upcoming fixtures found.")
        return []
    print(f"  Found {len(fixtures)} matches. Generating predictions...\n")

    results = []
    team_names = {t: {} for t in model.teams}   # find_team_match expects a dict
    for fix in fixtures:
        home_api, away_api = fix["HomeTeam"], fix["AwayTeam"]
        home_key = find_team_match(home_api, team_names)
        away_key = find_team_match(away_api, team_names)
        new_team = home_key is None or away_key is None
        # Unknown (e.g. newly promoted) teams are treated as league average
        # and are never bet on.
        m = model.score_matrix(home_key or home_api, away_key or away_api)
        lh, la = model.lambdas(home_key or home_api, away_key or away_api)
        probs = _shrink(market_probs(m), bundle["base"], bundle["shrink"])

        top = max(("H", "D", "A"), key=lambda k: probs[k])
        row = {
            "Date": fix["Date"], "League": league_name,
            "HomeTeam": home_api, "AwayTeam": away_api,
            "MostLikely": top, "TopProb": round(probs[top], 4),
            "HomeXG": round(lh, 2), "AwayXG": round(la, 2),
            "NewTeam": new_team,
            "ModelQuality": ("OK" if bundle["gates"].get("1X2") else "LOW"),
            "ApprovedMarkets": ",".join(k for k, v in bundle["gates"].items() if v),
        }
        for k in PROB_KEYS:
            row[k] = round(probs[k], 4)
            row[f"Fair_{k}"] = round(1 / probs[k], 2) if probs[k] > 0 else None
        # legacy column names kept for compatibility
        row.update({"ModelProb_H": row["H"], "ModelProb_D": row["D"], "ModelProb_A": row["A"],
                    "Predicted": top, "Confidence": row["TopProb"]})
        row["_odds"] = _lookup_odds(fix, home_key, away_key, manual_odds)
        results.append(row)

        flag = "  [NEW TEAM - no bets]" if new_team else ""
        print(f"  [{fix['Date']}]  {home_api} vs {away_api}{flag}")
        print(f"     H {probs['H']:.0%}  D {probs['D']:.0%}  A {probs['A']:.0%}   "
              f"xG {lh:.2f}-{la:.2f}   quality: {row['ModelQuality']}")
        print(f"     O1.5 {probs['Over15']:.0%} | O2.5 {probs['Over25']:.0%} | "
              f"U3.5 {probs['Under35']:.0%} | BTTS {probs['BTTS_Yes']:.0%}")
        print(f"     Fair odds  H {row['Fair_H']}  D {row['Fair_D']}  A {row['Fair_A']}\n")
    return results
