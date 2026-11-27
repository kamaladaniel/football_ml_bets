"""
Data Downloader for sports.bzzoiro.com
Downloads historical matches and upcoming fixtures for any league.
"""

import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone

from src.utils import RAW_DIR, get_api_token, get_enabled_leagues

BASE_URL = "https://sports.bzzoiro.com/api/v2"
HEADERS = {"Authorization": f"Token {get_api_token()}"}


# ----------------------------------------------------------------------
# Odds extraction (best effort - the API's odds schema is not documented
# here, so several common key names are tried). If nothing is found you
# can always supply odds yourself in data/odds.csv.
# ----------------------------------------------------------------------
_ODDS_KEYS = {
    "H": ["odds_home", "home_odds", "odds_1", "odds_home_win", "home_win_odds", "home", "1"],
    "D": ["odds_draw", "draw_odds", "odds_x", "draw", "x"],
    "A": ["odds_away", "away_odds", "odds_2", "odds_away_win", "away_win_odds", "away", "2"],
    "Over25": ["odds_over_25", "odds_over25", "over_25_odds", "over_2_5", "over25"],
    "Under25": ["odds_under_25", "odds_under25", "under_25_odds", "under_2_5", "under25"],
    "Over15": ["odds_over_15", "odds_over15", "over_15_odds", "over_1_5", "over15"],
    "Under35": ["odds_under_35", "odds_under35", "under_35_odds", "under_3_5", "under35"],
    "BTTS_Yes": ["odds_btts_yes", "btts_yes_odds", "btts_yes", "odds_btts"],
    "BTTS_No": ["odds_btts_no", "btts_no_odds", "btts_no"],
}


def _extract_odds(m: dict) -> dict:
    """Return {market: decimal_odds} found on an event payload (may be empty)."""
    sources = [m]
    for k in ("odds", "market_odds", "bookmaker_odds"):
        if isinstance(m.get(k), dict):
            sources.append(m[k])
    generic = {"home", "draw", "away", "1", "x", "2", "btts_yes", "btts_no"}
    found = {}
    for market, names in _ODDS_KEYS.items():
        for si, src in enumerate(sources):
            lowered = {str(k).lower(): v for k, v in src.items()}
            for name in names:
                # bare names like "home"/"1" are only trusted inside a nested odds dict
                if si == 0 and name in generic:
                    continue
                v = lowered.get(name)
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    continue
                if v > 1.0:
                    found[market] = v
                    break
            if market in found:
                break
    return found


def _get(url, params=None, retries=3):
    """Safe GET request with retries."""
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=30)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 10))
                print(f"  Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == retries - 1:
                print(f"  Request failed: {e}")
                return None
            time.sleep(2)
    return None


def get_current_season_id(league_id: int):
    """Get the current season ID for a league."""
    data = _get(f"{BASE_URL}/leagues/{league_id}/season/")
    if data and "id" in data:
        return data["id"]
    return None


def get_all_seasons(league_id: int):
    """Get list of available seasons (most recent first)."""
    data = _get(f"{BASE_URL}/leagues/{league_id}/seasons/")
    if data and "results" in data:
        return data["results"]
    if isinstance(data, list):
        return data
    return []


def download_finished_matches(league_id: int, season_id: int = None, max_pages: int = 20):
    """
    Download finished matches for a league (and optional season).
    Returns a list of match dicts.
    """
    matches = []
    offset = 0
    limit = 100

    params = {
        "league_id": league_id,
        "status": "finished",
        "limit": limit,
        "offset": offset,
    }
    if season_id:
        params["season_id"] = season_id

    for page in range(max_pages):
        params["offset"] = offset
        data = _get(f"{BASE_URL}/events/", params=params)
        if not data or "results" not in data:
            break

        results = data["results"]
        if not results:
            break

        for m in results:
            home_score = m.get("home_score")
            away_score = m.get("away_score")
            if home_score is None or away_score is None:
                continue

            matches.append({
                "Date": m.get("event_date", "")[:10],
                "HomeTeam": m.get("home_team") or m.get("home_team_name", ""),
                "AwayTeam": m.get("away_team") or m.get("away_team_name", ""),
                "FTHG": int(home_score),
                "FTAG": int(away_score),
                "FTR": "H" if home_score > away_score else ("A" if away_score > home_score else "D"),
                "league_id": league_id,
                "season_id": m.get("season_id"),
                "match_id": m.get("id"),
            })

        if data.get("next") is None:
            break
        offset += limit
        time.sleep(0.3)  # be polite

    return matches


def download_upcoming_fixtures(
    league_id: int,
    max_matches: int = 12,
    days_ahead: int = 50,
    target_date: str = None,
):
    """
    Download the nearest upcoming fixtures for a league.
    - Only looks within the next `days_ahead` days (ignored if target_date set)
    - If target_date (YYYY-MM-DD) is given, only that single day is requested
    - Sorts by date ascending so the soonest matches come first
    - Correctly preserves Home / Away from the API
    """
    today = datetime.now(timezone.utc).date()

    if target_date:
        date_from = target_date
        date_to = target_date
    else:
        date_from = today.strftime("%Y-%m-%d")
        date_to = (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    params = {
        "league_id": league_id,
        "status": "upcoming",
        "limit": 100,                    # fetch more, then take nearest
        "date_from": date_from,
        "date_to": date_to,
    }
    data = _get(f"{BASE_URL}/events/", params=params)
    if not data or "results" not in data:
        return []

    # Sort by actual kick-off time (soonest first)
    results = sorted(
        data["results"],
        key=lambda x: x.get("event_date") or "9999"
    )

    fixtures = []
    for m in results[:max_matches]:
        home = m.get("home_team") or m.get("home_team_name") or ""
        away = m.get("away_team") or m.get("away_team_name") or ""
        if not home or not away:
            continue
        fixtures.append({
            "Date": (m.get("event_date") or "")[:10],
            "HomeTeam": home.strip(),
            "AwayTeam": away.strip(),
            "match_id": m.get("id"),
            "league_id": league_id,
            "odds": _extract_odds(m),
        })
    return fixtures


def save_league_history(league_slug: str, matches: list):
    """Save matches to CSV in the format expected by the feature engineer."""
    if not matches:
        return None

    df = pd.DataFrame(matches)
    # Keep only needed columns and sort
    df = df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]].copy()
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])
    df = df.sort_values("Date").reset_index(drop=True)

    path = RAW_DIR / f"{league_slug}.csv"
    df.to_csv(path, index=False)
    print(f"  Saved {len(df)} matches → {path.name}")
    return path


def download_league_data(league: dict, seasons_back: int = 4):
    """
    Full download pipeline for one league.
    Downloads recent seasons of finished matches + upcoming fixtures.
    """
    name = league["name"]
    league_id = league["id"]
    slug = league["slug"]

    print(f"\n{'='*60}")
    print(f"DOWNLOADING: {name} (ID: {league_id})")
    print(f"{'='*60}")

    all_matches = []

    # Try current season first
    current_season_id = get_current_season_id(league_id)
    if current_season_id:
        print(f"  Current season ID: {current_season_id}")
        matches = download_finished_matches(league_id, current_season_id)
        print(f"  Current season finished matches: {len(matches)}")
        all_matches.extend(matches)

    # Optionally pull previous seasons for more history
    seasons = get_all_seasons(league_id)
    if seasons:
        # Skip the current one if we already have it
        extra_seasons = [s for s in seasons if s.get("id") != current_season_id][:seasons_back]
        for s in extra_seasons:
            sid = s.get("id")
            sname = s.get("name", sid)
            print(f"  Downloading season: {sname}")
            matches = download_finished_matches(league_id, sid)
            print(f"    → {len(matches)} matches")
            all_matches.extend(matches)
            time.sleep(0.4)

    # If still too few, try without season filter (last resort)
    if len(all_matches) < 50:
        print("  Trying broader finished matches pull...")
        matches = download_finished_matches(league_id, season_id=None, max_pages=30)
        all_matches.extend(matches)

    # Remove duplicates by match_id if present
    seen = set()
    unique = []
    for m in all_matches:
        key = m.get("match_id") or (m["Date"], m["HomeTeam"], m["AwayTeam"])
        if key not in seen:
            seen.add(key)
            unique.append(m)

    path = save_league_history(slug, unique)
    return path, unique


def download_all_enabled_leagues():
    """Download data for every enabled league in config."""
    leagues = get_enabled_leagues()
    results = {}
    for lg in leagues:
        try:
            path, matches = download_league_data(lg)
            results[lg["slug"]] = {"path": path, "matches": len(matches) if matches else 0}
        except Exception as e:
            print(f"  ERROR downloading {lg['name']}: {e}")
            results[lg["slug"]] = {"path": None, "matches": 0}
        time.sleep(0.5)
    return results


def has_fixtures_on_date(league_id: int, target_date: str) -> tuple[bool, int]:
    """
    Check whether a league has any fixtures (upcoming or live) on a given date.
    Returns (has_fixtures, count).
    """
    params = {
        "league_id": league_id,
        "status": "upcoming",
        "date_from": target_date,
        "date_to": target_date,
        "limit": 50,
    }
    data = _get(f"{BASE_URL}/events/", params=params)
    count = 0
    if data and "results" in data:
        count = len(data["results"])
        # Some APIs also put total in "count"
        if data.get("count") is not None:
            count = max(count, int(data["count"]))

    # Also check finished matches that might still be listed for today
    # (edge case: early-finished games) — not needed for prediction,
    # but we only care about upcoming for the active-leagues filter.
    return count > 0, count


def find_leagues_with_matches_today(all_leagues: list, target_date: str = None) -> list:
    """
    Scan every league in the provided list and return only those that have
    fixtures scheduled on target_date (defaults to today UTC).

    Returns list of league dicts augmented with 'today_fixtures' count.
    """
    if target_date is None:
        target_date = datetime.now(timezone.utc).date().strftime("%Y-%m-%d")

    print(f"\n{'='*70}")
    print(f"  SCANNING ALL LEAGUES FOR FIXTURES ON {target_date}")
    print(f"{'='*70}")
    print(f"  Checking {len(all_leagues)} competitions...\n")

    active = []
    for i, lg in enumerate(all_leagues, 1):
        name = lg.get("name", "?")
        lid = lg.get("id")
        try:
            has, count = has_fixtures_on_date(lid, target_date)
            status = f"{count} match(es)" if has else "none"
            print(f"  [{i:3d}/{len(all_leagues)}] {name:<40} → {status}")
            if has:
                entry = dict(lg)
                entry["today_fixtures"] = count
                entry["enabled"] = True
                active.append(entry)
        except Exception as e:
            print(f"  [{i:3d}/{len(all_leagues)}] {name:<40} → ERROR: {e}")
        time.sleep(0.25)  # be polite to the API

    print(f"\n  → Found {len(active)} league(s) with matches today.\n")
    return active
