"""
Betting layer.

A good probability model is NOT the same thing as a profitable betting system.
Bookmakers build in a margin (typically 4-8%), so you only win long-term if you
bet when your probability is better than the *price* offered, not merely when
you think a team will win.

This module therefore:
  * works only from real bookmaker odds you provide (API odds if available,
    or data/odds.csv),
  * removes the bookmaker margin where both sides of a market are known,
  * blends model and market probabilities (the market is usually right),
  * requires a minimum edge, sane odds range, and an approved market,
  * stakes with capped fractional Kelly and a daily exposure limit,
  * places at most one bet per match.
"""

from __future__ import annotations

# Groups of mutually exclusive outcomes, used to remove the bookmaker margin
DEVIG_GROUPS = [
    ("H", "D", "A"),
    ("Over25", "Under25"),
    ("BTTS_Yes", "BTTS_No"),
]

DEFAULT_SETTINGS = {
    "bankroll": 100.0,            # money set aside for betting (NOT your savings)
    "min_edge": 0.06,             # required expected return, e.g. 0.06 = +6%
    "single_sided_extra_edge": 0.03,
    "model_weight": 0.3,          # 0 = trust market only, 1 = trust model only
    "kelly_fraction": 0.20,       # fraction of full Kelly
    "max_stake_pct": 0.015,       # hard cap per bet (share of bankroll)
    "max_daily_exposure_pct": 0.05,
    "min_odds": 1.40,
    "max_odds": 4.00,
}

MARKET_LABELS = {
    "H": "Home Win", "D": "Draw", "A": "Away Win",
    "Over15": "Over 1.5", "Over25": "Over 2.5", "Under25": "Under 2.5",
    "Under35": "Under 3.5", "HomeOver15": "Home Over 1.5",
    "AwayOver15": "Away Over 1.5", "BTTS_Yes": "BTTS Yes", "BTTS_No": "BTTS No",
}


def gate_key(market: str) -> str:
    if market in ("H", "D", "A"):
        return "1X2"
    return {"Under25": "Over25", "BTTS_No": "BTTS_Yes"}.get(market, market)


def devig(odds: dict) -> dict:
    """Return margin-free market probabilities for every complete group."""
    out = {}
    for group in DEVIG_GROUPS:
        if all(odds.get(m, 0) > 1.0 for m in group):
            inv = {m: 1.0 / odds[m] for m in group}
            total = sum(inv.values())
            for m in group:
                out[m] = inv[m] / total
    return out


def find_value_bets(pred: dict, odds: dict, gates: dict, settings: dict) -> list[dict]:
    """
    pred  : prediction row (contains probabilities under the market keys)
    odds  : {market: decimal_odds}
    gates : {market_group: bool} from validation
    Returns the single best bet for the match (or an empty list).
    """
    s = {**DEFAULT_SETTINGS, **(settings or {})}
    if pred.get("NewTeam"):
        return []
    market_p = devig(odds)
    candidates = []

    for market, o in odds.items():
        if market not in MARKET_LABELS or not o or o <= 1.0:
            continue
        if not gates.get(gate_key(market), False):
            continue
        if not (s["min_odds"] <= o <= s["max_odds"]):
            continue
        p_model = pred.get(market)
        if p_model is None:
            continue

        if market in market_p:
            p_final = s["model_weight"] * p_model + (1 - s["model_weight"]) * market_p[market]
            needed = s["min_edge"]
        else:
            # one-sided odds: cannot remove the margin, be stricter
            p_final = 0.7 * p_model + 0.3 * (1.0 / o)
            needed = s["min_edge"] + s["single_sided_extra_edge"]

        edge = p_final * o - 1.0
        # model must itself agree there is value (not just the blend)
        if edge < needed or p_model * o - 1.0 <= 0:
            continue

        kelly = edge / (o - 1.0)
        stake_pct = min(s["max_stake_pct"], s["kelly_fraction"] * kelly)
        if stake_pct <= 0:
            continue
        candidates.append({
            "Date": pred["Date"], "League": pred["League"],
            "HomeTeam": pred["HomeTeam"], "AwayTeam": pred["AwayTeam"],
            "Market": MARKET_LABELS[market], "MarketKey": market,
            "Odds": round(o, 2),
            "ModelProb": round(p_model, 4),
            "BlendedProb": round(p_final, 4),
            "FairOdds": round(1.0 / p_final, 2),
            "Edge": round(edge, 4),
            "StakePct": round(stake_pct, 4),
            "Stake": round(stake_pct * s["bankroll"], 2),
        })

    if not candidates:
        return []
    return [max(candidates, key=lambda c: c["Edge"])]


def apply_daily_cap(bets: list[dict], settings: dict) -> list[dict]:
    """Keep the highest-edge bets until the daily exposure cap is reached."""
    s = {**DEFAULT_SETTINGS, **(settings or {})}
    cap = s["max_daily_exposure_pct"]
    kept, used = [], 0.0
    for b in sorted(bets, key=lambda b: b["Edge"], reverse=True):
        if used + b["StakePct"] <= cap + 1e-12:
            kept.append(b)
            used += b["StakePct"]
    return kept
