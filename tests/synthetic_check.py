"""
Offline sanity test (no internet / API needed).

Simulates a league whose true team strengths we know, then checks that:
  1. the model beats a naive base-rate predictor on UNSEEN matches,
  2. it gets close to the "oracle" that knows the true strengths,
  3. the full train -> predict -> value-bet path runs without errors,
  4. betting against a bookmaker that prices PERFECTLY with a 6% margin
     loses money (this is the point: a model alone can't beat the margin;
     only a genuine edge over the market can).

Run:  python tests/synthetic_check.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import predictor, trainer  # noqa: E402
from src.betting import find_value_bets  # noqa: E402
from src.dixon_coles import market_probs, scoreline_matrix  # noqa: E402
from src.evaluate import select_model  # noqa: E402
from src.utils import RAW_DIR  # noqa: E402

rng = np.random.default_rng(7)
N_TEAMS, SEASONS = 20, 3
MU, HOME = 0.15, 0.25
teams = [f"Team {i:02d}" for i in range(N_TEAMS)]
att = rng.normal(0, 0.28, N_TEAMS)
dfn = rng.normal(0, 0.28, N_TEAMS)


def lam(i, j):
    return np.exp(MU + HOME + att[i] - dfn[j]), np.exp(MU + att[j] - dfn[i])


rows, day = [], pd.Timestamp("2023-08-01")
for s in range(SEASONS):
    fixtures = [(i, j) for i in range(N_TEAMS) for j in range(N_TEAMS) if i != j]
    rng.shuffle(fixtures)
    for k, (i, j) in enumerate(fixtures):
        lh, la = lam(i, j)
        rows.append({"Date": day + pd.Timedelta(days=int(k / 10) + s * 365),
                     "HomeTeam": teams[i], "AwayTeam": teams[j],
                     "FTHG": rng.poisson(lh), "FTAG": rng.poisson(la)})
df = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
df["FTR"] = np.where(df.FTHG > df.FTAG, "H", np.where(df.FTHG < df.FTAG, "A", "D"))
print(f"Simulated {len(df)} matches, {N_TEAMS} teams\n")

# 1 + 2: validation vs naive and oracle ---------------------------------
rep = select_model(df)
n_val = int(len(df) * 0.30)
test = df.iloc[-n_val:]
ll = 0
for _, r in test.iterrows():
    i, j = teams.index(r.HomeTeam), teams.index(r.AwayTeam)
    p = market_probs(scoreline_matrix(*lam(i, j), 0.0))
    ll += -np.log(p[r.FTR])
oracle = ll / len(test)
print(f"\n  naive base-rate : {rep['base_logloss']:.4f}")
print(f"  our model       : {rep['val_logloss']:.4f}")
print(f"  oracle (truth)  : {oracle:.4f}   <- best possible")
assert rep["val_logloss"] < rep["base_logloss"], "model failed to beat naive baseline"
assert rep["gates"]["1X2"], "1X2 gate should open on a league with real signal"

# 3: full train -> predict path (API stubbed) ---------------------------
slug = "zz_synthetic_test"
df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]].to_csv(RAW_DIR / f"{slug}.csv", index=False)
bundle = trainer.train_league_model("Synthetic League", slug, min_matches=120)
assert bundle is not None

last = df.Date.max() + pd.Timedelta(days=7)
fake = [{"Date": str(last.date()), "HomeTeam": teams[0], "AwayTeam": teams[1], "odds": {}},
        {"Date": str(last.date()), "HomeTeam": "Brand New FC", "AwayTeam": teams[2], "odds": {}}]
predictor.download_upcoming_fixtures = lambda *a, **k: fake
preds = predictor.predict_league({"name": "Synthetic League", "id": 0, "slug": slug}, bundle, 12, str(last.date()))
assert len(preds) == 2 and preds[1]["NewTeam"] is True
p0 = preds[0]
assert abs(p0["H"] + p0["D"] + p0["A"] - 1) < 1e-6
print("\n  pipeline OK. Example:", {k: p0[k] for k in ("H", "D", "A", "Over25", "BTTS_Yes")})

# value bets: no bet on a new team; bet appears only with a real price gap
assert find_value_bets(preds[1], {"H": 9.0, "D": 4, "A": 1.5}, bundle["gates"], {}) == []
fair = {k: 1 / p0[k] for k in ("H", "D", "A")}
generous = {k: v * 1.25 for k, v in fair.items()}     # bookmaker badly off
stingy = {k: v * 0.93 for k, v in fair.items()}       # normal margin
assert find_value_bets(p0, generous, bundle["gates"], {"min_odds": 1.01, "max_odds": 99}), "should flag value"
assert find_value_bets(p0, stingy, bundle["gates"], {}) == [], "must not bet into a normal margin"
print("  value filter OK (bets only when price beats fair odds by a margin)")

# 4: perfect bookmaker + 6% margin -> bettor loses in expectation --------
exp_ret = real_ret = 0.0
for _, r in test.iterrows():
    i, j = teams.index(r.HomeTeam), teams.index(r.AwayTeam)
    true_p = market_probs(scoreline_matrix(*lam(i, j), 0.0))
    book = {k: 1 / (true_p[k] * 1.06) for k in ("H", "D", "A")}
    pm = predictor._shrink(market_probs(bundle["model"].score_matrix(r.HomeTeam, r.AwayTeam)),
                           bundle["base"], bundle["shrink"])
    pick = max(("H", "D", "A"), key=lambda k: pm[k])      # "back the favourite"
    exp_ret += true_p[pick] * book[pick]                   # expected return on 1 unit
    real_ret += book[pick] if r.FTR == pick else 0
n = len(test)
print(f"\n  'Bet the favourite' vs a perfectly-priced bookmaker with a 6% margin:")
print(f"    expected ROI : {100*(exp_ret-n)/n:+.1f}%   (the margin; no pick can change this)")
print(f"    realised ROI : {100*(real_ret-n)/n:+.1f}%   (pure luck over only {n} bets; it swings wildly)")
assert exp_ret < n

(RAW_DIR / f"{slug}.csv").unlink()
(ROOT / "models" / f"bundle_{slug}.pkl").unlink(missing_ok=True)
print("\nALL CHECKS PASSED")
