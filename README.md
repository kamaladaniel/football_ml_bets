# Football ML Prediction System — v2

**Read this first:** no model can guarantee profit. Bookmakers add a margin of
roughly 4–8% to every price, so even a *perfect* probability model loses money if
you simply bet on what it thinks will happen. v2 is built around that fact.
**Only bet money you can afford to lose completely — never savings, rent or debt money.**

## What changed from v1 (and why v1 lost money)

| Problem in v1 | Fix in v2 |
|---|---|
| Training features ≠ prediction features (the "goals scored" inputs were silently replaced by a different formula at prediction time), so the model was fed data it was never trained on | One consistent model; no separate feature pipeline to drift |
| 400 boosted trees on ~700 matches per league → heavy overfitting | Dixon-Coles goal model (≈45 parameters), ridge-regularised, time-weighted |
| `class_weight="balanced"` + isotonic calibration on tiny data → distorted probabilities (too many draws, jumpy values) | Probabilities come straight from a scoreline distribution, then are shrunk toward the league base rate by the amount that validated best |
| 1X2 from one model, goal markets from a crude separate formula (inconsistent) | Every market (1X2, O/U, BTTS, team totals) comes from the **same** scoreline matrix |
| Only 2 seasons of history, Elo never converged | 4 seasons by default, jointly-fitted attack/defence ratings + home advantage |
| Evaluated on one 15% split | **Walk-forward** validation: each test match predicted using only earlier matches |
| Output was "predicted winner + confidence" — no prices | Output is probabilities and **fair odds**; bets are flagged only where real bookmaker odds beat them |
| No staking rules | Capped fractional Kelly, per-bet cap, daily exposure cap, one bet per match |

## The safety gates

1. **Model gate (per league, per market).** On matches it never saw, the model must beat a
   naive "league base rate" predictor. If it doesn't, that league/market is marked `LOW` / not
   approved and **no bets are ever suggested for it**.
2. **Value gate.** A bet is only flagged if real odds are provided, the margin-free market
   probability blended with the model (model weight 0.3) shows ≥ 6% expected return, odds are
   between 1.40 and 4.00, and the model itself also sees value.
3. **Staking.** 20% Kelly, max 1.5% of bankroll per bet, max 5% per day.
4. Newly-promoted / unmatched teams are never bet on.

Most days the correct output is **"no value bets"**. That is the system working, not failing.

## How to run

```bash
pip install -r requirements.txt
export BSD_API_TOKEN="your_token"      # Windows PowerShell: $env:BSD_API_TOKEN="your_token"
python main.py                         # today's leagues, fresh download
python main.py --no-download           # reuse data/raw CSVs
python main.py --date 2026-12-01
python main.py --all-enabled           # use leagues enabled in config/leagues.yaml
```

Outputs (in `data/predictions/`):
- `latest_predictions.csv` — probabilities, expected goals, **Fair_\*** odds for every market,
  `ModelQuality`, `ApprovedMarkets`
- `latest_value_bets.csv` — only bets that passed every gate, with stake

## Giving it bookmaker odds (required for bet suggestions)

The code tries to read odds from the API, but that schema isn't guaranteed. The reliable way:
copy `data/odds_template.csv` to `data/odds.csv` and fill in decimal odds
(`Market` ∈ H, D, A, Over15, Over25, Under25, Under35, HomeOver15, AwayOver15, BTTS_Yes, BTTS_No).
Give **both sides** of a market (e.g. Over25 *and* Under25) so the margin can be removed.

No odds file? Use the `Fair_*` columns by hand: only consider a bet when the bookmaker's odds are
at least ~10% **above** the fair odds, and the market is `ApprovedMarkets`.

## Settings

`config/leagues.yaml` → `betting:` block (bankroll, min_edge, kelly_fraction, caps, odds range).
Set `bankroll` to the small amount you've set aside for betting. Loosening these raises risk.
Hyper-parameters (decay half-life, ridge, shrinkage) are chosen automatically per league.

## Project layout

```
main.py                 run this
src/dixon_coles.py      the goal model + all market probabilities
src/evaluate.py         walk-forward validation, model/market gates
src/trainer.py          per-league training
src/predictor.py        predictions + fair odds
src/betting.py          de-vig, edge, Kelly, caps
src/downloader.py       sports.bzzoiro.com client (+ best-effort odds)
config/leagues.yaml     leagues, settings, betting rules
tests/synthetic_check.py   offline test (no API needed)
```

Run `python tests/synthetic_check.py` to verify your install works offline.

## Honest limitations

- I could not access the live API while building this, so it was verified on simulated leagues
  (where it lands very close to the theoretical best achievable), **not** on your real data.
  Run it and read the `Validation` lines per league — they are your real-data report card.
- It uses only goals and results. Public-market prices already contain lineups, injuries and news,
  which this model doesn't see. That is why the value gate leans heavily on the market.
- Cups and international tournaments have little data and will usually fail the gate.
- Track results for 100s of bets (or paper-trade first) before trusting any edge. Small samples lie.
- The API token is still in `src/utils.py` as a fallback; prefer the environment variable and
  remove the fallback before sharing this code.

---

## Streamlit UI (phone-friendly)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL on your phone (same Wi-Fi) or deploy for remote access.

See **DEPLOY.md** for free hosting options (Streamlit Community Cloud recommended), secrets setup, and how to keep the free tier from timing out.
