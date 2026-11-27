"""
Model Trainer
Fits one Dixon-Coles model per league, with hyper-parameters and market
approval decided by walk-forward validation (see evaluate.py).
"""

import joblib
import numpy as np
import pandas as pd

from src.dixon_coles import DixonColes
from src.evaluate import _base_rates, clean_matches, select_model
from src.utils import MODEL_DIR, RAW_DIR

REQUIRED = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}


def train_league_model(league_name: str, slug: str, min_matches: int = 120):
    """
    Returns a bundle dict (model + validation report + gates) or None.
    """
    history_path = RAW_DIR / f"{slug}.csv"
    model_path = MODEL_DIR / f"bundle_{slug}.pkl"

    print(f"\n{'='*60}\nTRAINING: {league_name}\n{'='*60}")

    if not history_path.exists():
        print("  History file not found. Skipping.")
        return None
    try:
        df = pd.read_csv(history_path)
    except Exception as e:
        print(f"  Could not read history CSV: {e}")
        return None
    if missing := REQUIRED - set(df.columns):
        print(f"  Missing columns: {sorted(missing)}")
        return None

    df = clean_matches(df)
    print(f"  Matches: {len(df)} | teams: {len(set(df.HomeTeam) | set(df.AwayTeam))} "
          f"| {df.Date.min().date()} → {df.Date.max().date()}")
    if len(df) < min_matches:
        print(f"  Not enough data ({len(df)} < {min_matches}). Skipping.")
        return None

    report = select_model(df)
    if not report.get("ok"):
        print(f"  Validation failed: {report.get('reason')}. Skipping.")
        return None

    model = DixonColes(report["halflife"], report["reg"]).fit(
        df, ref_date=df.Date.max() + pd.Timedelta(days=3)
    )
    print(f"  Home advantage: {model.home_adv:+.3f} (log goals) | "
          f"avg goals/team: {float(np.exp(model.mu)):.2f} "
          f"| rho {model.rho:+.2f}")

    bundle = {
        "model": model,
        "league": league_name,
        "slug": slug,
        "shrink": report["shrink"],
        "gates": report["gates"],
        "detail": report["detail"],
        "val_logloss": report["val_logloss"],
        "base_logloss": report["base_logloss"],
        "val_accuracy": report["val_accuracy"],
        "n_val": report["n_val"],
        "n_train": len(df),
        "last_match": df.Date.max(),
        # base rates used for shrinkage at prediction time
        "base": _base_rates(df),
    }
    joblib.dump(bundle, model_path)
    print(f"  Saved → {model_path.name}")
    return bundle
