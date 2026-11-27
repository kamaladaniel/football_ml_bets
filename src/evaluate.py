"""
Honest out-of-sample evaluation.

Everything here is *walk-forward*: each validation match is predicted by a
model that only saw matches played BEFORE it. That is the only evaluation that
tells you what will happen when you use the model for real.

The result decides, per league and per market, whether the model is allowed to
recommend bets at all (the "gate"). If the model cannot beat a trivial
"league base-rate" predictor on matches it never saw, it does not get to
touch your money.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from src.dixon_coles import DixonColes, market_probs

HALFLIFE_GRID = [120, 240, 480, 900]     # days
REG_GRID = [1.0, 4.0, 12.0]               # ridge strength
SHRINK_GRID = [0.0, 0.10, 0.20, 0.30]     # blend toward base rate

BINARY = {
    "Over15":     lambda hg, ag: hg + ag >= 2,
    "Over25":     lambda hg, ag: hg + ag >= 3,
    "Under35":    lambda hg, ag: hg + ag <= 3,
    "HomeOver15": lambda hg, ag: hg >= 2,
    "AwayOver15": lambda hg, ag: ag >= 2,
    "BTTS_Yes":   lambda hg, ag: hg >= 1 and ag >= 1,
}
# markets that inherit the gate of their complement
GATE_ALIAS = {"Under25": "Over25", "BTTS_No": "BTTS_Yes"}

MIN_VAL_MATCHES = 60
MIN_IMPROVEMENT = 0.005   # log-loss improvement required over base rate


def clean_matches(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    df["HomeTeam"] = df["HomeTeam"].astype(str).str.strip()
    df["AwayTeam"] = df["AwayTeam"].astype(str).str.strip()
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)
    df = df.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"])
    return df.sort_values("Date").reset_index(drop=True)


def _base_rates(train: pd.DataFrame) -> dict:
    hg, ag = train["FTHG"].to_numpy(), train["FTAG"].to_numpy()
    r = {
        "H": float((hg > ag).mean()),
        "D": float((hg == ag).mean()),
        "A": float((hg < ag).mean()),
    }
    for name, fn in BINARY.items():
        r[name] = float(np.mean([fn(a, b) for a, b in zip(hg, ag)]))
    return r


def walk_forward(df: pd.DataFrame, halflife: float, reg: float,
                 val_frac: float = 0.30, max_chunks: int = 10) -> pd.DataFrame:
    """Refit on all past data before each chunk of matches and predict the chunk."""
    n = len(df)
    start = int(n * (1 - val_frac))
    chunk = max(15, int(np.ceil((n - start) / max_chunks)))
    rows = []
    for s in range(start, n, chunk):
        e = min(s + chunk, n)
        train, test = df.iloc[:s], df.iloc[s:e]
        model = DixonColes(halflife, reg).fit(train, ref_date=test["Date"].min())
        base = _base_rates(train)
        for _, r in test.iterrows():
            if not (model.knows(r["HomeTeam"]) and model.knows(r["AwayTeam"])):
                continue
            p = market_probs(model.score_matrix(r["HomeTeam"], r["AwayTeam"]))
            hg, ag = int(r["FTHG"]), int(r["FTAG"])
            row = {"hg": hg, "ag": ag}
            row.update({f"p_{k}": v for k, v in p.items()})
            row.update({f"b_{k}": v for k, v in base.items()})
            rows.append(row)
    return pd.DataFrame(rows)


def _ll_1x2(pred: pd.DataFrame, shrink: float):
    pm = pred[["p_H", "p_D", "p_A"]].to_numpy()
    pb = pred[["b_H", "b_D", "b_A"]].to_numpy()
    p = (1 - shrink) * pm + shrink * pb
    res = np.where(pred["hg"] > pred["ag"], 0, np.where(pred["hg"] == pred["ag"], 1, 2)).astype(int)
    idx = np.arange(len(res))
    ll_model = -np.mean(np.log(np.clip(p[idx, res], 1e-9, 1)))
    ll_base = -np.mean(np.log(np.clip(pb[idx, res], 1e-9, 1)))
    onehot = np.eye(3)[res]
    brier = float(np.mean(((p - onehot) ** 2).sum(axis=1)))
    acc = float(np.mean(p.argmax(axis=1) == res))
    return float(ll_model), float(ll_base), brier, acc


def _ll_binary(pred: pd.DataFrame, name: str, shrink: float):
    pm = pred[f"p_{name}"].to_numpy()
    pb = pred[f"b_{name}"].to_numpy()
    p = np.clip((1 - shrink) * pm + shrink * pb, 1e-6, 1 - 1e-6)
    pbc = np.clip(pb, 1e-6, 1 - 1e-6)
    y = np.array([BINARY[name](a, b) for a, b in zip(pred["hg"], pred["ag"])], dtype=float)
    ll_m = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    ll_b = -np.mean(y * np.log(pbc) + (1 - y) * np.log(1 - pbc))
    return float(ll_m), float(ll_b)


def select_model(df: pd.DataFrame, verbose: bool = True) -> dict:
    """
    Grid-search (half-life, ridge, shrink) by walk-forward 1X2 log-loss and
    return the winner plus its validation report and market gates.
    """
    best = None
    for hl, reg in itertools.product(HALFLIFE_GRID, REG_GRID):
        pred = walk_forward(df, hl, reg)
        if len(pred) < MIN_VAL_MATCHES:
            continue
        for sh in SHRINK_GRID:
            ll, ll_base, brier, acc = _ll_1x2(pred, sh)
            if best is None or ll < best["val_logloss"]:
                best = {"halflife": hl, "reg": reg, "shrink": sh,
                        "val_logloss": ll, "base_logloss": ll_base,
                        "val_brier": brier, "val_accuracy": acc,
                        "n_val": len(pred), "_pred": pred}
    if best is None:
        return {"ok": False, "reason": "not enough validation matches"}

    pred = best.pop("_pred")
    gates, detail = {}, {}
    improvement_1x2 = best["base_logloss"] - best["val_logloss"]
    gates["1X2"] = bool(improvement_1x2 >= MIN_IMPROVEMENT and best["n_val"] >= MIN_VAL_MATCHES)
    detail["1X2"] = {"model": best["val_logloss"], "base": best["base_logloss"],
                     "improvement": improvement_1x2}
    for name in BINARY:
        ll_m, ll_b = _ll_binary(pred, name, best["shrink"])
        gates[name] = bool(ll_b - ll_m >= MIN_IMPROVEMENT and best["n_val"] >= MIN_VAL_MATCHES)
        detail[name] = {"model": ll_m, "base": ll_b, "improvement": ll_b - ll_m}
    for alias, src in GATE_ALIAS.items():
        gates[alias] = gates[src]
        detail[alias] = detail[src]

    best.update({"ok": True, "gates": gates, "detail": detail})
    if verbose:
        print(f"  Best config: half-life {best['halflife']}d | ridge {best['reg']} | "
              f"shrink {best['shrink']:.2f}")
        print(f"  Validation ({best['n_val']} unseen matches, walk-forward):")
        print(f"    1X2 log-loss {best['val_logloss']:.4f} vs base-rate {best['base_logloss']:.4f} "
              f"(improvement {improvement_1x2:+.4f}) | accuracy {best['val_accuracy']*100:.1f}%")
        approved = [k for k, v in gates.items() if v]
        print(f"  Markets approved for betting: {', '.join(approved) if approved else 'NONE'}")
    return best
