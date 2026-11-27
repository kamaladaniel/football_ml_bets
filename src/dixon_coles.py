"""
Dixon-Coles style goal model (time-weighted, ridge-regularised).

Why this replaces the old LightGBM classifier
---------------------------------------------
* One coherent model produces a full scoreline distribution, so 1X2,
  Over/Under, BTTS and team-totals are all consistent with each other
  (the old code used one model for 1X2 and a crude separate formula for goals).
* Team attack/defence strengths are estimated jointly with home advantage,
  so strength of schedule is handled correctly.
* Recent matches count more (exponential time decay).
* Ridge shrinkage stops small-sample teams from getting extreme ratings.
* Few parameters -> far less overfitting than 400 boosted trees on ~700 rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

MAX_GOALS = 10
_G = np.arange(MAX_GOALS + 1)
_LOGFACT = gammaln(_G + 1)
_I, _J = np.indices((MAX_GOALS + 1, MAX_GOALS + 1))
_TOTAL = _I + _J


class DixonColes:
    def __init__(self, halflife_days: float = 365.0, reg: float = 4.0):
        self.halflife_days = float(halflife_days)
        self.reg = float(reg)
        self.teams: list[str] = []
        self.team_idx: dict[str, int] = {}
        self.mu = 0.0
        self.home_adv = 0.0
        self.attack = np.zeros(0)
        self.defence = np.zeros(0)
        self.rho = 0.0
        self.n_matches = 0
        self.fitted = False

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame, ref_date=None) -> "DixonColes":
        df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        if ref_date is None:
            ref_date = df["Date"].max()
        ref_date = pd.Timestamp(ref_date)

        teams = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        hi = df["HomeTeam"].map(idx).to_numpy()
        ai = df["AwayTeam"].map(idx).to_numpy()
        x = df["FTHG"].to_numpy(dtype=float)
        y = df["FTAG"].to_numpy(dtype=float)

        days_ago = (ref_date - df["Date"]).dt.days.to_numpy(dtype=float)
        days_ago = np.clip(days_ago, 0, None)
        w = 0.5 ** (days_ago / self.halflife_days)
        reg = self.reg

        def fun(theta):
            mu, h = theta[0], theta[1]
            a = theta[2:2 + n]
            d = theta[2 + n:]
            eh = mu + h + a[hi] - d[ai]
            ea = mu + a[ai] - d[hi]
            lh, la = np.exp(eh), np.exp(ea)
            nll = -(w * (x * eh - lh + y * ea - la)).sum() + reg * (a @ a + d @ d)
            gh = -w * (x - lh)
            ga = -w * (y - la)
            g = np.zeros_like(theta)
            g[0] = gh.sum() + ga.sum()
            g[1] = gh.sum()
            g[2:2 + n] = np.bincount(hi, gh, n) + np.bincount(ai, ga, n) + 2 * reg * a
            g[2 + n:] = -np.bincount(ai, gh, n) - np.bincount(hi, ga, n) + 2 * reg * d
            return nll, g

        theta0 = np.zeros(2 + 2 * n)
        theta0[0] = np.log(max(0.5, (x.mean() + y.mean()) / 2))
        res = minimize(fun, theta0, jac=True, method="L-BFGS-B",
                       options={"maxiter": 300})
        th = res.x
        self.teams, self.team_idx = teams, idx
        self.mu, self.home_adv = float(th[0]), float(th[1])
        self.attack, self.defence = th[2:2 + n], th[2 + n:]

        # Dixon-Coles low-score dependence parameter (1-D weighted search)
        lh = np.exp(self.mu + self.home_adv + self.attack[hi] - self.defence[ai])
        la = np.exp(self.mu + self.attack[ai] - self.defence[hi])
        best_rho, best_ll = 0.0, -np.inf
        for rho in np.linspace(-0.25, 0.10, 36):
            tau = np.ones_like(lh)
            m00 = (x == 0) & (y == 0)
            m01 = (x == 0) & (y == 1)
            m10 = (x == 1) & (y == 0)
            m11 = (x == 1) & (y == 1)
            tau[m00] = 1 - lh[m00] * la[m00] * rho
            tau[m01] = 1 + lh[m01] * rho
            tau[m10] = 1 + la[m10] * rho
            tau[m11] = 1 - rho
            if (tau <= 0).any():
                continue
            ll = (w * np.log(tau)).sum()
            if ll > best_ll:
                best_ll, best_rho = ll, rho
        self.rho = float(best_rho)
        self.n_matches = len(df)
        self.fitted = True
        return self

    # -------------------------------------------------------------- predict
    def knows(self, team: str) -> bool:
        return team in self.team_idx

    def lambdas(self, home: str, away: str) -> tuple[float, float]:
        """Expected goals. Unknown teams are treated as league-average."""
        ha = self.attack[self.team_idx[home]] if home in self.team_idx else 0.0
        hd = self.defence[self.team_idx[home]] if home in self.team_idx else 0.0
        aa = self.attack[self.team_idx[away]] if away in self.team_idx else 0.0
        ad = self.defence[self.team_idx[away]] if away in self.team_idx else 0.0
        lh = np.exp(self.mu + self.home_adv + ha - ad)
        la = np.exp(self.mu + aa - hd)
        return float(lh), float(la)

    def score_matrix(self, home: str, away: str) -> np.ndarray:
        lh, la = self.lambdas(home, away)
        return scoreline_matrix(lh, la, self.rho)


def scoreline_matrix(lh: float, la: float, rho: float = 0.0) -> np.ndarray:
    ph = np.exp(-lh + _G * np.log(lh) - _LOGFACT)
    pa = np.exp(-la + _G * np.log(la) - _LOGFACT)
    m = np.outer(ph, pa)
    m[0, 0] *= 1 - lh * la * rho
    m[0, 1] *= 1 + lh * rho
    m[1, 0] *= 1 + la * rho
    m[1, 1] *= 1 - rho
    m = np.clip(m, 0, None)
    return m / m.sum()


def market_probs(m: np.ndarray) -> dict:
    """All supported markets from one scoreline matrix."""
    return {
        "H": float(m[_I > _J].sum()),
        "D": float(m[_I == _J].sum()),
        "A": float(m[_I < _J].sum()),
        "Over15": float(m[_TOTAL >= 2].sum()),
        "Over25": float(m[_TOTAL >= 3].sum()),
        "Under25": float(m[_TOTAL <= 2].sum()),
        "Under35": float(m[_TOTAL <= 3].sum()),
        "HomeOver15": float(m[_I >= 2].sum()),
        "AwayOver15": float(m[_J >= 2].sum()),
        "BTTS_Yes": float(m[(_I >= 1) & (_J >= 1)].sum()),
        "BTTS_No": float(m[(_I == 0) | (_J == 0)].sum()),
    }
