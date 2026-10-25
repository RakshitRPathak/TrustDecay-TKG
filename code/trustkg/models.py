"""Predictors: audit-practice baselines, the prototype analogue and the proposed TKG model."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from .labels import tier
from .tkg import GRAPH, LOCAL, TEMPORAL

# feature vector of the original design: X = [VC, PD, FL, CD, BF, PE, CV, SE, HN, HT]
PROTOTYPE = ["vc_open", "patch_delay", "failed_logins", "drift", "backup_failed", "priv_escalations",
             "compliance_viol", "alerts", "nb_lowtrust", "ts_obs_slope"]
FEATURE_SETS = {
    "prototype": PROTOTYPE,
    "local": LOCAL,
    "local+temporal": LOCAL + TEMPORAL,
    "full": LOCAL + TEMPORAL + GRAPH,
}
GROUPS = {   # for explanations
    "vulnerability": ["vc_open", "max_cvss_open", "patch_delay", "kev_open", "kev_overdue"],
    "identity": ["failed_logins", "fl_ewm1", "fl_ewm4", "priv_escalations", "pe_ewm1", "pe_ewm4"],
    "configuration": ["config_changes", "cfg_ewm1", "cfg_ewm4", "drift", "own_ewm1", "own_ewm4"],
    "resilience": ["backup_failed", "bf_ewm1", "bf_ewm4"],
    "compliance": ["compliance_viol", "cv_ewm1", "cv_ewm4", "weeks_since_audit"],
    "detection": ["alerts", "al_ewm1", "al_ewm4"],
    "trust trend": ["ts_obs", "ts_obs_slope", "ts_obs_drop"],
    "graph context": GRAPH + ["nb_p_mean", "nb_p_max", "user_p_max"],
    "asset profile": ["criticality", "internet", "asset_type"],
}


def _pre(cols):
    cat = [c for c in cols if c == "asset_type"]
    num = [c for c in cols if c != "asset_type"]
    return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat), ("num", StandardScaler(), num)])


def make(name, cols, seed):
    if name == "RF":
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample",
                                     n_jobs=-1, random_state=seed)
    elif name == "LR":
        clf = LogisticRegression(max_iter=3000, class_weight="balanced")
    elif name == "HGB":
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                             l2_regularization=1.0, random_state=seed)
    else:
        raise ValueError(name)
    return Pipeline([("pre", _pre(cols)), ("clf", clf)])


def fit(model, X, y):
    if isinstance(model.named_steps["clf"], HistGradientBoostingClassifier):
        return model.fit(X, y, clf__sample_weight=compute_sample_weight("balanced", y))
    return model.fit(X, y)


def proba4(model, X):
    """Class probabilities over all four tiers (robust to classes absent in training)."""
    p = model.predict_proba(X)
    out = np.zeros((len(X), 4))
    out[:, model.classes_] = p
    return out


# ------------------------------------------------------------------ relational stacking
def neighbour_aggregates(D: pd.DataFrame, p_sev: np.ndarray, A: sp.csr_matrix, UA: sp.csr_matrix) -> pd.DataFrame:
    """Aggregate first-stage severe-risk predictions over dependency neighbours and shared users, per window."""
    out = np.zeros((len(D), 3))
    deg = np.maximum(np.asarray(A.sum(1)).ravel(), 1)
    t_vals = D.t.to_numpy(); a_vals = D.asset.to_numpy()
    N = A.shape[0]
    for t in np.unique(t_vals):
        m = t_vals == t
        p = np.zeros(N); p[a_vals[m]] = p_sev[m]
        mean = (A @ p) / deg
        mx = np.zeros(N)
        for i in range(N):
            s, e = A.indptr[i], A.indptr[i + 1]
            if e > s:
                mx[i] = p[A.indices[s:e]].max()
        u_max = UA.multiply(p[None, :]).max(axis=1).toarray().ravel()           # riskiest asset per user
        upm = np.asarray(UA.T.multiply(u_max[None, :]).max(axis=1).todense()).ravel()  # riskiest co-user exposure
        out[m, 0] = mean[a_vals[m]]; out[m, 1] = mx[a_vals[m]]; out[m, 2] = upm[a_vals[m]]
    return pd.DataFrame(out, index=D.index, columns=["nb_p_mean", "nb_p_max", "user_p_max"])


class StackedTKG:
    """Two-stage relational stacking: stage-1 HGB -> neighbour risk aggregates -> stage-2 HGB."""

    def __init__(self, cols, seed, A, UA, n_blocks=4):
        self.cols, self.seed, self.A, self.UA, self.n_blocks = cols, seed, A, UA, n_blocks

    def fit(self, D, y):
        y = np.asarray(y)
        oof = np.zeros(len(D))
        times = np.sort(D.t.unique()); blocks = np.array_split(times, self.n_blocks)
        for b in blocks:                                    # out-of-fold first-stage predictions
            te = D.t.isin(b).to_numpy()
            m = fit(make("HGB", self.cols, self.seed), D.loc[~te, self.cols], y[~te])
            oof[te] = proba4(m, D.loc[te, self.cols])[:, 2:].sum(1)
        self.s1 = fit(make("HGB", self.cols, self.seed), D[self.cols], y)
        X2 = pd.concat([D[self.cols], neighbour_aggregates(D, oof, self.A, self.UA)], axis=1)
        self.cols2 = list(X2.columns)
        self.s2 = fit(make("HGB", self.cols2, self.seed), X2, y)
        return self

    def stage2_frame(self, D):
        p1 = proba4(self.s1, D[self.cols])[:, 2:].sum(1)
        return pd.concat([D[self.cols], neighbour_aggregates(D, p1, self.A, self.UA)], axis=1)

    def predict_proba4(self, D, X2=None):
        X2 = self.stage2_frame(D) if X2 is None else X2
        return proba4(self.s2, X2)


# ------------------------------------------------------------------ rule baselines (risk scores)
def rule_scores(D: pd.DataFrame, H: int):
    """Higher = audit first. Returns dict name -> (score, predicted tier or None)."""
    pred_drop = D.ts_obs_drop + H * np.maximum(-D.ts_obs_slope, 0)
    return {
        "Audit rotation": (D.weeks_since_audit.to_numpy(), None),
        "CVSS-max": (D.max_cvss_open.to_numpy() + 0.01 * D.vc_open.to_numpy(), None),
        "KEV-first": (10 * D.kev_overdue.to_numpy() + 5 * D.kev_open.to_numpy() + D.max_cvss_open.to_numpy() / 10, None),
        "Trust-score rule": (pred_drop.to_numpy(), tier(pred_drop)),
    }
