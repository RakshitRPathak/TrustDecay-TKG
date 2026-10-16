"""Classification, ranking and audit-budget metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, recall_score, roc_auc_score


def classification(y, yhat):
    rec = recall_score(y, yhat, labels=[0, 1, 2, 3], average=None, zero_division=0)
    return dict(accuracy=accuracy_score(y, yhat), macro_f1=f1_score(y, yhat, average="macro", zero_division=0),
                weighted_f1=f1_score(y, yhat, average="weighted", zero_division=0),
                recall_high=rec[2], recall_critical=rec[3])


def ranking(D: pd.DataFrame, score: np.ndarray, budget: float):
    """Per-window top-B audit selection metrics, averaged over windows.

    D needs columns t, y (tier), criticality. Severe = tier >= High.
    """
    sev = (D.y.to_numpy() >= 2)
    out = dict(auc=roc_auc_score(sev, score) if 0 < sev.mean() < 1 else np.nan,
               ap=average_precision_score(sev, score) if sev.any() else np.nan)
    P, R, ND, CW, CR = [], [], [], [], []
    tmp = pd.DataFrame({"t": D.t.to_numpy(), "y": D.y.to_numpy(), "crit": D.criticality.to_numpy(), "s": score})
    for _, g in tmp.groupby("t"):
        k = max(1, int(round(budget * len(g))))
        top = g.nlargest(k, "s")
        sev_g = g.y >= 2
        P.append((top.y >= 2).mean())
        if sev_g.any():
            R.append((top.y >= 2).sum() / sev_g.sum())
            w = g.crit * sev_g
            CW.append((top.crit * (top.y >= 2)).sum() / w.sum())
        crit_g = g.y == 3
        if crit_g.any():
            CR.append((top.y == 3).sum() / min(crit_g.sum(), k))
        gains = (2.0 ** top.y.to_numpy() - 1)
        disc = 1 / np.log2(np.arange(2, k + 2))
        ideal = (2.0 ** np.sort(g.y.to_numpy())[::-1][:k] - 1)
        ND.append((gains * disc).sum() / max((ideal * disc).sum(), 1e-12))
    out.update(precision_at_B=np.mean(P), recall_at_B=np.mean(R) if R else np.nan, ndcg_at_B=np.mean(ND),
               crit_weighted_capture=np.mean(CW) if CW else np.nan, critical_hit_rate=np.mean(CR) if CR else np.nan)
    return out
