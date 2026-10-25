"""Trust-decay targets.

The audit question is how far an asset will have fallen from its last ASSURED state (its most
recent audit) by the end of the forecast horizon H:

    TDS*_{a,t} = max(0, TS*_{a,tau(a,t)} - min_{1<=k<=H} TS*_{a,t+k}),

where tau(a,t) is the last audit of asset a at or before t and TS* is the HIDDEN true trust score.
Tiers use the thresholds of the original design: Low <= 5 < Moderate <= 15 < High <= 30 < Critical.

Protocol A (index labels, as in the earlier prototype) applies the same formula to the OBSERVED
rule-based trust score TS_obs, which is a deterministic function of the model inputs.
"""
import numpy as np
import pandas as pd

THRESH = [5, 15, 30]
TIERS = ["Low", "Moderate", "High", "Critical"]


def tier(tds) -> np.ndarray:
    return np.digitize(np.asarray(tds, float), THRESH, right=True)


def decay_since_audit(D: pd.DataFrame, col: str, horizon: int) -> pd.Series:
    """D must contain t, asset, audited_now and `col`; returns TDS aligned with D.index."""
    D = D.sort_values(["asset", "t"])
    ref = D[col].where(D.audited_now > 0)
    first = D.groupby("asset")[col].transform("first")
    ref = ref.groupby(D.asset).ffill().fillna(first)
    g = D.groupby("asset")[col]
    fut = pd.concat([g.shift(-k) for k in range(1, horizon + 1)], axis=1).min(axis=1, skipna=False)
    return (ref - fut).clip(lower=0).reindex(D.index)


def future_onset(D: pd.DataFrame, horizon: int) -> pd.Series:
    """Cause of the first compromise onset within the horizon (0 none, 1 exposure, 2 credential, 3 pivot)."""
    D = D.sort_values(["asset", "t"])
    g = D.groupby("asset").true_onset
    fut = pd.concat([g.shift(-k) for k in range(1, horizon + 1)], axis=1).fillna(0).to_numpy()
    first = np.zeros(len(D), int)
    for k in range(fut.shape[1] - 1, -1, -1):
        first = np.where(fut[:, k] > 0, fut[:, k], first)
    return pd.Series(first, index=D.index).reindex(D.index)
