"""Audit-selection rules with miss-rate control.

Score s = predicted P(tier >= High). An asset is flagged for audit when s >= q.

Static (split) conformal: q is the alpha-quantile of the scores of SEVERE calibration
encounters, so P(flag | severe) >= 1 - alpha under exchangeability (class-conditional
split conformal; the guarantee is on the miss rate of severe assets).

Delayed-feedback adaptive conformal (ACI, Gibbs & Candes 2021, adapted to label delay H):
the working level alpha_t is updated once the labels of window t-H are revealed,

    alpha_{t+1} = alpha_t + gamma * (alpha - err_{t-H}),

where err_{t-H} is the fraction of severe assets of window t-H that were not flagged
at that time. alpha_t <= 0 flags every asset and alpha_t >= 1 flags none, so alpha_t
stays within [-(H+1)gamma, 1+(H+1)gamma] and, telescoping the updates,

    | (1/T) sum_t err_t - alpha | <= (1 + 2(H+1)gamma) / (gamma T)

for ANY sequence of scores and labels (no exchangeability), i.e., the long-run miss rate
converges to alpha. The threshold uses the severe scores of the most recent
labelled windows (rolling calibration).
"""
from __future__ import annotations

import numpy as np


def lower_quantile(scores, a):
    """Threshold q such that a fraction >= 1 - a of `scores` satisfies s >= q (finite-sample corrected)."""
    s = np.sort(np.asarray(scores))
    n = len(s)
    if n == 0 or a >= 1:
        return np.inf
    if a <= 0:
        return -np.inf
    k = int(np.floor(a * (n + 1)))          # number of calibration points allowed below q
    return -np.inf if k < 1 else s[k - 1] + 1e-12


def static_conformal(cal_scores_severe, alpha):
    return lower_quantile(cal_scores_severe, alpha)


def delayed_aci(stream, alpha, gamma, H, init_scores, window_memory=8):
    """Online audit selection.

    stream: list of (t, scores[np.array], severe[np.array bool]) in time order; labels of
            window t become usable at time t + H.
    init_scores: severe calibration scores available at deployment start.
    Returns per-window dicts with threshold, flagged fraction, realised miss rate and alpha_t.
    """
    a_t = alpha
    memory = [np.asarray(init_scores)]
    pending = []          # (reveal_time, scores, severe, threshold)
    out = []
    for t, s, sev in stream:
        # reveal labels whose delay has elapsed
        ready = [p for p in pending if p[0] <= t]
        pending = [p for p in pending if p[0] > t]
        for _, s_old, sev_old, q_old in ready:
            if sev_old.any():
                err = float(np.mean(s_old[sev_old] < q_old))
                a_t = a_t + gamma * (alpha - err)
                memory.append(s_old[sev_old])
                memory = memory[-window_memory:]
        q = -np.inf if a_t <= 0 else (np.inf if a_t >= 1 else lower_quantile(np.concatenate(memory), a_t))
        flag = s >= q
        miss = float(np.mean(~flag[sev])) if sev.any() else np.nan
        out.append(dict(t=t, alpha_t=a_t, threshold=q, flagged=float(flag.mean()), miss=miss, n_severe=int(sev.sum())))
        pending.append((t + H, s, sev, q))
    return out
