"""Temporal knowledge graph (TKG) store and trust-feature extraction.

Events are quadruples (s, r, o, t). The TKG is updated incrementally per
audit window, G_k = G_{k-1} U dV_k U dE_k, preserving history. Features
for asset a at window t use only quadruples with timestamp <= t.

Feature families
  local     : the per-window trust indicators of the original design
              (vulnerabilities, patch delay, failed logins, drift, backup
              failures, privilege escalation, compliance, alerts)
  temporal  : exponentially decayed (half-life 1 and 4 windows) histories,
              observed-trust trend, known-exploited (KEV) overdue count
  graph     : 1- and 2-hop aggregation over typed dependency relations and
              time-varying shared-identity paths (users who recently accessed
              both assets), attack-path exposure, high-risk neighbours
"""
from __future__ import annotations

import time

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp

W_TS = dict(vc=0.20, pd=0.15, fl=0.10, cd=0.15, bf=0.10, pe=0.10, cv=0.10, se=0.10)   # observed trust weights


class TemporalKG:
    """Incremental TKG (NetworkX MultiDiGraph with time-stamped edges)."""

    def __init__(self):
        self.G = nx.MultiDiGraph()
        self.update_ms = []

    def update(self, batch: pd.DataFrame):
        t0 = time.perf_counter()
        for s, r, o, t, v in batch[["s", "r", "o", "t", "value"]].itertuples(index=False):
            if s not in self.G:
                self.G.add_node(s)
            if o not in self.G:
                self.G.add_node(o)
            self.G.add_edge(s, o, key=(r, int(t)), relation=r, t=int(t), value=v)
        self.update_ms.append((time.perf_counter() - t0) * 1e3)

    def stats(self):
        return dict(nodes=self.G.number_of_nodes(), quadruples=self.G.number_of_edges(),
                    relations=len({k[0] for _, _, k in self.G.edges(keys=True)}),
                    mean_update_ms=float(np.mean(self.update_ms)), max_update_ms=float(np.max(self.update_ms)))


def _ewma(X, hl):
    lam = 0.5 ** (1.0 / hl)
    out = np.zeros_like(X, dtype=float)
    acc = np.zeros(X.shape[1])
    for t in range(X.shape[0]):
        acc = lam * acc + (1 - lam) * X[t]
        out[t] = acc
    return out


def extract_features(ent, events: pd.DataFrame, weeks: int, build_graph: bool = False):
    N = ent.n_assets
    aidx = {f"A{i:03d}": i for i in range(N)}
    T = weeks

    def counts(rel, col="o", use_value=True):
        e = events[(events.r == rel) & events[col].isin(aidx)]
        M = np.zeros((T, N))
        np.add.at(M, (e.t.to_numpy(), e[col].map(aidx).to_numpy()), e.value.to_numpy() if use_value else 1.0)
        return M

    FL = counts("failed_login"); CFG = counts("config_change"); BF = counts("backup_failed", use_value=False)
    PE = counts("privilege_escalation"); AL = counts("alert"); CV = counts("violated_by")
    AUD = counts("audited", use_value=False)
    OWN = counts("owner_changed", col="s", use_value=False)
    # last observed drift (carried forward)
    dr = events[events.r == "drift_from_baseline"]
    DR = np.full((T, N), np.nan)
    DR[dr.t.to_numpy(), dr.s.map(aidx).to_numpy()] = dr.value.to_numpy()
    DR = pd.DataFrame(DR).ffill().fillna(0).to_numpy()

    # vulnerability ledger as seen by scanners / KEV feed
    aff = events[events.r == "affects"][["s", "o", "t", "cvss"]].rename(columns={"s": "id", "o": "asset", "t": "t_found"})
    rem = events[events.r == "remediates"][["o", "t"]].rename(columns={"o": "id", "t": "t_rem"})
    kev = events[events.r == "listed_known_exploited"][["s", "t"]].rename(columns={"s": "id", "t": "t_kev"})
    V = aff.merge(rem, on="id", how="left").merge(kev, on="id", how="left")
    V["t_rem"] = V.t_rem.fillna(10 ** 6).astype(int); V["t_kev"] = V.t_kev.fillna(10 ** 6).astype(int)
    VC = np.zeros((T, N)); MAXC = np.zeros((T, N)); KEVO = np.zeros((T, N)); KEVD = np.zeros((T, N)); AGE = np.zeros((T, N))
    for _, v in V.iterrows():
        a = aidx[v.asset]; t0, t1 = int(v.t_found), min(int(v.t_rem), T)
        if t0 >= T or t1 <= t0:
            continue
        sl = slice(t0, t1)
        VC[sl, a] += 1
        MAXC[sl, a] = np.maximum(MAXC[sl, a], v.cvss)
        AGE[sl, a] += np.arange(t1 - t0)
        if v.t_kev < t1:
            k0 = max(int(v.t_kev), t0)
            KEVO[k0:t1, a] += 1
            KEVD[min(k0 + 2, t1):t1, a] += 1       # overdue vs a 2-week remediation rule
    PD = np.where(VC > 0, AGE / np.maximum(VC, 1), 0)

    # weeks since last audit
    WSA = np.zeros((T, N)); last = np.full(N, -26.0)
    for t in range(T):
        last = np.where(AUD[t] > 0, t, last); WSA[t] = t - last

    # observed trust score (rule of the original design) and trend
    z = dict(vc=np.minimum(VC / 10, 1), pd=np.minimum(PD / 12, 1), fl=np.minimum(FL / 50, 1), cd=np.minimum(DR, 1),
             bf=np.minimum(BF, 1), pe=np.minimum(PE / 3, 1), cv=np.minimum(CV / 3, 1), se=np.minimum(AL / 5, 1))
    TS_OBS = 100 * (1 - sum(W_TS[k] * z[k] for k in W_TS))
    SLOPE = np.zeros((T, N))
    for t in range(T):
        lo = max(0, t - 3)
        if t - lo >= 1:
            x = np.arange(lo, t + 1) - np.arange(lo, t + 1).mean()
            SLOPE[t] = (x[:, None] * (TS_OBS[lo:t + 1] - TS_OBS[lo:t + 1].mean(0))).sum(0) / (x ** 2).sum()

    # observed trust at the last audit (assurance reference), carried forward
    REF = np.zeros((T, N)); ref = TS_OBS[0].copy()
    for t in range(T):
        ref = np.where(AUD[t] > 0, TS_OBS[t], ref); REF[t] = ref
    DROP = REF - TS_OBS

    ew = {}
    for name, M in dict(fl=FL, cfg=CFG, bf=BF, pe=PE, al=AL, cv=CV, own=OWN).items():
        ew[f"{name}_ewm1"] = _ewma(M, 1); ew[f"{name}_ewm4"] = _ewma(M, 4)

    # ---------------------------------------------------------- graph features
    A = ent.A_dep.copy(); A.setdiag(0); A.eliminate_zeros()
    A2 = (A @ A); A2.setdiag(0); A2.data[:] = 1.0; A2 = A2.tocsr()
    deg = np.asarray(A.sum(1)).ravel()
    acc = events[events.r == "accesses"]
    uidx = {u: i for i, u in enumerate(sorted(acc.s.unique()))}
    g = {k: np.zeros((T, N)) for k in ["nb_max_al1", "nb_alert_jump", "nb_max_pe2", "hop2_max_al1", "nb_mean_al4", "nb_max_al4", "hop2_max_al4", "nb_max_kevo", "nb_sum_pe4",
                                       "shared_user_alert", "path_kev_exposure", "nb_lowtrust"]}
    inet = ent.internet.astype(float)
    for t in range(T):
        al4 = ew["al_ewm4"][t]; pe4 = ew["pe_ewm4"][t]
        g["nb_max_al1"][t] = _sparse_rowmax(A, AL[t])
        g["nb_alert_jump"][t] = _sparse_rowmax(A, np.maximum(ew["al_ewm1"][t] - ew["al_ewm4"][t], 0))
        g["nb_max_pe2"][t] = _sparse_rowmax(A, PE[t] + (PE[t - 1] if t else 0))
        g["hop2_max_al1"][t] = _sparse_rowmax(A2, AL[t])
        g["nb_mean_al4"][t] = (A @ al4) / np.maximum(deg, 1)
        g["nb_max_al4"][t] = _sparse_rowmax(A, al4)
        g["hop2_max_al4"][t] = _sparse_rowmax(A2, al4)
        g["nb_max_kevo"][t] = _sparse_rowmax(A, KEVO[t])
        g["nb_sum_pe4"][t] = A @ pe4
        g["nb_lowtrust"][t] = A @ (TS_OBS[t] < 50).astype(float)
        exposed = inet * (KEVO[t] > 0)
        g["path_kev_exposure"][t] = np.minimum((A @ exposed) + (A2 @ exposed), 1)
        a_t = acc[(acc.t <= t) & (acc.t > t - 4)]
        if len(a_t):
            U = sp.csr_matrix((np.ones(len(a_t)), (a_t.s.map(uidx).to_numpy(), a_t.o.map(aidx).to_numpy())),
                              shape=(len(uidx), N)); U.data[:] = 1.0
            hot = (ew["al_ewm1"][t] > 0.8).astype(float)
            u_hot = np.minimum(U @ hot, 1)
            touched = np.asarray(U.sum(0)).ravel()
            g["shared_user_alert"][t] = (U.T @ u_hot) / np.maximum(touched, 1)

    rows = []
    tt, aa = np.meshgrid(np.arange(T), np.arange(N), indexing="ij")
    F = pd.DataFrame({"t": tt.ravel(), "asset": aa.ravel()})
    base = dict(vc_open=VC, max_cvss_open=MAXC, kev_open=KEVO, kev_overdue=KEVD, patch_delay=PD,
                failed_logins=FL, config_changes=CFG, drift=DR, backup_failed=BF, priv_escalations=PE,
                alerts=AL, compliance_viol=CV, weeks_since_audit=WSA, ts_obs=TS_OBS, ts_obs_slope=SLOPE,
                ts_obs_ref=REF, ts_obs_drop=DROP, audited_now=AUD)
    for k, M in {**base, **ew, **g}.items():
        F[k] = M.ravel()
    F["asset_type"] = np.tile(ent.asset_type, T)
    F["criticality"] = np.tile(ent.criticality, T)
    F["internet"] = np.tile(ent.internet.astype(int), T)
    F["degree"] = np.tile(deg, T)

    tkg_stats = None
    if build_graph:
        kg = TemporalKG()
        for t in range(T):
            kg.update(events[events.t == t])
        tkg_stats = kg.stats()
    return F, tkg_stats


def _sparse_rowmax(A: sp.csr_matrix, x: np.ndarray) -> np.ndarray:
    out = np.zeros(A.shape[0])
    vals = x[A.indices]
    for i in range(A.shape[0]):
        s, e = A.indptr[i], A.indptr[i + 1]
        if e > s:
            out[i] = vals[s:e].max()
    return out


LOCAL = ["vc_open", "max_cvss_open", "patch_delay", "failed_logins", "config_changes", "drift", "backup_failed",
         "priv_escalations", "alerts", "compliance_viol", "weeks_since_audit", "criticality", "internet", "asset_type"]
TEMPORAL = ["kev_open", "kev_overdue", "ts_obs", "ts_obs_slope", "ts_obs_drop"] + [f"{k}_ewm{h}" for k in
                                                                     ["fl", "cfg", "bf", "pe", "al", "cv", "own"] for h in (1, 4)]
GRAPH = ["nb_max_al1", "nb_alert_jump", "nb_max_pe2", "hop2_max_al1", "nb_mean_al4", "nb_max_al4", "hop2_max_al4", "nb_max_kevo", "nb_sum_pe4", "shared_user_alert",
         "path_kev_exposure", "nb_lowtrust", "degree"]
