"""Latent-state enterprise simulator emitting a temporal-knowledge-graph event stream.

Every week each asset carries a HIDDEN state:
  * configuration drift d, consecutive backup-failure streak, privilege creep;
  * open vulnerabilities (some exploited in the wild, some later listed as
    known-exploited, each with an asset-specific remediation time);
  * a compromise stage c in {0 clean, 1 foothold, 2 escalated, 3 impact}
    driven by exploitable exposure, credential attacks and lateral movement
    along dependency and shared-identity paths.
The hidden state defines a TRUE trust score

    TS*_{a,t} = 100 * h_{a,t} * (1 - pi[c_{a,t}]),   pi = (0, .10, .45, .85)

where h is latent control health. The audit layer never sees TS*; it sees
only time-stamped quadruples (subject, relation, object, week) emitted with
asset-specific logging coverage, scanner delay and false positives.

An optional campaign reproduces a MOVEit-type mass-exploitation event: a zero-day
exploited on internet-facing file-transfer, web-application and edge assets before
any CVE is visible, disclosed one week later and listed as known-exploited after two weeks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .enterprise import VULN_RATE, Enterprise, build_enterprise

PI_STAGE = np.array([0.0, 0.10, 0.45, 0.85])


def _cvss(rng):
    u = rng.random()
    if u < 0.10:
        return rng.uniform(1.0, 3.9)
    if u < 0.45:
        return rng.uniform(4.0, 6.9)
    if u < 0.85:
        return rng.uniform(7.0, 8.9)
    return rng.uniform(9.0, 10.0)


def simulate(seed: int, n_assets: int = 400, n_users: int = 250, weeks: int = 60,
             campaign_week: int | None = None, threat_scale: float = 1.0):
    rng = np.random.default_rng(seed)
    ent: Enterprise = build_enterprise(rng, n_assets, n_users)
    N = n_assets
    ids = np.array([f"A{i:03d}" for i in range(N)])
    A = ent.A_dep
    UA = ent.user_access

    d = rng.uniform(0, 0.2, N); bf = np.zeros(N); pc = rng.uniform(0, 0.2, N)
    c = np.zeros(N, int); camp = np.zeros(N, int); camp_age = np.zeros(N, int)
    dwell = np.zeros(N, int); backlog = np.zeros(N); pivots = []
    neglect = np.zeros(N, bool)      # latent regime: ownership/maintenance neglect (slow-burn erosion)
    audit_iv = rng.integers(12, 21, N); last_audit = -rng.integers(0, 16, N)
    vulns = [[] for _ in range(N)]
    vid = 0
    ev = []          # (t, s, r, o, value, cvss, kev)
    truth = []

    def emit(t, s, r, o, value=1.0, cvss=np.nan, kev=0):
        ev.append((t, s, r, o, float(value), float(cvss), int(kev)))

    # static structure as quadruples at t = 0
    for u, v, k in ent.G.edges(keys=True):
        emit(0, u, k, v)

    u_camp = rng.random(N)          # drawn unconditionally so that runs with/without campaign coincide before it
    camp_assets = np.where(np.isin(ent.asset_type, ["file_transfer", "edge", "web_app"]) & ent.internet
                           & (u_camp < 0.7))[0] if campaign_week is not None else np.array([], int)

    for t in range(weeks):
        # -------------------------------------------------- vulnerabilities
        if campaign_week is not None and t == campaign_week:
            for a in camp_assets:
                due = t + 1 + (2 if ent.kev_policy[a] else rng.lognormal(np.log(ent.patch_median_wk[a]), 0.5))
                vulns[a].append(dict(id=f"CVE-CAMP-{vid}", cvss=9.8, expl=True, found=t + 1, kev=t + 2,
                                     due=due, start=t, mass=3.0)); vid += 1
        for a in range(N):
            for _ in range(rng.poisson(VULN_RATE[ent.asset_type[a]])):
                cv = _cvss(rng)
                expl = rng.random() < (0.12 if cv >= 9 else 0.04) * threat_scale
                found = t + rng.geometric(0.55 + 0.4 * ent.coverage[a]) - 1
                kev = (found + rng.geometric(0.5)) if (expl and rng.random() < 0.85) else 10 ** 6
                due = t + rng.lognormal(np.log(ent.patch_median_wk[a]), 0.6) * (3.0 if neglect[a] else 1.0)
                if kev < 10 ** 6 and ent.kev_policy[a]:
                    due = min(due, kev + 2)
                vulns[a].append(dict(id=f"CVE-{vid}", cvss=cv, expl=expl, found=found, kev=kev, due=due, start=t))
                vid += 1
        E = np.zeros(N); n_open_true = np.zeros(N); overdue = np.zeros(N)
        for a in range(N):
            keep = []
            for v in vulns[a]:
                if v["found"] == t:
                    emit(t, v["id"], "affects", ids[a], 1.0, v["cvss"], 0)
                if v["kev"] == t and v["found"] <= t:
                    emit(t, v["id"], "listed_known_exploited", "KEV", 1.0, v["cvss"], 1)
                if t >= v["due"]:
                    if v["found"] <= t:
                        emit(t, ids[a], "remediates", v["id"], 1.0, v["cvss"], 0)
                    continue
                keep.append(v); n_open_true[a] += 1
                overdue[a] += (t - v["start"]) >= 4
                if v["expl"] and t >= v["start"]:      # exploitation ramps up after publication
                    E[a] += (v["cvss"] / 10.0) * (3.0 if ent.internet[a] else 1.0) * min((t - v["start"] + 1) / 3, 1) \
                        * v.get("mass", 1.0)                       # automated mass exploitation
            vulns[a] = keep

        # -------------------------------------------------- control erosion (regime-switching)
        p_in = 0.008 + 0.012 * (1 - ent.maturity)
        enter = ~neglect & (rng.random(N) < p_in)
        neglect = np.where(neglect, rng.random(N) > 0.06, enter)
        for a in np.where(enter & (rng.random(N) < 0.6))[0]:      # CMDB records an ownership/maintenance change
            emit(t, ids[a], "owner_changed", f"TEAM{rng.integers(20):02d}")
        ng = neglect.astype(float)
        d += np.abs(rng.normal(0, (0.006 + 0.02 * (1 - ent.maturity)) * (1 + 3 * ng)))
        fix = (rng.random(N) < 0.04 + 0.06 * ent.maturity) & ~neglect   # routine change-management correction
        d = np.where(fix, d * 0.5, d)
        bfail = rng.random(N) < (0.02 + 0.10 * (1 - ent.maturity) + 0.10 * np.minimum(d, 1) + 0.25 * ng)
        bf = np.where(bfail, bf + 1, 0)
        pc += (0.005 + 0.01 * (1 - ent.maturity)) * (1 + 4 * ng) * rng.uniform(0, 2, N)

        # -------------------------------------------------- threats & compromise
        # multi-week credential campaigns: failed-login bursts precede any success
        start = (camp == 0) & (rng.random(N) < np.where(ent.internet, 0.03, 0.004) * threat_scale)
        camp = np.where(start, rng.integers(2, 6, N), camp)
        camp_age = np.where(start, 0, camp_age + (camp > 0))
        attack = camp > 0
        cred = attack & (camp_age >= 1) & (rng.random(N) < 0.10 * (1 - 0.5 * ent.maturity) * (1 + np.minimum(pc, 1)))
        camp = np.maximum(camp - 1, 0)
        comp = (c >= 1).astype(float)
        # attackers pivot after escalation: lateral pressure from escalated neighbours
        lat = A @ (c == 2).astype(float)
        shared = UA.T @ np.minimum(UA @ (c == 2).astype(float), 1)
        shared = shared / np.maximum(np.asarray(UA.sum(0)).ravel(), 1)
        newc = c.copy()
        ext = (c == 0) & ((rng.random(N) < 1 - np.exp(-(0.0003 + 0.05 * E))) | cred)
        # targeted lateral movement: pivots scheduled by escalated attackers execute 1-3 weeks later
        piv = np.zeros(N, bool)
        for (tt, src, dst) in list(pivots):
            if tt == t:
                pivots.remove((tt, src, dst))
                if c[src] >= 1 and c[dst] == 0 and not ext[dst]:
                    piv[dst] = True
        piv |= (c == 0) & ~ext & (rng.random(N) < 1 - np.exp(-(0.005 * lat + 0.005 * shared)))
        newc[ext] = 1
        onset_type = np.where(cred & ext, 2, np.where(ext, 1, np.where(piv, 3, 0)))   # 1 exposure, 2 credential, 3 pivot
        newc[piv] = np.where(rng.random(piv.sum()) < 0.6, 2, 1)      # pivots often carry elevated credentials
        # escalation after a typical dwell time
        esc = (c == 1) & (rng.random(N) < np.where(dwell >= 2, 0.8, np.where(dwell >= 1, 0.4, 0.05)) + 0.1 * np.minimum(pc, 1))
        newc[esc] = 2
        imp = (c == 2) & (rng.random(N) < 0.22 * (1 + (bf > 0)))
        newc[imp] = 3
        detected = (c >= 1) & (c < 3) & (rng.random(N) < 0.20 + 0.35 * ent.coverage * c / 3)
        newc[detected] = 0
        rec = (c == 3) & (rng.random(N) < 0.5)
        newc[rec] = 0
        for src in np.where((newc == 2) & (c != 2))[0]:              # newly escalated: plan pivots
            nb = A.indices[A.indptr[src]:A.indptr[src + 1]]
            admins = UA[:, src].nonzero()[0]
            co = np.unique(UA[admins].nonzero()[1]) if len(admins) else np.array([], int)
            cand = np.unique(np.concatenate([nb, co[rng.random(len(co)) < 0.3]]))
            cand = cand[cand != src]
            if len(cand):
                w = ent.criticality[cand].astype(float) ** 3                  # attackers go for high-value targets
                for dst in rng.choice(cand, size=min(len(cand), rng.integers(1, 5)), replace=False, p=w / w.sum()):
                    if rng.random() < 0.6:
                        pivots.append((t + int(rng.integers(1, 4)), src, int(dst)))
        dwell = np.where(newc == c, dwell + 1, 0)
        lat = 0.10 * lat                                             # for the probing signal below
        c = newc

        # -------------------------------------------------- periodic audits
        due_audit = (t - last_audit) >= audit_iv
        for a in np.where(due_audit)[0]:
            emit(t, "AUDITOR", "audited", ids[a])
            d[a] *= 0.3; pc[a] *= 0.3; last_audit[a] = t; neglect[a] = False
            if c[a] in (1, 2) and rng.random() < 0.5:
                c[a] = 0

        # -------------------------------------------------- latent trust
        backlog = 0.8 * backlog + 0.2 * overdue          # remediation backlog (smoothed)
        h = 1 - (0.35 * np.minimum(d, 1) + 0.10 * np.minimum(bf / 3, 1) + 0.25 * np.minimum(pc, 1)
                 + 0.15 * np.minimum(backlog / 10, 1))
        h = np.clip(h, 0, 1)
        ts = 100 * h * (1 - PI_STAGE[c])
        for a in range(N):
            truth.append((t, a, ts[a], h[a], c[a], E[a], n_open_true[a], d[a], bf[a], pc[a], int(neglect[a]),
                          int(onset_type[a])))

        # -------------------------------------------------- observable events
        cov = ent.coverage
        fl = rng.poisson((1.5 + attack * (10 + 15 * camp_age) + 6 * (c >= 1)) * cov)
        cfg = rng.poisson(0.6 + 1.5 * (1 - ent.maturity) + 2.0 * ng)
        drift_scan = rng.random(N) < 0.5
        pe = rng.poisson(0.08, N) + ((c == 2) & (rng.random(N) < 0.5 * cov)).astype(int)
        al = rng.poisson(0.15 + np.array([0, 1.0, 2.0, 3.0])[c] * cov + 0.5 * lat * cov)   # incl. weak lateral-probing signal
        cvio = rng.poisson(0.03 + 0.25 * (1 - h))
        for a in range(N):
            if fl[a]:
                emit(t, "EXTERNAL" if ent.internet[a] else "INTERNAL", "failed_login", ids[a], fl[a])
            if cfg[a]:
                emit(t, f"U{rng.integers(n_users):03d}", "config_change", ids[a], cfg[a])
            if drift_scan[a]:
                emit(t, ids[a], "drift_from_baseline", "BASELINE", max(d[a] + rng.normal(0, 0.05), 0))
            if bfail[a] and rng.random() < cov[a]:
                emit(t, "BACKUP", "backup_failed", ids[a])
            if pe[a]:
                emit(t, f"U{rng.integers(n_users):03d}", "privilege_escalation", ids[a], pe[a])
            if al[a]:
                emit(t, "SIEM", "alert", ids[a], al[a])
            if cvio[a]:
                emit(t, "POLICY", "violated_by", ids[a], cvio[a])
        # identity activity: a sample of users touch entitled assets
        active = rng.random(n_users) < 0.3
        ua = UA[active].tocoo()
        users = np.where(active)[0]
        for r_, a_ in zip(ua.row, ua.col):
            if rng.random() < 0.5:
                emit(t, f"U{users[r_]:03d}", "accesses", ids[a_])

    events = pd.DataFrame(ev, columns=["t", "s", "r", "o", "value", "cvss", "kev"])
    tr = pd.DataFrame(truth, columns=["t", "asset", "ts_true", "true_health", "true_stage", "true_exposure",
                                      "true_open_vulns", "true_drift", "true_bf_streak", "true_priv_creep",
                                      "true_neglect", "true_onset"])
    return ent, events, tr
