"""Reproduce every table and figure of the trust-decay manuscript.

    python main.py            # full study: 10 enterprises (seeds), 400 assets x 80 weeks each
    python main.py --quick    # smoke test: 2 seeds

Outputs are written to ../results (Code Ocean: /results).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, cohen_kappa_score, confusion_matrix

from trustkg import figures
from trustkg.conformal import delayed_aci, static_conformal
from trustkg.evaluate import classification, ranking
from trustkg.labels import decay_since_audit, future_onset, tier
from trustkg.models import FEATURE_SETS, GROUPS, StackedTKG, fit, make, proba4, rule_scores
from trustkg.simulate import simulate
from trustkg.tkg import extract_features

HERE = os.path.dirname(os.path.abspath(__file__))
H = 4                       # forecast horizon (windows)
WEEKS = 80
BURN = 24
TRAIN, CAL, TEST = (24, 43), (48, 51), (56, 75)   # 4-window embargo between blocks
CAMPAIGN = 64
BUDGETS = (0.05, 0.10)
ALPHAS, GAMMA = (0.10, 0.20, 0.30), 0.05
CANONICAL_SEED = 2026


def build(seed, campaign=None, graph_stats=False):
    ent, ev, tr = simulate(seed, weeks=WEEKS, campaign_week=campaign)
    t0 = time.perf_counter()
    F, st = extract_features(ent, ev, WEEKS, build_graph=graph_stats)
    fe_ms = (time.perf_counter() - t0) / WEEKS * 1e3
    D = F.merge(tr, on=["t", "asset"])
    D["tds_true"] = decay_since_audit(D, "ts_true", H)
    D["tds_obs"] = decay_since_audit(D, "ts_obs", H)
    D["onset_cause"] = future_onset(D, H)
    D = D[(D.t >= BURN) & D.tds_true.notna() & D.tds_obs.notna()].copy()
    D["y"] = tier(D.tds_true); D["yA"] = tier(D.tds_obs)
    D["asset_type"] = D.asset_type.astype(str)
    return ent, ev, D, st, fe_ms


def block(D, rng_):
    return D[(D.t >= rng_[0]) & (D.t <= rng_[1])]


def run_seed(seed, keep=False):
    ent, ev, D, st, fe_ms = build(seed, graph_stats=keep)
    _, _, Ds, _, _ = build(seed, campaign=CAMPAIGN)
    tr, ca, te = block(D, TRAIN), block(D, CAL), block(D, TEST)
    te_s = block(Ds, TEST)
    rows, rank_rows, extra = [], [], {}

    def evaluate(name, family, P=None, score=None, yhat=None, data=te):
        r = dict(method=name, family=family, seed=seed)
        if yhat is not None:
            r.update(classification(data.y, yhat))
        s = score if score is not None else P[:, 2:].sum(1)
        for b in BUDGETS:
            for k, v in ranking(data, s, b).items():
                r[f"{k}@{int(b * 100)}" if k not in ("auc", "ap") else k] = v
        rows.append(r)

    for name, (s, yh) in rule_scores(te, H).items():
        evaluate(name, "rule", score=s, yhat=yh)

    models = {}
    PROP = "TKG-HGB (proposed)"
    specs = [("Prototype RF", "RF", "prototype"), ("LR (full)", "LR", "full"), ("HGB (local)", "HGB", "local"),
             ("HGB (local+temporal)", "HGB", "local+temporal"), (PROP, "HGB", "full")]
    for name, algo, fs in specs:
        cols = FEATURE_SETS[fs]
        t0 = time.perf_counter()
        m = fit(make(algo, cols, seed), tr[cols], tr.y)
        if name == PROP:
            fit_s = time.perf_counter() - t0
            t0 = time.perf_counter(); P = proba4(m, te[cols]); inf_ms = (time.perf_counter() - t0) / len(te) * 1e3
        else:
            P = proba4(m, te[cols])
        models[name] = (m, cols)
        evaluate(name, "proposed" if name == PROP else "learned", P=P, yhat=P.argmax(1))
    m_prop, cols_prop = models[PROP]
    P_prop = proba4(m_prop, te[cols_prop]); p_te = P_prop[:, 2:].sum(1)
    stk = StackedTKG(FEATURE_SETS["full"], seed, ent.A_dep, ent.user_access).fit(tr, tr.y.to_numpy())
    Ps = stk.predict_proba4(te)
    evaluate("TKG-HGB + relational stacking", "learned", P=Ps, yhat=Ps.argmax(1))

    # -------------------------------------------- relational value by cause of future compromise
    for name, s in [("HGB (local+temporal)", proba4(*[models["HGB (local+temporal)"][0], te[FEATURE_SETS["local+temporal"]]])[:, 2:].sum(1)),
                    (PROP, p_te), ("TKG-HGB + relational stacking", Ps[:, 2:].sum(1))]:
        tmp = te.assign(s=s)
        tmp["top"] = tmp.groupby("t").s.rank(ascending=False, method="first") <= round(0.10 * 400)
        for cause, lab in [(0, "no compromise (control erosion)"), (1, "exposure exploit"), (2, "credential campaign"),
                           (3, "lateral pivot")]:
            sub = tmp[(tmp.y >= 2) & (tmp.onset_cause == cause)]
            rank_rows.append(dict(seed=seed, model=name, cause=lab, n=len(sub), recall_top10=sub.top.mean() if len(sub) else np.nan))

    # -------------------------------------------- conformal audit selection (proposed scores)
    p_ca = proba4(m_prop, ca[cols_prop])[:, 2:].sum(1)
    cal_sev = p_ca[ca.y.to_numpy() >= 2]
    conf = []
    for label, data in [("nominal", te), ("campaign", te_s)]:
        p = p_te if label == "nominal" else proba4(m_prop, data[cols_prop])[:, 2:].sum(1)
        stream = [(t, p[(data.t == t).to_numpy()], (data.y[data.t == t] >= 2).to_numpy()) for t in sorted(data.t.unique())]
        for alpha in ALPHAS:
            q = static_conformal(cal_sev, alpha)
            for t, s_, sev in stream:
                conf.append(dict(seed=seed, regime=label, alpha=alpha, rule="split conformal", t=t,
                                 flagged=float((s_ >= q).mean()),
                                 miss=float(np.mean(s_[sev] < q)) if sev.any() else np.nan, n_severe=int(sev.sum())))
            for r in delayed_aci(stream, alpha, GAMMA, H, cal_sev):
                r.update(seed=seed, regime=label, alpha=alpha, rule="delayed ACI"); conf.append(r)
        for t, s_, sev in stream:     # fixed budget for reference
            k = int(0.10 * len(s_)); thr = np.sort(s_)[-k]
            conf.append(dict(seed=seed, regime=label, alpha=np.nan, rule="top-10% budget", t=t,
                             flagged=float((s_ >= thr).mean()),
                             miss=float(np.mean(s_[sev] < thr)) if sev.any() else np.nan, n_severe=int(sev.sum())))

    # campaign regime: ranking quality of every method on the shifted enterprise
    shift_rows = []
    for name, (s, _) in rule_scores(te_s, H).items():
        shift_rows.append(dict(seed=seed, method=name, **ranking(te_s, s, 0.10)))
    for name, (m, cols) in models.items():
        shift_rows.append(dict(seed=seed, method=name, **ranking(te_s, proba4(m, te_s[cols])[:, 2:].sum(1), 0.10)))
    shift_rows.append(dict(seed=seed, method="TKG-HGB + relational stacking",
                           **ranking(te_s, stk.predict_proba4(te_s)[:, 2:].sum(1), 0.10)))

    # -------------------------------------------- Protocol A (index-derived labels)
    m_A = fit(make("RF", FEATURE_SETS["prototype"], seed), tr[FEATURE_SETS["prototype"]], tr.yA)
    yA_hat = m_A.predict(te[FEATURE_SETS["prototype"]])
    protoA = dict(seed=seed, rf_acc_vs_index=float(np.mean(yA_hat == te.yA)),
                  rf_wf1_vs_index=classification(te.yA, yA_hat)["weighted_f1"],
                  index_vs_truth_acc=float(np.mean(D.yA == D.y)), index_vs_truth_kappa=float(cohen_kappa_score(D.yA, D.y)),
                  index_severe_recall=float(np.mean(D.yA[D.y >= 2] >= 2)),
                  rf_index_model_ap_on_truth=float(average_precision_score(te.y >= 2, proba4(m_A, te[FEATURE_SETS["prototype"]])[:, 2:].sum(1))))

    if keep:
        # group permutation importance (drop in average precision) and occlusion faithfulness
        X2 = te[cols_prop]
        base_ap = average_precision_score(te.y >= 2, p_te)
        rng = np.random.default_rng(0); imp = []
        for g, cols in GROUPS.items():
            cols = [c for c in cols if c in X2.columns]
            if not cols:
                continue
            drops = []
            for _ in range(3):
                Xp = X2.copy(); perm = rng.permutation(len(Xp))
                Xp[cols] = Xp[cols].to_numpy()[perm]
                drops.append(base_ap - average_precision_score(te.y >= 2, proba4(m_prop, Xp)[:, 2:].sum(1)))
            imp.append(dict(group=g, mean=float(np.mean(drops)), std=float(np.std(drops))))
        extra["group_importance"] = pd.DataFrame(imp).sort_values("mean", ascending=False)
        # per-asset occlusion on the top-20 ranked assets of each window
        med = tr[cols_prop].median(numeric_only=True)
        topidx = te.assign(s=p_te).groupby("t").s.nlargest(20).index.get_level_values(-1)
        Xt = X2.loc[topidx]; base = proba4(m_prop, Xt)[:, 2:].sum(1)
        dd = {}
        for g, cols in GROUPS.items():
            cols = [c for c in cols if c in Xt.columns and c != "asset_type"]
            if not cols:
                continue
            Xo = Xt.copy(); Xo[cols] = med[cols].to_numpy()
            dd[g] = base - proba4(m_prop, Xo)[:, 2:].sum(1)
        dd = pd.DataFrame(dd, index=Xt.index)
        top_g = dd.idxmax(axis=1)
        rand_g = dd.apply(lambda r: r.sample(1, random_state=int(r.name) % 1000).iloc[0], axis=1)
        extra["faithfulness"] = dict(mean_drop_top_group=float(dd.max(axis=1).mean()), mean_drop_random_group=float(rand_g.mean()),
                                     top_group_share=top_g.value_counts(normalize=True).round(3).to_dict())
        extra["example"] = te.loc[topidx].assign(score=base, top_reason=top_g.values)[
            ["t", "asset", "asset_type", "criticality", "score", "y", "top_reason", "ts_obs", "ts_obs_drop", "kev_overdue",
             "al_ewm4", "nb_max_al4"]].head(40)
        P = P_prop
        extra["cm"] = confusion_matrix(te.y, P.argmax(1), labels=[0, 1, 2, 3]).tolist()
        extra["cm_A"] = confusion_matrix(D.y, D.yA, labels=[0, 1, 2, 3]).tolist()
        extra["tkg"] = st
        extra["runtime"] = dict(feature_extraction_ms_per_window=fe_ms, model_fit_s=fit_s, inference_ms_per_asset=inf_ms)
        extra["D"] = D; extra["Ds"] = Ds; extra["ent"] = ent
        extra["scores_te"] = te.assign(score=p_te)[["t", "asset", "y", "score"]]
    stats = dict(seed=seed, n=len(D), n_train=len(tr), n_cal=len(ca), n_test=len(te),
                 **{f"share_{k}": float(np.mean(D.y == i)) for i, k in enumerate(["low", "moderate", "high", "critical"])},
                 compromised_share=float(np.mean(D.true_stage > 0)), neglect_share=float(np.mean(D.true_neglect > 0)),
                 n_quadruples=int(len(ev)))
    return rows, rank_rows, conf, shift_rows, protoA, stats, extra


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--results", default=os.path.normpath(os.path.join(HERE, "..", "results")))
    ap.add_argument("--data", default=os.path.normpath(os.path.join(HERE, "..", "data")))
    a = ap.parse_args()
    if a.quick:
        a.seeds = 2
    os.makedirs(a.results, exist_ok=True)
    t_start = time.time()

    # canonical released data: the TKG quadruple stream and the asset-window table of one enterprise
    ent, ev, D, _, _ = build(CANONICAL_SEED)
    ev_path = os.path.join(a.results, f"tkg_quadruples_seed{CANONICAL_SEED}.csv")
    tab_path = os.path.join(a.results, f"asset_windows_seed{CANONICAL_SEED}.csv")
    ev.to_csv(ev_path, index=False, float_format="%.6g")
    D.drop(columns=[]).to_csv(tab_path, index=False, float_format="%.6g")
    data_check = {}
    for pth in (ev_path, tab_path):
        k = os.path.basename(pth); data_check[k] = {"generated_sha256": sha256(pth)}
        shipped = os.path.join(a.data, k)
        if os.path.exists(shipped):
            data_check[k]["shipped_sha256"] = sha256(shipped)
            data_check[k]["identical"] = data_check[k]["shipped_sha256"] == data_check[k]["generated_sha256"]

    R, RK, C, SH, PA, ST, extra = [], [], [], [], [], [], {}
    for s in range(a.seeds):
        print(f"seed {s}", flush=True)
        r, rk, c, sh, pa, st, ex = run_seed(1000 + s, keep=(s == 0))
        R += r; RK += rk; C += c; SH += sh; PA.append(pa); ST.append(st)
        if ex:
            extra = ex
    R, RK, C, SH, PA, ST = map(pd.DataFrame, (R, RK, C, SH, PA, ST))
    for name, d in [("main_per_seed", R), ("cause_per_seed", RK), ("conformal_stream", C), ("shift_per_seed", SH),
                    ("protocolA_per_seed", PA), ("dataset_stats_per_seed", ST)]:
        d.to_csv(os.path.join(a.results, f"{name}.csv"), index=False)
    metric_cols = [c for c in R.columns if c not in ("method", "family", "seed")]
    R.groupby(["method", "family"], sort=False)[metric_cols].agg(["mean", "std"]).to_csv(os.path.join(a.results, "table_main.csv"))
    RK.groupby(["model", "cause"], sort=False)[["recall_top10", "n"]].agg(["mean", "std", "sum"]).to_csv(os.path.join(a.results, "table_cause.csv"))
    C.groupby(["regime", "rule", "alpha"], dropna=False)[["miss", "flagged"]].agg(["mean", "std"]).to_csv(
        os.path.join(a.results, "table_conformal.csv"))
    SH.groupby("method", sort=False).agg(["mean", "std"]).drop(columns="seed", level=0).to_csv(os.path.join(a.results, "table_shift.csv"))
    extra["group_importance"].to_csv(os.path.join(a.results, "table_group_importance.csv"), index=False)
    extra["example"].to_csv(os.path.join(a.results, "table_example_priorities.csv"), index=False)

    summary = dict(seeds=a.seeds, horizon=H, weeks=WEEKS, splits=dict(train=TRAIN, cal=CAL, test=TEST), campaign_week=CAMPAIGN,
                   alphas=ALPHAS, gamma=GAMMA, budgets=BUDGETS, data_check=data_check,
                   protocol_a=PA.drop(columns="seed").agg(["mean", "std"]).to_dict(),
                   dataset=ST.drop(columns="seed").agg(["mean", "std"]).to_dict(),
                   tkg=extra["tkg"], runtime=extra["runtime"], faithfulness=extra["faithfulness"],
                   cm_proposed_seed0=extra["cm"], cm_index_vs_truth_seed0=extra["cm_A"],
                   environment=dict(python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__,
                                    sklearn=sklearn.__version__, platform=platform.platform()))
    figures.make_all(a.results, R, RK, C, SH, extra)
    summary["runtime_s"] = round(time.time() - t_start, 1)
    with open(os.path.join(a.results, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    show = ["accuracy", "macro_f1", "weighted_f1", "auc", "ap", "precision_at_B@10", "recall_at_B@10", "ndcg_at_B@10"]
    print(R.groupby("method", sort=False)[[c for c in show if c in R.columns]].mean().round(3).to_string())
    print(C.groupby(["regime", "rule", "alpha"], dropna=False)[["miss", "flagged"]].mean().round(3))
    print(PA.mean(numeric_only=True).round(3))
    print(f"done in {summary['runtime_s']} s")


if __name__ == "__main__":
    main()
