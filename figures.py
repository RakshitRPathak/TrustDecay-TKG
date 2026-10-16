"""Manuscript figures (300 dpi PNG + PDF)."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .labels import TIERS  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
GRAY, INK, MUTED = "#8a8985", "#0b0b0b", "#52514e"
TIER_COLORS = ["#cfe0f5", "#7fb0e8", "#2a78d6", "#123f78"]
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.5,
    "savefig.dpi": 300, "savefig.bbox": "tight", "figure.facecolor": "white"})
SINGLE, DOUBLE = 3.5, 7.16
ORDER = ["Audit rotation", "CVSS-max", "KEV-first", "Trust-score rule", "Prototype RF", "LR (full)", "HGB (local)",
         "HGB (local+temporal)", "TKG-HGB + relational stacking", "TKG-HGB (proposed)"]
FAMC = {"rule": GRAY, "learned": BLUE, "proposed": ORANGE}


def _save(fig, out, name):
    fig.savefig(os.path.join(out, f"{name}.png")); fig.savefig(os.path.join(out, f"{name}.pdf")); plt.close(fig)


def fig_trajectories(out, D):
    """True vs observed trust for illustrative assets of seed 0 (one compromise, one neglect, one stable)."""
    g = D.groupby("asset")
    comp = g.true_stage.max().idxmax()
    negl = g.true_neglect.mean().sort_values().index[-1]
    stable = g.ts_true.std().idxmin()
    fig, axs = plt.subplots(1, 3, figsize=(DOUBLE, 1.9), sharey=True)
    for ax, a, title in zip(axs, (comp, negl, stable), ("Compromise episode", "Maintenance neglect", "Stable asset")):
        d = D[D.asset == a].sort_values("t")
        ax.plot(d.t, d.ts_true, color=ORANGE, lw=1.6, label="True trust TS* (hidden)")
        ax.plot(d.t, d.ts_obs, color=BLUE, lw=1.2, label="Observed rule score TS_obs")
        for t in d.t[d.audited_now > 0]:
            ax.axvline(t, color=GRAY, lw=0.6, ls="--")
        ax.set_title(f"{title} (asset {a}, {d.asset_type.iloc[0]})", fontsize=7)
        ax.set_xlabel("Week")
    axs[0].set_ylabel("Trust score"); axs[0].legend(frameon=False, loc="lower left")
    _save(fig, out, "fig_trajectories")


def fig_egograph(out, ent, D):
    """Ego sub-graph of the asset with the most severe labels (seed 0): typed dependency relations."""
    cand = D[D.asset_type.isin(["server", "app", "web_app"])]
    a = cand.groupby("asset").y.mean().idxmax()
    aid = f"A{a:03d}"
    G = ent.G
    nodes = {aid} | set(G.successors(aid)) | set(G.predecessors(aid))
    for n in list(nodes):
        if n != aid and len(nodes) < 18:
            nodes |= set(list(G.successors(n))[:2]) | set(list(G.predecessors(n))[:2])
    H = G.subgraph(list(nodes)[:22]).copy()
    pos = nx.spring_layout(nx.Graph(H), seed=3, k=0.9)
    types = {n: ent.asset_type[int(n[1:])] for n in H.nodes}
    tcol = {"edge": RED, "web_app": ORANGE, "app": YELLOW, "server": BLUE, "database": VIOLET, "file_transfer": MAGENTA,
            "endpoint_pool": AQUA}
    fig, ax = plt.subplots(figsize=(SINGLE, 2.6))
    nx.draw_networkx_edges(nx.Graph(H), pos, ax=ax, edge_color="#bdbcb7", width=0.8)
    for n in H.nodes:
        ax.scatter(*pos[n], s=180 if n == aid else 70, color=tcol[types[n]], edgecolor=INK if n == aid else "white",
                   linewidth=1.2 if n == aid else 0.6, zorder=3)
    for u, v, k in H.edges(keys=True):
        if aid in (u, v):                     # label only the ego's own relations
            x, y = (pos[u] + pos[v]) / 2
            ax.text(x, y, k.replace("_", " "), fontsize=5.2, color=INK, ha="center",
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))
    ax.text(*pos[aid], f"  {aid}", fontsize=6.5, color=INK, va="center")
    for t_, c_ in tcol.items():
        ax.scatter([], [], color=c_, s=25, label=t_.replace("_", " "))
    ax.legend(frameon=False, fontsize=5.8, loc="center left", bbox_to_anchor=(1.0, 0.5), ncol=1)
    ax.axis("off")
    _save(fig, out, "fig_egograph")


def fig_main(out, R):
    g = R.groupby("method")[["ap", "recall_at_B@10", "ndcg_at_B@10"]].agg(["mean", "std"]).reindex(ORDER)
    fam = R.drop_duplicates("method").set_index("method").family
    fig, axs = plt.subplots(1, 3, figsize=(DOUBLE, 2.5), sharey=True)
    y = np.arange(len(ORDER))
    for ax, (col, lab) in zip(axs, [("ap", "Average precision (severe decay)"), ("recall_at_B@10", "Recall of severe @ 10% budget"),
                                    ("ndcg_at_B@10", "NDCG @ 10% budget")]):
        ax.barh(y, g[(col, "mean")], xerr=g[(col, "std")], color=[FAMC[fam[m]] for m in ORDER], height=0.7,
                error_kw=dict(lw=0.6, capsize=1.2))
        ax.set_xlabel(lab); ax.grid(axis="y", visible=False)
    axs[0].set_yticks(y, ORDER); axs[0].invert_yaxis()
    _save(fig, out, "fig_main")


def fig_cause(out, RK):
    g = RK.groupby(["cause", "model"]).recall_top10.mean().unstack()
    causes = ["no compromise (control erosion)", "exposure exploit", "credential campaign", "lateral pivot"]
    g = g.reindex(causes)
    fig, ax = plt.subplots(figsize=(SINGLE, 2.3))
    x = np.arange(len(causes)); w = 0.27
    for i, (m, c) in enumerate([("HGB (local+temporal)", GRAY), ("TKG-HGB (proposed)", ORANGE), ("TKG-HGB + relational stacking", BLUE)]):
        ax.bar(x + (i - 1) * w, g[m] * 100, w * 0.92, color=c, label=m)
    ax.set_xticks(x, ["control\nerosion", "exposure\nexploit", "credential\ncampaign", "lateral\npivot"])
    ax.set_ylabel("Severe cases in top-10% [%]"); ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6)
    _save(fig, out, "fig_cause")


def fig_conformal(out, C, campaign_week):
    fig, axs = plt.subplots(1, 3, figsize=(DOUBLE, 2.3), gridspec_kw=dict(wspace=0.45))
    # (a) realised miss vs nominal alpha
    for (regime, rule), c, ls in [(("nominal", "split conformal"), BLUE, "--"), (("campaign", "split conformal"), BLUE, "-"),
                                  (("nominal", "delayed ACI"), ORANGE, "--"), (("campaign", "delayed ACI"), ORANGE, "-")]:
        d = C[(C.regime == regime) & (C.rule == rule)].groupby("alpha").miss.mean()
        axs[0].plot(d.index, d.values, color=c, ls=ls, marker="o", ms=3, lw=1.4, label=f"{rule}, {regime}")
    axs[0].plot([0.08, 0.32], [0.08, 0.32], color=GRAY, lw=0.8, ls=":", label="target")
    axs[0].set_xlabel("Target miss rate alpha"); axs[0].set_ylabel("Realised miss rate")
    axs[0].legend(frameon=False, fontsize=5.4, loc="upper left")
    # (b,c) timeline at alpha = 0.2 under the campaign
    for rule, c in [("split conformal", BLUE), ("delayed ACI", ORANGE)]:
        d = C[(C.regime == "campaign") & (C.rule == rule) & (C.alpha == 0.2)].groupby("t")[["miss", "flagged"]].mean()
        axs[1].plot(d.index, d.miss * 100, color=c, lw=1.5, marker="o", ms=2.5, label=rule)
        axs[2].plot(d.index, d.flagged * 100, color=c, lw=1.5, marker="o", ms=2.5, label=rule)
    for ax in axs[1:]:
        ax.axvline(campaign_week, color=RED, lw=0.8, ls="--"); ax.set_xlabel("Week")
    axs[1].axhline(20, color=INK, lw=0.7, ls=":"); axs[1].set_ylabel("Missed severe [%], alpha = 0.2")
    axs[2].set_ylabel("Assets flagged for audit [%]"); axs[2].legend(frameon=False, fontsize=6, loc="lower right")
    axs[1].text(campaign_week + 0.3, axs[1].get_ylim()[0] + 0.3, "zero-day campaign", fontsize=6, color=RED, va="bottom")
    _save(fig, out, "fig_conformal")


def fig_groups(out, G):
    g = G.sort_values("mean")
    fig, ax = plt.subplots(figsize=(SINGLE, 2.2))
    ax.barh(g.group, g["mean"], xerr=g["std"], color=BLUE, height=0.7, error_kw=dict(lw=0.6, capsize=1.2))
    ax.set_xlabel("Drop in average precision when the group is permuted"); ax.grid(axis="y", visible=False)
    _save(fig, out, "fig_groups")


def fig_shift(out, SH):
    g = SH.groupby("method")[["ap"]].agg(["mean", "std"]).reindex(ORDER)
    fig, ax = plt.subplots(figsize=(SINGLE, 2.4))
    y = np.arange(len(ORDER))
    fam = {m: ("proposed" if "proposed" in m else ("rule" if m in ORDER[:4] else "learned")) for m in ORDER}
    ax.barh(y, g[("ap", "mean")], xerr=g[("ap", "std")], color=[FAMC[fam[m]] for m in ORDER], height=0.7,
            error_kw=dict(lw=0.6, capsize=1.2))
    ax.set_yticks(y, ORDER); ax.invert_yaxis(); ax.set_xlabel("Average precision under the campaign regime")
    ax.grid(axis="y", visible=False)
    _save(fig, out, "fig_shift")


def make_all(out, R, RK, C, SH, extra, campaign_week=64):
    fig_trajectories(out, extra["D"])
    fig_egograph(out, extra["ent"], extra["D"])
    fig_main(out, R)
    fig_cause(out, RK)
    fig_conformal(out, C, campaign_week)
    fig_groups(out, extra["group_importance"])
    fig_shift(out, SH)
