"""Synthetic enterprise topology: assets, users and dependency relations.

The topology is the *static skeleton* of the temporal knowledge graph (TKG):
applications run on servers, servers host databases, applications read
databases, edge devices protect segments, servers communicate with each
other and users access applications and servers. Asset-level latent
properties (patch discipline, operational maturity, logging coverage) are
hidden from every predictor.
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import scipy.sparse as sp

ASSET_TYPES = ["edge", "web_app", "app", "server", "database", "file_transfer", "endpoint_pool"]
TYPE_SHARE = [0.05, 0.12, 0.18, 0.30, 0.15, 0.03, 0.17]
VULN_RATE = {"edge": 0.40, "web_app": 0.55, "app": 0.45, "server": 0.60, "database": 0.30,
             "file_transfer": 0.50, "endpoint_pool": 0.80}           # new findings / week
INTERNET_P = {"edge": 1.0, "web_app": 0.6, "app": 0.05, "server": 0.05, "database": 0.0,
              "file_transfer": 1.0, "endpoint_pool": 0.0}
CRIT_RANGE = {"edge": (3, 5), "web_app": (2, 4), "app": (1, 4), "server": (2, 4), "database": (3, 5),
              "file_transfer": (4, 5), "endpoint_pool": (1, 3)}


@dataclass
class Enterprise:
    n_assets: int
    n_users: int
    asset_type: np.ndarray        # str
    internet: np.ndarray          # bool
    criticality: np.ndarray       # 1..5
    patch_median_wk: np.ndarray   # latent
    kev_policy: np.ndarray        # latent bool: follows accelerated KEV remediation
    maturity: np.ndarray          # latent 0..1
    coverage: np.ndarray          # latent logging coverage 0.6..1
    G: nx.MultiDiGraph            # typed dependency relations (asset ids "A###")
    A_dep: sp.csr_matrix          # symmetric asset-asset adjacency
    user_access: sp.csr_matrix    # user x asset entitlement matrix
    is_admin: np.ndarray


def build_enterprise(rng: np.random.Generator, n_assets: int = 400, n_users: int = 250) -> Enterprise:
    types = rng.choice(ASSET_TYPES, size=n_assets, p=TYPE_SHARE)
    internet = np.array([rng.random() < INTERNET_P[t] for t in types])
    crit = np.array([rng.integers(CRIT_RANGE[t][0], CRIT_RANGE[t][1] + 1) for t in types])
    patch = rng.lognormal(np.log(4.5), 0.55, n_assets) * np.where(types == "edge", 1.5, 1.0)
    kev_policy = rng.random(n_assets) < 0.7
    maturity = rng.beta(4, 2, n_assets)
    coverage = rng.uniform(0.6, 1.0, n_assets)

    G = nx.MultiDiGraph()
    ids = [f"A{i:03d}" for i in range(n_assets)]
    for i, t in enumerate(types):
        G.add_node(ids[i], kind="asset", asset_type=t)
    idx = {t: np.where(types == t)[0] for t in ASSET_TYPES}
    servers, dbs = idx["server"], idx["database"]
    apps = np.concatenate([idx["web_app"], idx["app"], idx["file_transfer"]])

    def rel(a, r, b):
        G.add_edge(ids[a], ids[b], key=r, relation=r)

    for a in apps:
        rel(a, "runs_on", rng.choice(servers))
        for d in rng.choice(dbs, size=rng.integers(1, 3), replace=False):
            rel(a, "reads", d)
    for d in dbs:
        rel(rng.choice(servers), "hosts", d)
    for e in idx["edge"]:   # each edge device protects a segment
        seg = rng.choice(np.concatenate([servers, apps]), size=min(12, len(servers)), replace=False)
        for s in seg:
            rel(e, "protects", s)
    for s in servers:       # sparse east-west communication
        for s2 in rng.choice(servers, size=rng.integers(1, 4), replace=False):
            if s2 != s:
                rel(s, "communicates_with", s2)
    for p in idx["endpoint_pool"]:
        for a in rng.choice(apps, size=rng.integers(2, 6), replace=False):
            rel(p, "connects_to", a)

    rows, cols = [], []
    pos = {n: i for i, n in enumerate(ids)}
    for u, v in G.edges():
        rows += [pos[u], pos[v]]; cols += [pos[v], pos[u]]
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_assets, n_assets))
    A.data[:] = 1.0

    is_admin = rng.random(n_users) < 0.15
    ur, uc = [], []
    for u in range(n_users):
        pool = np.concatenate([servers, dbs, apps]) if is_admin[u] else np.concatenate([apps, idx["endpoint_pool"]])
        k = rng.integers(4, 12) if is_admin[u] else rng.integers(2, 6)
        for a in rng.choice(pool, size=min(k, len(pool)), replace=False):
            ur.append(u); uc.append(a)
    UA = sp.csr_matrix((np.ones(len(ur)), (ur, uc)), shape=(n_users, n_assets))
    return Enterprise(n_assets, n_users, types, internet, crit, patch, kev_policy, maturity, coverage, G, A, UA, is_admin)
