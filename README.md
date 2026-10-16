# TrustDecay-TKG

Code and data for the manuscript

> R. R. Pathak, S. Vibhanshu, and H. Haidros Rahima Manzil, "Forecasting Information-System Trust Decay on Temporal Knowledge Graphs With Delayed-Feedback Conformal Audit Selection," submitted to *IEEE Access*, 2026.

The framework turns a continuous stream of audit and security events into **forecasts of how far each IT asset's trust will have decayed since its last audit**. It then converts those forecasts into an audit work queue with an explicit, adaptive guarantee on how many deteriorating assets are missed.

* **Assurance-referenced, latent ground truth.** A simulator keeps a hidden true trust state for every asset. That state covers configuration drift, privilege creep, backup failures, patch backlog, a maintenance-neglect regime and a staged compromise process with dwell times and targeted lateral pivots. Labels measure the decline of this hidden trust from the asset's last audit over the next H = 4 windows. Predictors only see emitted audit events.
* **Temporal knowledge graph (TKG).** Events become quadruples (s, r, o, t). The graph is updated incrementally, so history is preserved. Trust features come in local, temporal (decayed histories, decay since assurance) and relational (dependency and shared-identity neighbourhood) families.
* **TKG-Stack.** A two-stage relational-stacking gradient-boosting model.
* **Delayed-feedback adaptive conformal audit selection.** ACI adapted to audit outcomes that arrive H windows late. The long-run rate of missed severe assets converges to the target α for any data sequence.
* **Index-label audit (Protocol A).** Quantifies how far labels computed from the observed rule-based trust score disagree with the hidden truth.

| Item | Location |
|---|---|
| Source code (this repository) | https://github.com/RakshitRPathak/TrustDecay-TKG |
| Archived release + data (Zenodo) | https://doi.org/10.5281/zenodo.XXXXXXX *(fill in after the first release)* |
| Reproducible capsule (Code Ocean) | https://doi.org/10.24433/CO.XXXXXXX.v1 *(fill in after publishing the capsule)* |
| Licence | Code: MIT. Data: CC BY 4.0 |

## Layout (Code Ocean compatible)

```
code/
  run                 # Code Ocean entry point -> python main.py
  main.py             # reproduces every table and figure into ../results
  trustkg/
    enterprise.py     # synthetic enterprise: assets, users, typed dependency relations (hidden asset traits)
    simulate.py       # latent trust dynamics + audit/security event emission (TKG quadruples), zero-day campaign
    tkg.py            # incremental TKG store and local / temporal / graph trust-feature extraction
    labels.py         # assurance-referenced trust-decay targets (true and index-derived)
    models.py         # audit-practice baselines, prototype RF, LR/HGB, TKG-Stack (relational stacking)
    conformal.py      # split conformal and delayed-feedback adaptive conformal audit selection
    evaluate.py       # classification, ranking and audit-budget metrics
    figures.py        # manuscript figures
data/
  tkg_quadruples_seed2026.csv     # released canonical event stream (one enterprise, 80 weeks)
  asset_windows_seed2026.csv      # released asset-window table with features and labels
environment/                      # pinned Python 3.13 environment (Dockerfile, requirements.txt)
results/                          # written by the run (empty in the repository)
results_reference/                # outputs of the reference run reported in the paper
docs/fig_framework.*              # framework diagram (Graphviz source + render)
```

## Reproduce

```bash
pip install -r requirements.txt
cd code
python main.py           # full study: 10 enterprises x 400 assets x 80 weekly windows
python main.py --quick   # smoke test: 2 enterprises
```

On Code Ocean, press **Reproducible Run**. All randomness is seeded (enterprise seeds 1000–1009, canonical release seed 2026). `results/summary.json` records library versions, the runtime and a SHA-256 check that the regenerated canonical data match the files in `data/`.

## Data dictionary

`tkg_quadruples_seed2026.csv`: one row per quadruple, with columns `t` (week), `s` (subject), `r` (relation), `o` (object), `value` (count or measurement), `cvss` and `kev` (known-exploited flag). The relations are `affects`, `remediates`, `listed_known_exploited`, `failed_login`, `config_change`, `drift_from_baseline`, `backup_failed`, `privilege_escalation`, `alert`, `violated_by`, `owner_changed`, `audited`, `accesses`, plus the static `runs_on`, `hosts`, `reads`, `protects`, `communicates_with` and `connects_to`.

`asset_windows_seed2026.csv`: one row per asset and week. All observed feature columns are model inputs. The `true_*` and `ts_true` columns are hidden simulator truth and must never be used as inputs. The label columns are `tds_true` with tier `y` (Low/Moderate/High/Critical = 0–3), plus the index-derived `tds_obs` with tier `yA`.

## Scope

This is a TRL-4 study on synthetic data. It does not replace professional audit judgement or a production SIEM.
