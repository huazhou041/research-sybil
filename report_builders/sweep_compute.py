"""Follow-up cutoff sweep for Sybil-1.5.

Reconstruction of the original sweep_compute.py (which was not kept alongside
test_flat/sweep.json). The five method definitions are recovered from the report
prose + the variant labels in the v2 sweep.json, then *validated* by replaying
them against the v2 predictions and diffing cell-for-cell against that file --
see validate_vs_v2.py. Only once that diff is clean are the same definitions
applied to the Sybil-1.5 runs, which keeps the two reports comparable.

Cohort: the 14,549-scan modelling set (union of the 10 cross_val fold test.csv).
Scores: pred_6 (6-year cumulative probability) except method (a), which uses each
year's own column. `kind` selects calibrated (pred_k) vs raw (pred_k_raw).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RESULT = HERE.parent  # res_20260801 itself is the result dir for this axial-fix rerun
DATA = RESULT / "predictions"
MAXT, NFOLD = 6, 10
TAUS = [0.0, 0.5, 1.0, 1.5, 2.0]

METHOD_DESC = {
    "a": "Uniform observed-time filter, then per-year cumulative-mask AUC (years 1-6)",
    "b": "Flat binary: pos = diagnosed after cutoff; neg = never diagnosed, followed past cutoff",
    "c": "As (b), negative pool tightened to a minimum follow-up",
    "d": "Horizon matched: pos diagnosed in (cutoff, h]; neg followed at least h",
    "e": "Patient level: collapse each patient's eligible scans to one score",
}


def auc(y, s):
    """Rank AUC with tie correction. Returns None if either class is empty."""
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return None
    r = pd.Series(s).rank().values
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def load_cohort():
    """Cohort with BOTH time scales, which the original sweep mixes deliberately.

    `obs` is continuous observed time in years (time to diagnosis for cancers,
    time to last encounter otherwise) -- this is what the cutoff tau and the
    positive definitions use, and it is why tau can take half-year values.
    `tbin` is the discrete 0-5 `time_at_event` bucket that the Sybil training
    convention uses, and the per-year negative masks in method (a) key off it.
    """
    b = pd.read_csv(DATA / "cohort_base_14549.csv")
    b["ct_id"] = b["ct_id"].astype(str)
    dt = pd.to_datetime(b["ct_date"], errors="coerce")
    b["year_ct"] = dt.dt.year
    b["cancer"] = b["cancer"].astype(int)
    b["obs"] = b["time_ct_to_last_event_or_followup"].astype(float)
    b["tbin"] = b["time_at_event"].astype(int)
    return b


def load_scores(path, key):
    """-> dict ct_id -> (calibrated[6], raw[6]).

    v2 files key scans by `patient_id`; the v15 files carry an explicit `ct_id`.
    Both are the same identifier space (the CT folder name).
    """
    s = pd.read_csv(path)
    ids = s[key].astype(str)
    cal = s[[f"pred_{t + 1}" for t in range(MAXT)]].values
    raw = s[[f"pred_{t + 1}_raw" for t in range(MAXT)]].values
    return {c: (cal[i], raw[i]) for i, c in enumerate(ids)}


def attach(base, scores, kind):
    """Add score columns p1..p6 for the requested kind; drop unscored scans."""
    j = 0 if kind == "calibrated" else 1
    mat = np.full((len(base), MAXT), np.nan)
    for i, c in enumerate(base["ct_id"].values):
        v = scores.get(c)
        if v is not None:
            mat[i] = v[j]
    out = base.copy()
    for t in range(MAXT):
        out[f"p{t + 1}"] = mat[:, t]
    return out


def negyr(df):
    """Median scan year of the negative pool -- the control-pool drift probe."""
    v = df.loc[df["_y"] == 0, "year_ct"].dropna()
    return float(v.median()) if len(v) else None


# ---------------------------------------------------------------- methods

def _mask_year(e, k, masking="pure"):
    """Per-year mask. Positives use continuous time, negatives use the discrete bin.

    The two maskings share the positives and differ only in the control pool:
      pure       -> negatives are never-diagnosed only, so a patient who develops
                    cancer *later* than year k is dropped from the comparison.
      cumulative -> negatives are everyone still at risk at year k who is not yet a
                    positive, so those later-diagnosed patients are counted as
                    controls (and the model is penalised for scoring them high).
    """
    pos = (e["cancer"] == 1) & (e["obs"] <= k)
    at_risk = e["tbin"] >= k - 1
    if masking == "pure":
        neg = (e["cancer"] == 0) & at_risk
    else:
        neg = at_risk & ~pos
    return pos, neg


def method_a(df, tau, masking="pure"):
    """Observed-time filter, then Sybil-convention cumulative masking per year."""
    e = df[df["obs"] >= tau]
    per_year = []
    for k in range(1, MAXT + 1):
        pos, neg = _mask_year(e, k, masking)
        m = pos | neg
        per_year.append(auc(pos[m].astype(int), e.loc[m, f"p{k}"]))
    folds = []
    for f in range(NFOLD):
        ef = e[e["fold"] == f]
        vals = []
        for k in range(1, MAXT + 1):
            pos, neg = _mask_year(ef, k, masking)
            m = pos | neg
            a = auc(pos[m].astype(int), ef.loc[m, f"p{k}"])
            if a is not None:
                vals.append(a)
        if vals:
            folds.append(float(np.mean(vals)))
    ok = [v for v in per_year if v is not None]
    # headline counts are cohort-level: positives are the 6-year cases, and the
    # negative count is everything else in the eligible set (not the year-6 mask)
    pos6 = (e["cancer"] == 1) & (e["obs"] <= MAXT)
    sub = e.copy()
    sub["_y"] = pos6.astype(int)
    return {
        "per_year": per_year,
        "n": int(len(e)),
        "npos": int(pos6.sum()),
        "nneg": int(len(e) - pos6.sum()),
        "auc": float(np.mean(ok)) if ok else None,
        "foldavg": float(np.mean(folds)) if folds else None,
        "negyr": negyr(sub),
    }


def _flat(df, tau, horizon=None, negfu=None, masking="pure"):
    """Shared skeleton for (b)/(c): one fixed eligible set, scored once on pred_6.

    Eligibility is `obs > tau` for both classes. Within it, a scan is positive if
    it is a cancer diagnosed within `horizon`; cancers diagnosed *beyond* the
    horizon stay in the study as negatives rather than being dropped, which is
    what makes the 'dx within 6 yr' and 'dx any horizon' variants differ only in
    where those late cases are counted.
    """
    if negfu is None:
        # (b): one eligible set for both classes; late cancers stay in as negatives
        e = df[df["obs"] > tau]
        pos = (e["cancer"] == 1)
        if horizon is not None:
            pos &= e["obs"] <= horizon
        neg = ~pos
    else:
        # (c): the explicit follow-up requirement replaces the cutoff for controls,
        # so the negative pool is drawn from the whole cohort and does not move
        # with tau -- only the positive side is cut back.
        e = df
        pos = (e["cancer"] == 1) & (e["obs"] > tau)
        if horizon is not None:
            pos &= e["obs"] <= horizon
        meets_fu = e["obs"] >= negfu
        if masking == "pure":
            neg = (e["cancer"] == 0) & meets_fu
        else:
            neg = meets_fu & ~pos
    m = pos | neg
    sub = e[m].copy()
    sub["_y"] = pos[m].astype(int)
    return {
        "n": int(m.sum()),
        "npos": int(pos.sum()),
        "nneg": int(neg.sum()),
        "auc": auc(sub["_y"], sub["p6"]),
        "negyr": negyr(sub),
    }


def method_b(df, tau):
    return [("dx within 6 yr", _flat(df, tau, horizon=MAXT)),
            ("dx any horizon", _flat(df, tau, horizon=None))]


def method_c(df, tau, masking="pure"):
    return [(f"neg fu>={h} yr", _flat(df, tau, horizon=MAXT, negfu=h, masking=masking))
            for h in range(1, MAXT + 1)]


def method_d(df, tau, masking="pure"):
    """Both sides refer to the same horizon h."""
    out = []
    for h in range(1, MAXT + 1):
        if h <= tau:
            continue
        pos = (df["cancer"] == 1) & (df["obs"] > tau) & (df["obs"] <= h)
        meets_fu = df["obs"] >= h
        # pure drops anyone who is ever diagnosed; cumulative keeps the
        # later-diagnosed as controls provided they clear the same follow-up bar
        neg = ((df["cancer"] == 0) & meets_fu) if masking == "pure" else (meets_fu & ~pos)
        m = pos | neg
        sub = df[m].copy()
        sub["_y"] = pos[m].astype(int)
        lab = f"pos ({tau:g},{h}] / neg fu>={h}"
        out.append((lab, {"n": int(m.sum()), "npos": int(pos.sum()),
                          "nneg": int(neg.sum()), "auc": auc(sub["_y"], sub["p6"]),
                          "negyr": negyr(sub)}))
    return out


def method_e(df, tau):
    """Collapse each patient's eligible scans to a single score."""
    # cancers diagnosed beyond the 6-year horizon are dropped rather than counted
    # as controls, so a patient whose only eligible scan is such a case leaves the
    # analysis entirely instead of entering it with an unsafe negative label
    e = df[(df["obs"] > tau) & ~((df["cancer"] == 1) & (df["obs"] > MAXT))].copy()
    e["_ispos"] = ((e["cancer"] == 1) & (e["obs"] <= MAXT)).astype(int)
    out = []
    for agg in ["first scan", "mean", "median", "max", "min"]:
        if agg == "first scan":
            # Take the patient's genuinely first eligible scan and label THAT scan:
            # positive iff a diagnosis follows it within 6 years. Selecting from `e`
            # instead would be biased -- `e` has already dropped case scans taken
            # more than 6 years before diagnosis, so cases would be represented by a
            # later, easier scan while controls keep their true baseline.
            # mergesort keeps the order stable when a patient has same-day scans.
            b = df[df["obs"] > tau].sort_values("ct_date", kind="mergesort")
            f = b.groupby("pat_id").first()
            y = ((f["cancer"] == 1) & (f["obs"] <= MAXT)).astype(int)
            g = pd.DataFrame({"_y": y.values, "year_ct": f["year_ct"].values})
            score = f["p6"].values
        else:
            gg = e.groupby("pat_id")
            g = gg.agg(_y=("_ispos", "max"), year_ct=("year_ct", "median")).reset_index()
            score = g["pat_id"].map(getattr(gg["p6"], agg)()).values
        out.append((agg, {"n": int(len(g)), "npos": int((g["_y"] == 1).sum()),
                          "nneg": int((g["_y"] == 0).sum()),
                          "auc": auc(g["_y"].values, score), "negyr": negyr(g)}))
    return out


# ---------------------------------------------------------------- driver

def compute(models, base):
    rows = []
    for name, path, key in models:
        sc = load_scores(path, key)
        for kind in ["calibrated", "raw"]:
            df = attach(base, sc, kind)
            for tau in TAUS:
                # (a), (c) and (d) are computed under both control-pool rules;
                # (b) and (e) have only one sensible reading, so they carry "pure"
                for masking in ["pure", "cumulative"]:
                    r = method_a(df, tau, masking)
                    rows.append(dict(method="a", tau=tau, model=name, kind=kind,
                                     variant="per-year", masking=masking, **r))
                    for meth, fn in [("c", method_c), ("d", method_d)]:
                        for variant, res in fn(df, tau, masking):
                            rows.append(dict(method=meth, tau=tau, model=name,
                                             kind=kind, variant=variant,
                                             masking=masking, **res))
                for meth, fn in [("b", method_b), ("e", method_e)]:
                    for variant, res in fn(df, tau):
                        rows.append(dict(method=meth, tau=tau, model=name,
                                         kind=kind, variant=variant,
                                         masking="pure", **res))
    return rows


def main():
    base = load_cohort()
    # sybil_v1_reprocess (the previous-generation runs on reprocessed volumes) leads,
    # so every table reads baseline-first and the Sybil-1.5 rows are compared against
    # it rather than standing alone
    models = [
        ("sybil_v1_reprocess single", DATA / "sybil_v1_reprocess_single.csv", "patient_id"),
        ("sybil_v1_reprocess ensemble", DATA / "sybil_v1_reprocess_ensemble.csv", "patient_id"),
        ("sybil_v15 seed0", DATA / "sybil_v15_seed0.csv", "ct_id"),
        ("sybil_v15 seed1", DATA / "sybil_v15_seed1.csv", "ct_id"),
        ("sybil_v15 seed2", DATA / "sybil_v15_seed2.csv", "ct_id"),
        ("sybil_v15 ensemble", DATA / "sybil_v15_ensemble.csv", "ct_id"),
    ]
    rows = compute(models, base)
    out = {"meta": {"taus": TAUS, "models": [m[0] for m in models],
                    "methods": METHOD_DESC}, "rows": rows}
    p = RESULT / "sweep.json"
    json.dump(out, open(p, "w"))
    print(f"-> {p}  rows={len(rows)}")

    # flat export, same schema as the sybil_v1_reprocess sweep_all_cells.csv
    cols = ["method", "tau", "model", "kind", "variant", "masking", "npos", "nneg", "n",
            "negyr", "auc", "foldavg", "year1", "year2", "year3", "year4",
            "year5", "year6"]
    flat = []
    for r in rows:
        d = {c: r.get(c) for c in cols}
        for i, v in enumerate(r.get("per_year") or []):
            d[f"year{i + 1}"] = v
        flat.append(d)
    q = RESULT / "sweep_all_cells.csv"
    pd.DataFrame(flat, columns=cols).to_csv(q, index=False)
    print(f"-> {q}  rows={len(flat)}")


if __name__ == "__main__":
    main()
